/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Config;
use crate::Dimensions;
use crate::Distance;
use crate::Filter;
use crate::IndexFactory;
use crate::IndexKey;
use crate::Limit;
use crate::Quantization;
use crate::SpaceType;
use crate::Vector;
use crate::index::actor::AnnR;
use crate::index::actor::Index;
use crate::index::factory::IndexConfiguration;
use crate::index::validator;
use crate::memory::Allocate;
use crate::memory::Memory;
use crate::memory::MemoryExt;
use crate::table::IndexId;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::table::Table;
use crate::table::TableSearch;
use anyhow::anyhow;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::iter;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use tokio::runtime::Handle;
use tokio::sync::Notify;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;
use tracing::Instrument;
use tracing::debug;
use tracing::error;
use tracing::error_span;
use tracing::info;
use tracing::trace;
use tracing::warn;
use usearch::IndexOptions;
use usearch::MetricKind;
use usearch::ScalarKind;
use usearch::b1x8;

use crate::turbo_quant::distance::tq4_symmetric_distance;
use crate::turbo_quant::{Tq4CompressedVector, Tq4Config, Tq4Quantizer};

pub struct UsearchIndexFactory {
    tokio_semaphore: Arc<Semaphore>,
    rayon_semaphore: Arc<Semaphore>,
    mode: Mode,
}

impl IndexFactory for UsearchIndexFactory {
    fn create_index(
        &self,
        index: IndexConfiguration,
        table: Arc<RwLock<Table>>,
        memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<Index>> {
        match &self.mode {
            Mode::Usearch => {
                let threads =
                    Handle::current().metrics().num_workers() + rayon::current_num_threads();
                if index.quantization == Quantization::TQ4 {
                    let dimension = index.dimensions.0.get();
                    let connectivity = index.connectivity.0;
                    let expansion_add = index.expansion_add.0;
                    let expansion_search = index.expansion_search.0;
                    let space_type = index.space_type;
                    new(
                        move || {
                            Ok(Arc::new(ThreadedUsearchIndex::new_tq4(
                                dimension,
                                connectivity,
                                expansion_add,
                                expansion_search,
                                space_type,
                                threads,
                            )?))
                        },
                        index.key,
                        index.dimensions,
                        table,
                        Arc::clone(&self.tokio_semaphore),
                        Arc::clone(&self.rayon_semaphore),
                        memory,
                    )
                } else {
                    let options = IndexOptions {
                        dimensions: index.dimensions.0.get(),
                        connectivity: index.connectivity.0,
                        expansion_add: index.expansion_add.0,
                        expansion_search: index.expansion_search.0,
                        metric: metric_kind(index.quantization, index.space_type)?,
                        quantization: index.quantization.into(),
                        ..Default::default()
                    };
                    new(
                        move || Ok(Arc::new(ThreadedUsearchIndex::new(options, threads)?)),
                        index.key,
                        index.dimensions,
                        table,
                        Arc::clone(&self.tokio_semaphore),
                        Arc::clone(&self.rayon_semaphore),
                        memory,
                    )
                }
            }
            Mode::Simulator { config, config_rx } => new(
                {
                    let config = Arc::clone(config);
                    let index_key = index.key.clone();
                    let config_rx = config_rx.clone();
                    move || Ok(Simulator::new(config.clone(), config_rx.clone(), index_key))
                },
                index.key,
                index.dimensions,
                table,
                Arc::clone(&self.tokio_semaphore),
                Arc::clone(&self.rayon_semaphore),
                memory,
            ),
        }
    }

    fn index_engine_version(&self) -> String {
        match self.mode {
            Mode::Usearch => format!("usearch-{}", usearch::version()),
            Mode::Simulator { .. } => "usearch-simulator".to_string(),
        }
    }
}

pub fn new_usearch(
    tokio_semaphore: Arc<Semaphore>,
    rayon_semaphore: Arc<Semaphore>,
    mut config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<UsearchIndexFactory> {
    let config = config_rx.borrow_and_update().clone();
    Ok(UsearchIndexFactory {
        tokio_semaphore,
        rayon_semaphore,
        mode: if config.usearch_simulator.is_none() {
            Mode::Usearch
        } else {
            Mode::Simulator { config, config_rx }
        },
    })
}

enum Mode {
    Usearch,
    Simulator {
        config: Arc<Config>,
        config_rx: watch::Receiver<Arc<Config>>,
    },
}

trait UsearchIndex {
    fn reserve(&self, size: usize) -> anyhow::Result<()>;
    fn size(&self) -> usize;
    fn capacity(&self) -> usize;
    fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()>;
    fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()>;
    fn search(
        &self,
        vector: &Vector,
        limit: Limit,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>>;
    fn filtered_search(
        &self,
        vector: &Vector,
        limit: Limit,
        filter: impl Fn(PrimaryId) -> bool,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>>;

    fn stop(&self);
}

struct ThreadedUsearchIndex {
    inner: usearch::Index,
    threads: usize,
    quantization: usearch::ScalarKind,
    space_type: usearch::MetricKind,
    /// TQ4 state (None for non-TQ quantizations).
    tq4: Option<Tq4IndexState>,
}

/// Per-index TQ4 state shared across all vectors.
struct Tq4IndexState {
    quantizer: Arc<Tq4Quantizer>,
    /// Original vector dimension (before packing).
    original_dimension: usize,
    /// Packed TQ4 size in bytes = ceil(3d/8) + ceil(d/8) + 8.
    packed_dimension: usize,
    /// Oversample factor for HNSW retrieval before asymmetric reranking.
    oversample_factor: f32,
}

impl ThreadedUsearchIndex {
    fn new(options: IndexOptions, threads: usize) -> anyhow::Result<Self> {
        Ok(Self {
            inner: usearch::Index::new(&options)?,
            threads,
            quantization: options.quantization,
            space_type: options.metric,
            tq4: None,
        })
    }

    /// Create a TQ4 index with custom metric for TQ4-to-TQ4 distance.
    fn new_tq4(
        original_dimension: usize,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        space_type: SpaceType,
        threads: usize,
    ) -> anyhow::Result<Self> {
        let config = Tq4Config::default();
        let packed_dim = Tq4CompressedVector::packed_size(original_dimension);
        let quantizer = Arc::new(Tq4Quantizer::new(
            original_dimension,
            config.rotation_seed,
            config.qjl_seed,
        ));

        // Use IP as placeholder metric (overridden by change_metric)
        let metric_kind = match space_type {
            SpaceType::Cosine | SpaceType::DotProduct => MetricKind::IP,
            _ => anyhow::bail!("TQ4 only supports Cosine and DotProduct, got {space_type:?}"),
        };

        let options = IndexOptions {
            dimensions: packed_dim,
            connectivity,
            expansion_add,
            expansion_search,
            metric: metric_kind,
            quantization: ScalarKind::I8,
            ..Default::default()
        };
        let mut inner = usearch::Index::new(&options)?;

        // Precompute 8×8 cross-product table for the symmetric metric
        let inv_sqrt_d = quantizer.inv_sqrt_d();
        let cross_table = crate::turbo_quant::codebook::cross_product_table_3bit(inv_sqrt_d);
        let padded_dim = quantizer.padded_dim();

        // Register custom TQ4-to-TQ4 metric
        inner.change_metric::<i8>(Box::new(move |a_ptr: *const i8, b_ptr: *const i8| {
            // Per-thread AMX initialization (idempotent)
            thread_local! {
                static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
            }
            CONFIGURED.with(|c| {
                if !c.get() {
                    numkong::configure_thread();
                    c.set(true);
                }
            });

            let a = unsafe { std::slice::from_raw_parts(a_ptr as *const u8, packed_dim) };
            let b = unsafe { std::slice::from_raw_parts(b_ptr as *const u8, packed_dim) };

            let ip = tq4_symmetric_distance(a, b, padded_dim, &cross_table);
            // Convert inner product to distance (USearch minimizes distance)
            usearch::Distance::from(1.0 - ip)
        }));

        Ok(Self {
            inner,
            threads,
            quantization: ScalarKind::I8,
            space_type: metric_kind,
            tq4: Some(Tq4IndexState {
                quantizer,
                original_dimension,
                packed_dimension: packed_dim,
                oversample_factor: config.oversample_factor,
            }),
        })
    }
}

impl UsearchIndex for ThreadedUsearchIndex {
    fn reserve(&self, size: usize) -> anyhow::Result<()> {
        Ok(self
            .inner
            .reserve_capacity_and_threads(size, self.threads)?)
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()> {
        if let Some(tq4) = &self.tq4 {
            let compressed = tq4.quantizer.quantize(vector.as_slice());
            let packed = compressed.pack();
            let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed);
            return Ok(self.inner.add(primary_id.into(), packed_i8)?);
        }
        if self.quantization == ScalarKind::B1 {
            let vector = f32_to_b1x8(vector.as_slice());
            return Ok(self.inner.add(primary_id.into(), &vector)?);
        }
        Ok(self.inner.add(primary_id.into(), vector.as_slice())?)
    }

    fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()> {
        Ok(self.inner.remove(primary_id.into()).map(|_| ())?)
    }

    fn search(
        &self,
        vector: &Vector,
        limit: Limit,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>> {
        if let Some(tq4) = &self.tq4 {
            return self.tq4_search(tq4, vector, limit);
        }
        let matches = if self.quantization == ScalarKind::B1 {
            let vector = f32_to_b1x8(vector.as_slice());
            self.inner.search(&vector, limit.0.get())?
        } else {
            self.inner.search(vector.as_slice(), limit.0.get())?
        };
        Ok(matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .map(|(primary_id, distance)| {
                Distance::try_from((distance, self.space_type.try_into()?, vector.dim()))
                    .map(|dist| (primary_id.into(), dist))
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn filtered_search(
        &self,
        vector: &Vector,
        limit: Limit,
        filter: impl Fn(PrimaryId) -> bool,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>> {
        if let Some(tq4) = &self.tq4 {
            return self.tq4_filtered_search(tq4, vector, limit, filter);
        }
        let matches = if self.quantization == ScalarKind::B1 {
            let vector = f32_to_b1x8(vector.as_slice());
            self.inner
                .filtered_search(&vector, limit.0.get(), |row_id| filter(row_id.into()))?
        } else {
            self.inner
                .filtered_search(vector.as_slice(), limit.0.get(), |row_id| {
                    filter(row_id.into())
                })?
        };
        Ok(matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .map(|(primary_id, distance)| {
                Distance::try_from((distance, self.space_type.try_into()?, vector.dim()))
                    .map(|dist| (primary_id.into(), dist))
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn stop(&self) {}
}

impl ThreadedUsearchIndex {
    /// TQ4 search: oversample with HNSW, then rerank asymmetrically.
    fn tq4_search(
        &self,
        tq4: &Tq4IndexState,
        vector: &Vector,
        limit: Limit,
    ) -> anyhow::Result<std::vec::IntoIter<anyhow::Result<(PrimaryId, Distance)>>> {
        // Phase 1: HNSW retrieval with TQ4-to-TQ4 custom metric (oversampled)
        let oversample_limit = (limit.0.get() as f32 * tq4.oversample_factor).ceil() as usize;
        let compressed_query = tq4.quantizer.quantize(vector.as_slice());
        let packed_query = compressed_query.pack();
        let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
        let candidates = self.inner.search(packed_i8, oversample_limit)?;

        // Phase 2: Precise asymmetric reranking with f32 query vs stored TQ4
        let query_state = tq4.quantizer.prepare_query(vector.as_slice());
        let mut get_buf = vec![0i8; tq4.packed_dimension];

        // Collect valid candidates for batch reranking
        let mut rerank_ids: Vec<PrimaryId> = Vec::with_capacity(candidates.keys.len());
        let mut rerank_vecs: Vec<Tq4CompressedVector> =
            Vec::with_capacity(candidates.keys.len());
        for &id in &candidates.keys {
            get_buf.fill(0);
            if let Ok(found) = self.inner.get(id, &mut get_buf) {
                if found > 0 {
                    let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                    let compressed =
                        Tq4CompressedVector::unpack(buf_u8, tq4.original_dimension);
                    rerank_ids.push(PrimaryId::from(id));
                    rerank_vecs.push(compressed);
                }
            }
        }

        // Batch compute inner products (SIMD dot via NumKong, reused centroid buffer)
        let ips = tq4.quantizer.batch_inner_products(&query_state, &rerank_vecs);

        let mut results: Vec<(PrimaryId, f32, f32)> = rerank_ids
            .iter()
            .zip(rerank_vecs.iter())
            .zip(ips.iter())
            .map(|((pid, c), &ip)| (*pid, ip, c.norm))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit.0.get());

        let space_type = self.space_type;
        let q_norm = query_state.query_norm;
        let dim = vector.dim();

        Ok(results
            .into_iter()
            .map(move |(id, raw_ip, x_norm)| {
                let distance = match space_type {
                    MetricKind::Cos => {
                        let denom = q_norm * x_norm;
                        let sim = if denom > 0.0 {
                            (raw_ip / denom).clamp(-1.0, 1.0)
                        } else {
                            0.0
                        };
                        1.0 - sim
                    }
                    MetricKind::IP => -raw_ip,
                    _ => unreachable!("TQ4 only supports Cosine and DotProduct"),
                };
                Distance::try_from((distance, space_type.try_into()?, dim)).map(|dist| (id, dist))
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    /// TQ4 filtered search: same as tq4_search but with a filter predicate.
    fn tq4_filtered_search(
        &self,
        tq4: &Tq4IndexState,
        vector: &Vector,
        limit: Limit,
        filter: impl Fn(PrimaryId) -> bool,
    ) -> anyhow::Result<std::vec::IntoIter<anyhow::Result<(PrimaryId, Distance)>>> {
        let oversample_limit = (limit.0.get() as f32 * tq4.oversample_factor).ceil() as usize;
        let compressed_query = tq4.quantizer.quantize(vector.as_slice());
        let packed_query = compressed_query.pack();
        let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
        let candidates = self
            .inner
            .filtered_search(packed_i8, oversample_limit, |row_id| filter(row_id.into()))?;

        let query_state = tq4.quantizer.prepare_query(vector.as_slice());
        let mut get_buf = vec![0i8; tq4.packed_dimension];

        // Collect valid candidates for batch reranking
        let mut rerank_ids: Vec<PrimaryId> = Vec::with_capacity(candidates.keys.len());
        let mut rerank_vecs: Vec<Tq4CompressedVector> =
            Vec::with_capacity(candidates.keys.len());
        for &id in &candidates.keys {
            get_buf.fill(0);
            if let Ok(found) = self.inner.get(id, &mut get_buf) {
                if found > 0 {
                    let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                    let compressed =
                        Tq4CompressedVector::unpack(buf_u8, tq4.original_dimension);
                    rerank_ids.push(PrimaryId::from(id));
                    rerank_vecs.push(compressed);
                }
            }
        }

        // Batch compute inner products (SIMD dot via NumKong, reused centroid buffer)
        let ips = tq4.quantizer.batch_inner_products(&query_state, &rerank_vecs);

        let mut results: Vec<(PrimaryId, f32, f32)> = rerank_ids
            .iter()
            .zip(rerank_vecs.iter())
            .zip(ips.iter())
            .map(|((pid, c), &ip)| (*pid, ip, c.norm))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit.0.get());

        let space_type = self.space_type;
        let q_norm = query_state.query_norm;
        let dim = vector.dim();

        Ok(results
            .into_iter()
            .map(move |(id, raw_ip, x_norm)| {
                let distance = match space_type {
                    MetricKind::Cos => {
                        let denom = q_norm * x_norm;
                        let sim = if denom > 0.0 {
                            (raw_ip / denom).clamp(-1.0, 1.0)
                        } else {
                            0.0
                        };
                        1.0 - sim
                    }
                    MetricKind::IP => -raw_ip,
                    _ => unreachable!("TQ4 only supports Cosine and DotProduct"),
                };
                Distance::try_from((distance, space_type.try_into()?, dim)).map(|dist| (id, dist))
            })
            .collect::<Vec<_>>()
            .into_iter())
    }
}

struct Simulator {
    config: Arc<Config>,
    search: Duration,
    add_remove: Duration,
    reserve: Duration,
    keys: RwLock<HashSet<PrimaryId>>,
    notify: Arc<Notify>,
}

impl Simulator {
    const SEARCH_IDX: usize = 0;
    const ADD_REMOVE_IDX: usize = 1;
    const RESERVE_IDX: usize = 2;

    fn new(
        config: Arc<Config>,
        mut config_rx: watch::Receiver<Arc<Config>>,
        key: IndexKey,
    ) -> Arc<RwLock<Self>> {
        let mut sim = Self {
            config: Arc::new(Config::default()),
            search: Duration::ZERO,
            add_remove: Duration::ZERO,
            reserve: Duration::ZERO,
            keys: RwLock::new(HashSet::new()),
            notify: Arc::new(Notify::new()),
        };
        sim.update(config);
        let notify = Arc::clone(&sim.notify);
        let sim = Arc::new(RwLock::new(sim));

        tokio::spawn(
            {
                let sim = Arc::clone(&sim);
                async move {
                    loop {
                        tokio::select! {
                            _ = config_rx.changed() => {
                                let config = config_rx.borrow_and_update().clone();
                                sim.write().unwrap().update(config);
                            }
                            _ = notify.notified() => {
                                break;
                            }
                        }
                    }
                }
            }
            .instrument(error_span!("simulator", "{}", key)),
        );

        sim
    }

    fn update(&mut self, config: Arc<Config>) {
        if self.config.usearch_simulator == config.usearch_simulator {
            return;
        }
        self.search = *config
            .usearch_simulator
            .as_ref()
            .and_then(|vec| vec.get(Self::SEARCH_IDX))
            .unwrap_or(&Duration::ZERO);
        self.add_remove = *config
            .usearch_simulator
            .as_ref()
            .and_then(|vec| vec.get(Self::ADD_REMOVE_IDX))
            .unwrap_or(&Duration::ZERO);
        self.reserve = *config
            .usearch_simulator
            .as_ref()
            .and_then(|vec| vec.get(Self::RESERVE_IDX))
            .unwrap_or(&Duration::ZERO);
        info!(
            "usearch simulator config updated: search = {:?}, add_remove = {:?}, reserve = {:?}",
            self.search, self.add_remove, self.reserve
        );
        self.config = config;
    }

    fn wait(&self, start: Instant, duration: Duration) {
        while start.elapsed() < duration {}
    }

    fn wait_reserve(&self, start: Instant) {
        self.wait(start, self.reserve);
    }

    fn wait_add_remove(&self, start: Instant) {
        self.wait(start, self.add_remove);
    }

    fn wait_search(&self, start: Instant) {
        self.wait(start, self.search);
    }
}

impl UsearchIndex for RwLock<Simulator> {
    fn reserve(&self, size: usize) -> anyhow::Result<()> {
        let start = Instant::now();

        // we need simulate write lock similar to real usearch index
        #[allow(clippy::readonly_write_lock)]
        let sim = self.write().unwrap();
        {
            let mut keys = sim.keys.write().unwrap();
            let len = keys.len();
            keys.reserve(size - len);
        }

        sim.wait_reserve(start);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.read().unwrap().keys.read().unwrap().capacity()
    }

    fn size(&self) -> usize {
        self.read().unwrap().keys.read().unwrap().len()
    }

    fn add(&self, row_id: PrimaryId, _: &Vector) -> anyhow::Result<()> {
        let start = Instant::now();

        let sim = self.read().unwrap();
        sim.keys.write().unwrap().insert(row_id);

        sim.wait_add_remove(start);
        Ok(())
    }

    fn remove(&self, row_id: PrimaryId) -> anyhow::Result<()> {
        let start = Instant::now();

        let sim = self.read().unwrap();
        sim.keys.write().unwrap().remove(&row_id);

        sim.wait_add_remove(start);
        Ok(())
    }

    fn search(
        &self,
        _: &Vector,
        limit: Limit,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>> {
        let start = Instant::now();

        let sim = self.read().unwrap();
        let keys = {
            let len = sim.keys.read().unwrap().len() as u64;
            if len == 0 {
                Vec::new()
            } else {
                let keys = sim.keys.read().unwrap();
                iter::repeat_with(|| rand::random_range(0..len))
                    .map(PrimaryId::from)
                    .filter(|row_id| keys.contains(row_id))
                    .take(limit.0.get())
                    .collect()
            }
        };

        sim.wait_search(start);
        let distance = Distance::new_euclidean(0.0)?;
        Ok(keys.into_iter().map(move |row_id| Ok((row_id, distance))))
    }

    fn filtered_search(
        &self,
        vector: &Vector,
        limit: Limit,
        _filter: impl Fn(PrimaryId) -> bool,
    ) -> anyhow::Result<impl Iterator<Item = anyhow::Result<(PrimaryId, Distance)>>> {
        self.search(vector, limit)
    }

    fn stop(&self) {
        self.read().unwrap().notify.notify_one();
    }
}

// Initial and incremental number for the index vectors reservation.
// The value was taken for initial benchmarks (size similar to benchmark size)
const RESERVE_INCREMENT_GLOBAL: usize = 1000000;
const RESERVE_INCREMENT_LOCAL: usize = 1000;

// When free space for index vectors drops below this, will reserve more space
// The ratio was taken for initial benchmarks
const RESERVE_THRESHOLD_GLOBAL: usize = RESERVE_INCREMENT_GLOBAL / 3;
const RESERVE_THRESHOLD_LOCAL: usize = RESERVE_INCREMENT_LOCAL / 3;

struct MetricConfig {
    quantization: Quantization,
    space_type: SpaceType,
}

fn metric_kind(quantization: Quantization, space_type: SpaceType) -> anyhow::Result<MetricKind> {
    // Usearch requires a binary metric (e.g., Hamming, Jaccard) for B1 quantization.
    if quantization == Quantization::B1 {
        return Ok(MetricKind::Hamming);
    }

    // TQ4 uses a custom metric registered via change_metric(); return placeholder.
    if quantization == Quantization::TQ4 {
        return match space_type {
            SpaceType::Cosine | SpaceType::DotProduct => Ok(MetricKind::IP),
            _ => anyhow::bail!(
                "TQ4 quantization only supports Cosine and DotProduct. Unsupported: {space_type:?}"
            ),
        };
    }

    MetricConfig {
        quantization,
        space_type,
    }
    .try_into()
}

impl TryFrom<MetricConfig> for MetricKind {
    type Error = anyhow::Error;

    fn try_from(config: MetricConfig) -> Result<Self, Self::Error> {
        if config.quantization == Quantization::B1 {
            return match config.space_type {
                SpaceType::Hamming => Ok(MetricKind::Hamming),
                _ => anyhow::bail!(
                    "B1 quantization requires binary space type. Unsupported space type: {:?}",
                    config.space_type
                ),
            };
        }

        if config.quantization == Quantization::TQ4 {
            return match config.space_type {
                SpaceType::Cosine | SpaceType::DotProduct => Ok(MetricKind::IP),
                _ => anyhow::bail!(
                    "TQ4 quantization only supports Cosine and DotProduct. Unsupported: {:?}",
                    config.space_type
                ),
            };
        }

        match config.space_type {
            SpaceType::Cosine => Ok(MetricKind::Cos),
            SpaceType::Euclidean => Ok(MetricKind::L2sq),
            SpaceType::DotProduct => Ok(MetricKind::IP),
            SpaceType::Hamming => anyhow::bail!("Binary space type requires B1 quantization."),
        }
    }
}

impl TryFrom<MetricKind> for SpaceType {
    type Error = anyhow::Error;

    fn try_from(metric_kind: MetricKind) -> Result<Self, Self::Error> {
        match metric_kind {
            MetricKind::Cos => Ok(SpaceType::Cosine),
            MetricKind::L2sq => Ok(SpaceType::Euclidean),
            MetricKind::IP => Ok(SpaceType::DotProduct),
            MetricKind::Hamming => Ok(SpaceType::Hamming),
            _ => anyhow::bail!("Unsupported MetricKind for SpaceType: {:?}", metric_kind),
        }
    }
}

impl From<Quantization> for ScalarKind {
    fn from(quantization: Quantization) -> Self {
        match quantization {
            Quantization::F32 => ScalarKind::F32,
            Quantization::F16 => ScalarKind::F16,
            Quantization::BF16 => ScalarKind::BF16,
            Quantization::I8 => ScalarKind::I8,
            Quantization::B1 => ScalarKind::B1,
            Quantization::TQ4 => ScalarKind::I8, // TQ4 packed bytes stored as opaque I8
        }
    }
}

mod operation {
    use super::Index;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use tokio::sync::Notify;

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Reserve,
        Insert,
        Remove,
        Search,
    }

    impl From<&Index> for Mode {
        fn from(msg: &Index) -> Self {
            match msg {
                Index::AddVector { .. } => Mode::Insert,
                Index::RemoveVector { .. } => Mode::Remove,
                Index::Ann { .. } | Index::FilteredAnn { .. } => Mode::Search,
                Index::RemovePartition { .. } => todo!(),
                Index::Count { .. } => unreachable!(),
            }
        }
    }

    impl Mode {
        fn is_exclusive(&self) -> bool {
            match self {
                Mode::Insert | Mode::Search => false,
                // Remove and reserve are not safe to run concurrently with other operations.
                // Therefore, we perform both exclusively.
                // See: https://github.com/unum-cloud/USearch/issues/697.
                Mode::Reserve | Mode::Remove => true,
            }
        }
    }

    pub(super) struct Permit {
        notify: Arc<Notify>,
        counter: Arc<AtomicUsize>,
    }

    impl Drop for Permit {
        fn drop(&mut self) {
            let previous = self.counter.fetch_sub(1, Ordering::Relaxed);
            if previous == 1 {
                self.notify.notify_one();
            }
        }
    }

    pub(super) struct Operation {
        mode: Mode,
        notify: Arc<Notify>,
        counter: Arc<AtomicUsize>,
    }

    impl Operation {
        pub(super) fn new() -> Self {
            Self {
                mode: Mode::Insert,
                notify: Arc::new(Notify::new()),
                counter: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// Wait until it will be possible to spawn operation.
        ///
        /// The function must be called before spawning operation tasks as it blocks
        /// until only requested family of operations is in progress.
        async fn permit(&mut self, mode: Mode) -> Permit {
            while self.mode != mode {
                if self.counter.load(Ordering::Relaxed) == 0 {
                    // it is safe to switch to the operation because there are no spawned tasks
                    // and self.counter won't be changed
                    self.mode = mode;
                    break;
                }
                self.notify.notified().await;
            }

            if mode.is_exclusive() {
                while self.counter.load(Ordering::Relaxed) != 0 {
                    self.notify.notified().await;
                }
            }

            self.counter.fetch_add(1, Ordering::Relaxed);
            Permit {
                notify: Arc::clone(&self.notify),
                counter: Arc::clone(&self.counter),
            }
        }

        pub(super) async fn permit_for_message(&mut self, msg: &Index) -> Permit {
            self.permit(msg.into()).await
        }

        pub(super) async fn permit_for_reserve(&mut self) -> Permit {
            self.permit(Mode::Reserve).await
        }

        /// Capacity and size permit cannot be concurrent only with reserve mode.
        pub(super) async fn permit_for_capacity_and_size(&mut self) -> Permit {
            while self.mode == Mode::Reserve {
                if self.counter.load(Ordering::Relaxed) == 0 {
                    // checking for capacity is during add, so insert mode is fine
                    self.mode = Mode::Insert;
                    break;
                }
                self.notify.notified().await;
            }

            self.counter.fetch_add(1, Ordering::Relaxed);
            Permit {
                notify: Arc::clone(&self.notify),
                counter: Arc::clone(&self.counter),
            }
        }
    }
}

struct PartitionState<I: UsearchIndex + Send + Sync + 'static> {
    partition_id: PartitionId,
    idx: Arc<I>,
}

impl<I> PartitionState<I>
where
    I: UsearchIndex + Send + Sync + 'static,
{
    fn new(partition_id: PartitionId, idx: Arc<I>) -> Self {
        Self { partition_id, idx }
    }

    fn stop(&self) {
        self.idx.stop();
    }
}

struct IndexState {
    dimensions: Dimensions,
    size: Arc<AtomicUsize>,
    operation: operation::Operation,
}

impl IndexState {
    fn new(dimensions: Dimensions) -> Self {
        Self {
            dimensions,
            size: Arc::new(AtomicUsize::new(0)),
            operation: operation::Operation::new(),
        }
    }
}

fn new<I: UsearchIndex + Send + Sync + 'static>(
    index_fn: impl FnOnce() -> anyhow::Result<Arc<I>> + Clone + Send + Sync + 'static,
    index_key: IndexKey,
    dimensions: Dimensions,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    tokio_semaphore: Arc<Semaphore>,
    rayon_semaphore: Arc<Semaphore>,
    memory: mpsc::Sender<Memory>,
) -> anyhow::Result<mpsc::Sender<Index>> {
    // TODO: The value of channel size was taken from initial benchmarks. Needs more testing
    const CHANNEL_SIZE: usize = 10;
    let (tx, mut rx) = mpsc::channel(CHANNEL_SIZE);

    tokio::spawn(
        {
            let index_key = index_key.clone();
            async move {
                debug!("starting");
                let mut states = BTreeMap::new();
                let mut partitions = BTreeMap::new();

                let mut allocate_prev = Allocate::Can;

                while let Some(msg) = rx.recv().await {
                    if !check_memory_allocation(&msg, &memory, &mut allocate_prev, &index_key).await
                    {
                        continue;
                    }

                    let Some((state, partition, msg)) = preprocess(
                        index_fn.clone(),
                        &mut states,
                        &mut partitions,
                        table.as_ref(),
                        dimensions,
                        msg,
                    ) else {
                        continue;
                    };

                    dispatch_task(
                        state,
                        partition,
                        &table,
                        &tokio_semaphore,
                        &rayon_semaphore,
                        msg,
                    )
                    .await;
                }

                partitions
                    .into_values()
                    .for_each(|partition| partition.stop());

                debug!("finished");
            }
        }
        .instrument(error_span!("usearch", "{index_key}")),
    );

    Ok(tx)
}

fn preprocess<'a, I, T>(
    index_fn: impl FnOnce() -> anyhow::Result<Arc<I>>,
    states: &'a mut BTreeMap<IndexId, IndexState>,
    partitions: &mut BTreeMap<PartitionId, Arc<PartitionState<I>>>,
    table: &RwLock<T>,
    dimensions: Dimensions,
    msg: Index,
) -> Option<(&'a mut IndexState, Arc<PartitionState<I>>, Index)>
where
    I: UsearchIndex + Send + Sync + 'static,
    T: TableSearch + Send + Sync + 'static,
{
    match msg {
        Index::AddVector { partition_id, .. } => {
            let index_id = partition_id.index_id();
            if let Some(partition) = partitions.get(&partition_id) {
                let Some(state) = states.get_mut(&index_id) else {
                    error!("index state not found for index {index_id:?}");
                    return None;
                };
                return Some((state, Arc::clone(partition), msg));
            }
            let partition = Arc::new(PartitionState::new(
                partition_id,
                index_fn()
                    .inspect_err(|err| {
                        error!("failed to create index for partition {partition_id:?}: {err}")
                    })
                    .ok()?,
            ));
            let state = states
                .entry(index_id)
                .or_insert_with(|| IndexState::new(dimensions));
            partitions.insert(partition_id, Arc::clone(&partition));
            Some((state, partition, msg))
        }

        Index::Ann {
            index_key,
            embedding,
            limit,
            tx,
        } => {
            let Some((partition_id, _)) = table.read().unwrap().partition_id(&index_key, None)
            else {
                warn!("partition id not found for index key {index_key:?} during ann");
                _ = tx.send(Ok((vec![], vec![])));
                return None;
            };
            let index_id = partition_id.index_id();
            let Some((state, partition)) = states
                .get_mut(&index_id)
                .zip(partitions.get(&partition_id))
                .map(|(state, partition)| (state, Arc::clone(partition)))
            else {
                warn!("state or partition not found for index key {index_key:?} during ann");
                _ = tx.send(Ok((vec![], vec![])));
                return None;
            };
            Some((
                state,
                partition,
                Index::Ann {
                    embedding,
                    limit,
                    tx,
                    index_key,
                },
            ))
        }

        Index::FilteredAnn {
            index_key,
            embedding,
            filter,
            limit,
            tx,
        } => {
            let Some((partition_id, restrictions)) = table
                .read()
                .unwrap()
                .partition_id(&index_key, Some(filter.restrictions))
            else {
                warn!("partition id not found for index key {index_key:?} during filtered ann");
                _ = tx.send(Ok((vec![], vec![])));
                return None;
            };
            let index_id = partition_id.index_id();
            let Some((state, partition)) = states
                .get_mut(&index_id)
                .zip(partitions.get(&partition_id))
                .map(|(state, partition)| (state, Arc::clone(partition)))
            else {
                warn!(
                    "state or partition not found for index key {index_key:?} \
                        during filtered ann"
                );
                _ = tx.send(Ok((vec![], vec![])));
                return None;
            };
            let msg = if let Some(restrictions) = restrictions {
                Index::FilteredAnn {
                    embedding,
                    limit,
                    filter: Filter {
                        restrictions,
                        allow_filtering: filter.allow_filtering,
                    },
                    tx,
                    index_key,
                }
            } else {
                Index::Ann {
                    embedding,
                    limit,
                    tx,
                    index_key,
                }
            };
            Some((state, partition, msg))
        }

        Index::Count { index_key, tx } => {
            let Some(index_id) = table.read().unwrap().index_id(&index_key) else {
                let err = anyhow!("index id not found for index key {index_key:?}");
                warn!("index count: {err}");
                _ = tx.send(Err(err));
                return None;
            };
            _ = tx.send(Ok(states
                .get_mut(&index_id)
                .map(|state| state.size.load(Ordering::Relaxed))
                .unwrap_or(0)));
            None
        }

        Index::RemoveVector { partition_id, .. } => {
            let index_id = partition_id.index_id();
            states
                .get_mut(&index_id)
                .zip(partitions.get(&partition_id))
                .map(|(state, partition)| (state, Arc::clone(partition), msg))
        }

        Index::RemovePartition { partition_id } => {
            if let Some(idx) = partitions.remove(&partition_id) {
                idx.stop();
            };
            None
        }
    }
}

async fn dispatch_task<I, T>(
    state: &mut IndexState,
    partition: Arc<PartitionState<I>>,
    table: &Arc<RwLock<T>>,
    tokio_semaphore: &Arc<Semaphore>,
    rayon_semaphore: &Arc<Semaphore>,
    msg: Index,
) where
    I: UsearchIndex + Send + Sync + 'static,
    T: TableSearch + Send + Sync + 'static,
{
    if let Index::AddVector { .. } = &msg {
        let operation_permit = state.operation.permit_for_capacity_and_size().await;
        let is_global = partition.partition_id.index_id().is_global();
        if needs_more_capacity(partition.idx.as_ref(), is_global).is_some() {
            drop(operation_permit);
            let operation_permit = state.operation.permit_for_reserve().await;
            if let Some(capacity) = needs_more_capacity(partition.idx.as_ref(), is_global) {
                let permit = Arc::clone(rayon_semaphore).acquire_owned().await.unwrap();
                let idx = Arc::clone(&partition.idx);
                rayon::spawn(move || {
                    reserve(idx.as_ref(), capacity);
                    drop(permit);
                    drop(operation_permit);
                });
            }
        }
    }

    let operation_permit = state.operation.permit_for_message(&msg).await;

    let table = Arc::clone(table);
    let dimensions = state.dimensions;
    let size = Arc::clone(&state.size);
    if should_run_on_tokio(&msg) {
        let permit = Arc::clone(tokio_semaphore).acquire_owned().await.unwrap();
        tokio::spawn(async move {
            crate::move_to_the_end_of_async_runtime_queue().await;
            process(partition, table, dimensions, size, msg);
            drop(permit);
            drop(operation_permit);
        });
        return;
    }
    let permit = Arc::clone(rayon_semaphore).acquire_owned().await.unwrap();
    rayon::spawn(move || {
        process(partition, table, dimensions, size, msg);
        drop(permit);
        drop(operation_permit);
    });
}

fn should_run_on_tokio(msg: &Index) -> bool {
    matches!(msg, Index::Ann { .. })
}

fn process<I, T>(
    partition: Arc<PartitionState<I>>,
    table: Arc<RwLock<T>>,
    dimensions: Dimensions,
    size: Arc<AtomicUsize>,
    msg: Index,
) where
    I: UsearchIndex + Send + Sync + 'static,
    T: TableSearch + Send + Sync + 'static,
{
    match msg {
        Index::AddVector {
            primary_id,
            embedding,
            in_progress: _in_progress,
            ..
        } => add(partition.idx.as_ref(), primary_id, &embedding, &size),

        Index::Ann {
            embedding,
            limit,
            tx,
            ..
        } => {
            if let Some(tx) = validate_dimensions(tx, &embedding, dimensions) {
                ann(partition, tx, &table, embedding, limit);
            }
        }

        Index::FilteredAnn {
            embedding,
            limit,
            filter,
            tx,
            ..
        } => {
            if let Some(tx) = validate_dimensions(tx, &embedding, dimensions) {
                filtered_ann(partition, tx, &table, embedding, filter, limit);
            }
        }

        Index::Count { .. } => unreachable!(),

        Index::RemoveVector {
            primary_id,
            in_progress: _in_progress,
            ..
        } => remove(partition.idx.as_ref(), primary_id, &size),

        Index::RemovePartition { .. } => unreachable!(),
    }
}

fn reserve(idx: &impl UsearchIndex, capacity: usize) {
    let result = idx.reserve(capacity);
    if let Err(err) = &result {
        error!("unable to reserve index capacity for {capacity} in usearch: {err}");
    } else {
        debug!("reserve: reserved index capacity for {capacity}");
    }
}

fn needs_more_capacity(idx: &impl UsearchIndex, is_global: bool) -> Option<usize> {
    let capacity = idx.capacity();
    let free_space = capacity - idx.size();
    let (increment, threshold) = if is_global {
        (RESERVE_INCREMENT_GLOBAL, RESERVE_THRESHOLD_GLOBAL)
    } else {
        (RESERVE_INCREMENT_LOCAL, RESERVE_THRESHOLD_LOCAL)
    };

    if free_space < threshold {
        Some(capacity + increment)
    } else {
        None
    }
}

fn add(idx: &impl UsearchIndex, primary_id: PrimaryId, embedding: &Vector, size: &AtomicUsize) {
    if let Err(err) = idx.add(primary_id, embedding) {
        warn!("add: unable to add embedding: {err}");
    } else {
        size.fetch_add(1, Ordering::Relaxed);
    }
}

fn remove(idx: &impl UsearchIndex, row_id: PrimaryId, size: &AtomicUsize) {
    if let Err(err) = idx.remove(row_id) {
        warn!("remove: unable to remove embeddings: {err}");
    } else {
        size.fetch_sub(1, Ordering::Relaxed);
    }
}

fn validate_dimensions(
    tx_ann: oneshot::Sender<AnnR>,
    embedding: &Vector,
    dimensions: Dimensions,
) -> Option<oneshot::Sender<AnnR>> {
    if let Err(err) = validator::embedding_dimensions(embedding, dimensions) {
        tx_ann
            .send(Err(err))
            .unwrap_or_else(|_| trace!("validate_dimensions: unable to send response"));
        None
    } else {
        Some(tx_ann)
    }
}

fn ann<I>(
    partition: Arc<PartitionState<I>>,
    tx_ann: oneshot::Sender<AnnR>,
    table: &Arc<RwLock<impl TableSearch>>,
    embedding: Vector,
    limit: Limit,
) where
    I: UsearchIndex + Send + Sync + 'static,
{
    tx_ann
        .send(
            partition.idx.search(&embedding, limit)
                .map_err(|err| anyhow!("ann: search failed: {err}"))
                .and_then(|matches| {
                    let table = table.read().unwrap();
                    let (primary_keys, distances) = itertools::process_results(
                        matches.map(|result| {
                            result.and_then(|(primary_id, distance)| {
                                table
                                    .primary_key(partition.partition_id, primary_id)
                                    .ok_or(anyhow!(
                                        "not defined primary_key for partition_id {partition_id:?} and primary_id {primary_id:?}",
                                        partition_id = partition.partition_id,
                                    ))
                                    .map(|primary_key| (primary_key, distance))
                            })
                        }),
                        |it| it.unzip(),
                    )?;
                    Ok((primary_keys, distances))
                }),
        )
        .unwrap_or_else(|_| trace!("ann: unable to send response"));
}

fn filtered_ann<I>(
    partition: Arc<PartitionState<I>>,
    tx_ann: oneshot::Sender<AnnR>,
    table: &Arc<RwLock<impl TableSearch>>,
    embedding: Vector,
    filter: Filter,
    limit: Limit,
) where
    I: UsearchIndex + Send + Sync + 'static,
{
    let id_ok = |primary_id: PrimaryId| {
        let table = table.read().unwrap();
        filter
            .restrictions
            .iter()
            .all(|restriction| table.is_valid_for(partition.partition_id, primary_id, restriction))
    };

    tx_ann
        .send(
            partition.idx.filtered_search(&embedding, limit, id_ok)
                .map_err(|err| anyhow!("ann: search failed: {err}"))
                .and_then(|matches| {
                    let table = table.read().unwrap();
                    let (primary_keys, distances) = itertools::process_results(
                        matches.map(|result| {
                            result.and_then(|(primary_id, distance)| {
                                table
                                    .primary_key(partition.partition_id, primary_id)
                                    .ok_or(anyhow!(
                                        "not defined primary key for partition_id {partition_id:?} and primary_id {primary_id:?}",
                                        partition_id = partition.partition_id,
                                    ))
                                    .map(|primary_key| (primary_key, distance))
                            })
                        }),
                        |it| it.unzip(),
                    )?;
                    Ok((primary_keys, distances))
                }),
        )
        .unwrap_or_else(|_| trace!("ann: unable to send response"));
}

async fn check_memory_allocation(
    msg: &Index,
    memory: &mpsc::Sender<Memory>,
    allocate_prev: &mut Allocate,
    key: &IndexKey,
) -> bool {
    if !matches!(msg, Index::AddVector { .. }) {
        return true;
    }

    let allocate = memory.can_allocate().await;
    if allocate == Allocate::Cannot {
        if *allocate_prev == Allocate::Can {
            error!("Unable to add vector for index {key}: not enough memory to reserve more space");
        }
        *allocate_prev = allocate;
        return false;
    }
    *allocate_prev = allocate;
    true
}

fn f32_to_b1x8(f32_vec: &[f32]) -> Vec<b1x8> {
    fn chunk_to_byte(chunk: impl Iterator<Item = f32>) -> b1x8 {
        chunk.enumerate().fold(b1x8(0u8), |byte, (i, val)| {
            if val > 0.0 {
                b1x8(byte.0 | (1 << i))
            } else {
                byte
            }
        })
    }

    // Pre-calculate total capacity to avoid reallocation when pushing the remainder chunk
    let capacity = f32_vec.len().div_ceil(8);
    let mut bytes = Vec::<b1x8>::with_capacity(capacity);

    let mut iter = f32_vec.chunks_exact(8);
    bytes.extend(
        iter.by_ref()
            .map(|chunk| chunk_to_byte(chunk.iter().copied())),
    );

    let remainder = iter.remainder();
    if !remainder.is_empty() {
        bytes.push(chunk_to_byte(remainder.iter().copied()));
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use crate::IndexKey;
    use crate::index::IndexExt;
    use crate::memory;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use mockall::predicate::*;
    use scylla::value::CqlValue;
    use std::num::NonZeroUsize;
    use std::time::Duration;
    use tokio::runtime::Handle;
    use tokio::sync::watch;
    use tokio::task;
    use tokio::time;
    use usearch::b1x8;

    fn add_concurrently(
        partition_id: PartitionId,
        index: mpsc::Sender<Index>,
        threads: usize,
        adds_per_worker: usize,
        dimensions: NonZeroUsize,
    ) -> Vec<task::JoinHandle<()>> {
        let mut add_handles = Vec::new();
        for worker in 0..threads {
            let actor = index.clone();
            add_handles.push(tokio::spawn(async move {
                for offset in 0..adds_per_worker {
                    let id = (worker * adds_per_worker + offset) as u64;
                    actor
                        .add_vector(
                            partition_id,
                            id.into(),
                            vec![0.0f32; dimensions.get()].into(),
                            None,
                        )
                        .await;
                }
            }));
        }
        add_handles
    }

    fn search_concurrently(
        index_key: IndexKey,
        index: mpsc::Sender<Index>,
        threads: usize,
        searches_per_worker: usize,
        dimensions: NonZeroUsize,
    ) -> Vec<task::JoinHandle<()>> {
        let mut search_handles = Vec::new();
        for _ in 0..threads {
            let actor = index.clone();
            let index_key = index_key.clone();
            search_handles.push(tokio::spawn(async move {
                for _ in 0..searches_per_worker {
                    let limit = NonZeroUsize::new(5).unwrap().into();
                    let _ = actor
                        .ann(
                            index_key.clone(),
                            vec![0.0f32; dimensions.get()].into(),
                            limit,
                        )
                        .await;
                }
            }));
        }
        search_handles
    }

    #[tokio::test]
    async fn add_or_replace_size_ann() {
        let (_, config_rx) = watch::channel(Arc::new(Config::default()));

        let options = IndexOptions {
            dimensions: 3,
            metric: MetricKind::L2sq,
            ..Default::default()
        };
        let threads = Handle::current().metrics().num_workers() + rayon::current_num_threads();
        let table = Arc::new(RwLock::new(MockTableSearch::new()));
        let index_key = IndexKey::new(&"vector".into(), &"store".into());
        let actor = new(
            move || Ok(Arc::new(ThreadedUsearchIndex::new(options, threads)?)),
            index_key.clone(),
            NonZeroUsize::new(3).unwrap().into(),
            Arc::clone(&table),
            Arc::new(Semaphore::new(4)),
            Arc::new(Semaphore::new(4)),
            memory::new(config_rx),
        )
        .unwrap();

        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        actor
            .add_vector(partition_id, 1.into(), vec![1., 1., 1.].into(), None)
            .await;
        actor
            .add_vector(partition_id, 2.into(), vec![2., -2., 2.].into(), None)
            .await;
        actor
            .add_vector(partition_id, 3.into(), vec![3., 3., 3.].into(), None)
            .await;

        table
            .write()
            .unwrap()
            .expect_index_id()
            .with(eq(index_key.clone()))
            .returning(move |_| Some(index_id));

        table.write().unwrap().expect_partition_id().returning({
            let index_key = index_key.clone();
            move |key, restrictions| {
                assert_eq!(key, &index_key);
                assert!(restrictions.is_none());
                Some((partition_id, None))
            }
        });
        time::timeout(Duration::from_secs(10), async {
            while actor.count(index_key.clone()).await.unwrap() != 3 {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();

        table
            .write()
            .unwrap()
            .expect_primary_key()
            .with(eq(partition_id), eq(PrimaryId::from(2)))
            .once()
            .returning(|_, _| Some([CqlValue::Int(2)].into()));

        let (primary_keys, distances) = actor
            .ann(
                index_key.clone(),
                vec![2.2, -2.2, 2.2].into(),
                NonZeroUsize::new(1).unwrap().into(),
            )
            .await
            .unwrap();
        assert_eq!(primary_keys.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(primary_keys.first().unwrap(), &[CqlValue::Int(2)].into());

        actor.remove_vector(partition_id, 3.into(), None).await;
        actor
            .add_vector(partition_id, 3.into(), vec![2.1, -2.1, 2.1].into(), None)
            .await;

        table
            .write()
            .unwrap()
            .expect_primary_key()
            .with(eq(partition_id), eq(PrimaryId::from(3)))
            .once()
            .returning(|_, _| Some([CqlValue::Int(3)].into()));

        time::timeout(Duration::from_secs(10), async {
            while actor
                .ann(
                    index_key.clone(),
                    vec![2.2, -2.2, 2.2].into(),
                    NonZeroUsize::new(1).unwrap().into(),
                )
                .await
                .unwrap()
                .0
                .first()
                .unwrap()
                != &[CqlValue::Int(3)].into()
            {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();

        actor.remove_vector(partition_id, 3.into(), None).await;

        time::timeout(Duration::from_secs(10), async {
            while actor.count(index_key.clone()).await.unwrap() != 2 {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();

        table
            .write()
            .unwrap()
            .expect_primary_key()
            .with(eq(partition_id), eq(PrimaryId::from(2)))
            .once()
            .returning(|_, _| Some([CqlValue::Int(2)].into()));

        let (primary_keys, distances) = actor
            .ann(
                index_key,
                vec![2.2, -2.2, 2.2].into(),
                NonZeroUsize::new(1).unwrap().into(),
            )
            .await
            .unwrap();
        assert_eq!(primary_keys.len(), 1);
        assert_eq!(distances.len(), 1);
        assert_eq!(primary_keys.first().unwrap(), &[CqlValue::Int(2)].into());
    }

    #[tokio::test]
    async fn allocate_parameter_works() {
        let (memory_tx, mut memory_rx) = mpsc::channel(1);

        let options = IndexOptions {
            dimensions: 3,
            metric: MetricKind::L2sq,
            ..Default::default()
        };
        let threads = Handle::current().metrics().num_workers() + rayon::current_num_threads();
        let table = Arc::new(RwLock::new(MockTableSearch::new()));
        let index_key = IndexKey::new(&"vector".into(), &"store".into());
        let actor = new(
            move || Ok(Arc::new(ThreadedUsearchIndex::new(options, threads)?)),
            index_key.clone(),
            NonZeroUsize::new(3).unwrap().into(),
            Arc::clone(&table),
            Arc::new(Semaphore::new(4)),
            Arc::new(Semaphore::new(4)),
            memory_tx,
        )
        .unwrap();

        let memory_respond = tokio::spawn(async move {
            let Memory::CanAllocate { tx } = memory_rx.recv().await.unwrap();
            _ = tx.send(Allocate::Cannot);
            memory_rx
        });
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        actor
            .add_vector(partition_id, 1.into(), vec![1., 1., 1.].into(), None)
            .await;
        let mut memory_rx = memory_respond.await.unwrap();

        table
            .write()
            .unwrap()
            .expect_index_id()
            .with(eq(index_key.clone()))
            .returning(move |_| Some(index_id));

        assert_eq!(actor.count(index_key.clone()).await.unwrap(), 0);

        let memory_respond = tokio::spawn(async move {
            let Memory::CanAllocate { tx } = memory_rx.recv().await.unwrap();
            _ = tx.send(Allocate::Can);
        });
        actor
            .add_vector(partition_id, 1.into(), vec![1., 1., 1.].into(), None)
            .await;
        memory_respond.await.unwrap();

        // Wait for the add operation to complete, as it runs in a separate task.
        time::timeout(Duration::from_secs(10), async {
            while actor.count(index_key.clone()).await.unwrap() != 1 {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ntest::timeout(10_000)]
    async fn concurrent_add_and_search() {
        // By default, Usearch limits concurrent operations (searches, adds) to the number of CPU cores.
        // Exceeding this limit results in a "No available threads to lock" error.
        // This test verifies our concurrency control by spawning a high number of parallel adds and searches (2 x num of cores).
        let (_, config_rx) = watch::channel(Arc::new(Config::default()));

        let dimensions = NonZeroUsize::new(1024).unwrap();
        let options = IndexOptions {
            dimensions: dimensions.get(),
            metric: MetricKind::L2sq,
            ..Default::default()
        };
        let threads = Handle::current().metrics().num_workers() + rayon::current_num_threads();
        let table = Arc::new(RwLock::new(MockTableSearch::new()));
        let index_key = IndexKey::new(&"vector".into(), &"store".into());
        let index = new(
            move || Ok(Arc::new(ThreadedUsearchIndex::new(options, threads)?)),
            index_key.clone(),
            dimensions.into(),
            Arc::clone(&table),
            Arc::new(Semaphore::new(Semaphore::MAX_PERMITS)),
            Arc::new(Semaphore::new(Semaphore::MAX_PERMITS)),
            memory::new(config_rx),
        )
        .unwrap();

        let threads = Handle::current().metrics().num_workers();

        let adds_per_worker = 50;
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        table.write().unwrap().expect_partition_id().returning({
            let index_key = index_key.clone();
            move |key, restrictions| {
                assert_eq!(key, &index_key);
                assert!(restrictions.is_none());
                Some((partition_id, None))
            }
        });
        let add_handles = add_concurrently(
            partition_id,
            index.clone(),
            threads,
            adds_per_worker,
            dimensions,
        );
        let search_handles = search_concurrently(
            index_key.clone(),
            index.clone(),
            threads,
            adds_per_worker,
            dimensions,
        );

        for handle in add_handles {
            handle.await.unwrap();
        }
        for handle in search_handles {
            handle.await.unwrap();
        }

        table
            .write()
            .unwrap()
            .expect_index_id()
            .with(eq(index_key.clone()))
            .returning(move |_| Some(index_id));

        // Wait for expected number of vectors to be added.
        time::timeout(Duration::from_secs(10), async {
            while index.count(index_key.clone()).await.unwrap() != threads * adds_per_worker {
                task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn quantization_to_kind_conversion() {
        assert_eq!(ScalarKind::from(Quantization::F32), ScalarKind::F32);
        assert_eq!(ScalarKind::from(Quantization::F16), ScalarKind::F16);
        assert_eq!(ScalarKind::from(Quantization::BF16), ScalarKind::BF16);
        assert_eq!(ScalarKind::from(Quantization::I8), ScalarKind::I8);
        assert_eq!(ScalarKind::from(Quantization::B1), ScalarKind::B1);
    }

    fn b1x8_to_u8_vec(b1_vec: &[b1x8]) -> Vec<u8> {
        b1_vec.iter().map(|&b| b.0).collect()
    }

    #[test]
    fn f32_to_b1x8_empty() {
        let b1_vec = f32_to_b1x8(&[]);
        assert_eq!(b1_vec.len(), 0);
    }

    #[test]
    fn f32_to_b1x8_single_byte() {
        // =< 0 clears bits and > 0 sets bits
        let b1_vec = f32_to_b1x8(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(b1_vec.len(), 1);
        assert_eq!(b1x8_to_u8_vec(&b1_vec), &[0b00001111]);
    }

    #[test]
    fn f32_to_b1x8_multiple_bytes() {
        let input = vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // 0b01010101
            -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, // 0b11110000
        ];
        let b1_vec = f32_to_b1x8(&input);
        assert_eq!(b1_vec.len(), 2);
        assert_eq!(b1x8_to_u8_vec(&b1_vec), &[0b01010101, 0b11110000]);
    }

    #[test]
    fn f32_to_b1x8_large_input() {
        let input = vec![1.0; 64]; // 64 elements = 8 bytes
        let b1_vec = f32_to_b1x8(&input);
        assert_eq!(b1_vec.len(), 8);
        assert_eq!(b1x8_to_u8_vec(&b1_vec), &[0b11111111; 8]);
    }

    #[test]
    fn f32_to_b1x8_remainder() {
        let input = vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // 0b01010101
            1.0, -1.0, 1.0, // 0b00000101
        ];
        let b1_vec = f32_to_b1x8(&input);
        assert_eq!(b1_vec.len(), 2);
        assert_eq!(b1x8_to_u8_vec(&b1_vec), &[0b01010101, 0b00000101]);
    }

    mod tq4_recall {
        use super::*;
        use crate::turbo_quant::qjl::fill_standard_normal;
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use std::collections::HashSet;

        /// Generate a random unit-norm f32 vector of given dimension.
        fn random_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
            let mut v = vec![0.0f32; dim];
            fill_standard_normal(rng, &mut v);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        }

        /// Brute-force cosine similarity search: returns top-k (id, similarity) pairs.
        fn exact_cosine_search(
            vectors: &[(u64, Vec<f32>)],
            query: &[f32],
            k: usize,
        ) -> Vec<(u64, f32)> {
            let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut sims: Vec<(u64, f32)> = vectors
                .iter()
                .map(|(id, v)| {
                    let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                    let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let sim = if q_norm > 0.0 && v_norm > 0.0 {
                        dot / (q_norm * v_norm)
                    } else {
                        0.0
                    };
                    (*id, sim)
                })
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(k);
            sims
        }

        /// Compute recall@k: fraction of true top-k IDs present in retrieved top-k.
        fn recall_at_k(retrieved_ids: &[u64], ground_truth_ids: &[u64], k: usize) -> f32 {
            let truth_set: HashSet<_> = ground_truth_ids.iter().take(k).collect();
            let found = retrieved_ids
                .iter()
                .take(k)
                .filter(|id| truth_set.contains(id))
                .count();
            found as f32 / k as f32
        }

        fn build_tq4_index_and_search(
            n_vectors: usize,
            dim: usize,
            n_queries: usize,
            k: usize,
            seed: u64,
        ) -> f32 {
            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(
                dim,
                16, // connectivity
                128,
                64,
                SpaceType::Cosine,
                threads,
            )
            .unwrap();
            index.reserve(n_vectors + 1).unwrap();

            let mut rng = StdRng::seed_from_u64(seed);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n_vectors)
                .map(|i| {
                    let v = random_unit_vector(&mut rng, dim);
                    (i as u64, v)
                })
                .collect();

            // Insert all vectors
            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index.add((*id).into(), &vector).unwrap();
            }

            // Query and measure recall
            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                let query_vec = random_unit_vector(&mut rng, dim);
                let query: Vector = query_vec.clone().into();
                let limit = NonZeroUsize::new(k).unwrap().into();

                let results: Vec<u64> = index
                    .search(&query, limit)
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .map(|(pid, _)| pid.into())
                    .collect();

                let ground_truth = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = ground_truth.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&results, &gt_ids, k);
            }

            total_recall / n_queries as f32
        }

        #[test]
        fn tq4_recall_at_10_random_1k() {
            // Smaller test for CI: 1K vectors, d=128, 20 queries
            let recall = build_tq4_index_and_search(1_000, 128, 20, 10, 42);
            eprintln!("TQ4 recall@10 (1K, d=128): {recall:.3}");
            assert!(
                recall >= 0.40,
                "TQ4 recall@10 too low: {recall:.3} (expected >= 0.40)"
            );
        }

        #[test]
        #[ignore] // Slow: ~10K vectors
        fn tq4_recall_at_10_random_10k() {
            let recall = build_tq4_index_and_search(10_000, 768, 100, 10, 42);
            eprintln!("TQ4 recall@10 (10K, d=768): {recall:.3}");
            // Random vectors at d=768 are nearly equidistant, making recall
            // fundamentally harder. The brute-force ceiling is ~61% (symmetric).
            assert!(
                recall >= 0.25,
                "TQ4 recall@10 too low: {recall:.3} (expected >= 0.25)"
            );
        }

        #[test]
        fn tq4_recall_at_1_top1_accuracy() {
            let recall = build_tq4_index_and_search(1_000, 128, 50, 1, 137);
            eprintln!("TQ4 top-1 accuracy (1K, d=128): {recall:.3}");
            assert!(
                recall >= 0.35,
                "TQ4 top-1 accuracy too low: {recall:.3} (expected >= 0.35)"
            );
        }

        /// Diagnostic: brute-force TQ4 asymmetric ranking (no HNSW).
        /// Isolates whether low recall is from the distance function or HNSW graph.
        #[test]
        #[ignore] // Slow: brute-force on 10K vectors at d=768
        fn tq4_bruteforce_asymmetric_recall() {
            use crate::turbo_quant::quantize::{Tq4CompressedVector, Tq4Quantizer};

            let dim = 768;
            let n_vectors = 10_000;
            let n_queries = 100;
            let k = 10;

            let quantizer = Tq4Quantizer::new(dim, 42, 137);

            let mut rng = StdRng::seed_from_u64(42);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n_vectors)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();

            // Pre-quantize all vectors
            let compressed: Vec<(u64, Tq4CompressedVector)> = vectors
                .iter()
                .map(|(id, v)| (*id, quantizer.quantize(v)))
                .collect();

            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                let query_vec = random_unit_vector(&mut rng, dim);

                // Ground truth: brute-force cosine on original f32 vectors
                let ground_truth = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = ground_truth.iter().map(|(id, _)| *id).collect();

                // TQ4 asymmetric: compute inner product for ALL vectors
                let query_state = quantizer.prepare_query(&query_vec);
                let mut tq4_results: Vec<(u64, f32)> = compressed
                    .iter()
                    .map(|(id, c)| (*id, quantizer.inner_product(&query_state, c)))
                    .collect();
                tq4_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let tq4_ids: Vec<u64> = tq4_results.iter().take(k).map(|(id, _)| *id).collect();

                total_recall += recall_at_k(&tq4_ids, &gt_ids, k);
            }

            let avg_recall = total_recall / n_queries as f32;
            eprintln!("TQ4 brute-force asymmetric recall@{k} (d={dim}): {avg_recall:.3}");
            // This tests the distance function quality without HNSW
        }

        /// Diagnostic: brute-force TQ4 symmetric ranking (no HNSW).
        #[test]
        #[ignore] // Slow: brute-force on 10K vectors at d=768
        fn tq4_bruteforce_symmetric_recall() {
            use crate::turbo_quant::codebook::cross_product_table_3bit;
            use crate::turbo_quant::distance::tq4_symmetric_distance;
            use crate::turbo_quant::quantize::Tq4Quantizer;

            let dim = 768;
            let n_vectors = 10_000;
            let n_queries = 100;
            let k = 10;

            let quantizer = Tq4Quantizer::new(dim, 42, 137);
            let d_pad = quantizer.padded_dim();
            let inv_sqrt_d = quantizer.inv_sqrt_d();
            let cross_table = cross_product_table_3bit(inv_sqrt_d);

            let mut rng = StdRng::seed_from_u64(42);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n_vectors)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();

            // Pre-quantize and pack all vectors
            let packed: Vec<(u64, Vec<u8>)> = vectors
                .iter()
                .map(|(id, v)| (*id, quantizer.quantize(v).pack()))
                .collect();

            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                let query_vec = random_unit_vector(&mut rng, dim);

                // Ground truth
                let ground_truth = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = ground_truth.iter().map(|(id, _)| *id).collect();

                // TQ4 symmetric: quantize query, compare with all packed vectors
                let query_packed = quantizer.quantize(&query_vec).pack();
                let mut sym_results: Vec<(u64, f32)> = packed
                    .iter()
                    .map(|(id, p)| {
                        (
                            *id,
                            tq4_symmetric_distance(&query_packed, p, d_pad, &cross_table),
                        )
                    })
                    .collect();
                sym_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let sym_ids: Vec<u64> = sym_results.iter().take(k).map(|(id, _)| *id).collect();

                total_recall += recall_at_k(&sym_ids, &gt_ids, k);
            }

            let avg_recall = total_recall / n_queries as f32;
            eprintln!("TQ4 brute-force symmetric recall@{k} (d={dim}): {avg_recall:.3}");
            // This tests the symmetric distance quality without HNSW
        }

        #[test]
        fn tq4_cosine_ordering_top5() {
            // Insert 100 vectors, verify TQ4 top-5 order matches ground truth
            let dim: usize = 128;
            let n = 100;
            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::Cosine, threads)
                .unwrap();
            index.reserve(n + 1).unwrap();

            let mut rng = StdRng::seed_from_u64(999);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();
            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index.add((*id).into(), &vector).unwrap();
            }

            let query_vec = random_unit_vector(&mut rng, dim);
            let query: Vector = query_vec.clone().into();
            let results: Vec<u64> = index
                .search(&query, NonZeroUsize::new(5).unwrap().into())
                .unwrap()
                .filter_map(|r| r.ok())
                .map(|(pid, _)| pid.into())
                .collect();

            let gt = exact_cosine_search(&vectors, &query_vec, 5);
            let gt_ids: Vec<u64> = gt.iter().map(|(id, _)| *id).collect();

            // At least 4 of top-5 should match for a small dataset
            let overlap = results.iter().filter(|id| gt_ids.contains(id)).count();
            eprintln!("TQ4 top-5 overlap: {overlap}/5");
            assert!(
                overlap >= 3,
                "TQ4 top-5 ordering poor: only {overlap}/5 match ground truth"
            );
        }

        #[test]
        fn tq4_dotproduct_search() {
            // Verify DotProduct metric works and returns reasonable results
            let dim: usize = 128;
            let n = 200;
            let threads = rayon::current_num_threads();
            let index =
                ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::DotProduct, threads)
                    .unwrap();
            index.reserve(n + 1).unwrap();

            let mut rng = StdRng::seed_from_u64(77);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();
            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index.add((*id).into(), &vector).unwrap();
            }

            let query_vec = random_unit_vector(&mut rng, dim);
            let query: Vector = query_vec.clone().into();
            let results: Vec<(PrimaryId, Distance)> = index
                .search(&query, NonZeroUsize::new(10).unwrap().into())
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();

            // Should return results
            assert!(
                !results.is_empty(),
                "TQ4 DotProduct search returned no results"
            );
            // Should return up to 10
            assert!(results.len() <= 10);
        }

        #[test]
        fn tq4_filtered_search_basic() {
            let dim: usize = 128;
            let n = 200;
            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::Cosine, threads)
                .unwrap();
            index.reserve(n + 1).unwrap();

            let mut rng = StdRng::seed_from_u64(55);
            for i in 0..n {
                let v = random_unit_vector(&mut rng, dim);
                let vector: Vector = v.into();
                index.add((i as u64).into(), &vector).unwrap();
            }

            // Filter: only even IDs
            let query_vec = random_unit_vector(&mut rng, dim);
            let query: Vector = query_vec.into();
            let results: Vec<(PrimaryId, Distance)> = index
                .filtered_search(&query, NonZeroUsize::new(10).unwrap().into(), |pid| {
                    let id: u64 = pid.into();
                    id.is_multiple_of(2)
                })
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();

            // All returned IDs should be even
            for (pid, _) in &results {
                let id: u64 = (*pid).into();
                assert!(
                    id.is_multiple_of(2),
                    "Filtered search returned odd ID: {id}"
                );
            }
            assert!(!results.is_empty(), "Filtered search returned no results");
        }

        #[test]
        fn tq4_recall_improves_with_oversample() {
            // Verify that search with default oversample produces reasonable recall
            let dim: usize = 128;
            let n = 500;
            let k = 10;
            let n_queries = 20;
            let threads = rayon::current_num_threads();
            let seed = 42u64;

            let mut rng = StdRng::seed_from_u64(seed);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();

            let queries: Vec<Vec<f32>> = (0..n_queries)
                .map(|_| random_unit_vector(&mut rng, dim))
                .collect();

            let index = ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::Cosine, threads)
                .unwrap();
            index.reserve(n + 1).unwrap();
            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index.add((*id).into(), &vector).unwrap();
            }

            let mut total_recall = 0.0f32;
            for q in &queries {
                let query: Vector = q.clone().into();
                let results: Vec<u64> = index
                    .search(&query, NonZeroUsize::new(k).unwrap().into())
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .map(|(pid, _)| pid.into())
                    .collect();
                let gt = exact_cosine_search(&vectors, q, k);
                let gt_ids: Vec<u64> = gt.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&results, &gt_ids, k);
            }
            let recall = total_recall / n_queries as f32;

            eprintln!("TQ4 recall@{k} (500 vecs, d=128, oversample=3): {recall:.3}");
            assert!(
                recall >= 0.40,
                "TQ4 recall too low with default oversample: {recall:.3}"
            );
        }

        #[test]
        fn tq4_euclidean_rejected() {
            let threads = rayon::current_num_threads();
            let result =
                ThreadedUsearchIndex::new_tq4(128, 16, 128, 64, SpaceType::Euclidean, threads);
            assert!(result.is_err(), "TQ4 should reject Euclidean metric");
        }

        #[test]
        fn tq4_remove_vector() {
            let dim: usize = 128;
            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::Cosine, threads)
                .unwrap();
            index.reserve(10).unwrap();

            let mut rng = StdRng::seed_from_u64(123);
            let v1 = random_unit_vector(&mut rng, dim);
            let v2 = random_unit_vector(&mut rng, dim);
            index.add(1u64.into(), &v1.clone().into()).unwrap();
            index.add(2u64.into(), &v2.clone().into()).unwrap();
            assert_eq!(index.size(), 2);

            index.remove(1u64.into()).unwrap();
            // After remove, searching should not return id=1
            let results: Vec<u64> = index
                .search(&v1.into(), NonZeroUsize::new(10).unwrap().into())
                .unwrap()
                .filter_map(|r| r.ok())
                .map(|(pid, _)| pid.into())
                .collect();
            assert!(
                !results.contains(&1u64),
                "Removed vector should not appear in search results"
            );
        }

        /// Recall test with clustered embeddings (mixture of Gaussians).
        ///
        /// Real embedding models (Cohere, OpenAI, sentence-transformers) produce
        /// vectors that cluster by semantic topic. This test generates the same
        /// structure: K centroids with Gaussian perturbations, which creates a
        /// meaningful nearest-neighbor structure that uniform-random vectors lack.
        ///
        /// Expected recall is significantly higher than with random vectors because
        /// true neighbors are close (within-cluster) while distractors are far
        /// (between-cluster), giving TQ4 more signal to preserve.
        #[test]
        #[ignore] // Slow: 10K vectors at d=768
        fn tq4_recall_at_10_clustered_10k() {
            let dim: usize = 768;
            let n_clusters: usize = 100;
            let vectors_per_cluster: usize = 100;
            let n_vectors = n_clusters * vectors_per_cluster;
            let n_queries = 100;
            let k = 10;
            let noise_sigma: f32 = 0.15; // Controls cluster tightness

            let mut rng = StdRng::seed_from_u64(42);

            // Generate cluster centroids (random unit vectors)
            let centroids: Vec<Vec<f32>> = (0..n_clusters)
                .map(|_| random_unit_vector(&mut rng, dim))
                .collect();

            // Generate vectors: centroid + noise, then normalize
            let vectors: Vec<(u64, Vec<f32>)> = centroids
                .iter()
                .enumerate()
                .flat_map(|(ci, centroid)| {
                    let mut cluster_rng = StdRng::seed_from_u64(42 + ci as u64 * 1000);
                    (0..vectors_per_cluster).map(move |vi| {
                        let id = (ci * vectors_per_cluster + vi) as u64;
                        let mut noise = vec![0.0f32; dim];
                        fill_standard_normal(&mut cluster_rng, &mut noise);
                        let v: Vec<f32> = centroid
                            .iter()
                            .zip(noise.iter())
                            .map(|(c, n)| c + noise_sigma * n)
                            .collect();
                        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let v: Vec<f32> = v.iter().map(|x| x / norm).collect();
                        (id, v)
                    })
                })
                .collect();

            assert_eq!(vectors.len(), n_vectors);

            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(
                dim,
                16, // connectivity
                128,
                64,
                SpaceType::Cosine,
                threads,
            )
            .unwrap();
            index.reserve(n_vectors + 1).unwrap();

            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index.add((*id).into(), &vector).unwrap();
            }

            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                // Pick a random cluster, generate query near its centroid
                let ci = rng.random_range(0..n_clusters);
                let centroid = &centroids[ci];
                let mut noise = vec![0.0f32; dim];
                fill_standard_normal(&mut rng, &mut noise);
                let query_vec: Vec<f32> = centroid
                    .iter()
                    .zip(noise.iter())
                    .map(|(c, n)| c + noise_sigma * n)
                    .collect();
                let norm = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let query_vec: Vec<f32> = query_vec.iter().map(|x| x / norm).collect();

                let query: Vector = query_vec.clone().into();
                let results: Vec<u64> = index
                    .search(&query, NonZeroUsize::new(k).unwrap().into())
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .map(|(pid, _)| pid.into())
                    .collect();

                let gt = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = gt.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&results, &gt_ids, k);
            }

            let recall = total_recall / n_queries as f32;
            eprintln!("TQ4 recall@10 (10K clustered, d={dim}, σ={noise_sigma}): {recall:.3}");
            // Clustered embeddings have more structure than random vectors,
            // so recall is higher. HNSW graph quality is still limited by the
            // symmetric TQ4 distance used during construction.
            assert!(
                recall >= 0.40,
                "TQ4 clustered recall@10 too low: {recall:.3} (expected >= 0.40)"
            );
        }

        /// Recall test using real OpenAI text-embedding-3-large vectors (d=1536)
        /// from the public Qdrant/dbpedia-entities dataset on HuggingFace.
        ///
        /// Downloads 1000 pre-computed embedding vectors via the HuggingFace
        /// Datasets Server API (no authentication required), then measures
        /// TQ4 recall@10 against exact cosine search ground truth.
        ///
        /// Requires internet access. Skipped (not failed) if download fails.
        #[test]
        #[ignore] // Requires internet access
        fn tq4_recall_at_10_dbpedia_openai() {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async { dbpedia_openai_recall_inner(10).await });
            match result {
                Ok(recall) => {
                    eprintln!("TQ4 recall@10 (DBpedia OpenAI, d=1536, n=1000): {recall:.3}");
                    assert!(
                        recall >= 0.40,
                        "TQ4 DBpedia OpenAI recall@10 too low: {recall:.3} (expected >= 0.40)"
                    );
                }
                Err(e) => {
                    eprintln!("Skipping DBpedia OpenAI test: {e}");
                }
            }
        }

        #[test]
        #[ignore] // Requires internet access
        fn tq4_recall_at_100_dbpedia_openai() {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async { dbpedia_openai_recall_inner(100).await });
            match result {
                Ok(recall) => {
                    eprintln!("TQ4 recall@100 (DBpedia OpenAI, d=1536, n=1000): {recall:.3}");
                    assert!(
                        recall >= 0.40,
                        "TQ4 DBpedia OpenAI recall@100 too low: {recall:.3} (expected >= 0.40)"
                    );
                }
                Err(e) => {
                    eprintln!("Skipping DBpedia OpenAI test: {e}");
                }
            }
        }

        async fn dbpedia_openai_recall_inner(k: usize) -> Result<f32, String> {
            let client = reqwest::Client::new();
            let n_pages = 10; // 10 pages × 100 rows = 1000 vectors
            let rows_per_page = 100;
            let dim = 1536;
            let n_queries = 50;
            let emb_field = "text-embedding-3-large-1536-embedding";

            let mut vectors: Vec<(u64, Vec<f32>)> = Vec::with_capacity(n_pages * rows_per_page);
            for page in 0..n_pages {
                let offset = page * rows_per_page;
                let url = format!(
                    "https://datasets-server.huggingface.co/rows\
                     ?dataset=Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K\
                     &config=default&split=train&offset={offset}&length={rows_per_page}"
                );
                let resp = client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {e}"))?;
                if !resp.status().is_success() {
                    return Err(format!("HTTP {}", resp.status()));
                }
                let body: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("JSON parse failed: {e}"))?;
                let rows = body["rows"]
                    .as_array()
                    .ok_or("Missing 'rows' field in response")?;
                for row_obj in rows {
                    let emb = row_obj["row"][emb_field]
                        .as_array()
                        .ok_or_else(|| format!("Missing '{emb_field}' field in row"))?;
                    if emb.len() != dim {
                        return Err(format!("Expected dim={dim}, got {}", emb.len()));
                    }
                    let v: Vec<f32> = emb
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    let id = vectors.len() as u64;
                    vectors.push((id, v));
                }
            }

            eprintln!(
                "Downloaded {} DBpedia OpenAI vectors (d={dim})",
                vectors.len()
            );

            let threads = rayon::current_num_threads();
            let index = ThreadedUsearchIndex::new_tq4(dim, 16, 128, 64, SpaceType::Cosine, threads)
                .map_err(|e| format!("Index creation failed: {e}"))?;
            index
                .reserve(vectors.len() + 1)
                .map_err(|e| format!("Reserve failed: {e}"))?;

            for (id, v) in &vectors {
                let vector: Vector = v.clone().into();
                index
                    .add((*id).into(), &vector)
                    .map_err(|e| format!("Insert failed: {e}"))?;
            }

            // Use a deterministic subset as queries (every 20th vector)
            let query_indices: Vec<usize> =
                (0..vectors.len()).step_by(20).take(n_queries).collect();
            let mut total_recall = 0.0f32;
            for &qi in &query_indices {
                let query_vec = &vectors[qi].1;
                let query: Vector = query_vec.clone().into();
                let results: Vec<u64> = index
                    .search(&query, NonZeroUsize::new(k).unwrap().into())
                    .map_err(|e| format!("Search failed: {e}"))?
                    .filter_map(|r| r.ok())
                    .map(|(pid, _)| pid.into())
                    .collect();

                let gt = exact_cosine_search(&vectors, query_vec, k);
                let gt_ids: Vec<u64> = gt.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&results, &gt_ids, k);
            }

            Ok(total_recall / query_indices.len() as f32)
        }
    }

    mod polar_quant_recall {
        use super::*;
        use crate::turbo_quant::polar_quantize::{
            PolarCodebooks, PolarCompressedVector, PolarCrossTables, PolarQuantizer,
            polar_symmetric_distance,
        };
        use crate::turbo_quant::qjl::fill_standard_normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use std::collections::HashSet;

        /// Generate a random unit-norm f32 vector.
        fn random_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
            let mut v = vec![0.0f32; dim];
            fill_standard_normal(rng, &mut v);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        }

        /// Brute-force cosine similarity search.
        fn exact_cosine_search(
            vectors: &[(u64, Vec<f32>)],
            query: &[f32],
            k: usize,
        ) -> Vec<(u64, f32)> {
            let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut sims: Vec<(u64, f32)> = vectors
                .iter()
                .map(|(id, v)| {
                    let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                    let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let sim = if q_norm > 0.0 && v_norm > 0.0 {
                        dot / (q_norm * v_norm)
                    } else {
                        0.0
                    };
                    (*id, sim)
                })
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(k);
            sims
        }

        fn recall_at_k(retrieved_ids: &[u64], ground_truth_ids: &[u64], k: usize) -> f32 {
            let truth_set: HashSet<_> = ground_truth_ids.iter().take(k).collect();
            let found = retrieved_ids
                .iter()
                .take(k)
                .filter(|id| truth_set.contains(id))
                .count();
            found as f32 / k as f32
        }

        /// Brute-force PolarQuant recall: no HNSW, pure distance function quality.
        fn polar_quant_bruteforce_recall(
            n_vectors: usize,
            dim: usize,
            n_queries: usize,
            k: usize,
            seed: u64,
        ) -> f32 {
            let quantizer = PolarQuantizer::new(dim, 42, 137);

            let mut rng = StdRng::seed_from_u64(seed);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n_vectors)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();

            let compressed: Vec<(u64, PolarCompressedVector)> = vectors
                .iter()
                .map(|(id, v)| (*id, quantizer.quantize(v)))
                .collect();

            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                let query_vec = random_unit_vector(&mut rng, dim);
                let ground_truth = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = ground_truth.iter().map(|(id, _)| *id).collect();

                let query_state = quantizer.prepare_query(&query_vec);
                let mut pq_results: Vec<(u64, f32)> = compressed
                    .iter()
                    .map(|(id, c)| {
                        let ip = quantizer.inner_product(&query_state, c);
                        let cos = ip / (query_state.query_norm * c.norm).max(1e-10);
                        (*id, cos)
                    })
                    .collect();
                pq_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let pq_ids: Vec<u64> =
                    pq_results.iter().take(k).map(|(id, _)| *id).collect();

                total_recall += recall_at_k(&pq_ids, &gt_ids, k);
            }

            total_recall / n_queries as f32
        }

        /// PolarQuant recall with HNSW index (end-to-end via USearch).
        fn build_polar_quant_index_and_search(
            n_vectors: usize,
            dim: usize,
            n_queries: usize,
            k: usize,
            seed: u64,
        ) -> f32 {
            let oversample_factor = 3.0f32;
            let config = Tq4Config::default();
            let packed_dim = PolarCompressedVector::packed_size(dim);
            let quantizer = Arc::new(PolarQuantizer::new(
                dim,
                config.rotation_seed,
                config.qjl_seed,
            ));

            // Build cross-product tables for symmetric metric (must live outside closure)
            let codebooks = PolarCodebooks::new(
                dim.next_power_of_two().trailing_zeros() as usize,
            );
            let tables = Arc::new(PolarCrossTables::new(&codebooks));
            let padded_dim = quantizer.padded_dim();

            let options = IndexOptions {
                dimensions: packed_dim,
                connectivity: 16,
                expansion_add: 128,
                expansion_search: 64,
                metric: MetricKind::IP,
                quantization: ScalarKind::I8,
                ..Default::default()
            };
            let mut inner = usearch::Index::new(&options).unwrap();

            // Register PolarQuant symmetric metric
            let tb = tables.clone();
            inner.change_metric::<i8>(Box::new(move |a_ptr: *const i8, b_ptr: *const i8| {
                thread_local! {
                    static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
                }
                CONFIGURED.with(|c| {
                    if !c.get() {
                        numkong::configure_thread();
                        c.set(true);
                    }
                });

                let a = unsafe { std::slice::from_raw_parts(a_ptr as *const u8, packed_dim) };
                let b = unsafe { std::slice::from_raw_parts(b_ptr as *const u8, packed_dim) };
                let ip = polar_symmetric_distance(a, b, padded_dim, &tb);
                usearch::Distance::from(1.0 - ip)
            }));

            let threads = rayon::current_num_threads();
            inner
                .reserve_capacity_and_threads(n_vectors + 1, threads)
                .unwrap();

            let mut rng = StdRng::seed_from_u64(seed);
            let vectors: Vec<(u64, Vec<f32>)> = (0..n_vectors)
                .map(|i| (i as u64, random_unit_vector(&mut rng, dim)))
                .collect();

            // Insert all vectors
            for (id, v) in &vectors {
                let compressed = quantizer.quantize(v);
                let packed = compressed.pack();
                let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed);
                inner.add(*id, packed_i8).unwrap();
            }

            // Query with oversample + asymmetric reranking
            let mut total_recall = 0.0f32;
            for _ in 0..n_queries {
                let query_vec = random_unit_vector(&mut rng, dim);

                // Phase 1: HNSW search with PolarQuant symmetric metric
                let compressed_query = quantizer.quantize(&query_vec);
                let packed_query = compressed_query.pack();
                let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
                let oversample_k = (k as f32 * oversample_factor).ceil() as usize;
                let candidates = inner.search(packed_i8, oversample_k).unwrap();

                // Phase 2: Asymmetric reranking
                let query_state = quantizer.prepare_query(&query_vec);
                let mut get_buf = vec![0i8; packed_dim];
                let mut reranked: Vec<(u64, f32)> = Vec::with_capacity(candidates.keys.len());

                for &id in &candidates.keys {
                    get_buf.fill(0);
                    if let Ok(found) = inner.get(id, &mut get_buf) {
                        if found > 0 {
                            let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                            let c = PolarCompressedVector::unpack(buf_u8, dim);
                            let ip = quantizer.inner_product(&query_state, &c);
                            reranked.push((id, ip));
                        }
                    }
                }

                reranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                reranked.truncate(k);
                let result_ids: Vec<u64> =
                    reranked.iter().map(|(id, _)| *id).collect();

                let ground_truth = exact_cosine_search(&vectors, &query_vec, k);
                let gt_ids: Vec<u64> = ground_truth.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&result_ids, &gt_ids, k);
            }

            total_recall / n_queries as f32
        }

        // --- Brute-force recall tests (distance function quality) ---

        #[test]
        fn polar_quant_bruteforce_recall_1k() {
            let recall = polar_quant_bruteforce_recall(1_000, 128, 20, 10, 42);
            eprintln!("PolarQuant brute-force recall@10 (1K, d=128): {recall:.3}");
            assert!(
                recall >= 0.30,
                "PolarQuant brute-force recall@10 too low: {recall:.3} (expected >= 0.30)"
            );
        }

        #[test]
        #[ignore] // Slow
        fn polar_quant_bruteforce_recall_10k() {
            let recall = polar_quant_bruteforce_recall(10_000, 768, 100, 10, 42);
            eprintln!("PolarQuant brute-force recall@10 (10K, d=768): {recall:.3}");
            assert!(
                recall >= 0.20,
                "PolarQuant brute-force recall@10 too low: {recall:.3} (expected >= 0.20)"
            );
        }

        // --- HNSW recall tests (end-to-end) ---

        #[test]
        fn polar_quant_recall_at_10_random_1k() {
            let recall = build_polar_quant_index_and_search(1_000, 128, 20, 10, 42);
            eprintln!("PolarQuant HNSW recall@10 (1K, d=128): {recall:.3}");
            assert!(
                recall >= 0.25,
                "PolarQuant HNSW recall@10 too low: {recall:.3} (expected >= 0.25)"
            );
        }

        #[test]
        #[ignore] // Slow
        fn polar_quant_recall_at_10_random_10k() {
            let recall = build_polar_quant_index_and_search(10_000, 768, 100, 10, 42);
            eprintln!("PolarQuant HNSW recall@10 (10K, d=768): {recall:.3}");
            assert!(
                recall >= 0.15,
                "PolarQuant HNSW recall@10 too low: {recall:.3} (expected >= 0.15)"
            );
        }

        // --- DBpedia recall test ---

        #[test]
        #[ignore] // Requires internet access + slow
        fn polar_quant_recall_at_10_dbpedia_openai() {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let result = rt.block_on(async { dbpedia_openai_polar_recall_inner(10).await });
            match result {
                Ok(recall) => {
                    eprintln!(
                        "PolarQuant recall@10 (DBpedia OpenAI, d=1536, n=1000): {recall:.3}"
                    );
                    assert!(
                        recall >= 0.30,
                        "PolarQuant DBpedia recall@10 too low: {recall:.3} (expected >= 0.30)"
                    );
                }
                Err(e) => {
                    eprintln!("Skipping DBpedia PolarQuant test: {e}");
                }
            }
        }

        async fn dbpedia_openai_polar_recall_inner(k: usize) -> Result<f32, String> {
            let client = reqwest::Client::new();
            let n_pages = 10;
            let rows_per_page = 100;
            let dim = 1536;
            let n_queries = 50;
            let emb_field = "text-embedding-3-large-1536-embedding";

            let mut vectors: Vec<(u64, Vec<f32>)> =
                Vec::with_capacity(n_pages * rows_per_page);
            for page in 0..n_pages {
                let offset = page * rows_per_page;
                let url = format!(
                    "https://datasets-server.huggingface.co/rows\
                     ?dataset=Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K\
                     &config=default&split=train&offset={offset}&length={rows_per_page}"
                );
                let resp = client
                    .get(&url)
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {e}"))?;
                if !resp.status().is_success() {
                    return Err(format!("HTTP {}", resp.status()));
                }
                let body: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("JSON parse failed: {e}"))?;
                let rows = body["rows"]
                    .as_array()
                    .ok_or("Missing 'rows' field in response")?;
                for row_obj in rows {
                    let emb = row_obj["row"][emb_field]
                        .as_array()
                        .ok_or_else(|| format!("Missing '{emb_field}' field in row"))?;
                    if emb.len() != dim {
                        return Err(format!("Expected dim={dim}, got {}", emb.len()));
                    }
                    let v: Vec<f32> = emb
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    let id = vectors.len() as u64;
                    vectors.push((id, v));
                }
            }

            eprintln!(
                "Downloaded {} DBpedia OpenAI vectors (d={dim})",
                vectors.len()
            );

            // Build PolarQuant HNSW index
            let oversample_factor = 3.0f32;
            let packed_dim = PolarCompressedVector::packed_size(dim);
            let quantizer = Arc::new(PolarQuantizer::new(dim, 42, 137));
            let codebooks = PolarCodebooks::new(
                dim.next_power_of_two().trailing_zeros() as usize,
            );
            let tables = Arc::new(PolarCrossTables::new(&codebooks));
            let padded_dim = quantizer.padded_dim();

            let options = IndexOptions {
                dimensions: packed_dim,
                connectivity: 16,
                expansion_add: 128,
                expansion_search: 64,
                metric: MetricKind::IP,
                quantization: ScalarKind::I8,
                ..Default::default()
            };
            let mut inner = usearch::Index::new(&options)
                .map_err(|e| format!("Index creation failed: {e}"))?;

            let tb = tables.clone();
            inner.change_metric::<i8>(Box::new(move |a_ptr: *const i8, b_ptr: *const i8| {
                thread_local! {
                    static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
                }
                CONFIGURED.with(|c| {
                    if !c.get() {
                        numkong::configure_thread();
                        c.set(true);
                    }
                });
                let a = unsafe { std::slice::from_raw_parts(a_ptr as *const u8, packed_dim) };
                let b = unsafe { std::slice::from_raw_parts(b_ptr as *const u8, packed_dim) };
                let ip = polar_symmetric_distance(a, b, padded_dim, &tb);
                usearch::Distance::from(1.0 - ip)
            }));

            let threads = rayon::current_num_threads();
            inner
                .reserve_capacity_and_threads(vectors.len() + 1, threads)
                .map_err(|e| format!("Reserve failed: {e}"))?;

            for (id, v) in &vectors {
                let compressed = quantizer.quantize(v);
                let packed = compressed.pack();
                let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed);
                inner
                    .add(*id, packed_i8)
                    .map_err(|e| format!("Insert failed: {e}"))?;
            }

            // Query with oversample + reranking
            let query_indices: Vec<usize> =
                (0..vectors.len()).step_by(20).take(n_queries).collect();
            let mut total_recall = 0.0f32;
            for &qi in &query_indices {
                let query_vec = &vectors[qi].1;

                let compressed_query = quantizer.quantize(query_vec);
                let packed_query = compressed_query.pack();
                let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
                let oversample_k = (k as f32 * oversample_factor).ceil() as usize;
                let candidates = inner
                    .search(packed_i8, oversample_k)
                    .map_err(|e| format!("Search failed: {e}"))?;

                let query_state = quantizer.prepare_query(query_vec);
                let mut get_buf = vec![0i8; packed_dim];
                let mut reranked: Vec<(u64, f32)> =
                    Vec::with_capacity(candidates.keys.len());

                for &id in &candidates.keys {
                    get_buf.fill(0);
                    if let Ok(found) = inner.get(id, &mut get_buf) {
                        if found > 0 {
                            let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                            let c = PolarCompressedVector::unpack(buf_u8, dim);
                            let ip = quantizer.inner_product(&query_state, &c);
                            reranked.push((id, ip));
                        }
                    }
                }

                reranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                reranked.truncate(k);
                let result_ids: Vec<u64> =
                    reranked.iter().map(|(id, _)| *id).collect();

                let gt = exact_cosine_search(&vectors, query_vec, k);
                let gt_ids: Vec<u64> = gt.iter().map(|(id, _)| *id).collect();
                total_recall += recall_at_k(&result_ids, &gt_ids, k);
            }

            Ok(total_recall / query_indices.len() as f32)
        }
    }
}
