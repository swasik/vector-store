/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Dimensions;
use crate::Distance;
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
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::Instrument;
use tracing::debug;
use tracing::debug_span;
use tracing::error;
use tracing::trace;
use tracing::warn;

/// Trait abstracting the low-level vector index operations for cuVS.
///
/// In this first version, a brute-force CPU implementation is provided.
/// When real GPU support is added, a CUDA-backed implementation will
/// implement this same trait, allowing the actor logic to remain unchanged.
#[cfg_attr(not(test), allow(dead_code))]
trait CuvsVectorIndex: Send + Sync {
    fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()>;
    fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()>;
    fn search(
        &self,
        vector: &Vector,
        limit: Limit,
    ) -> anyhow::Result<Vec<(PrimaryId, Distance)>>;
    fn size(&self) -> usize;
    /// Search multiple queries in a single batch. GPU backends override
    /// this to batch queries into a single kernel launch.
    fn search_batch(
        &self,
        queries: &[(&Vector, Limit)],
    ) -> Vec<anyhow::Result<Vec<(PrimaryId, Distance)>>> {
        queries.iter().map(|(v, l)| self.search(v, l.clone())).collect()
    }
}

/// Brute-force CPU implementation of the cuVS vector index.
///
/// Stores vectors in memory and performs exhaustive nearest-neighbor search.
/// This is the initial implementation used for integration testing and
/// development. A GPU-accelerated implementation (CAGRA/IVF) will replace
/// the inner search logic when the `gpu` feature is enabled.
#[cfg_attr(feature = "gpu", allow(dead_code))]
struct BruteForceIndex {
    vectors: RwLock<BTreeMap<PrimaryId, Vec<f32>>>,
    space_type: SpaceType,
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
impl BruteForceIndex {
    fn new(space_type: SpaceType) -> Self {
        Self {
            vectors: RwLock::new(BTreeMap::new()),
            space_type,
        }
    }
}

impl CuvsVectorIndex for BruteForceIndex {
    fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()> {
        self.vectors
            .write()
            .unwrap()
            .insert(primary_id, vector.as_ref().to_vec());
        Ok(())
    }

    fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()> {
        self.vectors.write().unwrap().remove(&primary_id);
        Ok(())
    }

    fn search(
        &self,
        vector: &Vector,
        limit: Limit,
    ) -> anyhow::Result<Vec<(PrimaryId, Distance)>> {
        let vectors = self.vectors.read().unwrap();
        let query = vector.as_ref();

        let mut results: Vec<(PrimaryId, Distance)> = vectors
            .iter()
            .map(|(&id, stored)| {
                compute_distance(query, stored, self.space_type).map(|dist| (id, dist))
            })
            .collect::<Result<Vec<_>, _>>()?;

        results.sort_by(|a, b| {
            let a_val: f32 = a.1.into();
            let b_val: f32 = b.1.into();
            a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit.0.get());
        Ok(results)
    }

    fn size(&self) -> usize {
        self.vectors.read().unwrap().len()
    }
}

/// Creates the appropriate vector index implementation for a new partition.
///
/// When the `gpu` feature is enabled, attempts to load libcuvs_c.so and create
/// a CAGRA graph-based ANN index. If libcuvs is not available, falls back to
/// GPU brute-force via cuBLAS. Without `gpu`, uses CPU brute-force.
fn create_partition_index(
    space_type: SpaceType,
    dimensions: Dimensions,
) -> Arc<dyn CuvsVectorIndex> {
    #[cfg(feature = "gpu")]
    {
        match gpu::load_cuvs_lib() {
            Ok(lib) => {
                debug!("using CAGRA index (libcuvs_c.so loaded)");
                Arc::new(gpu::CagraIndex::new(
                    space_type,
                    dimensions.0.get(),
                    lib,
                ))
            }
            Err(err) => {
                warn!("libcuvs_c.so not available ({err}), falling back to GPU brute-force");
                Arc::new(gpu::GpuBruteForceIndex::new(
                    space_type,
                    dimensions.0.get(),
                ))
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = dimensions;
        Arc::new(BruteForceIndex::new(space_type))
    }
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
fn compute_distance(a: &[f32], b: &[f32], space_type: SpaceType) -> anyhow::Result<Distance> {
    match space_type {
        SpaceType::Euclidean => {
            let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            Distance::new_euclidean(sum)
        }
        SpaceType::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos_sim = if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot / (norm_a * norm_b)
            };
            // Cosine distance in [0.0, 2.0], matching usearch convention
            Distance::new_cosine(1.0 - cos_sim)
        }
        SpaceType::DotProduct => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            // Inner product distance, matching usearch convention: distance = 1 - dot
            Distance::new_dot_product(1.0 - dot)
        }
        SpaceType::Hamming => {
            anyhow::bail!("cuVS backend does not support Hamming distance")
        }
    }
}

/// Configuration for batching mutations before dispatching to the index.
///
/// GPU-accelerated backends achieve peak throughput when operations are batched
/// into a single kernel launch. Mutations are accumulated until either
/// `batch_size` is reached or `batch_timeout` elapses, then flushed together
/// in one blocking task.
pub(crate) struct BatchConfig {
    pub batch_size: usize,
    pub batch_timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            batch_timeout: Duration::from_millis(5),
        }
    }
}

struct PendingMutation {
    partition: Arc<PartitionState>,
    size: Arc<AtomicUsize>,
    op: MutationOp,
}

enum MutationOp {
    Add {
        primary_id: PrimaryId,
        embedding: Vector,
        _in_progress: Option<crate::AsyncInProgress>,
    },
    Remove {
        primary_id: PrimaryId,
        _in_progress: Option<crate::AsyncInProgress>,
    },
}

pub struct CuvsIndexFactory {
    tokio_semaphore: Arc<Semaphore>,
    batch_config: BatchConfig,
}

impl IndexFactory for CuvsIndexFactory {
    fn create_index(
        &self,
        index: IndexConfiguration,
        table: Arc<RwLock<Table>>,
        memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<Index>> {
        if index.space_type == SpaceType::Hamming {
            anyhow::bail!("cuVS backend does not support Hamming space type");
        }
        if index.quantization == Quantization::B1 {
            anyhow::bail!("cuVS backend does not support B1 quantization");
        }

        new(
            index.space_type,
            index.key,
            index.dimensions,
            table,
            Arc::clone(&self.tokio_semaphore),
            memory,
            &self.batch_config,
        )
    }

    fn index_engine_version(&self) -> String {
        #[cfg(feature = "gpu")]
        {
            "cuvs-gpu-cagra-v1".to_string()
        }
        #[cfg(not(feature = "gpu"))]
        {
            "cuvs-brute-force-v1".to_string()
        }
    }
}

pub fn new_cuvs(tokio_semaphore: Arc<Semaphore>, batch_config: BatchConfig) -> CuvsIndexFactory {
    CuvsIndexFactory {
        tokio_semaphore,
        batch_config,
    }
}

struct PartitionState {
    partition_id: PartitionId,
    idx: Arc<dyn CuvsVectorIndex>,
}

impl PartitionState {
    fn new(partition_id: PartitionId, idx: Arc<dyn CuvsVectorIndex>) -> Self {
        Self { partition_id, idx }
    }
}

struct IndexState {
    dimensions: Dimensions,
    size: Arc<AtomicUsize>,
}

impl IndexState {
    fn new(dimensions: Dimensions) -> Self {
        Self {
            dimensions,
            size: Arc::new(AtomicUsize::new(0)),
        }
    }
}

fn new(
    space_type: SpaceType,
    index_key: IndexKey,
    dimensions: Dimensions,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    tokio_semaphore: Arc<Semaphore>,
    memory: mpsc::Sender<Memory>,
    batch_config: &BatchConfig,
) -> anyhow::Result<mpsc::Sender<Index>> {
    const CHANNEL_SIZE: usize = 10;
    let (tx, mut rx) = mpsc::channel(CHANNEL_SIZE);
    let batch_size = batch_config.batch_size;
    let batch_timeout = batch_config.batch_timeout;

    tokio::spawn(
        {
            let index_key = index_key.clone();
            async move {
                debug!("starting (batch_size={batch_size}, batch_timeout={batch_timeout:?})");
                let mut states: BTreeMap<IndexId, IndexState> = BTreeMap::new();
                let mut partitions: BTreeMap<PartitionId, Arc<PartitionState>> = BTreeMap::new();
                let mut allocate_prev = Allocate::Can;
                let mut pending: Vec<PendingMutation> = Vec::new();
                let mut flush_deadline: Option<tokio::time::Instant> = None;

                loop {
                    let msg = tokio::select! {
                        biased;
                        msg = rx.recv() => {
                            match msg {
                                Some(msg) => msg,
                                None => {
                                    flush_batch(&mut pending, &tokio_semaphore).await;
                                    break;
                                }
                            }
                        }
                        _ = sleep_until_deadline(flush_deadline) => {
                            flush_batch(&mut pending, &tokio_semaphore).await;
                            flush_deadline = None;
                            continue;
                        }
                    };

                    if !check_memory_allocation(&msg, &memory, &mut allocate_prev, &index_key)
                        .await
                    {
                        continue;
                    }

                    let Some((state, partition, msg)) = preprocess(
                        space_type,
                        &mut states,
                        &mut partitions,
                        table.as_ref(),
                        dimensions,
                        msg,
                    ) else {
                        continue;
                    };

                    match msg {
                        Index::AddVector {
                            primary_id,
                            embedding,
                            in_progress,
                            ..
                        } => {
                            pending.push(PendingMutation {
                                partition,
                                size: Arc::clone(&state.size),
                                op: MutationOp::Add {
                                    primary_id,
                                    embedding,
                                    _in_progress: in_progress,
                                },
                            });
                            if flush_deadline.is_none() {
                                flush_deadline =
                                    Some(tokio::time::Instant::now() + batch_timeout);
                            }
                            if pending.len() >= batch_size {
                                flush_batch(&mut pending, &tokio_semaphore).await;
                                flush_deadline = None;
                            }
                        }
                        Index::RemoveVector {
                            primary_id,
                            in_progress,
                            ..
                        } => {
                            pending.push(PendingMutation {
                                partition,
                                size: Arc::clone(&state.size),
                                op: MutationOp::Remove {
                                    primary_id,
                                    _in_progress: in_progress,
                                },
                            });
                            if flush_deadline.is_none() {
                                flush_deadline =
                                    Some(tokio::time::Instant::now() + batch_timeout);
                            }
                            if pending.len() >= batch_size {
                                flush_batch(&mut pending, &tokio_semaphore).await;
                                flush_deadline = None;
                            }
                        }
                        msg @ (Index::Ann { .. } | Index::FilteredAnn { .. }) => {
                            // Collect this search and any immediately-available
                            // ones for batch dispatch. Pending mutations are NOT
                            // flushed before search — searches use the last
                            // committed index state (stale-read optimisation).
                            let dims = state.dimensions;
                            let mut search_items: Vec<(
                                Dimensions,
                                Arc<PartitionState>,
                                Index,
                            )> = vec![(dims, partition, msg)];

                            // Drain the channel to coalesce searches arriving
                            // at roughly the same time into one GPU batch.
                            while let Ok(next_msg) = rx.try_recv() {
                                let Some((ns, np, nm)) = preprocess(
                                    space_type,
                                    &mut states,
                                    &mut partitions,
                                    table.as_ref(),
                                    dimensions,
                                    next_msg,
                                ) else {
                                    continue;
                                };
                                match nm {
                                    m @ (Index::Ann { .. }
                                    | Index::FilteredAnn { .. }) => {
                                        search_items
                                            .push((ns.dimensions, np, m));
                                    }
                                    Index::AddVector {
                                        primary_id,
                                        embedding,
                                        in_progress,
                                        ..
                                    } => {
                                        pending.push(PendingMutation {
                                            partition: np,
                                            size: Arc::clone(&ns.size),
                                            op: MutationOp::Add {
                                                primary_id,
                                                embedding,
                                                _in_progress: in_progress,
                                            },
                                        });
                                        if flush_deadline.is_none() {
                                            flush_deadline = Some(
                                                tokio::time::Instant::now()
                                                    + batch_timeout,
                                            );
                                        }
                                        if pending.len() >= batch_size {
                                            flush_batch(
                                                &mut pending,
                                                &tokio_semaphore,
                                            )
                                            .await;
                                            flush_deadline = None;
                                        }
                                    }
                                    Index::RemoveVector {
                                        primary_id,
                                        in_progress,
                                        ..
                                    } => {
                                        pending.push(PendingMutation {
                                            partition: np,
                                            size: Arc::clone(&ns.size),
                                            op: MutationOp::Remove {
                                                primary_id,
                                                _in_progress: in_progress,
                                            },
                                        });
                                        if flush_deadline.is_none() {
                                            flush_deadline = Some(
                                                tokio::time::Instant::now()
                                                    + batch_timeout,
                                            );
                                        }
                                        if pending.len() >= batch_size {
                                            flush_batch(
                                                &mut pending,
                                                &tokio_semaphore,
                                            )
                                            .await;
                                            flush_deadline = None;
                                        }
                                    }
                                    _ => unreachable!("handled by preprocess"),
                                }
                            }

                            dispatch_search_batch(
                                search_items,
                                &table,
                                &tokio_semaphore,
                            )
                            .await;
                        }
                        Index::Count { .. } | Index::RemovePartition { .. } => {
                            unreachable!("handled by preprocess")
                        }
                    }
                }

                debug!("finished");
            }
        }
        .instrument(debug_span!("cuvs", "{index_key}")),
    );

    Ok(tx)
}

fn preprocess<'a, T>(
    space_type: SpaceType,
    states: &'a mut BTreeMap<IndexId, IndexState>,
    partitions: &mut BTreeMap<PartitionId, Arc<PartitionState>>,
    table: &RwLock<T>,
    dimensions: Dimensions,
    msg: Index,
) -> Option<(&'a mut IndexState, Arc<PartitionState>, Index)>
where
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
            let idx = create_partition_index(space_type, dimensions);
            let partition = Arc::new(PartitionState::new(partition_id, idx));
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
                warn!(
                    "partition id not found for index key {index_key:?} during filtered ann"
                );
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
            // If all restrictions were resolved by the partition lookup
            // (e.g. partition key equality), downgrade to a plain Ann.
            // Otherwise reject: cuVS does not support filtered search.
            let msg = if let Some(_restrictions) = restrictions {
                warn!(
                    "cuVS: rejecting filtered ann for index {index_key:?}: \
                     unresolvable non-partition-key restrictions remain"
                );
                _ = tx.send(Err(anyhow!(
                    "cuVS backend does not support filtered search (index {index_key:?})"
                )));
                return None;
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
            partitions.remove(&partition_id);
            None
        }
    }
}

async fn sleep_until_deadline(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

/// Flush all pending mutations in a single blocking task.
///
/// This batching approach is designed for GPU acceleration: when a real GPU
/// backend is used, the entire batch is transferred to device memory and
/// processed in one kernel launch, amortizing the host-device transfer cost.
async fn flush_batch(pending: &mut Vec<PendingMutation>, tokio_semaphore: &Arc<Semaphore>) {
    if pending.is_empty() {
        return;
    }
    let ops = std::mem::take(pending);
    let batch_len = ops.len();
    let permit = Arc::clone(tokio_semaphore).acquire_owned().await.unwrap();

    let result = tokio::task::spawn_blocking(move || {
        trace!("flushing batch of {batch_len} mutations");
        for op in ops {
            match op.op {
                MutationOp::Add {
                    primary_id,
                    embedding,
                    _in_progress,
                } => {
                    if let Err(err) = op.partition.idx.add(primary_id, &embedding) {
                        warn!("batch add: unable to add embedding: {err}");
                    } else {
                        op.size.fetch_add(1, Ordering::Relaxed);
                    }
                }
                MutationOp::Remove {
                    primary_id,
                    _in_progress,
                } => {
                    if let Err(err) = op.partition.idx.remove(primary_id) {
                        warn!("batch remove: unable to remove embedding: {err}");
                    } else {
                        op.size.fetch_sub(1, Ordering::Relaxed);
                    }
                }
            }
        }
        drop(permit);
    });
    if let Err(err) = result.await {
        error!("cuVS batch flush task panicked: {err}");
    }
}

async fn dispatch_search_batch<T>(
    items: Vec<(Dimensions, Arc<PartitionState>, Index)>,
    table: &Arc<RwLock<T>>,
    tokio_semaphore: &Arc<Semaphore>,
) where
    T: TableSearch + Send + Sync + 'static,
{
    if items.is_empty() {
        return;
    }
    let permit = Arc::clone(tokio_semaphore).acquire_owned().await.unwrap();
    let table = Arc::clone(table);
    let batch_len = items.len();

    tokio::spawn(async move {
        crate::move_to_the_end_of_async_runtime_queue().await;
        let result = tokio::task::spawn_blocking(move || {
            trace!("dispatching search batch of {batch_len} queries");
            process_search_batch(items, table);
        })
        .await;
        if let Err(err) = result {
            error!("cuVS search batch task panicked: {err}");
        }
        drop(permit);
    });
}

fn process_search_batch<T>(
    items: Vec<(Dimensions, Arc<PartitionState>, Index)>,
    table: Arc<RwLock<T>>,
) where
    T: TableSearch + Send + Sync + 'static,
{
    // Single-item fast path: reuse existing process_search.
    if items.len() == 1 {
        let (dims, partition, msg) = items.into_iter().next().unwrap();
        process_search(partition, table, dims, msg);
        return;
    }

    // Group by partition for batch GPU search.
    let mut groups: Vec<(Arc<PartitionState>, Vec<(Dimensions, Index)>)> = Vec::new();
    for (dims, partition, msg) in items {
        let pid = partition.partition_id;
        if let Some(group) = groups.iter_mut().find(|g| g.0.partition_id == pid) {
            group.1.push((dims, msg));
        } else {
            groups.push((partition, vec![(dims, msg)]));
        }
    }

    for (partition, group_items) in groups {
        if group_items.len() == 1 {
            let (dims, msg) = group_items.into_iter().next().unwrap();
            process_search(partition, table.clone(), dims, msg);
            continue;
        }

        // Collect valid queries for batch search.
        struct BatchEntry {
            embedding: Vector,
            limit: Limit,
            tx: oneshot::Sender<AnnR>,
        }
        let mut entries: Vec<BatchEntry> = Vec::new();
        let dims = group_items[0].0;

        for (_, msg) in group_items {
            match msg {
                Index::Ann {
                    embedding,
                    limit,
                    tx,
                    ..
                } => {
                    if let Err(err) = validator::embedding_dimensions(&embedding, dims) {
                        warn!("validate_dimensions: {err}");
                        let _ = tx.send(Err(err));
                        continue;
                    }
                    entries.push(BatchEntry {
                        embedding,
                        limit,
                        tx,
                    });
                }
                Index::FilteredAnn { tx, index_key, .. } => {
                    warn!(
                        "cuVS backend does not support filtered search \
                         (index {index_key:?})"
                    );
                    let _ = tx.send(Err(anyhow!(
                        "cuVS backend does not support filtered search \
                         (index {index_key:?})"
                    )));
                }
                _ => unreachable!("only search operations in batch"),
            }
        }

        if entries.is_empty() {
            continue;
        }

        // Batch search via the index.
        let queries: Vec<(&Vector, Limit)> =
            entries.iter().map(|e| (&e.embedding, e.limit.clone())).collect();
        let results = partition.idx.search_batch(&queries);

        // Map results through the table and send responses.
        let table_guard = table.read().unwrap();
        for (entry, result) in entries.into_iter().zip(results) {
            let mapped = result.and_then(|matches| {
                let (primary_keys, distances): (Vec<_>, Vec<_>) = matches
                    .into_iter()
                    .filter_map(|(primary_id, distance)| {
                        table_guard
                            .primary_key(partition.partition_id, primary_id)
                            .map(|pk| (pk, distance))
                    })
                    .unzip();
                Ok((primary_keys, distances))
            });
            if let Err(ref err) = mapped {
                warn!(
                    "batch ann search failed for partition {:?}: {err:#}",
                    partition.partition_id
                );
            }
            entry
                .tx
                .send(mapped)
                .unwrap_or_else(|_| debug!("ann: unable to send response (receiver dropped)"));
        }
    }
}

fn process_search<T>(
    partition: Arc<PartitionState>,
    table: Arc<RwLock<T>>,
    dimensions: Dimensions,
    msg: Index,
) where
    T: TableSearch + Send + Sync + 'static,
{
    match msg {
        Index::Ann {
            embedding,
            limit,
            tx,
            ..
        } => {
            if let Some(tx) = validate_dimensions(tx, &embedding, dimensions) {
                ann(&partition, tx, &table, embedding, limit);
            }
        }

        Index::FilteredAnn { tx, index_key, .. } => {
            warn!(
                "cuVS backend does not support filtered search (index {index_key:?})"
            );
            _ = tx.send(Err(anyhow!(
                "cuVS backend does not support filtered search (index {index_key:?})"
            )));
        }

        _ => unreachable!("only search operations should be dispatched here"),
    }
}

fn validate_dimensions(
    tx_ann: oneshot::Sender<AnnR>,
    embedding: &Vector,
    dimensions: Dimensions,
) -> Option<oneshot::Sender<AnnR>> {
    if let Err(err) = validator::embedding_dimensions(embedding, dimensions) {
        warn!("validate_dimensions: {err}");
        tx_ann
            .send(Err(err))
            .unwrap_or_else(|_| warn!("validate_dimensions: unable to send response"));
        None
    } else {
        Some(tx_ann)
    }
}

fn ann<T>(
    partition: &PartitionState,
    tx_ann: oneshot::Sender<AnnR>,
    table: &Arc<RwLock<T>>,
    embedding: Vector,
    limit: Limit,
) where
    T: TableSearch + Send + Sync + 'static,
{
    let result = partition
        .idx
        .search(&embedding, limit)
        .and_then(|matches| {
            let table = table.read().unwrap();
            let (primary_keys, distances): (Vec<_>, Vec<_>) = matches
                .into_iter()
                .filter_map(|(primary_id, distance)| {
                    table
                        .primary_key(partition.partition_id, primary_id)
                        .map(|primary_key| (primary_key, distance))
                })
                .unzip();
            Ok((primary_keys, distances))
        });
    if let Err(ref err) = result {
        warn!(
            "ann search failed for partition {:?}: {err:#}",
            partition.partition_id
        );
    }
    tx_ann
        .send(result)
        .unwrap_or_else(|_| debug!("ann: unable to send response (receiver dropped)"));
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

/// GPU-accelerated search backends using NVIDIA CUDA.
///
/// When compiled with `--features gpu`, this module provides:
/// - `GpuBruteForceIndex`: exhaustive search via cuBLAS SGEMM
/// - `CagraIndex`: graph-based approximate nearest-neighbor search via
///   NVIDIA cuVS CAGRA, dynamically loaded from `libcuvs_c.so`
///
/// The `cudarc` crate dynamically loads CUDA libraries at runtime via
/// `fallback-dynamic-loading`, so the binary compiles without any CUDA
/// toolkit installed and fails gracefully at runtime if no GPU is available.
///
/// CAGRA (CUDA ANN Graph-based) builds a fixed-degree k-NN graph on the GPU
/// and traverses it during search. It requires `libcuvs_c.so` at runtime,
/// which can be installed via conda/mamba (rapidsai channel).
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};

    /// GPU-accelerated brute-force vector index using cuBLAS.
    ///
    /// Stores vectors in host memory. On search, copies the full dataset and
    /// query to the GPU, computes inner products via SGEMM, copies results
    /// back, and derives final distances on the CPU.
    pub(super) struct GpuBruteForceIndex {
        vectors: RwLock<BTreeMap<PrimaryId, Vec<f32>>>,
        space_type: SpaceType,
        dimensions: usize,
    }

    impl GpuBruteForceIndex {
        pub fn new(space_type: SpaceType, dimensions: usize) -> Self {
            Self {
                vectors: RwLock::new(BTreeMap::new()),
                space_type,
                dimensions,
            }
        }
    }

    impl CuvsVectorIndex for GpuBruteForceIndex {
        fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()> {
            self.vectors
                .write()
                .unwrap()
                .insert(primary_id, vector.as_ref().to_vec());
            Ok(())
        }

        fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()> {
            self.vectors.write().unwrap().remove(&primary_id);
            Ok(())
        }

        fn search(
            &self,
            vector: &Vector,
            limit: Limit,
        ) -> anyhow::Result<Vec<(PrimaryId, Distance)>> {
            let vectors = self.vectors.read().unwrap();
            if vectors.is_empty() {
                return Ok(vec![]);
            }

            let n = vectors.len();
            let d = self.dimensions;
            let k = limit.0.get().min(n);

            // Build contiguous dataset row-major [n, d] and PrimaryId map.
            let mut flat = Vec::with_capacity(n * d);
            let mut ids: Vec<PrimaryId> = Vec::with_capacity(n);
            for (&pid, v) in vectors.iter() {
                flat.extend_from_slice(v);
                ids.push(pid);
            }
            drop(vectors);

            let query = vector.as_ref();

            // --- GPU: compute inner products via cuBLAS SGEMM ---
            //
            // Row-major dataset[n,d] viewed as column-major is A_cm[d,n].
            // Row-major query[1,d] viewed as column-major is B_cm[d,1].
            //
            // We want: inner[n,1] = dataset[n,d] * query[d,1]
            //        = A_cm^T[n,d] * B_cm[d,1]
            //
            // SGEMM: C = alpha * op(A)[m,k] * op(B)[k,n_blas] + beta * C
            //   transa=T → op(A) = A^T = [n, d]  (A is [d, n] col-major)
            //   transb=N → op(B) = B   = [d, 1]  (B is [d, 1] col-major)
            //   m=n, n_blas=1, k=d
            let ctx = CudaContext::new(0)
                .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
            let stream = ctx.default_stream();

            let d_dataset = stream
                .clone_htod(&flat)
                .map_err(|e| anyhow!("failed to copy dataset to GPU: {e}"))?;
            let d_query = stream
                .clone_htod(query)
                .map_err(|e| anyhow!("failed to copy query to GPU: {e}"))?;
            let mut d_inner = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| anyhow!("failed to allocate GPU output: {e}"))?;

            let blas = CudaBlas::new(stream.clone())
                .map_err(|e| anyhow!("failed to create cuBLAS handle: {e}"))?;

            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: 1,
                k: d as i32,
                alpha: 1.0f32,
                lda: d as i32,
                ldb: d as i32,
                beta: 0.0f32,
                ldc: n as i32,
            };

            // SAFETY: cuBLAS SGEMM operates on device memory allocated above.
            unsafe {
                blas.gemm(cfg, &d_dataset, &d_query, &mut d_inner)
                    .map_err(|e| anyhow!("cuBLAS SGEMM failed: {e}"))?;
            }

            let inner_products = stream
                .clone_dtoh(&d_inner)
                .map_err(|e| anyhow!("failed to copy results from GPU: {e}"))?;

            // --- Derive distances from inner products on the CPU ---
            let mut indexed: Vec<(usize, f32)> = match self.space_type {
                SpaceType::Euclidean => {
                    // L2² = ||q||² + ||d_i||² − 2·(q·d_i)
                    let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();
                    (0..n)
                        .map(|i| {
                            let d_norm_sq: f32 =
                                flat[i * d..(i + 1) * d].iter().map(|x| x * x).sum();
                            (i, q_norm_sq + d_norm_sq - 2.0 * inner_products[i])
                        })
                        .collect()
                }
                SpaceType::Cosine => {
                    // cosine_distance = 1 − (q·d_i) / (||q|| · ||d_i||)
                    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    (0..n)
                        .map(|i| {
                            let d_norm: f32 = flat[i * d..(i + 1) * d]
                                .iter()
                                .map(|x| x * x)
                                .sum::<f32>()
                                .sqrt();
                            let dist = if q_norm == 0.0 || d_norm == 0.0 {
                                1.0
                            } else {
                                1.0 - inner_products[i] / (q_norm * d_norm)
                            };
                            (i, dist)
                        })
                        .collect()
                }
                SpaceType::DotProduct => {
                    (0..n).map(|i| (i, 1.0 - inner_products[i])).collect()
                }
                SpaceType::Hamming => {
                    anyhow::bail!("GPU backend does not support Hamming distance")
                }
            };

            // Sort by distance and take top-k.
            indexed.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            indexed.truncate(k);

            let mut results = Vec::with_capacity(k);
            for (idx, dist_val) in indexed {
                let pid = ids[idx];
                let distance = match self.space_type {
                    SpaceType::Euclidean => Distance::new_euclidean(dist_val)?,
                    SpaceType::Cosine => Distance::new_cosine(dist_val)?,
                    SpaceType::DotProduct => Distance::new_dot_product(dist_val)?,
                    SpaceType::Hamming => unreachable!(),
                };
                results.push((pid, distance));
            }

            Ok(results)
        }

        fn size(&self) -> usize {
            self.vectors.read().unwrap().len()
        }
    }

    // ---- cuVS CAGRA FFI bindings (dynamic loading) ----

    /// FFI types and function pointers for the NVIDIA cuVS C API.
    ///
    /// These `#[repr(C)]` structs mirror the C headers in
    /// `<cuvs/core/c_api.h>`, `<cuvs/neighbors/cagra.h>`,
    /// `<cuvs/neighbors/common.h>` and `<dlpack/dlpack.h>`.
    /// The library is loaded at runtime via `libloading` so the binary
    /// compiles without libcuvs installed.
    pub(super) mod cuvs_ffi {
        use std::ffi::c_void;

        // -- cuvsError_t --
        pub const CUVS_SUCCESS: u32 = 1;

        pub type CuvsError = u32;
        pub type CuvsResources = usize; // uintptr_t

        // -- DLPack types (v1) --

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct DLDevice {
            pub device_type: i32,
            pub device_id: i32,
        }

        pub const KDL_CPU: i32 = 1;
        #[allow(dead_code)]
        pub const KDL_CUDA: i32 = 2;

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct DLDataType {
            pub code: u8,
            pub bits: u8,
            pub lanes: u16,
        }

        /// float32 DLPack data type: code=kDLFloat(2), bits=32, lanes=1
        pub const DL_FLOAT32: DLDataType = DLDataType {
            code: 2,
            bits: 32,
            lanes: 1,
        };

        /// uint32 DLPack data type: code=kDLUInt(1), bits=32, lanes=1
        pub const DL_UINT32: DLDataType = DLDataType {
            code: 1,
            bits: 32,
            lanes: 1,
        };

        #[repr(C)]
        pub struct DLTensor {
            pub data: *mut c_void,
            pub device: DLDevice,
            pub ndim: i32,
            pub dtype: DLDataType,
            pub shape: *mut i64,
            pub strides: *mut i64,
            pub byte_offset: u64,
        }

        #[repr(C)]
        pub struct DLManagedTensor {
            pub dl_tensor: DLTensor,
            pub manager_ctx: *mut c_void,
            pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
        }

        // -- cuVS CAGRA types --

        #[repr(C)]
        pub struct CuvsCagraIndex {
            pub addr: usize,
            pub dtype: DLDataType,
        }

        pub type CuvsCagraIndexPtr = *mut CuvsCagraIndex;

        // -- CAGRA search params (mirrors struct cuvsCagraSearchParams) --

        #[repr(C)]
        pub struct CuvsCagraSearchParams {
            pub max_queries: usize,
            pub itopk_size: usize,
            pub max_iterations: usize,
            pub algo: i32, // cuvsCagraSearchAlgo
            pub team_size: usize,
            pub search_width: usize,
            pub min_iterations: usize,
            pub thread_block_size: usize,
            pub hashmap_mode: i32, // cuvsCagraHashMode
            pub hashmap_min_bitlen: usize,
            pub hashmap_max_fill_rate: f32,
            pub num_random_samplings: u32,
            pub rand_xor_mask: u64,
            pub persistent: bool,
            pub persistent_lifetime: f32,
            pub persistent_device_usage: f32,
        }

        // cuvsDistanceType enum values used by CAGRA
        pub const L2_EXPANDED: u32 = 0;
        pub const COSINE_EXPANDED: u32 = 2;
        pub const INNER_PRODUCT: u32 = 6;

        // cuvsFilterType::NO_FILTER
        pub const NO_FILTER: u32 = 0;

        /// cuvsFilter struct: { addr: uintptr_t, type: cuvsFilterType }
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct CuvsFilter {
            pub addr: usize,
            pub filter_type: u32,
        }

        impl CuvsFilter {
            pub fn none() -> Self {
                Self {
                    addr: 0,
                    filter_type: NO_FILTER,
                }
            }
        }

        /// Loaded function pointers from `libcuvs_c.so`.
        ///
        /// All functions are resolved once on library load and cached.
        /// Each function pointer matches the C API signature from the
        /// cuVS headers.
        #[allow(dead_code)]
        pub struct CuvsLib {
            _lib: libloading::Library,

            // Resources
            pub resources_create:
                unsafe extern "C" fn(*mut CuvsResources) -> CuvsError,
            pub resources_destroy: unsafe extern "C" fn(CuvsResources) -> CuvsError,

            // Stream
            pub stream_sync: unsafe extern "C" fn(CuvsResources) -> CuvsError,

            // Error text
            pub get_last_error_text:
                unsafe extern "C" fn() -> *const std::ffi::c_char,

            // CAGRA index lifecycle
            pub cagra_index_create:
                unsafe extern "C" fn(*mut CuvsCagraIndexPtr) -> CuvsError,
            pub cagra_index_destroy:
                unsafe extern "C" fn(CuvsCagraIndexPtr) -> CuvsError,

            // CAGRA index params
            pub cagra_index_params_create:
                unsafe extern "C" fn(*mut *mut c_void) -> CuvsError,
            pub cagra_index_params_destroy:
                unsafe extern "C" fn(*mut c_void) -> CuvsError,

            // CAGRA search params
            pub cagra_search_params_create:
                unsafe extern "C" fn(*mut *mut c_void) -> CuvsError,
            pub cagra_search_params_destroy:
                unsafe extern "C" fn(*mut c_void) -> CuvsError,

            // CAGRA extend params
            pub cagra_extend_params_create:
                unsafe extern "C" fn(*mut *mut c_void) -> CuvsError,
            pub cagra_extend_params_destroy:
                unsafe extern "C" fn(*mut c_void) -> CuvsError,

            // Build / Extend / Search
            pub cagra_build: unsafe extern "C" fn(
                CuvsResources,
                *mut c_void, // cuvsCagraIndexParams_t
                *mut DLManagedTensor,
                CuvsCagraIndexPtr,
            ) -> CuvsError,
            pub cagra_extend: unsafe extern "C" fn(
                CuvsResources,
                *mut c_void, // cuvsCagraExtendParams_t
                *mut DLManagedTensor,
                CuvsCagraIndexPtr,
            ) -> CuvsError,
            pub cagra_search: unsafe extern "C" fn(
                CuvsResources,
                *mut c_void, // cuvsCagraSearchParams_t
                CuvsCagraIndexPtr,
                *mut DLManagedTensor,
                *mut DLManagedTensor,
                *mut DLManagedTensor,
                CuvsFilter,
            ) -> CuvsError,
        }

        // SAFETY: CuvsLib holds a loaded library and function pointers.
        // The libcuvs C API is thread-safe for distinct resources, and we
        // ensure each call uses its own cuvsResources_t.
        unsafe impl Send for CuvsLib {}
        unsafe impl Sync for CuvsLib {}

        impl CuvsLib {
            /// Dynamically load `libcuvs_c.so` and resolve all needed symbols.
            ///
            /// Searches standard library paths (`LD_LIBRARY_PATH`, system dirs).
            /// Returns an error if the library cannot be found or any symbol
            /// is missing.
            pub fn load() -> anyhow::Result<Self> {
                // SAFETY: We load a well-known NVIDIA shared library and resolve
                // documented C API symbols. The function pointers are used only
                // through the typed wrappers defined here.
                unsafe {
                    let lib = libloading::Library::new("libcuvs_c.so")
                        .map_err(|e| anyhow::anyhow!(
                            "failed to load libcuvs_c.so: {e}. \
                             Install libcuvs (e.g. `mamba install libcuvs`) and \
                             set LD_LIBRARY_PATH to include the lib directory."
                        ))?;

                    macro_rules! sym {
                        ($name:expr) => {{
                            let f: libloading::Symbol<*const ()> = lib.get($name)
                                .map_err(|e| anyhow::anyhow!(
                                    "symbol {} not found in libcuvs_c.so: {e}",
                                    String::from_utf8_lossy($name)
                                ))?;
                            std::mem::transmute(*f)
                        }};
                    }

                    Ok(Self {
                        resources_create: sym!(b"cuvsResourcesCreate\0"),
                        resources_destroy: sym!(b"cuvsResourcesDestroy\0"),
                        stream_sync: sym!(b"cuvsStreamSync\0"),
                        get_last_error_text: sym!(b"cuvsGetLastErrorText\0"),
                        cagra_index_create: sym!(b"cuvsCagraIndexCreate\0"),
                        cagra_index_destroy: sym!(b"cuvsCagraIndexDestroy\0"),
                        cagra_index_params_create: sym!(b"cuvsCagraIndexParamsCreate\0"),
                        cagra_index_params_destroy: sym!(b"cuvsCagraIndexParamsDestroy\0"),
                        cagra_search_params_create: sym!(b"cuvsCagraSearchParamsCreate\0"),
                        cagra_search_params_destroy: sym!(b"cuvsCagraSearchParamsDestroy\0"),
                        cagra_extend_params_create: sym!(b"cuvsCagraExtendParamsCreate\0"),
                        cagra_extend_params_destroy: sym!(b"cuvsCagraExtendParamsDestroy\0"),
                        cagra_build: sym!(b"cuvsCagraBuild\0"),
                        cagra_extend: sym!(b"cuvsCagraExtend\0"),
                        cagra_search: sym!(b"cuvsCagraSearch\0"),
                        _lib: lib,
                    })
                }
            }
        }

        /// Check a cuVS C API return code and convert to `anyhow::Result`.
        pub fn check(
            lib: &CuvsLib,
            status: CuvsError,
            context: &str,
        ) -> anyhow::Result<()> {
            if status == CUVS_SUCCESS {
                Ok(())
            } else {
                // cuvsGetLastErrorText is thread-local; safe to call here.
                let msg = unsafe {
                    let ptr = (lib.get_last_error_text)();
                    if ptr.is_null() {
                        "unknown error".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(ptr)
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                anyhow::bail!("{context}: {msg}")
            }
        }

        /// Helper: create a `DLManagedTensor` wrapping a host `f32` slice.
        ///
        /// The tensor is non-owning: the caller must ensure `data` outlives
        /// the returned tensor. Shape is `[rows, cols]`, row-major, on CPU.
        pub fn make_host_f32_tensor(
            data: &mut [f32],
            shape: &mut [i64; 2],
        ) -> DLManagedTensor {
            DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.as_mut_ptr() as *mut c_void,
                    device: DLDevice {
                        device_type: KDL_CPU,
                        device_id: 0,
                    },
                    ndim: 2,
                    dtype: DL_FLOAT32,
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            }
        }

        /// Helper: create a `DLManagedTensor` wrapping a host `u32` slice.
        #[allow(dead_code)]
        pub fn make_host_u32_tensor(
            data: &mut [u32],
            shape: &mut [i64; 2],
        ) -> DLManagedTensor {
            DLManagedTensor {
                dl_tensor: DLTensor {
                    data: data.as_mut_ptr() as *mut c_void,
                    device: DLDevice {
                        device_type: KDL_CPU,
                        device_id: 0,
                    },
                    ndim: 2,
                    dtype: DL_UINT32,
                    shape: shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            }
        }
    }

    // ---- CAGRA Index ----

    /// GPU-accelerated approximate nearest-neighbor index using CAGRA.
    ///
    /// **Strategy**: vectors are accumulated in a host-side `BTreeMap`. When a
    /// search is requested and the dataset has changed since the last build
    /// (tracked via a dirty flag), the CAGRA graph index is rebuilt from the
    /// full dataset. Subsequent searches reuse the cached CAGRA graph until
    /// the next mutation invalidates it.
    ///
    /// CAGRA requires a minimum dataset size to build a meaningful graph (at
    /// least `graph_degree + 1` vectors, where `graph_degree` defaults to 64).
    /// When the dataset is too small, search falls back to CPU brute-force.
    ///
    /// All cuVS API calls happen on the calling thread (from `spawn_blocking`)
    /// because CUDA contexts are thread-local. A per-call `cuvsResources_t`
    /// is created and destroyed to avoid cross-thread issues.
    pub(super) struct CagraIndex {
        /// Canonical store of all vectors, keyed by PrimaryId.
        vectors: RwLock<BTreeMap<PrimaryId, Vec<f32>>>,
        space_type: SpaceType,
        dimensions: usize,
        /// Loaded cuVS library handle (shared across all CagraIndex instances).
        lib: Arc<cuvs_ffi::CuvsLib>,
        /// `true` when vectors have been mutated since last CAGRA build.
        dirty: RwLock<bool>,
        /// Cached flat data + built CAGRA graph index. Invalidated when dirty.
        cached_build: RwLock<Option<CagraCachedBuild>>,
        /// Serializes CAGRA graph builds so only one thread builds at a time
        /// (prevents thundering-herd when many searches arrive concurrently).
        build_lock: std::sync::Mutex<()>,
    }

    /// Cached CAGRA build: flat data snapshot, PrimaryId mapping, and the
    /// built CAGRA graph index pointer. The CAGRA graph lives in GPU
    /// memory and is reused across searches until the next rebuild.
    struct CagraCachedBuild {
        /// Row-major [n, d] dataset.
        flat: Vec<f32>,
        /// Ordered PrimaryId for each row.
        ids: Vec<PrimaryId>,
        /// cuVS resources handle (CUDA stream) used for build.
        /// Kept alive because the CAGRA graph may reference memory allocated
        /// on this stream's memory pool.
        res: cuvs_ffi::CuvsResources,
        /// Built CAGRA index (null if build failed or n < MIN_CAGRA_BUILD_SIZE).
        index_ptr: cuvs_ffi::CuvsCagraIndexPtr,
        /// cuVS library handle for cleanup.
        lib: Arc<cuvs_ffi::CuvsLib>,
    }

    // SAFETY: The CuvsCagraIndexPtr is an opaque handle managed by the cuVS
    // library. It is safe to send across threads because the CAGRA graph is
    // stored in GPU memory (device-global), and we only access it through
    // cuVS API calls that take their own per-call cuvsResources_t (CUDA
    // stream). Our RwLock ensures no concurrent mutation.
    unsafe impl Send for CagraCachedBuild {}
    unsafe impl Sync for CagraCachedBuild {}

    impl Drop for CagraCachedBuild {
        fn drop(&mut self) {
            // SAFETY: destroy in reverse creation order — index first, then
            // the resources that may hold the underlying memory pool.
            unsafe {
                if !self.index_ptr.is_null() {
                    let _ = (self.lib.cagra_index_destroy)(self.index_ptr);
                }
                if self.res != 0 {
                    let _ = (self.lib.resources_destroy)(self.res);
                }
            }
        }
    }

    /// Minimum number of vectors required to build a CAGRA index.
    /// Below this threshold we fall back to CPU brute-force search.
    /// This needs to be > graph_degree (default 64).
    const MIN_CAGRA_BUILD_SIZE: usize = 128;

    impl CagraIndex {
        pub fn new(
            space_type: SpaceType,
            dimensions: usize,
            lib: Arc<cuvs_ffi::CuvsLib>,
        ) -> Self {
            Self {
                vectors: RwLock::new(BTreeMap::new()),
                space_type,
                dimensions,
                lib,
                dirty: RwLock::new(true),
                cached_build: RwLock::new(None),
                build_lock: std::sync::Mutex::new(()),
            }
        }

        /// Map our `SpaceType` to the cuVS `cuvsDistanceType` enum value.
        fn distance_type(&self) -> anyhow::Result<u32> {
            match self.space_type {
                SpaceType::Euclidean => Ok(cuvs_ffi::L2_EXPANDED),
                SpaceType::Cosine => Ok(cuvs_ffi::COSINE_EXPANDED),
                SpaceType::DotProduct => Ok(cuvs_ffi::INNER_PRODUCT),
                SpaceType::Hamming => {
                    anyhow::bail!("CAGRA does not support Hamming distance")
                }
            }
        }

        /// Build or rebuild the CAGRA index from the current vector set.
        ///
        /// Returns the cuVS resources handle and the CAGRA index pointer.
        /// Both must be kept alive: the resources handle owns the CUDA stream
        /// and potentially a memory pool that the CAGRA graph references.
        ///
        /// The caller must eventually destroy both via `cagra_index_destroy`
        /// and `resources_destroy` (handled by `CagraCachedBuild::drop`).
        fn build_index(
            &self,
            flat: &mut Vec<f32>,
            n: usize,
            d: usize,
        ) -> anyhow::Result<(cuvs_ffi::CuvsResources, cuvs_ffi::CuvsCagraIndexPtr)> {
            let lib = &self.lib;
            let metric = self.distance_type()?;

            // Create cuVS resources (wraps a CUDA stream)
            let mut res: cuvs_ffi::CuvsResources = 0;
            // SAFETY: cuvsResourcesCreate is a well-defined C API call that
            // initializes CUDA resources. We pass a valid pointer.
            unsafe {
                cuvs_ffi::check(
                    lib,
                    (lib.resources_create)(&mut res),
                    "cuvsResourcesCreate",
                )?;
            }

            // Create index params with defaults, then set metric
            let mut index_params: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: cuvsCagraIndexParamsCreate allocates and zero-initializes
            // the params struct. We destroy it after cuvsCagraBuild.
            unsafe {
                cuvs_ffi::check(
                    lib,
                    (lib.cagra_index_params_create)(&mut index_params),
                    "cuvsCagraIndexParamsCreate",
                )?;
                // The first field of cuvsCagraIndexParams is `metric` (u32).
                // Set it directly via pointer cast.
                *(index_params as *mut u32) = metric;
            }

            // Create the index object
            let mut index_ptr: cuvs_ffi::CuvsCagraIndexPtr = std::ptr::null_mut();
            // SAFETY: cuvsCagraIndexCreate allocates the index struct.
            unsafe {
                cuvs_ffi::check(
                    lib,
                    (lib.cagra_index_create)(&mut index_ptr),
                    "cuvsCagraIndexCreate",
                )?;
            }

            // Build the dataset DLManagedTensor
            let mut shape = [n as i64, d as i64];
            let mut dataset_tensor = cuvs_ffi::make_host_f32_tensor(flat, &mut shape);

            // SAFETY: cuvsCagraBuild reads the dataset tensor, builds the CAGRA
            // graph on the GPU, and stores the result in index_ptr. The tensor
            // data (flat) must remain valid during this call.
            unsafe {
                let status = (lib.cagra_build)(
                    res,
                    index_params,
                    &mut dataset_tensor,
                    index_ptr,
                );
                // Clean up params regardless of build result
                let _ = (lib.cagra_index_params_destroy)(index_params);
                cuvs_ffi::check(lib, status, "cuvsCagraBuild")?;
                // Sync the CUDA stream to ensure the build is complete
                cuvs_ffi::check(
                    lib,
                    (lib.stream_sync)(res),
                    "cuvsStreamSync after build",
                )?;
            }

            Ok((res, index_ptr))
        }

        /// Execute a CAGRA search on a pre-built index.
        ///
        /// Uses the same `cuvsResources_t` that was used to build the CAGRA
        /// graph. The CAGRA index stores internal references to the build
        /// resource's memory pool and stream, so using a different resources
        /// handle causes "unknown error" from `cuvsCagraSearch`. Since the
        /// actor serializes all index operations, reusing the build `res` is
        /// safe.
        ///
        /// The cuVS CAGRA search API requires query, neighbors, and distances
        /// tensors on a CUDA device (`kDLCUDA`). We use cudarc to allocate
        /// device memory, copy the query to the GPU, run the search, and
        /// copy results back.
        fn search_cagra(
            &self,
            res: cuvs_ffi::CuvsResources,
            index_ptr: cuvs_ffi::CuvsCagraIndexPtr,
            query: &[f32],
            k: usize,
            ids: &[PrimaryId],
        ) -> anyhow::Result<Vec<(PrimaryId, Distance)>> {
            let lib = &self.lib;
            let d = self.dimensions;

            // Create search params
            let mut search_params: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: cuvsCagraSearchParamsCreate allocates default params.
            unsafe {
                cuvs_ffi::check(
                    lib,
                    (lib.cagra_search_params_create)(&mut search_params),
                    "cuvsCagraSearchParamsCreate",
                )?;
                // Ensure itopk_size >= k. The default is 64, which is too
                // small for large top-k queries (e.g. VectorDBBench k=100).
                let params = search_params as *mut cuvs_ffi::CuvsCagraSearchParams;
                if (*params).itopk_size < k {
                    (*params).itopk_size = k;
                }
            }

            // Allocate GPU memory for query, neighbors, distances via cudarc.
            let ctx = CudaContext::new(0)
                .map_err(|e| anyhow!("failed to create CUDA context: {e}"))?;
            let stream = ctx.default_stream();

            let d_query = stream
                .clone_htod(query)
                .map_err(|e| anyhow!("CAGRA search: failed to copy query to GPU: {e}"))?;
            let mut d_neighbors = stream
                .alloc_zeros::<u32>(k)
                .map_err(|e| anyhow!("CAGRA search: failed to alloc neighbors: {e}"))?;
            let mut d_distances = stream
                .alloc_zeros::<f32>(k)
                .map_err(|e| anyhow!("CAGRA search: failed to alloc distances: {e}"))?;

            // Build DLManagedTensor wrappers for GPU memory.
            // DLDevice { device_type: kDLCUDA=2, device_id: 0 }
            let cuda_device = cuvs_ffi::DLDevice {
                device_type: cuvs_ffi::KDL_CUDA,
                device_id: 0,
            };

            // Synchronize the cudarc stream so that the H2D copy of the
            // query and the zero-fill of neighbors/distances are complete
            // before cuVS accesses this memory on the build stream.
            ctx.synchronize()
                .map_err(|e| anyhow!("CAGRA search: CUDA ctx sync failed: {e}"))?;

            // Get raw device pointers. The SyncOnDrop guards ensure the stream
            // is synchronized when they are dropped.
            let (query_ptr, _query_guard) = d_query.device_ptr(&stream);
            let (neighbors_ptr, _neighbors_guard) = d_neighbors.device_ptr_mut(&stream);
            let (distances_ptr, _distances_guard) = d_distances.device_ptr_mut(&stream);

            let mut query_shape = [1i64, d as i64];
            let mut query_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: query_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_FLOAT32,
                    shape: query_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let mut neighbors_shape = [1i64, k as i64];
            let mut neighbors_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: neighbors_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_UINT32,
                    shape: neighbors_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let mut distances_shape = [1i64, k as i64];
            let mut distances_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: distances_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_FLOAT32,
                    shape: distances_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let filter = cuvs_ffi::CuvsFilter::none();

            // SAFETY: cuvsCagraSearch reads the query from GPU memory, searches
            // the CAGRA graph, and writes neighbor indices and distances to GPU
            // memory. All buffers are valid for the duration of this call.
            unsafe {
                let status = (lib.cagra_search)(
                    res,
                    search_params,
                    index_ptr,
                    &mut query_tensor,
                    &mut neighbors_tensor,
                    &mut distances_tensor,
                    filter,
                );
                // Check status BEFORE destroying params — the destroy call
                // clears cuVS's thread-local error text.
                cuvs_ffi::check(lib, status, "cuvsCagraSearch")?;
                let _ = (lib.cagra_search_params_destroy)(search_params);
                cuvs_ffi::check(
                    lib,
                    (lib.stream_sync)(res),
                    "cuvsStreamSync after search",
                )?;
            }

            // Synchronize device so cuVS results are visible to cudarc.
            ctx.synchronize()
                .map_err(|e| anyhow!("CAGRA search: CUDA ctx sync after search failed: {e}"))?;

            // Release mutable borrow guards before copying data back.
            drop(_query_guard);
            drop(_neighbors_guard);
            drop(_distances_guard);

            // Copy results back from GPU
            let neighbors_data = stream
                .clone_dtoh(&d_neighbors)
                .map_err(|e| anyhow!("CAGRA search: failed to copy neighbors: {e}"))?;
            let distances_data = stream
                .clone_dtoh(&d_distances)
                .map_err(|e| anyhow!("CAGRA search: failed to copy distances: {e}"))?;

            // Convert results to (PrimaryId, Distance) pairs.
            // CAGRA returns u32 indices into the flat dataset array.
            let n = ids.len();
            let mut results = Vec::with_capacity(k);
            for i in 0..k {
                let idx = neighbors_data[i] as usize;
                if idx >= n {
                    // CAGRA may return sentinel values for unfilled slots
                    continue;
                }
                let dist_val = distances_data[i];
                let distance = match self.space_type {
                    SpaceType::Euclidean => Distance::new_euclidean(dist_val)?,
                    SpaceType::Cosine => Distance::new_cosine(dist_val)?,
                    SpaceType::DotProduct => Distance::new_dot_product(dist_val)?,
                    SpaceType::Hamming => unreachable!(),
                };
                results.push((ids[idx], distance));
            }

            Ok(results)
        }

        /// CPU brute-force fallback for small datasets.
        fn search_brute_force(
            &self,
            query: &[f32],
            flat: &[f32],
            ids: &[PrimaryId],
            k: usize,
        ) -> anyhow::Result<Vec<(PrimaryId, Distance)>> {
            let d = self.dimensions;
            let n = ids.len();

            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let row = &flat[i * d..(i + 1) * d];
                    let dist = compute_distance(query, row, self.space_type)
                        .map(|d| -> f32 { d.into() })
                        .unwrap_or(f32::MAX);
                    (i, dist)
                })
                .collect();

            scored.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);

            let mut results = Vec::with_capacity(k);
            for (idx, dist_val) in scored {
                let distance = match self.space_type {
                    SpaceType::Euclidean => Distance::new_euclidean(dist_val)?,
                    SpaceType::Cosine => Distance::new_cosine(dist_val)?,
                    SpaceType::DotProduct => Distance::new_dot_product(dist_val)?,
                    SpaceType::Hamming => unreachable!(),
                };
                results.push((ids[idx], distance));
            }
            Ok(results)
        }
    }

    impl CagraIndex {
        /// Ensure the CAGRA index is up-to-date with the latest mutations.
        ///
        /// Uses double-checked locking: only one thread rebuilds at a time;
        /// others wait and reuse the result. Called once before a search
        /// batch so the rebuild cost is amortised across all queries.
        fn ensure_fresh(&self) {
            let dirty = *self.dirty.read().unwrap();
            if !dirty {
                return;
            }
            let _build_guard = self.build_lock.lock().unwrap();
            if !*self.dirty.read().unwrap() {
                return;
            }

            let vectors = self.vectors.read().unwrap();
            let n = vectors.len();
            let d = self.dimensions;

            let mut flat = Vec::with_capacity(n * d);
            let mut ids = Vec::with_capacity(n);
            for (&pid, v) in vectors.iter() {
                flat.extend_from_slice(v);
                ids.push(pid);
            }
            drop(vectors);

            let (res, index_ptr) = if n >= MIN_CAGRA_BUILD_SIZE {
                match self.build_index(&mut flat, n, d) {
                    Ok((res, ptr)) => {
                        tracing::info!(
                            "CAGRA graph built for {n} vectors ({d} dims)"
                        );
                        (res, ptr)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "CAGRA build failed ({e:#}), falling back to brute-force"
                        );
                        (0, std::ptr::null_mut())
                    }
                }
            } else {
                (0, std::ptr::null_mut())
            };

            *self.cached_build.write().unwrap() = Some(CagraCachedBuild {
                flat,
                ids,
                res,
                index_ptr,
                lib: Arc::clone(&self.lib),
            });
            *self.dirty.write().unwrap() = false;
        }

        /// Multi-query CAGRA search: runs all queries in a single GPU kernel.
        fn search_cagra_batch(
            &self,
            res: cuvs_ffi::CuvsResources,
            index_ptr: cuvs_ffi::CuvsCagraIndexPtr,
            queries: &[&[f32]],
            k: usize,
            ids: &[PrimaryId],
        ) -> anyhow::Result<Vec<Vec<(PrimaryId, Distance)>>> {
            let lib = &self.lib;
            let d = self.dimensions;
            let n_queries = queries.len();

            let mut search_params: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: cuvsCagraSearchParamsCreate allocates default params.
            unsafe {
                cuvs_ffi::check(
                    lib,
                    (lib.cagra_search_params_create)(&mut search_params),
                    "cuvsCagraSearchParamsCreate",
                )?;
                let params =
                    search_params as *mut cuvs_ffi::CuvsCagraSearchParams;
                if (*params).itopk_size < k {
                    (*params).itopk_size = k;
                }
            }

            // Build combined query matrix [n_queries, d].
            let mut combined_query: Vec<f32> =
                Vec::with_capacity(n_queries * d);
            for q in queries {
                combined_query.extend_from_slice(q);
            }

            let ctx = CudaContext::new(0)
                .map_err(|e| anyhow!("CAGRA batch: CUDA context: {e}"))?;
            let stream = ctx.default_stream();

            let d_query = stream
                .clone_htod(&combined_query)
                .map_err(|e| anyhow!("CAGRA batch: query H2D: {e}"))?;
            let mut d_neighbors = stream
                .alloc_zeros::<u32>(n_queries * k)
                .map_err(|e| anyhow!("CAGRA batch: alloc neighbors: {e}"))?;
            let mut d_distances = stream
                .alloc_zeros::<f32>(n_queries * k)
                .map_err(|e| anyhow!("CAGRA batch: alloc distances: {e}"))?;

            let cuda_device = cuvs_ffi::DLDevice {
                device_type: cuvs_ffi::KDL_CUDA,
                device_id: 0,
            };

            ctx.synchronize()
                .map_err(|e| anyhow!("CAGRA batch: ctx sync: {e}"))?;

            let (query_ptr, _qg) = d_query.device_ptr(&stream);
            let (neighbors_ptr, _ng) =
                d_neighbors.device_ptr_mut(&stream);
            let (distances_ptr, _dg) =
                d_distances.device_ptr_mut(&stream);

            let mut query_shape = [n_queries as i64, d as i64];
            let mut query_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: query_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_FLOAT32,
                    shape: query_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let mut neighbors_shape = [n_queries as i64, k as i64];
            let mut neighbors_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: neighbors_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_UINT32,
                    shape: neighbors_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let mut distances_shape = [n_queries as i64, k as i64];
            let mut distances_tensor = cuvs_ffi::DLManagedTensor {
                dl_tensor: cuvs_ffi::DLTensor {
                    data: distances_ptr as *mut std::ffi::c_void,
                    device: cuda_device,
                    ndim: 2,
                    dtype: cuvs_ffi::DL_FLOAT32,
                    shape: distances_shape.as_mut_ptr(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            };

            let filter = cuvs_ffi::CuvsFilter::none();

            // SAFETY: cuvsCagraSearch reads queries from GPU memory, searches
            // the graph, writes neighbor indices and distances to GPU memory.
            unsafe {
                let status = (lib.cagra_search)(
                    res,
                    search_params,
                    index_ptr,
                    &mut query_tensor,
                    &mut neighbors_tensor,
                    &mut distances_tensor,
                    filter,
                );
                cuvs_ffi::check(lib, status, "cuvsCagraSearch")?;
                let _ = (lib.cagra_search_params_destroy)(search_params);
                cuvs_ffi::check(
                    lib,
                    (lib.stream_sync)(res),
                    "cuvsStreamSync after search",
                )?;
            }

            ctx.synchronize().map_err(|e| {
                anyhow!("CAGRA batch: ctx sync after search: {e}")
            })?;

            drop(_qg);
            drop(_ng);
            drop(_dg);

            let neighbors_data = stream
                .clone_dtoh(&d_neighbors)
                .map_err(|e| anyhow!("CAGRA batch: neighbors D2H: {e}"))?;
            let distances_data = stream
                .clone_dtoh(&d_distances)
                .map_err(|e| anyhow!("CAGRA batch: distances D2H: {e}"))?;

            let n = ids.len();
            let mut all_results = Vec::with_capacity(n_queries);
            for qi in 0..n_queries {
                let mut results = Vec::with_capacity(k);
                for i in 0..k {
                    let idx = neighbors_data[qi * k + i] as usize;
                    if idx >= n {
                        continue;
                    }
                    let dist_val = distances_data[qi * k + i];
                    let distance = match self.space_type {
                        SpaceType::Euclidean => {
                            Distance::new_euclidean(dist_val)?
                        }
                        SpaceType::Cosine => {
                            Distance::new_cosine(dist_val)?
                        }
                        SpaceType::DotProduct => {
                            Distance::new_dot_product(dist_val)?
                        }
                        SpaceType::Hamming => unreachable!(),
                    };
                    results.push((ids[idx], distance));
                }
                all_results.push(results);
            }

            Ok(all_results)
        }
    }

    impl CuvsVectorIndex for CagraIndex {
        fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()> {
            self.vectors
                .write()
                .unwrap()
                .insert(primary_id, vector.as_ref().to_vec());
            // Invalidate cached build — next search will rebuild.
            *self.cached_build.write().unwrap() = None;
            *self.dirty.write().unwrap() = true;
            Ok(())
        }

        fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()> {
            self.vectors.write().unwrap().remove(&primary_id);
            // Invalidate cached build — next search will rebuild.
            *self.cached_build.write().unwrap() = None;
            *self.dirty.write().unwrap() = true;
            Ok(())
        }

        fn search(
            &self,
            vector: &Vector,
            limit: Limit,
        ) -> anyhow::Result<Vec<(PrimaryId, Distance)>> {
            self.ensure_fresh();

            let cache = self.cached_build.read().unwrap();
            let cached = cache.as_ref().ok_or_else(|| {
                anyhow::anyhow!("CAGRA index has no cached data")
            })?;

            let n = cached.ids.len();
            if n == 0 {
                return Ok(vec![]);
            }

            let k = limit.0.get().min(n);
            let query = vector.as_ref();

            if cached.index_ptr.is_null() {
                return self.search_brute_force(
                    query,
                    &cached.flat,
                    &cached.ids,
                    k,
                );
            }

            self.search_cagra(
                cached.res,
                cached.index_ptr,
                query,
                k,
                &cached.ids,
            )
        }

        fn search_batch(
            &self,
            queries: &[(&Vector, Limit)],
        ) -> Vec<anyhow::Result<Vec<(PrimaryId, Distance)>>> {
            if queries.is_empty() {
                return vec![];
            }

            // Rebuild once for the whole batch.
            self.ensure_fresh();

            let cache = self.cached_build.read().unwrap();
            let Some(cached) = cache.as_ref() else {
                return queries
                    .iter()
                    .map(|_| {
                        Err(anyhow::anyhow!("CAGRA index has no cached data"))
                    })
                    .collect();
            };

            let n = cached.ids.len();
            if n == 0 {
                return queries.iter().map(|_| Ok(vec![])).collect();
            }

            // Brute-force fallback for small datasets.
            if cached.index_ptr.is_null() {
                return queries
                    .iter()
                    .map(|(v, l)| {
                        let k = l.0.get().min(n);
                        self.search_brute_force(
                            v.as_ref(),
                            &cached.flat,
                            &cached.ids,
                            k,
                        )
                    })
                    .collect();
            }

            // Multi-query CAGRA search in a single GPU call.
            let max_k = queries
                .iter()
                .map(|(_, l)| l.0.get().min(n))
                .max()
                .unwrap_or(0);
            let query_slices: Vec<&[f32]> =
                queries.iter().map(|(v, _)| {
                    let r: &Vec<f32> = v.as_ref();
                    r.as_slice()
                }).collect();

            match self.search_cagra_batch(
                cached.res,
                cached.index_ptr,
                &query_slices,
                max_k,
                &cached.ids,
            ) {
                Ok(mut batch_results) => batch_results
                    .iter_mut()
                    .zip(queries.iter())
                    .map(|(results, (_, limit))| {
                        results.truncate(limit.0.get().min(n));
                        Ok(std::mem::take(results))
                    })
                    .collect(),
                Err(e) => {
                    warn!(
                        "CAGRA batch search failed ({e:#}), \
                         falling back to individual searches"
                    );
                    drop(cache);
                    queries
                        .iter()
                        .map(|(v, l)| self.search(v, l.clone()))
                        .collect()
                }
            }
        }

        fn size(&self) -> usize {
            self.vectors.read().unwrap().len()
        }
    }

    /// Load the cuVS library (singleton per process).
    static CUVS_LIB: std::sync::OnceLock<Result<Arc<cuvs_ffi::CuvsLib>, String>> =
        std::sync::OnceLock::new();

    /// Get or load the cuVS shared library.
    pub(super) fn load_cuvs_lib() -> anyhow::Result<Arc<cuvs_ffi::CuvsLib>> {
        let result = CUVS_LIB.get_or_init(|| {
            cuvs_ffi::CuvsLib::load()
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });
        match result {
            Ok(lib) => Ok(Arc::clone(lib)),
            Err(msg) => anyhow::bail!("{msg}"),
        }
    }
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
    use scylla::value::CqlValue;
    use std::num::NonZeroUsize;
    use std::time::Duration;
    use tokio::sync::watch;

    fn make_vector(values: Vec<f32>) -> Vector {
        Vector::from(values)
    }

    fn make_dimensions(n: usize) -> Dimensions {
        Dimensions::from(NonZeroUsize::new(n).unwrap())
    }

    fn make_limit(n: usize) -> Limit {
        Limit::from(NonZeroUsize::new(n).unwrap())
    }

    // --- BruteForceIndex tests ---

    #[test]
    fn test_brute_force_add_and_size() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);
        assert_eq!(index.size(), 0);

        index.add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0])).unwrap();
        assert_eq!(index.size(), 1);

        index.add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0])).unwrap();
        assert_eq!(index.size(), 2);
    }

    #[test]
    fn test_brute_force_add_and_remove() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);

        index.add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0])).unwrap();
        index.add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0])).unwrap();
        assert_eq!(index.size(), 2);

        index.remove(PrimaryId::from(1u64)).unwrap();
        assert_eq!(index.size(), 1);
    }

    #[test]
    fn test_brute_force_search_euclidean() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);

        // Three vectors in 2D space
        index.add(PrimaryId::from(1u64), &make_vector(vec![0.0, 0.0])).unwrap();
        index.add(PrimaryId::from(2u64), &make_vector(vec![1.0, 0.0])).unwrap();
        index.add(PrimaryId::from(3u64), &make_vector(vec![10.0, 10.0])).unwrap();

        // Search near origin - should find (0,0) and (1,0) first
        let results = index
            .search(&make_vector(vec![0.0, 0.0]), make_limit(2))
            .unwrap();

        assert_eq!(results.len(), 2);
        // Closest should be the origin itself (distance 0)
        assert_eq!(results[0].0, PrimaryId::from(1u64));
        let d0: f32 = results[0].1.into();
        assert_eq!(d0, 0.0);
        // Second closest should be (1,0) (distance 1.0 = squared L2)
        assert_eq!(results[1].0, PrimaryId::from(2u64));
        let d1: f32 = results[1].1.into();
        assert_eq!(d1, 1.0);
    }

    #[test]
    fn test_brute_force_search_cosine() {
        let index = BruteForceIndex::new(SpaceType::Cosine);

        // Two vectors pointing in different directions
        index.add(PrimaryId::from(1u64), &make_vector(vec![1.0, 0.0])).unwrap();
        index.add(PrimaryId::from(2u64), &make_vector(vec![0.0, 1.0])).unwrap();

        // Search with a vector pointing along x-axis
        let results = index
            .search(&make_vector(vec![1.0, 0.0]), make_limit(2))
            .unwrap();

        assert_eq!(results.len(), 2);
        // Closest should be (1,0) - same direction, cosine distance = 0
        assert_eq!(results[0].0, PrimaryId::from(1u64));
        let d0: f32 = results[0].1.into();
        assert!(d0.abs() < 1e-6, "expected ~0.0, got {d0}");
        // (0,1) is orthogonal, cosine distance = 1.0
        assert_eq!(results[1].0, PrimaryId::from(2u64));
        let d1: f32 = results[1].1.into();
        assert!((d1 - 1.0).abs() < 1e-6, "expected ~1.0, got {d1}");
    }

    #[test]
    fn test_brute_force_search_dot_product() {
        let index = BruteForceIndex::new(SpaceType::DotProduct);

        index.add(PrimaryId::from(1u64), &make_vector(vec![1.0, 0.0])).unwrap();
        index.add(PrimaryId::from(2u64), &make_vector(vec![0.5, 0.0])).unwrap();

        // Search with [1, 0] - dot product with [1,0] is 1.0, distance = 0.0
        let results = index
            .search(&make_vector(vec![1.0, 0.0]), make_limit(2))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, PrimaryId::from(1u64));
        let d0: f32 = results[0].1.into();
        assert!(d0.abs() < 1e-6, "expected ~0.0, got {d0}");
    }

    #[test]
    fn test_brute_force_search_empty_index() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);

        let results = index
            .search(&make_vector(vec![1.0, 2.0]), make_limit(5))
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_brute_force_search_limit_larger_than_size() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);

        index.add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0])).unwrap();

        let results = index
            .search(&make_vector(vec![1.0, 2.0]), make_limit(100))
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_brute_force_update_vector() {
        let index = BruteForceIndex::new(SpaceType::Euclidean);

        index.add(PrimaryId::from(1u64), &make_vector(vec![0.0, 0.0])).unwrap();
        // Overwrite with a new vector
        index.add(PrimaryId::from(1u64), &make_vector(vec![10.0, 10.0])).unwrap();

        assert_eq!(index.size(), 1);

        let results = index
            .search(&make_vector(vec![10.0, 10.0]), make_limit(1))
            .unwrap();
        let d: f32 = results[0].1.into();
        assert_eq!(d, 0.0);
    }

    // --- Distance computation tests ---

    #[test]
    fn test_euclidean_distance_computation() {
        let d = compute_distance(&[0.0, 0.0], &[3.0, 4.0], SpaceType::Euclidean).unwrap();
        let val: f32 = d.into();
        // squared L2: 9 + 16 = 25
        assert_eq!(val, 25.0);
    }

    #[test]
    fn test_cosine_distance_computation_identical() {
        let d = compute_distance(&[1.0, 0.0], &[1.0, 0.0], SpaceType::Cosine).unwrap();
        let val: f32 = d.into();
        assert!(val.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_computation_opposite() {
        let d = compute_distance(&[1.0, 0.0], &[-1.0, 0.0], SpaceType::Cosine).unwrap();
        let val: f32 = d.into();
        assert!((val - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_distance_computation() {
        // dot product of [1,0] and [1,0] = 1.0, distance = 1.0 - 1.0 = 0.0
        let d = compute_distance(&[1.0, 0.0], &[1.0, 0.0], SpaceType::DotProduct).unwrap();
        let val: f32 = d.into();
        assert!(val.abs() < 1e-6);
    }

    // --- Factory / validation tests ---

    #[test]
    fn test_hamming_distance_not_supported_by_cuvs() {
        let result = compute_distance(&[1.0, 0.0], &[0.0, 1.0], SpaceType::Hamming);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Hamming"));
    }

    #[test]
    fn test_engine_version() {
        let factory = CuvsIndexFactory {
            tokio_semaphore: Arc::new(Semaphore::new(1)),
            batch_config: BatchConfig::default(),
        };
        #[cfg(feature = "gpu")]
        assert_eq!(
            factory.index_engine_version(),
            "cuvs-gpu-cagra-v1"
        );
        #[cfg(not(feature = "gpu"))]
        assert_eq!(factory.index_engine_version(), "cuvs-brute-force-v1");
    }

    // --- Integration tests using the actor ---

    #[tokio::test]
    async fn test_add_and_search_via_actor() {
        let dimensions = make_dimensions(3);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        let primary_id = PrimaryId::from(0u64);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));
        mock_table
            .expect_primary_key()
            .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

        let table = Arc::new(RwLock::new(mock_table));

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &BatchConfig {
                batch_size: 1,
                ..BatchConfig::default()
            },
        )
        .unwrap();

        // Add a vector
        index_tx
            .add_vector(
                partition_id,
                primary_id,
                make_vector(vec![1.0, 2.0, 3.0]),
                None,
            )
            .await;

        // Give the actor time to process the add
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Search
        let (keys, distances) = index_tx
            .ann(index_key.clone(), make_vector(vec![1.0, 2.0, 3.0]), make_limit(1))
            .await
            .unwrap();

        assert_eq!(keys.len(), 1);
        assert_eq!(distances.len(), 1);
        let d: f32 = distances[0].into();
        assert_eq!(d, 0.0);
    }

    #[tokio::test]
    async fn test_add_remove_and_search_via_actor() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));
        mock_table
            .expect_primary_key()
            .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

        let table = Arc::new(RwLock::new(mock_table));

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &BatchConfig::default(),
        )
        .unwrap();

        let pid0 = PrimaryId::from(0u64);
        let pid1 = PrimaryId::from(1u64);

        // Add two vectors
        index_tx
            .add_vector(partition_id, pid0, make_vector(vec![1.0, 0.0]), None)
            .await;
        index_tx
            .add_vector(partition_id, pid1, make_vector(vec![0.0, 1.0]), None)
            .await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Remove first vector
        index_tx.remove_vector(partition_id, pid0, None).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Search should only find the remaining vector
        let (keys, _distances) = index_tx
            .ann(index_key.clone(), make_vector(vec![1.0, 0.0]), make_limit(10))
            .await
            .unwrap();

        assert_eq!(keys.len(), 1);
    }

    #[tokio::test]
    async fn test_search_empty_index_via_actor() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| None);

        let table = Arc::new(RwLock::new(mock_table));

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &BatchConfig::default(),
        )
        .unwrap();

        // Search on empty index should return empty results
        let (keys, distances) = index_tx
            .ann(index_key, make_vector(vec![1.0, 2.0]), make_limit(5))
            .await
            .unwrap();

        assert!(keys.is_empty());
        assert!(distances.is_empty());
    }

    #[tokio::test]
    async fn test_wrong_dimensions_rejected() {
        let dimensions = make_dimensions(3);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));

        let table = Arc::new(RwLock::new(mock_table));

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &BatchConfig::default(),
        )
        .unwrap();

        // Add a vector to create the partition
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(0u64),
                make_vector(vec![1.0, 2.0, 3.0]),
                None,
            )
            .await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Search with wrong dimensions
        let result = index_tx
            .ann(index_key, make_vector(vec![1.0, 2.0]), make_limit(1))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_count_via_actor() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));

        let table = Arc::new(RwLock::new(mock_table));

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &BatchConfig::default(),
        )
        .unwrap();

        // Count on empty index
        let count = index_tx.count(index_key.clone()).await.unwrap();
        assert_eq!(count, 0);

        // Add vectors
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(0u64),
                make_vector(vec![1.0, 2.0]),
                None,
            )
            .await;
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(1u64),
                make_vector(vec![3.0, 4.0]),
                None,
            )
            .await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        let count = index_tx.count(index_key).await.unwrap();
        assert_eq!(count, 2);
    }

    // --- Batching tests ---

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.batch_timeout, Duration::from_millis(5));
    }

    /// Verify that mutations are flushed when batch_timeout elapses,
    /// even if the batch is not full.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_batch_flush_on_timeout() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));
        mock_table
            .expect_primary_key()
            .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

        let table = Arc::new(RwLock::new(mock_table));

        // Large batch_size so the batch won't fill, short timeout to trigger flush
        let batch_config = BatchConfig {
            batch_size: 10000,
            batch_timeout: Duration::from_millis(20),
        };

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &batch_config,
        )
        .unwrap();

        // Add a single vector (well below batch_size)
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(0u64),
                make_vector(vec![1.0, 2.0]),
                None,
            )
            .await;

        // Poll until the timeout-triggered flush makes the vector visible.
        // Must use tokio::time::sleep (not std::thread::sleep) so the actor
        // task can make progress on the multi-thread runtime.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut found = false;
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let (keys, _) = index_tx
                .ann(index_key.clone(), make_vector(vec![1.0, 2.0]), make_limit(1))
                .await
                .unwrap();
            if keys.len() == 1 {
                found = true;
                break;
            }
        }
        assert!(found, "vector should be searchable after timeout flush");
    }

    /// Verify that the stale-read optimisation means search does NOT flush
    /// pending mutations. Vectors become visible only after a batch flush
    /// (by size or timeout).
    #[tokio::test]
    async fn test_search_triggers_batch_flush() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));
        mock_table
            .expect_primary_key()
            .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

        let table = Arc::new(RwLock::new(mock_table));

        // Very large batch_size and timeout so neither trigger
        let batch_config = BatchConfig {
            batch_size: 10000,
            batch_timeout: Duration::from_secs(60),
        };

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &batch_config,
        )
        .unwrap();

        // Add vectors (they will sit in the pending batch)
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(0u64),
                make_vector(vec![1.0, 0.0]),
                None,
            )
            .await;
        index_tx
            .add_vector(
                partition_id,
                PrimaryId::from(1u64),
                make_vector(vec![0.0, 1.0]),
                None,
            )
            .await;

        // Search should flush the batch first, then return results
        let (keys, _) = index_tx
            .ann(index_key, make_vector(vec![1.0, 0.0]), make_limit(10))
            .await
            .unwrap();

        assert_eq!(keys.len(), 0, "stale read: search should not flush pending batch");
    }

    /// Verify that the batch flushes when batch_size is reached.
    #[tokio::test]
    async fn test_batch_flush_on_size() {
        let dimensions = make_dimensions(2);
        let config_rx = watch::channel(Arc::new(Config::default())).1;
        let memory_tx = memory::new(config_rx.clone());

        let mut id_gen = IndexIdGenerator::new();
        let index_id = id_gen.next(true).unwrap();
        let partition_id = PartitionId::global(index_id);

        let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

        let mut mock_table = MockTableSearch::new();
        let ik = index_key.clone();
        mock_table
            .expect_partition_id()
            .withf(move |key, _| *key == ik)
            .returning(move |_, _| Some((partition_id, None)));
        mock_table
            .expect_index_id()
            .returning(move |_| Some(index_id));
        mock_table
            .expect_primary_key()
            .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

        let table = Arc::new(RwLock::new(mock_table));

        // Small batch_size = 3, long timeout so only size triggers flush
        let batch_config = BatchConfig {
            batch_size: 3,
            batch_timeout: Duration::from_secs(60),
        };

        let index_tx = new(
            SpaceType::Euclidean,
            index_key.clone(),
            dimensions,
            table,
            Arc::new(Semaphore::new(4)),
            memory_tx,
            &batch_config,
        )
        .unwrap();

        // Add exactly batch_size vectors
        for i in 0..3u64 {
            index_tx
                .add_vector(
                    partition_id,
                    PrimaryId::from(i),
                    make_vector(vec![i as f32, 0.0]),
                    None,
                )
                .await;
        }

        // Give the flush time to complete (it should fire immediately at size=3)
        tokio::time::sleep(Duration::from_millis(50)).await;

        let count = index_tx.count(index_key).await.unwrap();
        assert_eq!(count, 3, "batch should have flushed when batch_size reached");
    }

    // --- GPU brute-force tests (require `gpu` feature and NVIDIA GPU) ---

    #[cfg(feature = "gpu")]
    mod gpu_tests {
        use super::*;
        use crate::index::cuvs::gpu::GpuBruteForceIndex;
        use crate::index::cuvs::gpu::CagraIndex;
        use crate::index::cuvs::gpu::cuvs_ffi;

        #[test]
        fn test_gpu_add_and_size() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 3);
            assert_eq!(index.size(), 0);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0]))
                .unwrap();
            assert_eq!(index.size(), 1);

            index
                .add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0]))
                .unwrap();
            assert_eq!(index.size(), 2);
        }

        #[test]
        fn test_gpu_add_and_remove() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 3);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0]))
                .unwrap();
            assert_eq!(index.size(), 2);

            index.remove(PrimaryId::from(1u64)).unwrap();
            assert_eq!(index.size(), 1);
        }

        #[test]
        fn test_gpu_search_euclidean() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 2);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![0.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![1.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(3u64), &make_vector(vec![10.0, 10.0]))
                .unwrap();

            let results = index
                .search(&make_vector(vec![0.0, 0.0]), make_limit(2))
                .unwrap();

            assert_eq!(results.len(), 2);
            // Closest should be the origin (distance 0)
            assert_eq!(results[0].0, PrimaryId::from(1u64));
            let d0: f32 = results[0].1.into();
            assert_eq!(d0, 0.0);
            // Second closest: (1,0), squared L2 = 1.0
            assert_eq!(results[1].0, PrimaryId::from(2u64));
            let d1: f32 = results[1].1.into();
            assert_eq!(d1, 1.0);
        }

        #[test]
        fn test_gpu_search_cosine() {
            let index = GpuBruteForceIndex::new(SpaceType::Cosine, 2);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![0.0, 1.0]))
                .unwrap();

            let results = index
                .search(&make_vector(vec![1.0, 0.0]), make_limit(2))
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, PrimaryId::from(1u64));
            let d0: f32 = results[0].1.into();
            assert!(d0.abs() < 1e-6, "expected ~0.0, got {d0}");
            assert_eq!(results[1].0, PrimaryId::from(2u64));
            let d1: f32 = results[1].1.into();
            assert!((d1 - 1.0).abs() < 1e-6, "expected ~1.0, got {d1}");
        }

        #[test]
        fn test_gpu_search_dot_product() {
            let index = GpuBruteForceIndex::new(SpaceType::DotProduct, 2);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![0.5, 0.0]))
                .unwrap();

            let results = index
                .search(&make_vector(vec![1.0, 0.0]), make_limit(2))
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, PrimaryId::from(1u64));
            let d0: f32 = results[0].1.into();
            assert!(d0.abs() < 1e-6, "expected ~0.0, got {d0}");
        }

        #[test]
        fn test_gpu_search_empty() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 2);

            let results = index
                .search(&make_vector(vec![1.0, 2.0]), make_limit(5))
                .unwrap();

            assert!(results.is_empty());
        }

        #[test]
        fn test_gpu_search_limit_larger_than_size() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 2);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0]))
                .unwrap();

            let results = index
                .search(&make_vector(vec![1.0, 2.0]), make_limit(100))
                .unwrap();

            assert_eq!(results.len(), 1);
        }

        #[test]
        fn test_gpu_update_vector() {
            let index = GpuBruteForceIndex::new(SpaceType::Euclidean, 2);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![0.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(1u64), &make_vector(vec![10.0, 10.0]))
                .unwrap();

            assert_eq!(index.size(), 1);

            let results = index
                .search(&make_vector(vec![10.0, 10.0]), make_limit(1))
                .unwrap();
            let d: f32 = results[0].1.into();
            assert_eq!(d, 0.0);
        }

        #[test]
        fn test_gpu_engine_version() {
            let factory = CuvsIndexFactory {
                tokio_semaphore: Arc::new(Semaphore::new(1)),
                batch_config: BatchConfig::default(),
            };
            assert_eq!(
                factory.index_engine_version(),
                "cuvs-gpu-cagra-v1"
            );
        }

        /// Verify GPU results match CPU BruteForceIndex for the same data.
        #[test]
        fn test_gpu_matches_cpu_euclidean() {
            let cpu = BruteForceIndex::new(SpaceType::Euclidean);
            let gpu = GpuBruteForceIndex::new(SpaceType::Euclidean, 3);

            let vectors = vec![
                (1u64, vec![1.0, 0.0, 0.0]),
                (2u64, vec![0.0, 1.0, 0.0]),
                (3u64, vec![0.0, 0.0, 1.0]),
                (4u64, vec![0.5, 0.5, 0.0]),
            ];

            for (id, v) in &vectors {
                let vec = make_vector(v.clone());
                cpu.add(PrimaryId::from(*id), &vec).unwrap();
                gpu.add(PrimaryId::from(*id), &vec).unwrap();
            }

            let query = make_vector(vec![1.0, 0.0, 0.0]);
            let cpu_results = cpu.search(&query, make_limit(4)).unwrap();
            let gpu_results = gpu.search(&query, make_limit(4)).unwrap();

            assert_eq!(cpu_results.len(), gpu_results.len());
            for (cpu_r, gpu_r) in cpu_results.iter().zip(gpu_results.iter()) {
                assert_eq!(cpu_r.0, gpu_r.0, "PrimaryId mismatch");
                let cpu_d: f32 = cpu_r.1.into();
                let gpu_d: f32 = gpu_r.1.into();
                assert!(
                    (cpu_d - gpu_d).abs() < 1e-5,
                    "distance mismatch: cpu={cpu_d} gpu={gpu_d}"
                );
            }
        }

        /// Integration test: GPU index through the actor with batching.
        #[tokio::test]
        async fn test_gpu_add_and_search_via_actor() {
            let dimensions = make_dimensions(3);
            let config_rx = watch::channel(Arc::new(Config::default())).1;
            let memory_tx = memory::new(config_rx.clone());

            let mut id_gen = IndexIdGenerator::new();
            let index_id = id_gen.next(true).unwrap();
            let partition_id = PartitionId::global(index_id);
            let primary_id = PrimaryId::from(0u64);

            let index_key = IndexKey::new(&"test_ks".into(), &"test_idx".into());

            let mut mock_table = MockTableSearch::new();
            let ik = index_key.clone();
            mock_table
                .expect_partition_id()
                .withf(move |key, _| *key == ik)
                .returning(move |_, _| Some((partition_id, None)));
            mock_table
                .expect_index_id()
                .returning(move |_| Some(index_id));
            mock_table
                .expect_primary_key()
                .returning(move |_, _| Some(crate::PrimaryKey::from(vec![CqlValue::Int(0)])));

            let table = Arc::new(RwLock::new(mock_table));

            let index_tx = new(
                SpaceType::Euclidean,
                index_key.clone(),
                dimensions,
                table,
                Arc::new(Semaphore::new(4)),
                memory_tx,
                &BatchConfig {
                    batch_size: 1,
                    ..BatchConfig::default()
                },
            )
            .unwrap();

            index_tx
                .add_vector(
                    partition_id,
                    primary_id,
                    make_vector(vec![1.0, 2.0, 3.0]),
                    None,
                )
                .await;

            tokio::time::sleep(Duration::from_millis(100)).await;

            let (keys, distances) = index_tx
                .ann(
                    index_key.clone(),
                    make_vector(vec![1.0, 2.0, 3.0]),
                    make_limit(1),
                )
                .await
                .unwrap();

            assert_eq!(keys.len(), 1);
            assert_eq!(distances.len(), 1);
            let d: f32 = distances[0].into();
            assert_eq!(d, 0.0);
        }

        // --- CAGRA index tests ---
        //
        // These tests require libcuvs_c.so in LD_LIBRARY_PATH.
        // They are skipped gracefully if the library is not available.

        /// Helper to load the cuVS library, skipping the test if unavailable.
        fn try_load_cuvs() -> Option<Arc<cuvs_ffi::CuvsLib>> {
            match gpu::load_cuvs_lib() {
                Ok(lib) => Some(lib),
                Err(err) => {
                    eprintln!("skipping CAGRA test: {err}");
                    None
                }
            }
        }

        #[test]
        fn test_cagra_add_and_size() {
            let Some(lib) = try_load_cuvs() else { return };
            let index = CagraIndex::new(SpaceType::Euclidean, 3, lib);
            assert_eq!(index.size(), 0);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0]))
                .unwrap();
            assert_eq!(index.size(), 1);

            index
                .add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0]))
                .unwrap();
            assert_eq!(index.size(), 2);
        }

        #[test]
        fn test_cagra_add_and_remove() {
            let Some(lib) = try_load_cuvs() else { return };
            let index = CagraIndex::new(SpaceType::Euclidean, 3, lib);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![1.0, 2.0, 3.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![4.0, 5.0, 6.0]))
                .unwrap();
            assert_eq!(index.size(), 2);

            index.remove(PrimaryId::from(1u64)).unwrap();
            assert_eq!(index.size(), 1);
        }

        /// Test CAGRA search with enough vectors to trigger graph build.
        /// Generates 200 random-ish vectors, inserts a known target, and
        /// verifies it's found as the nearest neighbor.
        #[test]
        fn test_cagra_search_euclidean() {
            let Some(lib) = try_load_cuvs() else { return };
            let d = 16;
            let n = 200;
            let index = CagraIndex::new(SpaceType::Euclidean, d, lib);

            // Insert `n` vectors: vector i = [i as f32 / n; d]
            for i in 0..n {
                let val = i as f32 / n as f32;
                let v: Vec<f32> = vec![val; d];
                index
                    .add(PrimaryId::from(i as u64), &make_vector(v))
                    .unwrap();
            }

            // Query is exactly vector 0 → should find PrimaryId(0) closest.
            let query = make_vector(vec![0.0; d]);
            let results = index.search(&query, make_limit(5)).unwrap();

            assert!(!results.is_empty(), "expected at least 1 result");
            // The nearest neighbor should be PrimaryId(0) (distance ~0)
            assert_eq!(
                results[0].0,
                PrimaryId::from(0u64),
                "expected PrimaryId(0) as nearest"
            );
            let d0: f32 = results[0].1.into();
            assert!(d0 < 1e-4, "expected distance ~0, got {d0}");
        }

        #[test]
        fn test_cagra_search_cosine() {
            let Some(lib) = try_load_cuvs() else { return };
            let d = 16;
            let n = 200;
            let index = CagraIndex::new(SpaceType::Cosine, d, lib);

            // Insert vectors with varying directions
            for i in 0..n {
                let mut v = vec![0.0f32; d];
                v[i % d] = 1.0;
                v[(i + 1) % d] = (i as f32) / n as f32;
                index
                    .add(PrimaryId::from(i as u64), &make_vector(v))
                    .unwrap();
            }

            // Query along first dimension
            let mut query_v = vec![0.0f32; d];
            query_v[0] = 1.0;
            let query = make_vector(query_v);
            let results = index.search(&query, make_limit(5)).unwrap();

            assert!(!results.is_empty(), "expected at least 1 result");
            // The result should have small cosine distance
            let d0: f32 = results[0].1.into();
            assert!(d0 < 0.5, "expected small cosine distance, got {d0}");
        }

        #[test]
        fn test_cagra_search_dot_product() {
            let Some(lib) = try_load_cuvs() else { return };
            let d = 16;
            let n = 200;
            let index = CagraIndex::new(SpaceType::DotProduct, d, lib);

            for i in 0..n {
                let val = (i as f32 + 1.0) / n as f32;
                let v: Vec<f32> = vec![val; d];
                index
                    .add(PrimaryId::from(i as u64), &make_vector(v))
                    .unwrap();
            }

            // Query with all 1.0s → highest dot product with the largest vector
            let query = make_vector(vec![1.0; d]);
            let results = index.search(&query, make_limit(5)).unwrap();

            assert!(!results.is_empty(), "expected at least 1 result");
            // The nearest (smallest distance=1-dot) should be the largest vector
            assert_eq!(
                results[0].0,
                PrimaryId::from((n - 1) as u64),
                "expected PrimaryId({}) as nearest",
                n - 1
            );
        }

        #[test]
        fn test_cagra_search_empty() {
            let Some(lib) = try_load_cuvs() else { return };
            let index = CagraIndex::new(SpaceType::Euclidean, 3, lib);

            let results = index
                .search(&make_vector(vec![1.0, 2.0, 3.0]), make_limit(5))
                .unwrap();

            assert!(results.is_empty());
        }

        /// Test that small datasets (< MIN_CAGRA_BUILD_SIZE) fall back to
        /// brute-force and still return correct results.
        #[test]
        fn test_cagra_small_dataset_fallback() {
            let Some(lib) = try_load_cuvs() else { return };
            let index = CagraIndex::new(SpaceType::Euclidean, 2, lib);

            index
                .add(PrimaryId::from(1u64), &make_vector(vec![0.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(2u64), &make_vector(vec![1.0, 0.0]))
                .unwrap();
            index
                .add(PrimaryId::from(3u64), &make_vector(vec![10.0, 10.0]))
                .unwrap();

            let results = index
                .search(&make_vector(vec![0.0, 0.0]), make_limit(2))
                .unwrap();

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, PrimaryId::from(1u64));
            let d0: f32 = results[0].1.into();
            assert_eq!(d0, 0.0);
        }

        /// Verify CAGRA results match CPU BruteForceIndex for the same data
        /// (approximate — CAGRA is ANN so we allow some tolerance in ranking).
        #[test]
        fn test_cagra_matches_cpu_top1() {
            let Some(lib) = try_load_cuvs() else { return };
            let d = 16;
            let n = 200;

            let cpu = BruteForceIndex::new(SpaceType::Euclidean);
            let cagra = CagraIndex::new(SpaceType::Euclidean, d, lib);

            for i in 0..n {
                let val = i as f32 / n as f32;
                let v = make_vector(vec![val; d]);
                cpu.add(PrimaryId::from(i as u64), &v).unwrap();
                cagra.add(PrimaryId::from(i as u64), &v).unwrap();
            }

            let query = make_vector(vec![0.0; d]);
            let cpu_results = cpu.search(&query, make_limit(1)).unwrap();
            let cagra_results = cagra.search(&query, make_limit(1)).unwrap();

            assert_eq!(cpu_results.len(), 1);
            assert_eq!(cagra_results.len(), 1);
            // Top-1 result should match for this simple dataset
            assert_eq!(
                cpu_results[0].0, cagra_results[0].0,
                "top-1 PrimaryId mismatch"
            );
        }

        #[test]
        fn test_cagra_update_vector() {
            let Some(lib) = try_load_cuvs() else { return };
            let d = 16;
            let n = 200;
            let index = CagraIndex::new(SpaceType::Euclidean, d, lib);

            // Fill with baseline vectors
            for i in 0..n {
                let v: Vec<f32> = vec![1.0; d];
                index
                    .add(PrimaryId::from(i as u64), &make_vector(v))
                    .unwrap();
            }

            // Update vector 0 to be at origin
            index
                .add(PrimaryId::from(0u64), &make_vector(vec![0.0; d]))
                .unwrap();

            assert_eq!(index.size(), n);

            // Search for origin → should find PrimaryId(0)
            let results = index
                .search(&make_vector(vec![0.0; d]), make_limit(1))
                .unwrap();
            assert_eq!(results[0].0, PrimaryId::from(0u64));
        }
    }
}
