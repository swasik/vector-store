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
/// When the `gpu` feature is enabled, returns a GPU-accelerated brute-force
/// index using cuBLAS SGEMM for distance computation. Otherwise falls back
/// to the CPU implementation.
fn create_partition_index(
    space_type: SpaceType,
    dimensions: Dimensions,
) -> Arc<dyn CuvsVectorIndex> {
    #[cfg(feature = "gpu")]
    {
        Arc::new(gpu::GpuBruteForceIndex::new(
            space_type,
            dimensions.0.get(),
        ))
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
            "cuvs-gpu-brute-force-v1".to_string()
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
                            // Flush pending mutations before search to ensure
                            // the index is up-to-date.
                            flush_batch(&mut pending, &tokio_semaphore).await;
                            flush_deadline = None;
                            dispatch_search(state, partition, &table, &tokio_semaphore, msg)
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

        // Filtered search is not supported in v1 of the cuVS backend.
        Index::FilteredAnn {
            index_key, tx, ..
        } => {
            _ = tx.send(Err(anyhow!(
                "cuVS backend does not support filtered search (index {index_key:?})"
            )));
            None
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

async fn dispatch_search<T>(
    state: &mut IndexState,
    partition: Arc<PartitionState>,
    table: &Arc<RwLock<T>>,
    tokio_semaphore: &Arc<Semaphore>,
    msg: Index,
) where
    T: TableSearch + Send + Sync + 'static,
{
    let permit = Arc::clone(tokio_semaphore).acquire_owned().await.unwrap();
    let table = Arc::clone(table);
    let dimensions = state.dimensions;

    tokio::spawn(async move {
        crate::move_to_the_end_of_async_runtime_queue().await;
        let result =
            tokio::task::spawn_blocking(move || process_search(partition, table, dimensions, msg))
                .await;
        if let Err(err) = result {
            error!("cuVS search task panicked: {err}");
        }
        drop(permit);
    });
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
        tx_ann
            .send(Err(err))
            .unwrap_or_else(|_| trace!("validate_dimensions: unable to send response"));
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
    tx_ann
        .send(
            partition
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

/// GPU-accelerated brute-force search backend using cuBLAS SGEMM.
///
/// When compiled with `--features gpu`, this module provides a
/// `GpuBruteForceIndex` that accelerates exhaustive nearest-neighbor search
/// on NVIDIA GPUs. Vectors are stored in host memory and transferred to the
/// GPU for each search. All-pairs inner products are computed via a single
/// cuBLAS SGEMM call (`dataset × query^T`), then distances are derived on
/// the host from the inner products.
///
/// The `cudarc` crate dynamically loads CUDA libraries at runtime via
/// `fallback-dynamic-loading`, so the binary compiles without any CUDA
/// toolkit installed and fails gracefully at runtime if no GPU is available.
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::CudaContext;

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
            "cuvs-gpu-brute-force-v1"
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
            &BatchConfig::default(),
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
    #[tokio::test]
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

        // Wait longer than batch_timeout so the flush fires
        tokio::time::sleep(Duration::from_millis(100)).await;

        let (keys, _) = index_tx
            .ann(index_key, make_vector(vec![1.0, 2.0]), make_limit(1))
            .await
            .unwrap();

        assert_eq!(keys.len(), 1, "vector should be searchable after timeout flush");
    }

    /// Verify that pending mutations are flushed before a search executes,
    /// ensuring consistency.
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

        assert_eq!(keys.len(), 2, "search should flush pending batch and find all vectors");
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
                "cuvs-gpu-brute-force-v1"
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
                &BatchConfig::default(),
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
    }
}
