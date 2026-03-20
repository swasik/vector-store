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
/// the inner search logic when the `cuvs` feature is enabled.
struct BruteForceIndex {
    vectors: RwLock<BTreeMap<PrimaryId, Vec<f32>>>,
    space_type: SpaceType,
}

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

pub struct CuvsIndexFactory {
    tokio_semaphore: Arc<Semaphore>,
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
        )
    }

    fn index_engine_version(&self) -> String {
        "cuvs-brute-force-v1".to_string()
    }
}

pub fn new_cuvs(tokio_semaphore: Arc<Semaphore>) -> CuvsIndexFactory {
    CuvsIndexFactory { tokio_semaphore }
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
) -> anyhow::Result<mpsc::Sender<Index>> {
    const CHANNEL_SIZE: usize = 10;
    let (tx, mut rx) = mpsc::channel(CHANNEL_SIZE);

    tokio::spawn(
        {
            let index_key = index_key.clone();
            async move {
                debug!("starting");
                let mut states: BTreeMap<IndexId, IndexState> = BTreeMap::new();
                let mut partitions: BTreeMap<PartitionId, Arc<PartitionState>> = BTreeMap::new();
                let mut allocate_prev = Allocate::Can;

                while let Some(msg) = rx.recv().await {
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

                    dispatch_task(state, partition, &table, &tokio_semaphore, msg).await;
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
            let idx = Arc::new(BruteForceIndex::new(space_type));
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

async fn dispatch_task<T>(
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
    let size = Arc::clone(&state.size);

    tokio::spawn(async move {
        crate::move_to_the_end_of_async_runtime_queue().await;
        // cuVS operations are dispatched via spawn_blocking to avoid blocking
        // the tokio runtime. In future GPU implementation, these will map to
        // CUDA kernel launches.
        let result =
            tokio::task::spawn_blocking(move || process(partition, table, dimensions, size, msg))
                .await;
        if let Err(err) = result {
            error!("cuVS task panicked: {err}");
        }
        drop(permit);
    });
}

fn process<T>(
    partition: Arc<PartitionState>,
    table: Arc<RwLock<T>>,
    dimensions: Dimensions,
    size: Arc<AtomicUsize>,
    msg: Index,
) where
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
                ann(&partition, tx, &table, embedding, limit);
            }
        }

        Index::FilteredAnn { tx, index_key, .. } => {
            _ = tx.send(Err(anyhow!(
                "cuVS backend does not support filtered search (index {index_key:?})"
            )));
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

fn add(idx: &dyn CuvsVectorIndex, primary_id: PrimaryId, embedding: &Vector, size: &AtomicUsize) {
    if let Err(err) = idx.add(primary_id, embedding) {
        warn!("add: unable to add embedding: {err}");
    } else {
        size.fetch_add(1, Ordering::Relaxed);
    }
}

fn remove(idx: &dyn CuvsVectorIndex, primary_id: PrimaryId, size: &AtomicUsize) {
    if let Err(err) = idx.remove(primary_id) {
        warn!("remove: unable to remove embedding: {err}");
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
        };
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
}
