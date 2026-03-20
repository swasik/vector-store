# Design: NVIDIA cuVS Backend for Vector Store

## Summary

Add NVIDIA cuVS (CUDA Vector Search) as an alternative vector index backend alongside the existing usearch and OpenSearch backends. cuVS provides GPU-accelerated approximate nearest neighbor (ANN) search using algorithms like CAGRA (a graph-based GPU ANN algorithm), IVF-Flat, and IVF-PQ, offering significant throughput improvements for workloads that can leverage GPU hardware.

## Current Architecture

The vector store uses a trait-based factory pattern to abstract over index backends:

### Key abstractions

- **`IndexFactory` trait** ([crates/vector-store/src/index/factory.rs](../crates/vector-store/src/index/factory.rs)):
  Central abstraction that each backend implements. Has two methods:
  - `create_index(IndexConfiguration, table, memory) -> Sender<Index>` — creates an index actor
  - `index_engine_version() -> String` — returns engine version string

- **`IndexConfiguration` struct** ([crates/vector-store/src/index/factory.rs](../crates/vector-store/src/index/factory.rs)):
  Holds index parameters: `key`, `dimensions`, `connectivity`, `expansion_add`, `expansion_search`, `space_type`, `quantization`.

- **`Index` enum** ([crates/vector-store/src/index/actor.rs](../crates/vector-store/src/index/actor.rs)):
  Actor messages: `AddVector`, `RemoveVector`, `RemovePartition`, `Ann`, `FilteredAnn`, `Count`.

- **`UsearchIndex` trait** (internal to `index/usearch.rs`):
  Low-level operations: `reserve`, `size`, `capacity`, `add`, `remove`, `search`, `filtered_search`, `stop`.

### Existing backends

| Backend | File | Selection |
|---------|------|-----------|
| usearch (default) | `crates/vector-store/src/index/usearch.rs` | Default when no `VECTOR_STORE_OPENSEARCH_URI` is set |
| OpenSearch | `crates/vector-store/src/index/opensearch.rs` | `VECTOR_STORE_OPENSEARCH_URI` env var |

### Backend selection flow

1. `config_manager.rs` reads `VECTOR_STORE_OPENSEARCH_URI` env var into `Config.opensearch_addr`
2. `main.rs` checks `config.opensearch_addr`:
   - `Some(addr)` → `new_index_factory_opensearch(addr, config_rx)`
   - `None` → `new_index_factory_usearch(config_rx)`
3. The factory is passed as `Box<dyn IndexFactory + Send + Sync>` to `engine::new()`

## Proposed Changes

### 1. New Rust crate for cuVS C/C++ FFI bindings (optional)

**Path:** `crates/cuvs-sys/` (if no suitable crate on crates.io)

cuVS is a C++/CUDA library. We need Rust bindings. Options:
- **Option A (preferred): Use `cuvs` crate** — NVIDIA publishes `cuvs-rs` (Rust bindings). If a suitable version exists on crates.io, use it directly as a workspace dependency.
- **Option B: Thin FFI crate** — Create `crates/cuvs-sys/` with `bindgen` or hand-written `extern "C"` bindings to `libcuvs`. This gives full control but requires maintenance.

In either case, add the dependency to `Cargo.toml` workspace dependencies and gate it behind a **Cargo feature flag** (see below).

### 2. Cargo feature flag: `cuvs`

**Files:** `Cargo.toml` (workspace), `crates/vector-store/Cargo.toml`

Add an optional feature so that cuVS support is opt-in at compile time. This avoids requiring CUDA SDK/GPU drivers for builds that don't need it.

```toml
# crates/vector-store/Cargo.toml
[features]
default = []
dev-tools = []
cuvs = ["dep:cuvs"]       # <-- new

[dependencies]
cuvs = { workspace = true, optional = true }  # <-- new
```

```toml
# Cargo.toml (workspace)
[workspace.dependencies]
cuvs = "0.x.y"   # or path = "crates/cuvs-sys"
```

### 3. New module: `crates/vector-store/src/index/cuvs.rs`

**Pattern:** Mirror `index/usearch.rs` structure — it already defines a clean internal trait + factory.

#### 3.1 Internal `CuvsIndex` trait

Define a trait analogous to `UsearchIndex` but adapted for cuVS semantics:

```rust
// crates/vector-store/src/index/cuvs.rs

trait CuvsIndex: Send + Sync {
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
```

**Key design considerations for cuVS:**

- **GPU memory management:** cuVS indexes live in GPU memory. The `reserve` implementation must allocate GPU memory via CUDA. Consider tracking GPU memory usage separately from the existing `Memory` actor (which tracks CPU/system memory). A new `GpuMemory` concern or extending the existing `Memory` enum may be needed.

- **Batch operations:** cuVS (especially CAGRA) achieves peak throughput with batched operations. Unlike usearch where single add/search operations are fine, cuVS benefits from batching multiple vectors in a single GPU kernel call. The implementation should consider a batching layer:
  - Accumulate add/remove operations in a buffer
  - Flush the buffer either when it reaches a configurable batch size or after a timeout
  - For search: batch multiple concurrent ANN queries together before dispatching to GPU

- **Index build vs. search:** CAGRA builds the graph on GPU. There are two phases:
  1. **Build phase:** Load all vectors, build the index (GPU-intensive, one-time cost per index)
  2. **Search phase:** Query the built index (GPU-accelerated)
  
  Incremental adds after initial build may require periodic index rebuild, or use IVF-PQ which supports incremental additions more naturally. This should be configurable via an environment variable or index parameter.

- **Host ↔ Device transfer:** Vectors arrive from ScyllaDB as CPU-side data (`Vec<f32>`). They need to be transferred to GPU memory. This transfer cost must be accounted for. Consider using pinned (page-locked) host memory for faster transfers.

- **cuVS algorithm selection:** cuVS supports multiple algorithms. The algorithm choice should be configurable. Options:
  - **CAGRA** — Graph-based, highest throughput for large indexes, GPU-only build + search
  - **IVF-Flat** — Inverted file index, supports incremental add, lower memory overhead
  - **IVF-PQ** — With product quantization, most memory-efficient, supports incremental add

#### 3.2 `CuvsIndexFactory` struct

```rust
pub struct CuvsIndexFactory {
    tokio_semaphore: Arc<Semaphore>,
    gpu_device: i32,            // CUDA device ordinal
    algorithm: CuvsAlgorithm,   // CAGRA | IVF_FLAT | IVF_PQ
}

enum CuvsAlgorithm {
    Cagra,
    IvfFlat,
    IvfPq,
}

impl IndexFactory for CuvsIndexFactory {
    fn create_index(
        &self,
        index: IndexConfiguration,
        table: Arc<RwLock<Table>>,
        memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<Index>> {
        // 1. Initialize cuVS index with appropriate parameters
        // 2. Map IndexConfiguration fields to cuVS equivalents:
        //    - dimensions  → cuVS dimensions
        //    - space_type  → cuVS metric (L2, InnerProduct, Cosine)
        //    - connectivity → CAGRA graph_degree (or IVF n_lists)
        //    - expansion_add → CAGRA intermediate_graph_degree
        //    - expansion_search → CAGRA itopk_size
        //    - quantization → cuVS supports f32, f16, int8
        // 3. Spawn actor task (same pattern as usearch.rs `new()` function)
        // 4. Return mpsc::Sender<Index>
        todo!()
    }

    fn index_engine_version(&self) -> String {
        format!("cuvs-{}", /* cuvs version */)
    }
}
```

#### 3.3 Space type / metric mapping

```rust
fn map_space_type(space_type: SpaceType) -> anyhow::Result<CuvsMetric> {
    match space_type {
        SpaceType::Euclidean => Ok(CuvsMetric::L2),
        SpaceType::Cosine => Ok(CuvsMetric::Cosine),
        SpaceType::DotProduct => Ok(CuvsMetric::InnerProduct),
        SpaceType::Hamming => anyhow::bail!("cuVS does not support Hamming distance"),
    }
}
```

#### 3.4 Distance conversion

cuVS returns raw distance values. The conversion to the `Distance` enum should use the same logic as usearch — using `Distance::try_from((raw_distance, space_type, dimensions))`.

#### 3.5 Actor structure

Reuse the same actor pattern from `usearch.rs`. The internal `new()` function spawns a tokio task that:
1. Receives `Index` messages via `mpsc::channel`
2. Preprocesses them (same `preprocess()` logic — partition routing, index creation)
3. Dispatches to GPU via `CuvsIndex` trait methods

The key difference from usearch: GPU operations should be dispatched via tokio's `spawn_blocking` or a dedicated GPU thread pool (not rayon) since CUDA calls are blocking.

### 4. Register the module

**File:** `crates/vector-store/src/index/mod.rs`

```rust
pub(crate) mod usearch;
pub(crate) mod opensearch;
#[cfg(feature = "cuvs")]
pub(crate) mod cuvs;
```

### 5. Configuration

**File:** `crates/vector-store/src/config_manager.rs`

Add new environment variables:

| Environment Variable | Type | Description |
|---------------------|------|-------------|
| `VECTOR_STORE_CUVS_DEVICE` | `i32` | CUDA device ordinal (default: `0`) |
| `VECTOR_STORE_CUVS_ALGORITHM` | `String` | Algorithm: `CAGRA`, `IVF_FLAT`, `IVF_PQ` (default: `CAGRA`) |
| `VECTOR_STORE_CUVS_BATCH_SIZE` | `usize` | Max batch size for GPU operations (default: `1024`) |
| `VECTOR_STORE_CUVS_BATCH_TIMEOUT_MS` | `u64` | Max wait time to fill a batch (default: `5`) |

**File:** `crates/vector-store/src/lib.rs`

Add to `Config` struct:

```rust
pub struct Config {
    // ... existing fields ...
    pub cuvs_device: Option<i32>,
    pub cuvs_algorithm: Option<String>,
    pub cuvs_batch_size: Option<usize>,
    pub cuvs_batch_timeout: Option<Duration>,
}
```

### 6. Backend selection in `main.rs`

**File:** `crates/vector-store/src/main.rs`

Extend the backend selection logic. Use an explicit `VECTOR_STORE_BACKEND` env var for clarity, or maintain backward compatibility with the existing `VECTOR_STORE_OPENSEARCH_URI` approach:

**Option A: Explicit backend env var (recommended)**

```rust
let index_factory: Box<dyn IndexFactory + Send + Sync> = match config.backend.as_deref() {
    Some("opensearch") => {
        let addr = config.opensearch_addr.as_ref()
            .ok_or_else(|| anyhow!("VECTOR_STORE_OPENSEARCH_URI required for opensearch backend"))?;
        tracing::info!("Using OpenSearch index factory at {addr}");
        vector_store::new_index_factory_opensearch(addr.clone(), config_rx.clone())?
    }
    Some("cuvs") => {
        tracing::info!("Using cuVS index factory");
        vector_store::new_index_factory_cuvs(config_rx.clone())?
    }
    Some("usearch") | None => {
        tracing::info!("Using Usearch index factory");
        vector_store::new_index_factory_usearch(config_rx.clone())?
    }
    Some(other) => {
        anyhow::bail!("Unknown backend: {other}. Supported: usearch, opensearch, cuvs");
    }
};
```

**Option B: Backward-compatible layered approach (if we want to avoid breaking existing deployments)**

Keep the existing `opensearch_addr` check, and add a new `VECTOR_STORE_CUVS_ENABLED=true` check:

```rust
let index_factory = if let Some(addr) = opensearch_addr {
    tracing::info!("Using OpenSearch index factory at {addr}");
    vector_store::new_index_factory_opensearch(addr, config_rx.clone())?
} else if config.cuvs_enabled {
    tracing::info!("Using cuVS index factory");
    vector_store::new_index_factory_cuvs(config_rx.clone())?
} else {
    tracing::info!("Using Usearch index factory");
    vector_store::new_index_factory_usearch(config_rx.clone())?
};
```

### 7. Public API in `lib.rs`

**File:** `crates/vector-store/src/lib.rs`

Add the factory constructor (gated behind feature flag):

```rust
#[cfg(feature = "cuvs")]
pub fn new_index_factory_cuvs(
    config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<Box<dyn IndexFactory + Send + Sync>> {
    let config = config_rx.borrow().clone();
    let device = config.cuvs_device.unwrap_or(0);
    let algorithm = config.cuvs_algorithm.as_deref().unwrap_or("CAGRA");

    // GPU operations are not CPU-bound in the same way as usearch.
    // Use a semaphore to limit concurrent GPU submissions to avoid OOM on the GPU.
    let gpu_concurrency = 4; // configurable
    let tokio_semaphore = Arc::new(Semaphore::new(gpu_concurrency));

    Ok(Box::new(index::cuvs::new_cuvs(
        tokio_semaphore,
        device,
        algorithm.parse()?,
        config_rx,
    )?))
}
```

### 8. Quantization mapping

cuVS supports a subset of quantization types. Map from the existing `Quantization` enum:

| Vector Store `Quantization` | cuVS Support | Notes |
|----------------------------|--------------|-------|
| `F32` | Yes | Native support |
| `F16` | Yes | Native support |
| `BF16` | Partial | CAGRA may not support; IVF-PQ does |
| `I8` | Yes | Via IVF-PQ SQ (scalar quantization) |
| `B1` | No | Not supported — return error |

The factory's `create_index` should validate that the requested quantization is supported and return a clear error if not.

### 9. Filtered search implementation

cuVS (as of RAPIDS 24.x) has limited built-in filtering support. Two approaches:

- **Pre-filtering:** Apply filter to get a candidate set, then search only within those candidates. Requires the cuVS index to support an allow-list. CAGRA supports a `sample_filter` parameter.
- **Post-filtering with over-fetch:** Request `limit * overfetch_factor` results from cuVS, then apply the filter on the CPU side. Simple to implement but wastes GPU compute.

For the initial implementation, **post-filtering with over-fetch** is recommended as it requires no cuVS-specific filtering API changes. The overfetch factor should be configurable.

### 10. Memory management considerations

**GPU memory tracking:**

The existing `Memory` actor tracks system RAM. GPU memory is a separate resource. Options:

- **Extend `Memory` enum** with a `GpuCanAllocate` / `GpuAllocate` variant
- **Separate GPU memory check** inside `CuvsIndexFactory` using CUDA's `cudaMemGetInfo` to check free/total GPU memory before allocating

For the initial implementation, use `cudaMemGetInfo` directly in the cuVS module. Integration with the `Memory` actor can be done as a follow-up.

### 11. Thread/runtime model

usearch uses rayon for CPU-bound add/remove operations. cuVS is different:

- **GPU operations** are dispatched from a host thread and execute asynchronously on the GPU.
- **CUDA streams** allow concurrent GPU operations.
- Use `tokio::task::spawn_blocking` for cuVS operations rather than rayon, since they are blocking I/O-like calls to the CUDA runtime (not CPU-compute-bound).
- The `tokio_semaphore` limits how many GPU operations are in-flight concurrently (to prevent GPU OOM).
- The `rayon_semaphore` from usearch is **not needed** for cuVS.

### 12. Tests

**File:** `crates/vector-store/src/index/cuvs.rs` (same file, `#[cfg(test)]` module)

- Unit tests with mock `CuvsIndex` implementation (no GPU required)
- Metric mapping tests
- Quantization validation tests
- Batch accumulation tests

**File:** `crates/vector-store/tests/` (integration tests)

- Feature-gated integration tests that require a GPU (`#[cfg(feature = "cuvs")]`)
- End-to-end test: create index, add vectors, search, verify results
- Test fallback: cuVS factory should fail gracefully with a clear error message when no GPU is available

### 13. Build & CI considerations

- cuVS requires CUDA toolkit at build time and NVIDIA GPU drivers at runtime.
- CI pipeline needs a GPU-enabled runner (or the cuVS tests run only in a nightly/special pipeline).
- Feature flag ensures `cargo build` and `cargo test` work without CUDA by default.
- Add documentation in `docs/` explaining how to build with cuVS support.
- The SBOM (`.cdx.json`) will need updating to include the cuVS dependency.

## File Change Summary

| File | Change Type | Description |
|------|------------|-------------|
| `Cargo.toml` | Modify | Add `cuvs` to workspace dependencies |
| `crates/vector-store/Cargo.toml` | Modify | Add `cuvs` feature flag and optional dependency |
| `crates/vector-store/src/index/mod.rs` | Modify | Add `#[cfg(feature = "cuvs")] pub(crate) mod cuvs;` |
| `crates/vector-store/src/index/cuvs.rs` | **New** | cuVS backend implementation (~500-800 lines) |
| `crates/vector-store/src/lib.rs` | Modify | Add `Config` fields, `new_index_factory_cuvs()` function |
| `crates/vector-store/src/config_manager.rs` | Modify | Parse `VECTOR_STORE_CUVS_*` env vars |
| `crates/vector-store/src/main.rs` | Modify | Extend backend selection logic |

## Implementation Order

1. **Add feature flag and dependency** — `Cargo.toml` changes
2. **Implement `CuvsIndex` trait and `CuvsIndexFactory`** — `index/cuvs.rs` (start with CAGRA only)
3. **Register module** — `index/mod.rs`
4. **Add configuration** — `config_manager.rs`, `lib.rs` (Config struct + env vars)
5. **Wire up backend selection** — `main.rs`, `lib.rs` (factory constructor)
6. **Unit tests** — Mock-based tests in `index/cuvs.rs`
7. **Integration tests** — Feature-gated GPU tests
8. **Documentation** — Build instructions, configuration reference
9. **Follow-up: Batching layer** — Optimize throughput with GPU batch operations
10. **Follow-up: GPU memory integration** — Wire GPU memory tracking into `Memory` actor

## Open Questions

1. **cuVS Rust bindings maturity:** Check the current state of `cuvs-rs` on crates.io. If immature, we may need our own FFI crate.
2. **Multi-GPU support:** Should we support multiple GPUs (one index per GPU, or sharding across GPUs)? Defer to a follow-up.
3. **Hybrid CPU/GPU:** Should we support a mode where build happens on GPU and search falls back to CPU if GPU is busy? CAGRA supports exporting to HNSW for CPU search.
4. **Index persistence:** cuVS CAGRA indexes can be serialized. Should we persist them to disk to avoid rebuild on restart?
5. **Incremental updates:** CAGRA does not natively support incremental add/remove well. If the workload has frequent updates, IVF-Flat/IVF-PQ may be better. Should the algorithm auto-select based on workload?
