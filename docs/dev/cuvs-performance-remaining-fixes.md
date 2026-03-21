# cuVS Performance: Remaining Optimisations

These items were identified as root causes for the cuVS throughput gap
versus usearch but have **not yet been implemented**. Fixes #2 (cache
CudaContext), #3 (stale reads), and #4 (query batching) are already applied.

---

## Fix #1 — Decouple CAGRA graph rebuild from the search path (CRITICAL)

**Problem:** Every mutation sets `dirty = true` and clears `cached_build`.
The next search rebuilds the *entire* CAGRA graph from scratch. Under
sustained write + read traffic this means a full GPU rebuild on nearly
every search.

**Suggested approach:**

1. Remove the `cached_build = None` invalidation from `add()` / `remove()`.
   Instead, keep the last-known good graph alive.
2. Introduce a background rebuild task (spawned from the actor loop) that
   periodically checks `dirty` and rebuilds asynchronously.
3. While the rebuild is in progress, searches continue using the stale
   CAGRA graph. Newly added vectors that are not yet in the graph are
   accumulated in a small secondary brute-force index and merged into
   search results.
4. When the rebuild finishes, atomically swap `cached_build` and clear the
   secondary index.

**Files to modify:**

- `crates/vector-store/src/index/cuvs.rs` — `CagraIndex::{add, remove, search, ensure_fresh}`

---

## ~~Fix #2 — Cache CudaContext per index (HIGH)~~ ✅ DONE

Implemented: `OnceLock<Arc<CudaContext>>` in both `CagraIndex` and
`GpuBruteForceIndex`, created lazily on first search and reused.
`bind_to_thread()` called on each search to ensure correctness across
blocking-pool threads.

---

## Fix #5 — Increase actor channel size (LOW-MEDIUM)

**Problem:** `CHANNEL_SIZE = 10` is small for GPU workloads where each
operation has higher latency than CPU usearch. Under load the channel fills
up and HTTP handlers block on `tx.send()`, creating back-pressure that
limits throughput.

**Suggested approach:**

- Make `CHANNEL_SIZE` configurable via `BatchConfig` (or a new config
  field) and default to a larger value, e.g. 128 or 256.
- Expose via env var `VECTOR_STORE_CUVS_CHANNEL_SIZE`.

**Files to modify:**

- `crates/vector-store/src/index/cuvs.rs` — `fn new()`, `BatchConfig`
- `crates/vector-store/src/config_manager.rs` — new env var
- `crates/vector-store/src/lib.rs` — wire through to `new_index_factory_cuvs`

---

## Fix #6 — Pre-compute vector norms for brute-force (LOW)

**Problem:** The CPU brute-force fallback (used when dataset < 128 vectors)
recomputes `||d_i||²` for every stored vector on every search. The GPU
brute-force index (`GpuBruteForceIndex`) also computes norms on the CPU
after the SGEMM.

**Suggested approach:**

1. Maintain a parallel `Vec<f32>` (or `BTreeMap<PrimaryId, f32>`) of
   pre-computed norms, updated on `add()` / `remove()`.
2. In `search()`, use the cached norms instead of recomputing them.
3. For `GpuBruteForceIndex`, upload norms to GPU alongside the dataset so
   the distance derivation can also run on the GPU (or at least avoid the
   per-search CPU norm loop).

**Files to modify:**

- `crates/vector-store/src/index/cuvs.rs` — `BruteForceIndex`,
  `CagraIndex::search_brute_force`, `GpuBruteForceIndex`
