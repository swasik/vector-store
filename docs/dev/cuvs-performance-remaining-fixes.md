# cuVS Performance: Remaining Optimisations

These items were identified as root causes for the cuVS throughput gap
versus usearch but have **not yet been implemented**. Fixes #1 (decouple
rebuild), #2 (cache CudaContext), #3 (stale reads), #4 (query batching),
and #5 (channel size) are already applied.

---

## ~~Fix #1 — Decouple CAGRA graph rebuild from the search path (CRITICAL)~~ ✅ DONE

Implemented: mutations no longer clear `cached_build` or trigger a rebuild.
Instead, a `DeltaIndex` tracks added/removed vectors since the last build.
Searches merge the (potentially stale) CAGRA graph results with a
brute-force scan of the delta, filtering out removed/replaced entries.
`rebuild()` (called from `Index::Flush`) builds a fresh CAGRA graph and
atomically swaps it in, clearing the delta.

---

## ~~Fix #2 — Cache CudaContext per index (HIGH)~~ ✅ DONE

Implemented: `OnceLock<Arc<CudaContext>>` in both `CagraIndex` and
`GpuBruteForceIndex`, created lazily on first search and reused.
`bind_to_thread()` called on each search to ensure correctness across
blocking-pool threads.

---

## Fix #5 — Increase actor channel size (LOW-MEDIUM) ✅ DONE

Channel size is now configurable via `BatchConfig::channel_size` (default 128,
up from the old hardcoded 10). Exposed as env var
`VECTOR_STORE_CUVS_CHANNEL_SIZE`.

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
