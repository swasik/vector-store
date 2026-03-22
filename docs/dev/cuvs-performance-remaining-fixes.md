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

## ~~Fix #6 — Pre-compute vector norms for brute-force (LOW)~~ ✅ DONE

Implemented: all index types (`BruteForceIndex`, `GpuBruteForceIndex`,
`CagraIndex`/`DeltaIndex`) now store precomputed squared norms (`||v||²`)
alongside vectors, updated on `add()`/`remove()`. Search paths use cached
norms instead of recomputing O(d) per stored vector per query. The GPU
brute-force post-SGEMM distance derivation also uses cached norms.
