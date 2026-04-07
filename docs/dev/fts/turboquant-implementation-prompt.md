# TurboQuant Implementation Prompt

## Task

Implement 4-bit TurboQuant (`TQ4`) vector quantization with QJL residual correction in the
ScyllaDB Vector Store Rust codebase, following the implementation plan at
`docs/dev/fts/turboquant-implementation-plan.md` exactly.

## Context

- **Codebase**: Rust, async (Tokio), USearch 2.22 for HNSW indexing
- **Plan**: `docs/dev/fts/turboquant-implementation-plan.md` — the authoritative spec
- **Coding conventions**: `docs/rust_instructions.md` and `CONTRIBUTING.md`
- **Key papers**: TurboQuant (arXiv:2504.19874), QJL (arXiv:2406.03482)
- **New dependency**: NumKong v7.4 (Rust crate, trait-based SIMD API)

## Implementation Order

Execute the 6 phases in strict dependency order. Each phase must compile and pass its own
tests before moving to the next. Do NOT skip ahead.

### Phase 1: Core TurboQuant Module (`crates/vector-store/src/turbo_quant/`)

Create these files in order:

1. **`codebook.rs`** — Compile-time Lloyd-Max centroids and boundaries for 3-bit quantization.
   `encode_scalar_3bit()`, `decode_scalar_3bit()`, `encode_vector_3bit()`, `decode_vector_3bit()`.
   Branchless boundary comparisons. Big-endian 3-bit packing (8 indices → 3 bytes).

2. **`rotation.rs`** — `RotationMatrix` with seeded RHT (Hadamard butterfly + diagonal ±1 signs).
   `from_seed()`, `forward()`, `inverse()`. In-place `hadamard_transform()` for power-of-2
   dimensions. Zero-pad to `d_pad` internally; return first `d` coordinates.
   **Critical**: `forward()` must post-scale by `1/√d_pad` after the butterfly to preserve norms.
   `inverse()` pre-scales by `1/√d_pad` before the butterfly. The diagonal D entries remain ±1.0.
   Own a reusable `Vec<f32>` scratchpad (length `d_pad`) to avoid per-call allocation.

3. **`qjl.rs`** — `QjlProjection` with dense seeded random matrix S ∈ ℝ^{d×d}.
   `from_seed()`, `quantize()` (sign bits), `project_query()` (S·v for the **rotated** query),
   `inner_product_term()` (sign·float dot product × π/2d × γ).

4. **`quantize.rs`** — `Tq4CompressedVector` (packed storage: MSE indices + QJL signs + γ + norm),
   `Tq4Quantizer` (orchestrates rotation → codebook → QJL). `quantize()`, `dequantize_f32()`,
   `pack()`, `unpack()`. Total storage: `d/2 + 8` bytes per vector.

5. **`distance.rs`** — `cross_product_table_3bit()`, `tq4_symmetric_distance()` (for HNSW metric),
   `accumulate_cross_products()` (streaming, zero-allocation). `Tq4QueryState`, `prepare_query()`,
   `inner_product()`, `cosine_similarity()`, `batch_inner_products()`.

6. **`mod.rs`** — Module root with re-exports.

### Phase 2: NumKong Integration

1. Add `numkong = "7.4"` to `crates/vector-store/Cargo.toml`.
2. Use NumKong traits throughout Phase 1 code:
   - `Dot` trait for `f32::dot()` and `i8::dot()` (dot products with widened accumulators)
   - `Hamming` trait for `u1x8::hamming()` (QJL sign-bit Hamming distance)
   - `EachScale` for element-wise multiply (diagonal D signs, post-butterfly scaling)
   - `configure_thread()` via `thread_local` guard in the USearch custom metric closure

### Phase 3: USearch Integration

1. **`lib.rs`** — Add `TQ4` variant to `Quantization` enum. Map `TQ4 → ScalarKind::I8` with
   `dimensions = packed_dim` (not the original vector dimension).

2. **`index/usearch.rs`** — Add `tq4: Option<Tq4IndexState>` to `ThreadedUsearchIndex`.
   - **`create_tq4_index()`**: Create USearch index with `ScalarKind::I8`, register custom
     TQ4-to-TQ4 metric via `change_metric::<i8>()`. Precompute cross-product table.
     Include `thread_local` guard for `configure_thread()` inside the closure.
   - **Index reload**: `change_metric()` must be re-executed after `Index::load()` because
     USearch does not persist custom metric closures. Ensure the reload path calls
     `create_tq4_index` logic (or at minimum re-registers the metric).
   - **`add()`**: TQ4 branch — quantize → pack → cast to `&[i8]` → `inner.add()`.
   - **`search()`**: TQ4 branch — quantize query → HNSW search (oversampled) → retrieve
     TQ4 codes via `inner.get()` → asymmetric rerank with `prepare_query()` +
     `inner_product()` → sort → truncate → convert to Distance. Pre-allocate `get_buf`
     outside the loop to avoid per-candidate heap allocation.
   - **`filtered_search()`**: Same pattern as `search()` with filter applied. Return type
     must be `Box<dyn Iterator<...> + '_>` (lifetime `'_`, not `'static`). Pre-allocate
     `get_buf` outside the loop.
   - **`remove()`**: Direct passthrough to `inner.remove()` — no side buffer.

3. **`engine.rs`** — When `IndexMetadata.quantization == TQ4`, create `Tq4Config::default()`,
   build `Tq4Quantizer`, pass to `ThreadedUsearchIndex`.

4. **`index/factory.rs`** — Handle `TQ4` in `IndexConfiguration`.

### Phase 4: API Surface

1. Add `TQ4` to CQL type mapping (string `"TQ4"` → `Quantization::TQ4`).
2. Reject `TQ4` with `SpaceType::Euclidean` or `SpaceType::Hamming` (return error).
3. Run `cargo openapi` to regenerate `api/openapi.json`.

### Phase 5: Testing (see Definition of Done below)

### Phase 6: Configuration

1. Add `Tq4Config` struct with hardcoded defaults (`oversample_factor: 3.0`, seeds: 42/137).
2. No new CQL parameters — `Tq4Config::default()` used everywhere.

## Definition of Done

The implementation is complete when ALL of the following are satisfied:

### 1. Compilation & Static Analysis

- [ ] `cargo build --all-targets` succeeds with zero warnings (`RUSTFLAGS=-Dwarnings`)
- [ ] `cargo fmt --all -- --check` passes (no formatting issues)
- [ ] `cargo clippy --all-targets` passes (no lint warnings)

### 2. Unit Tests (Phase 1 modules)

All unit tests below must pass. Write them as `#[cfg(test)] mod tests` in each file.

| Test | File | Assertion |
|------|------|-----------|
| Codebook symmetry | `codebook.rs` | Centroids symmetric around 0, boundaries correct |
| Encode/decode roundtrip | `codebook.rs` | `decode(encode(x)) == nearest_centroid(x)` for all inputs |
| 3-bit packing roundtrip | `codebook.rs` | Pack → unpack is identity for all valid index values |
| RHT norm preservation | `rotation.rs` | `‖forward(x)‖ ≈ ‖x‖` (tolerance: 1e-5) for 100 random vectors |
| RHT invertibility | `rotation.rs` | `inverse(forward(x)) ≈ x` (tolerance: 1e-5) for 100 random vectors |
| RHT distributional | `rotation.rs` | Coords of `forward(x̂)` pass Kolmogorov-Smirnov test for Beta(d/2, 1/2) |
| RHT determinism | `rotation.rs` | Same seed → identical output across two `RotationMatrix` instances |
| Non-power-of-2 dim | `rotation.rs` | `forward()` works correctly for d=768, d=1536 (not powers of 2) |
| QJL unbiasedness | `qjl.rs` | `mean(⟨y, Q_qjl⁻¹(Q_qjl(x))⟩) ≈ ⟨y,x⟩` over 1000 trials (tolerance: 5%) |
| QJL variance bound | `qjl.rs` | Empirical variance ≤ π/(2d) · ‖y‖² (with 10% slack) |
| QJL determinism | `qjl.rs` | Same seed → identical projection matrix |
| TQ4 MSE bound | `quantize.rs` | `mean(‖x − dequant(quant(x))‖²) ≤ 0.012` over 1000 unit vectors |
| TQ4 pack/unpack | `quantize.rs` | `unpack(pack(v)) == v` for random compressed vectors |
| TQ4 storage size | `quantize.rs` | `pack().len() == d/2 + 8` for d ∈ {768, 1024, 1536} |
| IP unbiasedness | `distance.rs` | `mean(ip_estimate) ≈ true_ip` (tolerance: 5%) over 1000 trials |
| IP variance | `distance.rs` | Empirical variance matches paper bounds |
| Cross-product table | `distance.rs` | `table[i][j] == centroids[i] * centroids[j] / d` for all i,j |
| Symmetric distance | `distance.rs` | `tq4_symmetric_distance(a, b) ≈ tq4_symmetric_distance(b, a)` |

### 3. Recall & Quality Tests (standalone, no ScyllaDB)

Write these as integration tests in `crates/vector-store/tests/` or as `#[test]` functions
in a dedicated test module. They exercise `ThreadedUsearchIndex` directly with synthetic data.

| Test | Setup | Pass Criterion |
|------|-------|----------------|
| recall@10 / 10K random | 10K random unit-norm d=768, 100 queries | recall@10 ≥ 0.90 |
| recall@10 / 50K random | 50K random unit-norm d=768, 100 queries | recall@10 ≥ 0.92 |
| recall@1 (top-1 accuracy) | 10K vectors, 100 queries | accuracy ≥ 0.85 |
| TQ4 vs I8 recall | Same 10K dataset, compare recall@10 | TQ4 ≥ I8 × 0.95 |
| Recall vs oversample | Sweep oversample ∈ {1,2,3,4} on 10K | recall@10 monotonically increases |
| Cosine ordering | 100 vectors, 1 query, compare top-5 order vs F32 | top-5 order matches |
| DotProduct ordering | 100 vectors, 1 query, compare top-5 order vs F32 | top-5 order matches |
| High-dim recall | 5K vectors d=1536, 50 queries | recall@10 ≥ 0.88 |
| Symmetric vs asymmetric | Compare MSE of both distances to true f32 | asymmetric MSE < symmetric MSE |

**Note**: Numeric thresholds are preliminary. Tests MUST print the measured recall value
regardless of pass/fail, enabling data-driven calibration. If a threshold needs adjustment
after measurement, document the rationale.

### 4. Integration Tests (via DbBasic mock, no ScyllaDB)

Use the existing `setup_store_with_quantization()` helper pattern from the F32/I8/B1 tests.

| Test | What |
|------|------|
| Index create with TQ4 | Create → verify config roundtrip |
| Add + search TQ4 | Insert via mock CDC → search via HTTP → verify results |
| TQ4 + Cosine metric | Cosine similarity values are in [0, 1] range |
| TQ4 + DotProduct metric | DotProduct search returns expected ordering |
| TQ4 + filtered search | Filtered search returns correct subset |
| TQ4 remove | Insert → remove → verify not found |
| Euclidean rejection | `TQ4` + `Euclidean` metric → error |

### 5. Performance Validation

| Check | Method | Criterion |
|-------|--------|-----------|
| Memory compression | Measure RSS with 100K vectors at d=768 | TQ4 RSS ≤ F32 RSS / 6 |
| Search latency | Benchmark p50 search latency on 100K d=768 | TQ4 p50 ≤ 2× F32 p50 |
| Encode throughput | Benchmark `quantize()` on 10K vectors d=768 | ≥ 50K vectors/sec |
| No regressions | Run existing F32/I8/B1 test suites | All pass unchanged |

### 6. Code Quality

- [ ] Every public type implements `Debug` (per Rust API guidelines)
- [ ] All public APIs have `///` rustdoc with meaningful descriptions
- [ ] Error paths use `Result<T, E>` with `anyhow` — no `unwrap()` on fallible ops
- [ ] No `unsafe` outside of the USearch FFI boundary (i.e., only in `from_raw_parts` for
      the custom metric pointer decoding)
- [ ] All `unsafe` blocks have `// SAFETY:` comments
- [ ] No per-candidate heap allocations in the search hot path (reuse buffers)
- [ ] `cargo test --verbose` passes all tests (unit + integration + recall)

## Constraints

- **No changes to existing quantization paths** (F32, F16, BF16, I8, B1 must be unaffected).
- **No changes to USearch source** — use only the public Rust API.
- **NumKong is a hard dependency** — no fallback. If NumKong traits don't compile, fix the
  usage, don't remove the dependency.
- **Single PR per phase** — each phase should be a self-contained, compilable, tested commit.
- **Seeds are deterministic** — same seeds = identical quantization across restarts. Verify
  this property in tests.
- **QJL operates on rotated vectors** — `S · q'` (where `q' = Π·q`), NOT `S · q`. This is
  because QJL sign bits were computed from the rotated residual. Violating this will produce
  silently wrong inner product estimates.
- **Hadamard normalization** — `forward()` post-scales by `1/√d_pad`. Without this, all
  norms are inflated by `√d_pad` and the codebook boundaries are wrong.
- **Index reload** — after `Index::load()`, `change_metric()` must be re-registered.

## Key Files to Modify

| File | Change |
|------|--------|
| `crates/vector-store/Cargo.toml` | Add `numkong = "7.4"`, add `bytemuck` if not present |
| `crates/vector-store/src/lib.rs` | Add `TQ4` to `Quantization` enum |
| `crates/vector-store/src/turbo_quant/` | NEW: entire module (6 files) |
| `crates/vector-store/src/index/usearch.rs` | Add `Tq4IndexState`, TQ4 branches in add/search/filtered_search |
| `crates/vector-store/src/index/factory.rs` | Handle `TQ4` in `IndexConfiguration` |
| `crates/vector-store/src/engine.rs` | Create `Tq4Quantizer` when TQ4 selected |
| `crates/vector-store/tests/` | New recall & integration tests |

## Reference: NumKong API Patterns

```rust
use numkong::{Dot, Hamming, EachScale, configure_thread};

// Dot product (f32 → f64 widened accumulator)
let ip: f64 = f32::dot(&a, &b).expect("dot product");

// Hamming distance on packed bits
let dist: f64 = numkong::u1x8::hamming(&a_bits, &b_bits).expect("hamming");

// Per-thread AMX init (call once per thread)
thread_local! { static INIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
INIT.with(|c| if !c.get() { configure_thread(); c.set(true); });
```
