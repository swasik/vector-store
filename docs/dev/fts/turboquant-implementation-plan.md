# TurboQuant Implementation Plan for Vector Store

## References

- **TurboQuant paper**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (Google Research, ICLR 2026)
- **QJL paper**: [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) — Zandieh, Daliri, Han (AAAI 2025)
- **PolarQuant paper**: [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) — Han et al. (AISTATS 2026)
- **NumKong**: [github.com/ashvardanian/NumKong](https://github.com/ashvardanian/NumKong) — SIMD-accelerated mixed-precision numeric operations (Rust crate)
- **Google Research blog**: [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## 1. Overview

### What is TurboQuant?

TurboQuant is a **data-oblivious online vector quantization** algorithm that compresses
high-dimensional vectors to low bit-widths (e.g., 4 bits per coordinate) while achieving
**near-optimal distortion rates** — within a factor of ≈2.7 of the Shannon information-theoretic
lower bound.

The algorithm is a two-stage process (**TurboQuant_prod** variant, optimized for inner product):

1. **MSE-optimal scalar quantization** (b−1 bits): randomly rotate the input vector via a
   shared rotation matrix Π, inducing a concentrated Beta distribution on each coordinate.
   Apply precomputed Lloyd-Max scalar quantizers (codebook lookup) per coordinate independently.
2. **QJL residual correction** (1 bit): compute the quantization residual `r = x − x̃_mse`,
   apply a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform `sign(S · r/‖r‖)` to produce
   an unbiased inner product estimator. Store the residual norm `γ = ‖r‖`.

### Why TurboQuant for Vector Store?

| Benefit | Detail |
|---------|--------|
| **8× memory reduction** | 4 bits/dim vs 32 bits/dim (F32), enabling larger indexes in RAM |
| **Near-optimal quality** | MSE distortion ≈0.009 at 4-bit; provably close to information-theoretic limit |
| **Unbiased inner products** | TurboQuant_prod provides unbiased dot product estimates — critical for ranking |
| **Zero indexing overhead** | Data-oblivious: no k-means, no dataset-specific training, no codebook learning |
| **Online quantization** | Vectors can be quantized one-at-a-time as they arrive (ideal for both full scan and CDC ingestion) |
| **SIMD-friendly** | Per-coordinate codebook lookup + dot product — maps directly to SIMD via NumKong |

### Scope

- **Initial target**: 4-bit TurboQuant_prod (`TQ4`) with QJL residual correction.
- **Distance metrics**: Cosine similarity and Dot Product (Euclidean prepared for later).
- **Both ingestion and query** vectors are quantized.
- **Lower bit-widths** (TQ2, TQ3) to be added later with minimal code changes.

## 2. Algorithm Details

### 2.1 TurboQuant_prod Quantization (Algorithm 2 from the paper)

**Setup** (once per index, shared across all vectors):

```
Input: dimension d, bit-width b=4
1. Generate random rotation matrix Π ∈ ℝ^{d×d}    // seeded PRNG, regenerated on startup
2. Instantiate TurboQuant_mse with bit-width b−1=3  // 8 centroids per coordinate
3. Construct codebook: find c₁,...,c₈ ∈ [−1,1] minimizing Lloyd-Max cost for Beta(d/2, 1/2)
4. Generate random projection matrix S ∈ ℝ^{d×d} with i.i.d. entries ~ N(0,1)  // for QJL
```

**Quantize** (per vector):

```
Input: x ∈ ℝ^d with ‖x‖ known
1. Normalize: x̂ = x / ‖x‖                        // project onto unit sphere
2. Rotate: y = Π · x̂                              // y follows Beta distribution per coord
3. For each j ∈ [d]:
     idx_j = argmin_k |y_j − c_k|                  // nearest centroid index (3 bits)
4. Dequantize MSE part: ỹ_j = c[idx_j] for each j
5. Residual: r = y − ỹ                             // in rotated space
6. γ = ‖r‖₂                                        // residual norm (1 float)
7. QJL signs: qjl = sign(S · (r/γ))                // d sign bits (1 bit each)
8. Store: (idx[d] @ 3 bits, qjl[d] @ 1 bit, γ @ f32, ‖x‖ @ f32)
```

**Storage per vector**: `d × 3/8 bytes` (MSE indices) + `d/8 bytes` (QJL signs) + `8 bytes`
(γ + norm) = **d/2 + 8 bytes**.
For d=1536: 776 bytes vs 6144 bytes (F32) → **7.9× compression**.

### 2.2 Inner Product Estimation (search-time)

For query q and stored TQ4 vector (idx, qjl, γ, norm_x):

```
1. Pre-rotate query: q' = Π · q                    // once per search, O(d²) or O(d log d)
2. Pre-project query: q_s = S · q'                  // once per search, for QJL term
                                                     // NOTE: S operates on the ROTATED query q',
                                                     // not the original q, because QJL sign bits
                                                     // were computed from the rotated residual r = y − ỹ.

For each candidate vector (idx, qjl, γ, norm_x):
3. MSE term: mse_ip = Σ_j c[idx_j] · q'_j          // codebook-lookup dot product
4. QJL term: qjl_ip = (π/2)/d · γ · Σ_j qjl_j · q_s_j   // sign×float dot product
5. Inner product estimate: ⟨q, x⟩ ≈ norm_x · (mse_ip + qjl_ip)
                                                     // norm_x rescales from unit-sphere estimate;
                                                     // ‖q‖ is already baked into mse_ip and qjl_ip
                                                     // because q' = Π·q (not Π·q̂) — the raw query
                                                     // is rotated without prior normalization.
```

**Key optimization**: The query is rotated and projected once (steps 1-2). Per-candidate cost
is just two dot products: one codebook-lookup dot product (step 3) and one sign-bit dot
product (step 4). Both are highly SIMD-friendly via NumKong.

### 2.3 Precomputed Lloyd-Max Codebooks

The codebook for the Beta(d/2, 1/2) distribution is computed offline by solving the 1D k-means
problem via the Lloyd-Max iterative algorithm. For moderate-to-high dimensions (d ≥ 64), the
distribution converges to N(0, 1/d), and the codebooks stabilize.

**3-bit codebook** (8 centroids, for the MSE stage of 4-bit TurboQuant_prod):
Centroids for N(0, 1/d): `{±0.1528/√d, ±0.4528/√d, ±0.7914/√d, ±1.224/√d}` (approximate).
Exact values precomputed and stored as compile-time constants for a range of dimensions.

**Boundaries**: midpoints between consecutive centroids (7 boundaries for 8 centroids).
Quantization is a binary search or branchless comparison against boundaries.

### 2.4 Structured Rotation (Performance Optimization)

A dense random rotation Π·x costs O(d²). For high dimensions this dominates. We use a
**Randomized Hadamard Transform (RHT)** instead:

```
Π · x = H_d · D · x
```

where H_d is the Walsh-Hadamard matrix (O(d log d) butterfly) and D is a diagonal matrix of
random ±1 signs (generated from a seed). The RHT preserves the distributional properties
needed by TurboQuant (coordinates of H·D·x follow the same Beta distribution as a fully
random rotation for unit-norm inputs).

**Cost**: O(d log d) vs O(d²) — a 100× speedup for d=1536.

**Dimension padding**: The Walsh-Hadamard transform requires power-of-2 dimensions. When d
is not a power of 2 (e.g., d=1536), the input is zero-padded to the next power of 2
(d_pad=2048) before the transform. After the transform, only the first d coordinates are
retained for quantization. The extra (d_pad − d) coordinates are discarded. This is
equivalent to applying a d×d sub-matrix of H_{d_pad}, which preserves the key property
that coordinates of the output follow an approximately Beta(d/2, 1/2) distribution for
unit-norm inputs. The padding is a temporary working buffer — it does not affect packed
storage size (still d/2 + 8 bytes) or compression ratio. The QJL matrix S operates on the
original d dimensions (not d_pad) since it receives the already-truncated rotated vector.

**Codebook scaling note**: After the padded Hadamard transform, each output coordinate has
variance ≈ 1/d_pad (not 1/d). For d=768 (d_pad=1024) this is a ≈15% standard-deviation
mismatch when the codebook uses `inv_sqrt_d`. In practice this is acceptable because
Lloyd-Max quantizers are robust to small distribution shifts — the centroid boundaries
remain effective over a range of variances. For maximum precision, use `inv_sqrt_d_pad`
instead; this is a one-line change in `Tq4Quantizer::new()` and can be validated
empirically during recall testing.

NumKong does not provide Hadamard transforms directly, but the butterfly structure is trivially
implementable with SIMD and should be a standalone kernel in the TQ module. NumKong is used
for the subsequent dot products, reductions, and scalar ops.

### 2.5 QJL Projection (Performance Optimization)

The QJL matrix S ∈ ℝ^{d×d} with i.i.d. N(0,1) entries requires O(d²) storage and O(d²)
for S·x. For the inner product estimator, we need both `sign(S · r)` (at quantization time)
and `S · q'` (at query time, where q' is the rotated query).

**Optimization**: Use a **structured random projection** (e.g., SRHT = S·H·D) which costs
O(d log d) per multiply and needs only O(d) seed storage, while preserving QJL's theoretical
guarantees. Alternatively, use a block-diagonal random matrix with block size matching SIMD
width.

For the initial implementation, use a **pseudo-random dense matrix** generated from a seeded
PRNG at startup. Optimize to structured form in a follow-up.

## 3. Architecture

### 3.1 Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  crates/vector-store/src/                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  turbo_quant/                        (NEW MODULE)                  │ │
│  │  ├── mod.rs          Module root, TurboQuantConfig, TurboQuantIndex│ │
│  │  ├── codebook.rs     Precomputed Lloyd-Max centroids + boundaries  │ │
│  │  ├── rotation.rs     Seeded RHT (Hadamard + diagonal signs)       │ │
│  │  ├── qjl.rs          QJL projection, sign quantization, estimator │ │
│  │  ├── quantize.rs     TQ4 encode/decode, packed storage format     │ │
│  │  └── distance.rs     TQ4 inner product / cosine distance compute  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │  lib.rs       │  │  index/      │  │  httproutes.rs               │   │
│  │  Quantization │  │  usearch.rs  │  │  Search/Add endpoints        │   │
│  │  enum + TQ4   │  │  TQ4 add/    │  │  (unchanged API, new quant   │   │
│  │               │  │  search path │  │   type accepted)             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘   │
│                            │                                             │
│                    ┌───────┴───────┐                                     │
│                    │  USearch 2.22 │  (UNMODIFIED)                       │
│                    │  HNSW index   │                                     │
│                    │  I8 ScalarKind│  ← TQ4 packed codes stored as      │
│                    │  custom metric│    opaque byte arrays; custom      │
│                    │  via          │    metric computes TQ4-to-TQ4      │
│                    │  change_      │    distance directly from packed   │
│                    │  metric()     │    representations                  │
│                    └───────────────┘                                     │
│                                                                          │
│              ┌──────────────────────────────┐                            │
│              │  NumKong (Rust crate)         │  (NEW DEPENDENCY)         │
│              │  - dot products (f32, i8)     │                           │
│              │  - vector reductions          │                           │
│              │  - mixed-precision arithmetic │                           │
│              └──────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow: Ingestion

```
f32 vector from ScyllaDB (full scan or CDC)
  │
  ▼
normalize to unit norm, store ‖x‖
  │
  ▼
RHT rotation: y = H·D·x               ← Hadamard + diagonal (O(d log d))
  │
  ▼
scalar quantize each y_j:              ← branchless boundary comparisons
  idx_j = codebook_encode(y_j)           3 bits per coordinate
  │
  ▼
dequantize MSE: ỹ_j = codebook[idx_j]
compute residual: r = y − ỹ
  │
  ▼
QJL: qjl = sign(S · (r/‖r‖))          ← matrix-vector multiply, sign extraction
store γ = ‖r‖                            1 bit per coordinate + 1 float
  │
  ▼
pack (idx, qjl, γ, ‖x‖) into TQ4 packed byte array
  │
  ▼
USearch.add(primary_id, tq4_packed)     ← HNSW graph built with TQ4-to-TQ4
                                           custom metric via change_metric()
```

### 3.3 Data Flow: Search

```
f32 query vector
  │
  ▼
pre-rotate: q' = H·D·q                 ← once per search (O(d log d))
pre-project: q_s = S·q'                ← once per search (O(d²) initially)
                                          S operates on rotated query (same space as residuals)
  │
  ▼
quantize query to TQ4 packed byte array (same pipeline as ingestion)
  │
  ▼
USearch.search(q_tq4, limit × oversample_factor)  ← HNSW traversal using
  │                                                    TQ4-to-TQ4 custom metric
  ▼                                     
candidate set (primary_ids + approximate TQ4-to-TQ4 distances)
  │
  ▼
rerank candidates using precise asymmetric TQ4 inner product estimator:
  for each candidate primary_id:
    tq4_bytes = USearch.get(primary_id)            ← retrieve stored TQ4 codes
    decode (idx, qjl, γ, norm_x) from tq4_bytes
    mse_ip = Σ_j codebook[idx_j] · q'_j           ← NumKong dot product
    qjl_ip = (π/2)/d · γ · Σ_j qjl_j · q_s_j     ← NumKong sign·float dot product (q_s = S·q')
    ip = norm_x · (mse_ip + qjl_ip)
  │
  ▼
sort by precise TQ4 distance, return top limit results
```

**Two-tier distance design**:
- **HNSW traversal** (TQ4-to-TQ4 symmetric metric): Fast approximate distance using
  codebook lookup table + Hamming-based QJL correction. O(d) per comparison.
- **Final reranking** (f32 query vs TQ4 asymmetric): Precise unbiased inner product
  estimator using the pre-rotated/pre-projected f32 query against stored TQ4 codes
  retrieved via `USearch.get()`. This is the original TurboQuant estimator.

### 3.4 USearch Integration Strategy

**Key principle**: USearch stores TQ4-packed byte codes directly and uses a custom distance
function registered via `Index::change_metric()`. No I8 proxy, no side buffer.

**True 8× compression**: Each vector stored as `d/2 + 8` bytes — 3-bit MSE indices
(`3d/8` bytes) + 1-bit QJL signs (`d/8` bytes) + gamma (`4` bytes) + norm (`4` bytes).
For d=768 this is 392 bytes vs 3072 bytes for F32.

**USearch configuration for TQ4**:
- `ScalarKind::I8` with `dimensions = d/2 + 8` — each "dimension" corresponds to one byte
  of the TQ4 packed representation. USearch treats them as opaque bytes.
- `change_metric()` registers a custom closure that decodes two TQ4 packed buffers and
  computes their distance.

**TQ4-to-TQ4 symmetric distance** (used by HNSW for both graph construction and traversal):

The custom metric computes an approximate inner product from two TQ4 packed representations:

```
⟨a, b⟩ ≈ norm_a · norm_b · (mse_term + qjl_term)

mse_term = (1/d) · Σ_j  c_{a_j} · c_{b_j}
         = (1/d) · Σ_j  CROSS_TABLE[a_j][b_j]      ← precomputed 8×8 lookup table

qjl_term = gamma_a · gamma_b · cos(π · hamming(z_a, z_b) / d)
         ← popcount on XOR of packed QJL sign bits, O(d/8)
```

The MSE term uses a precomputed 8×8 cross-product table `CROSS_TABLE[i][j] = c_i · c_j`
(64 f32 values, fits in a cache line). The QJL term uses Hamming distance between packed
sign bits (a fast popcount operation) combined with the sign random projection identity:
`E[sign(Sx)·sign(Sy)] = 1 − (2/π)·arccos(⟨x,y⟩/‖x‖‖y‖)`.

**Note on cross terms**: The full expansion of `⟨y_a, y_b⟩` includes cross terms
`⟨ỹ_a, r_b⟩` and `⟨r_a, ỹ_b⟩`. Their magnitude is bounded by `|⟨ỹ_a, r_b⟩| ≤ γ_b`
and `|⟨r_a, ỹ_b⟩| ≤ γ_a` (by Cauchy-Schwarz, since `‖ỹ‖ ≤ 1` and `‖r‖ = γ`). For
3-bit Lloyd-Max quantization at moderate-to-high dimensions (d ≥ 256), the expected
residual norm is `E[γ] ≈ √(MSE · d) ≈ √(0.009 · d/d) ≈ 0.095` (TurboQuant paper,
Table 1, b=3). Since the MSE term itself has magnitude ≈ 1/d per coordinate (summing
to ≈ 1.0 total), the omitted cross terms contribute at most `2 × 0.095 / 1.0 ≈ 19%`
relative error to the symmetric distance. This approximation error is compensated by:
(1) the QJL correction term which captures part of the residual structure, and
(2) the oversample + asymmetric reranking strategy that recovers exact TurboQuant_prod
estimates for the final ranking.

**Asymmetric reranking** (f32 query vs stored TQ4, post-HNSW):
After HNSW returns oversampled candidates, retrieve TQ4 codes via `USearch::get()` and
compute the full unbiased TurboQuant inner product estimator using the pre-rotated and
pre-projected f32 query. This provides maximum precision for the final ranking.

The oversample factor (e.g., 2×−4×) ensures high recall despite the approximate TQ4-to-TQ4
metric in graph traversal. This two-phase approach (approximate retrieval + precise reranking)
is standard in production vector search systems (FAISS IVF+PQ, ScaNN).

**Ingestion path neutrality**: In the existing architecture, both the initial full scan
(token-range parallel read of the ScyllaDB table) and CDC (ongoing change stream) send
`DbEmbedding` values through the same `mpsc` channel to `monitor_items::add()`, which
calls `index.add_vector()`. TQ4 quantization is applied inside the index's `add()` method,
meaning both paths are handled identically with zero special-casing. The rotation matrix
and QJL projection are initialized at index creation time and remain constant regardless
of whether vectors arrive via full scan or CDC.

## 4. Implementation Phases

### Phase 1: Core TurboQuant Module

**Goal**: Standalone TQ4 encode/decode with correctness tests.

#### 4.1.1 `turbo_quant/codebook.rs`

Precomputed Lloyd-Max codebooks for the Beta(d/2, 1/2) distribution.

```rust
/// Precomputed Lloyd-Max centroids for b-bit scalar quantization
/// of coordinates following Beta(d/2, 1/2) distribution.
///
/// For moderate d (≥64), the distribution approximates N(0, 1/d).
/// Centroids are stored as multiples of 1/√d for numerical stability.
///
/// Reference: TurboQuant paper, Section 3.1, Equation (4).

/// 3-bit codebook: 8 centroids (used as MSE stage of 4-bit TurboQuant_prod)
pub const CENTROIDS_3BIT: [f32; 8] = [
    -1.2240, -0.7914, -0.4528, -0.1528,
     0.1528,  0.4528,  0.7914,  1.2240,
];
// These are c_k / √d — multiply by 1/√d to get the actual centroid values.

/// Boundaries between centroids (midpoints). 7 boundaries for 8 centroids.
pub const BOUNDARIES_3BIT: [f32; 7] = [
    -1.0077, -0.6221, -0.3028,
     0.0,
     0.3028,  0.6221,  1.0077,
];

/// Encode a single scalar coordinate to a 3-bit centroid index.
/// Uses branchless binary search against precomputed boundaries.
pub fn encode_scalar_3bit(value: f32, inv_sqrt_d: f32) -> u8;

/// Decode a 3-bit centroid index back to the centroid value.
pub fn decode_scalar_3bit(index: u8, inv_sqrt_d: f32) -> f32;

/// Batch-encode d coordinates. Returns packed 3-bit indices.
/// Output: d*3/8 bytes (with padding to byte boundary).
pub fn encode_vector_3bit(rotated: &[f32], inv_sqrt_d: f32) -> Vec<u8>;

/// Batch-decode packed 3-bit indices to f32 centroid values.
pub fn decode_vector_3bit(packed: &[u8], d: usize, inv_sqrt_d: f32) -> Vec<f32>;
```

**Compile-time tables**: Store codebooks for bit-widths 1, 2, 3, 4 and several dimension
ranges. The 3-bit codebook is the primary one for TQ4. The 1-bit, 2-bit, and 4-bit
codebooks are stored for future TQ2, TQ3, TQ5 support.

Codebook values are computed offline by solving the Lloyd-Max iteration for the Gaussian
N(0, 1) distribution (the high-d limit of Beta(d/2, 1/2)). For low dimensions (d < 64),
dimension-specific codebooks can be computed at index creation time as a fallback.

#### 4.1.2 `turbo_quant/rotation.rs`

Randomized Hadamard Transform for fast, reproducible random rotation.

```rust
/// Randomized Hadamard Transform (RHT): Π = H_d · D
///
/// H_d: Walsh-Hadamard matrix (recursive butterfly, O(d log d))
/// D: diagonal of random ±1 signs from seeded PRNG
///
/// The transform preserves norms and induces Beta-distributed coordinates
/// on unit-norm inputs, as required by TurboQuant.

pub struct RotationMatrix {
    signs: Vec<f32>,    // diagonal D: +1.0 or -1.0 per coordinate, length = d_pad
    dimension: usize,   // original dimension d
    padded_dim: usize,  // next power of 2 ≥ dimension (e.g., 2048 for d=1536)
    // No dense matrix stored — Hadamard is computed via butterfly
}

impl RotationMatrix {
    /// Create from a deterministic seed. Same seed = same rotation.
    pub fn from_seed(dimension: usize, seed: u64) -> Self;

    /// Forward transform: y = H · D · x, truncated to first `dimension` coords.
    /// Internally zero-pads x to `padded_dim`, applies butterfly, returns first d.
    /// Output length = dimension (not padded_dim). O(d_pad log d_pad).
    pub fn forward(&self, x: &mut [f32]);

    /// Inverse transform: x = D · H · y (in-place, O(d_pad log d_pad))
    /// Internally zero-pads y to `padded_dim`, applies H then D, returns first d.
    /// (H is symmetric and orthogonal, H^{-1} = H; D^{-1} = D)
    pub fn inverse(&self, y: &mut [f32]);
}

/// In-place Walsh-Hadamard transform. O(d log d) butterfly operations.
/// Dimension must be a power of 2 (pad with zeros if needed).
fn hadamard_transform(data: &mut [f32]);
```

**Normalization convention**: The `hadamard_transform()` function computes the
**unnormalized** Walsh-Hadamard butterfly (entries ±1, no division). This means
`‖H_raw · x‖ = √d_pad · ‖x‖`. To obtain an orthonormal rotation that preserves norms
(required by TurboQuant), `forward()` applies a single post-scaling multiply by
`1/√d_pad` after the butterfly: `y = (1/√d_pad) · H · D · x_padded`. The inverse
`inverse()` pre-scales by `1/√d_pad` before the butterfly. This is one scalar-×-vector
multiply per transform — negligible compared to the O(d_pad log d_pad) butterfly. The
diagonal D entries remain ±1.0 (not ±1/√d_pad) to keep the sign-flip as a branchless
operation.

**Performance note**: The Hadamard butterfly involves `d_pad log₂(d_pad)` additions/subtractions
plus one element-wise multiply by `1/√d_pad` for normalization. For d=1536 (not a power of
2), the internal working buffer is padded to d_pad=2048 with zeros. Only the first d=1536
coordinates of the output are returned; the padding is discarded. The `RotationMatrix` owns
a reusable `Vec<f32>` of length d_pad as an internal scratchpad to avoid per-call
allocation. NumKong's SIMD vector ops can accelerate the inner butterfly. Use SIMD for the
diagonal sign multiply (element-wise `x[i] *= signs[i]`) and the post-butterfly scaling
(element-wise `x[i] *= inv_sqrt_d_pad`) via NumKong's `EachScale` trait.

#### 4.1.3 `turbo_quant/qjl.rs`

QJL (Quantized Johnson-Lindenstrauss) 1-bit projection.

```rust
/// QJL: 1-bit inner product quantizer based on the Johnson-Lindenstrauss lemma.
///
/// Q_qjl(x) = sign(S · x) where S ∈ ℝ^{d×d}, S_{ij} ~ N(0,1)
/// Q_qjl^{-1}(z) = (π/2)/d · S^T · z
///
/// Reference: QJL paper (arXiv:2406.03482), Definition 1.

pub struct QjlProjection {
    /// Projection matrix S, stored row-major.
    /// Generated from seed at startup. O(d²) storage.
    /// Future optimization: structured SRHT for O(d) storage + O(d log d) multiply.
    matrix: Vec<f32>,  // d × d, row-major
    dimension: usize,
}

impl QjlProjection {
    /// Create from deterministic seed.
    pub fn from_seed(dimension: usize, seed: u64) -> Self;

    /// Quantize: sign(S · x). Returns packed sign bits (d/8 bytes).
    pub fn quantize(&self, x: &[f32]) -> Vec<u8>;

    /// Project a vector through S: returns S · v (d-dimensional, not quantized).
    /// At query time, pass the **rotated** query q' = Π·q (not the original q)
    /// so that the projection matches the space of the QJL sign bits.
    pub fn project_query(&self, q: &[f32]) -> Vec<f32>;

    /// Compute QJL inner product term:
    ///   (π/2)/d · γ · Σ_j sign_j · projected_query_j
    ///
    /// `signs`: packed sign bits from quantize()
    /// `projected_query`: output of project_query()
    /// `gamma`: residual norm ‖r‖
    pub fn inner_product_term(
        &self,
        signs: &[u8],
        projected_query: &[f32],
        gamma: f32,
    ) -> f32;
}
```

**NumKong usage**:
- `project_query()`: matrix-vector multiply S·q → use NumKong's `Dot` trait per row
  (`f32::dot(&row, &q)`), or pack S into a `PackedMatrix` and use `dots_packed()` for
  batch matrix-vector multiply.
- `inner_product_term()`: sign-float dot product. Unpack signs to ±1.0 f32, then
  `f32::dot(&signs_f32, &projected_query)`. Alternative: map signs to i8 {-1,+1}, use
  `i8::dot()` for integer SIMD accumulation.

#### 4.1.4 `turbo_quant/quantize.rs`

Full TQ4 encode/decode pipeline orchestrating codebook + rotation + QJL.

**3-bit packing format**: MSE codebook indices use 3 bits per coordinate. Eight consecutive
3-bit indices are packed into 3 bytes as a big-endian bitstream:

```
Indices:  [i₀][i₁][i₂][i₃][i₄][i₅][i₆][i₇]   (each 3 bits, values 0–7)
Byte 0:   [i₀₂ i₀₁ i₀₀ | i₁₂ i₁₁ i₁₀ | i₂₂ i₂₁]   (MSB first)
Byte 1:   [i₂₀ | i₃₂ i₃₁ i₃₀ | i₄₂ i₄₁ i₄₀ | i₅₂]
Byte 2:   [i₅₁ i₅₀ | i₆₂ i₆₁ i₆₀ | i₇₂ i₇₁ i₇₀]
```

Extraction of index j: `bit_offset = j * 3`, `byte = packed[bit_offset / 8]`,
shift and mask across 1–2 bytes. Groups of 8 indices → 3 bytes enables batch
extraction aligned to byte boundaries. If d is not a multiple of 8, the final
group is zero-padded to the next byte boundary.

```rust
/// Packed TQ4 representation of a single vector.
pub struct Tq4CompressedVector {
    /// 3-bit codebook indices, packed as big-endian bitstream: ceil(d*3/8) bytes.
    /// See "3-bit packing format" above for the layout.
    pub mse_indices: Vec<u8>,
    /// QJL sign bits, packed: ceil(d/8) bytes. Bit j of byte j/8 = sign of coord j.
    pub qjl_signs: Vec<u8>,
    /// Residual L2 norm (γ)
    pub gamma: f32,
    /// Original vector L2 norm
    pub norm: f32,
}

impl Tq4CompressedVector {
    /// Total storage in bytes.
    pub fn storage_bytes(&self) -> usize;

    /// Pack into a contiguous byte array for USearch storage.
    /// Layout: [mse_indices | qjl_signs | gamma(4 bytes LE) | norm(4 bytes LE)]
    /// Total: d/2 + 8 bytes.
    pub fn pack(&self) -> Vec<u8>;

    /// Unpack from a contiguous byte array retrieved via USearch::get().
    pub fn unpack(bytes: &[u8], dimension: usize) -> Self;
}

/// TQ4 quantizer state, shared across all vectors in an index.
pub struct Tq4Quantizer {
    rotation: RotationMatrix,
    qjl: QjlProjection,
    dimension: usize,
    inv_sqrt_d: f32,
}

impl Tq4Quantizer {
    /// Initialize with dimension and a pair of seeds (rotation + QJL).
    pub fn new(dimension: usize, rotation_seed: u64, qjl_seed: u64) -> Self;

    /// Quantize a single f32 vector to TQ4 representation.
    pub fn quantize(&self, vector: &[f32]) -> Tq4CompressedVector;

    /// Dequantize TQ4 back to approximate f32 vector.
    /// Used for debugging, testing, and MSE measurement.
    pub fn dequantize_f32(&self, compressed: &Tq4CompressedVector) -> Vec<f32>;
}
```

#### 4.1.5 `turbo_quant/distance.rs`

TQ4 distance computation for both HNSW metric and asymmetric reranking.

```rust
/// Precomputed 8×8 cross-product table for TQ4-to-TQ4 MSE inner product.
/// CROSS_TABLE[i][j] = centroid_i * centroid_j * inv_sqrt_d² = centroid_i * centroid_j / d
/// Used by the custom USearch metric for fast symmetric distance.
pub fn cross_product_table_3bit(inv_sqrt_d: f32) -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            table[i][j] = CENTROIDS_3BIT[i] * inv_sqrt_d * CENTROIDS_3BIT[j] * inv_sqrt_d;
        }
    }
    table
}

/// TQ4-to-TQ4 symmetric distance for USearch custom metric.
///
/// Computes approximate inner product from two packed TQ4 representations:
///   ⟨a, b⟩ ≈ norm_a · norm_b · (mse_term + qjl_term)
///
/// MSE term: codebook cross-product lookup table, O(d) index lookups + accumulate
/// QJL term: Hamming distance (popcount XOR), O(d/8) + one cos() call
///
/// This function is called millions of times during HNSW traversal —
/// every instruction counts. Hot path for NumKong SIMD acceleration.
pub fn tq4_symmetric_distance(
    a: &[u8],        // packed TQ4 codes
    b: &[u8],        // packed TQ4 codes
    dim: usize,      // original vector dimension
    cross_table: &[[f32; 8]; 8],  // precomputed codebook cross products
) -> f32 {
    // Layout: [mse_indices(3d/8 bytes) | qjl_signs(d/8 bytes) | gamma(4) | norm(4)]
    let mse_len = (dim * 3 + 7) / 8;
    let qjl_len = (dim + 7) / 8;

    let a_mse = &a[..mse_len];
    let b_mse = &b[..mse_len];
    let a_qjl = &a[mse_len..mse_len + qjl_len];
    let b_qjl = &b[mse_len..mse_len + qjl_len];
    let a_gamma = f32::from_le_bytes(a[mse_len + qjl_len..][..4].try_into().unwrap());
    let b_gamma = f32::from_le_bytes(b[mse_len + qjl_len..][..4].try_into().unwrap());
    let a_norm = f32::from_le_bytes(a[mse_len + qjl_len + 4..][..4].try_into().unwrap());
    let b_norm = f32::from_le_bytes(b[mse_len + qjl_len + 4..][..4].try_into().unwrap());

    // MSE term: streaming accumulation over 3-bit index pairs.
    // Extract pairs of 3-bit indices, look up CROSS_TABLE[a_j][b_j], and
    // accumulate directly into a running f32 sum. No heap allocation.
    //
    // Process in groups of 8 index pairs (= 3 bytes per stream) for byte-aligned
    // extraction. Each group: unpack 8 indices from a_mse and b_mse, look up 8
    // cross-product values, accumulate into a local SIMD-width f32 accumulator.
    let mse_term = accumulate_cross_products(a_mse, b_mse, dim, cross_table);

    // QJL term: Hamming distance between sign bit vectors via NumKong.
    // u1x8::hamming() returns the number of differing bits (popcount of XOR).
    let a_qjl_bits: &[numkong::u1x8] = bytemuck::cast_slice(a_qjl);
    let b_qjl_bits: &[numkong::u1x8] = bytemuck::cast_slice(b_qjl);
    let hamming = numkong::u1x8::hamming(a_qjl_bits, b_qjl_bits)
        .expect("hamming distance computation");
    // hamming() returns normalized distance in [0,1]; convert to bit count
    let hamming_bits = (hamming * (dim as f64)) as f32;
    let qjl_term = a_gamma * b_gamma
        * f32::cos(std::f32::consts::PI * hamming_bits / dim as f32);

    a_norm * b_norm * (mse_term + qjl_term)
}

/// Streaming accumulation of 3-bit cross-product lookups. Zero allocation.
/// Processes 8 index pairs at a time (3 bytes each from a and b).
/// Returns the sum of CROSS_TABLE[a_j][b_j] for all j ∈ [0, dim).
fn accumulate_cross_products(
    a_mse: &[u8],
    b_mse: &[u8],
    dim: usize,
    cross_table: &[[f32; 8]; 8],
) -> f32;

/// Precomputed query state for efficient asymmetric TQ4 distance computation.
/// Computed once per search query, reused across all candidates during reranking.
pub struct Tq4QueryState {
    /// Rotated query: q' = H · D · q
    pub rotated_query: Vec<f32>,
    /// QJL-projected query: S · q' (S applied to the ROTATED query, not original)
    pub projected_query: Vec<f32>,
    /// Original query norm
    pub query_norm: f32,
}

impl Tq4Quantizer {
    /// Prepare query for TQ4 distance computations.
    /// O(d log d) for rotation + O(d²) for QJL projection.
    pub fn prepare_query(&self, query: &[f32]) -> Tq4QueryState;

    /// Compute inner product estimate between prepared query and a TQ4 vector.
    /// O(d) per candidate — two dot products.
    pub fn inner_product(
        &self,
        query_state: &Tq4QueryState,
        compressed: &Tq4CompressedVector,
    ) -> f32;

    /// Compute cosine similarity from TQ4 inner product.
    pub fn cosine_similarity(
        &self,
        query_state: &Tq4QueryState,
        compressed: &Tq4CompressedVector,
    ) -> f32;

    /// Batch-compute inner products for multiple candidates.
    /// Enables NumKong's many-to-many operations for throughput.
    pub fn batch_inner_products(
        &self,
        query_state: &Tq4QueryState,
        candidates: &[Tq4CompressedVector],
    ) -> Vec<f32>;
}
```

**NumKong usage for `inner_product()`**:

```rust
use numkong::Dot;

// Step 1: MSE dot product — codebook-lookup dot product
// Decode each 3-bit index → centroid value, then dot with rotated query
// This is: Σ_j codebook[idx_j] · rotated_query[j]
//
// Optimization: gather centroids into a temporary buffer, then use NumKong dot.
//   1. Gather centroids: centroid_values[j] = CENTROIDS_3BIT[idx_j] * inv_sqrt_d
//   2. f32::dot(&centroid_values, &rotated_query) → f64 result (widened accumulator)
let mse_ip = f32::dot(&centroid_values, &query_state.rotated_query)
    .expect("dot product") as f32;

// Step 2: QJL correction — sign-bit dot product
// Unpack qjl_signs to ±1.0 f32 vector, then dot with projected query.
// Alternative: map signs to i8 {-1, +1}, use i8::dot() for integer SIMD path.
let qjl_correction = f32::dot(&unpacked_signs_f32, &query_state.projected_query)
    .expect("dot product") as f32;
let qjl_ip = (std::f32::consts::FRAC_PI_2 / d as f32) * compressed.gamma * qjl_correction;

// Step 3: Full inner product
let ip = compressed.norm * (mse_ip + qjl_ip);
```

#### 4.1.6 `turbo_quant/mod.rs`

Module root. Exports public types and re-exports.

```rust
pub mod codebook;
pub mod distance;
pub mod qjl;
pub mod quantize;
pub mod rotation;

pub use quantize::{Tq4CompressedVector, Tq4Quantizer};
pub use distance::Tq4QueryState;
```

### Phase 2: NumKong Integration

**Goal**: Add NumKong as a Rust dependency and use it for all performance-critical vector ops.

#### 4.2.1 Cargo.toml Changes

```toml
# In crates/vector-store/Cargo.toml
[dependencies]
numkong = "7.4"  # SIMD-accelerated dot products, reductions, casts
```

NumKong provides via its Rust crate (trait-based API):
- `f32::dot(&a, &b)` / `f16::dot(&a, &b)` — dot products with wider accumulators (`Dot` trait)
- `i8::dot(&a, &b)` — integer dot products with i32 accumulation (`Dot` trait)
- `f32::angular(&a, &b)` — angular (cosine) distance (`Angular` trait)
- `u1x8::hamming(&a, &b)` — Hamming distance on packed bits (`Hamming` trait)
- `EachScale`, `EachFMA` — element-wise multiply and fused multiply-add
- `ReduceMoments` — sum and sum-of-squares reductions
- `configure_thread()` — must be called once per thread before using AMX operations
- Automatic SIMD dispatch (AVX-512, AVX2, NEON, SVE) at runtime

#### 4.2.2 Usage Points

| Operation | NumKong API | Where |
|-----------|------------|-------|
| Dot product (f32) | `f32::dot(&a, &b)` → f64 (`Dot` trait) | Asymmetric MSE term in TQ4 distance |
| Dot product (i8) | `i8::dot(&a, &b)` → i32 (`Dot` trait) | QJL sign·float optimization (signs as i8) |
| Hamming distance | `u1x8::hamming(&a, &b)` (`Hamming` trait) | TQ4-to-TQ4 QJL term (HNSW metric) |
| Element-wise multiply | `Tensor::try_blend_tensor()` or `EachScale` | Diagonal sign multiply in RHT |
| Moments reduction | `ReduceMoments::try_moments_all()` | Norm computation (sum-of-squares) |
| Angular distance | `f32::angular(&a, &b)` (`Angular` trait) | USearch metric replacement (optional) |

### Phase 3: USearch Integration

**Goal**: Wire TQ4 into the existing USearch-based index using `change_metric()` for true 8× compression.

#### 4.3.1 Quantization Enum Extension

In `lib.rs`:

```rust
pub enum Quantization {
    F32,
    F16,
    BF16,
    I8,
    B1,
    TQ4,  // NEW: 4-bit TurboQuant with QJL residual
}
```

The `From<Quantization> for ScalarKind` mapping:

```rust
Quantization::TQ4 => ScalarKind::I8,  // USearch stores TQ4 packed bytes (1 byte per "dimension")
// dimensions = d/2 + 8 (packed TQ4 size), NOT the original vector dimension
```

#### 4.3.2 TurboQuantIndex Wrapper

In `index/usearch.rs`, extend the existing `ThreadedUsearchIndex`:

```rust
struct ThreadedUsearchIndex {
    inner: usearch::Index,
    quantization: ScalarKind,
    space_type: SpaceType,
    threads: usize,
    // NEW: TQ4 state (None for non-TQ quantizations)
    tq4: Option<Tq4IndexState>,
}

struct Tq4IndexState {
    quantizer: Arc<Tq4Quantizer>,
    /// Original vector dimension (before packing)
    original_dimension: usize,
    /// Packed TQ4 size in bytes = ceil(3d/8) + ceil(d/8) + 8
    /// For d divisible by 8 (768, 1024, 1536, 3072) this equals d/2 + 8.
    /// For other d values it may be up to 1 byte larger.
    packed_dimension: usize,
    /// Oversample factor for HNSW retrieval before asymmetric reranking (e.g., 3.0)
    oversample_factor: f32,
}
// Note: No DashMap side buffer. TQ4 codes are stored directly in USearch
// and retrieved via USearch::get() for reranking.
```

#### 4.3.3 Index Initialization with Custom Metric

When creating a TQ4 index, register the custom TQ4-to-TQ4 distance function:

```rust
fn create_tq4_index(
    dimension: usize,
    quantizer: Arc<Tq4Quantizer>,
    space_type: SpaceType,
) -> usearch::Index {
    let packed_dim = (dimension * 3 + 7) / 8 + (dimension + 7) / 8 + 8;  // exact: ceil(3d/8) + ceil(d/8) + 8
    let options = IndexOptions {
        dimensions: packed_dim,
        metric: MetricKind::IP,   // placeholder, overridden by change_metric
        quantization: ScalarKind::I8,
        ..Default::default()
    };
    let mut index = usearch::Index::new(&options).unwrap();

    // Precompute 8×8 cross-product table: CROSS[i][j] = centroid_i * centroid_j / d
    let cross_table = codebook::cross_product_table_3bit(1.0 / (dimension as f32).sqrt());
    let dim = dimension;

    // Register custom TQ4-to-TQ4 metric — internally uses NumKong for:
    //   - accumulate_cross_products() for MSE codebook accumulation
    //   - u1x8::hamming() (Hamming trait) for QJL sign-bit Hamming distance
    //
    // NOTE: configure_thread() must be called on each USearch worker thread
    // before the first metric invocation to enable AMX (if available).
    // This can be done via a thread_local guard inside the closure.
    index.change_metric::<i8>(Box::new(move |a_ptr: *const i8, b_ptr: *const i8| {
        // Per-thread AMX initialization (idempotent, thread_local guard)
        thread_local! { static CONFIGURED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
        CONFIGURED.with(|c| if !c.get() { numkong::configure_thread(); c.set(true); });

        // Decode TQ4 packed layouts from raw pointers
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const u8, packed_dim) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const u8, packed_dim) };

        let ip = tq4_symmetric_distance(a, b, dim, &cross_table);

        // Convert inner product to distance (USearch minimizes distance)
        // For IP/Cosine: distance = 1.0 - ip
        Distance::from(1.0 - ip)
    })).unwrap();

    index
}
```

**Index reload**: After a restart, USearch indexes are restored via `Index::load()`, which
does **not** persist the custom metric closure. The `create_tq4_index` logic — specifically
the `change_metric()` call with the precomputed cross-product table — must be re-executed
on every index reload path (e.g., in the `ThreadedUsearchIndex::load()` or equivalent
reconstruction code). The `Tq4Quantizer` is also regenerated deterministically from the
same seeds, so the rotation and QJL matrices match the ones used during original ingestion.

Where `tq4_symmetric_distance()` computes the approximate inner product from two packed
TQ4 representations using the codebook cross-product table and Hamming-based QJL correction.

#### 4.3.4 Modified Add Path

```rust
fn add(&self, primary_id: PrimaryId, vector: &Vector) -> anyhow::Result<()> {
    match &self.tq4 {
        Some(tq4_state) => {
            // TQ4 path: quantize → pack → store in USearch with custom metric
            let compressed = tq4_state.quantizer.quantize(vector.as_slice());
            let packed = compressed.pack();  // → Vec<u8> of size d/2 + 8
            // Cast &[u8] to &[i8] for USearch I8 scalar kind
            let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed);
            Ok(self.inner.add(primary_id.into(), packed_i8)?)
        }
        None if self.quantization == ScalarKind::B1 => {
            let vector = f32_to_b1x8(vector.as_slice());
            Ok(self.inner.add(primary_id.into(), &vector)?)
        }
        None => {
            Ok(self.inner.add(primary_id.into(), vector.as_slice())?)
        }
    }
}
```

#### 4.3.5 Modified Search Path

```rust
fn search(&self, vector: &Vector, limit: Limit) -> anyhow::Result<SearchResults> {
    match &self.tq4 {
        Some(tq4_state) => {
            // Phase 1: HNSW retrieval with TQ4-to-TQ4 custom metric (oversampled)
            let oversample_limit = (limit.0.get() as f32 * tq4_state.oversample_factor) as usize;
            let compressed_query = tq4_state.quantizer.quantize(vector.as_slice());
            let packed_query = compressed_query.pack();
            let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
            let candidates = self.inner.search(packed_i8, oversample_limit)?;

            // Phase 2: Precise asymmetric reranking with f32 query vs stored TQ4
            let query_state = tq4_state.quantizer.prepare_query(vector.as_slice());
            let packed_dim = tq4_state.packed_dimension;
            // Reusable buffer for USearch get() — avoids per-candidate heap allocation
            let mut get_buf = vec![0i8; packed_dim];
            // Phase 2 collects (id, raw_ip, candidate_norm) for correct cosine conversion
            let mut results: Vec<(PrimaryId, f32, f32)> = candidates.keys
                .iter()
                .filter_map(|&id| {
                    get_buf.fill(0);
                    let found = self.inner.get(id, &mut get_buf).ok()?;
                    if found == 0 { return None; }
                    let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                    let compressed = Tq4CompressedVector::unpack(buf_u8, tq4_state.original_dimension);
                    let pid = PrimaryId::from(id);
                    let x_norm = compressed.norm;
                    let ip = tq4_state.quantizer.inner_product(&query_state, &compressed);
                    Some((pid, ip, x_norm))
                })
                .collect();

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit.0.get());

            Ok(results.into_iter().map(|(id, raw_ip, x_norm)| {
                // Convert raw inner product to Distance for the caller.
                // raw_ip ≈ ⟨q, x⟩ = ‖q‖·‖x‖·cos(θ) (norms already baked in).
                // The existing Distance::try_from() expects USearch-style distances:
                //   Cosine: distance = 1.0 - cos(θ) = 1.0 - raw_ip/(‖q‖·‖x‖)
                //   DotProduct: distance = -ip (USearch convention)
                let distance = match self.space_type {
                    SpaceType::Cosine => {
                        let denom = query_state.query_norm * x_norm;
                        let sim = if denom > 0.0 { raw_ip / denom } else { 0.0 };
                        1.0 - sim.clamp(-1.0, 1.0)
                    }
                    SpaceType::DotProduct => -raw_ip,
                    _ => unreachable!("TQ4 only supports Cosine and DotProduct"),
                };
                let dist = Distance::try_from((distance, self.space_type.into(), vector.dim()))
                    .expect("valid distance conversion");
                (id, dist)
            }).collect())
        }
        None => { /* existing search path */ }
    }
}
```

#### 4.3.7 Modified Filtered Search Path

The `filtered_search` follows the same TQ4 pattern as `search`: quantize the query,
oversample with the filter applied, then rerank asymmetrically.

```rust
fn filtered_search(
    &self,
    vector: &Vector,
    limit: Limit,
    filter: impl Fn(PrimaryId) -> bool,
) -> anyhow::Result<Box<dyn Iterator<Item = anyhow::Result<(PrimaryId, Distance)>> + '_>> {
    match &self.tq4 {
        Some(tq4_state) => {
            // Phase 1: Filtered HNSW retrieval with TQ4-to-TQ4 custom metric (oversampled)
            let oversample_limit = (limit.0.get() as f32 * tq4_state.oversample_factor) as usize;
            let compressed_query = tq4_state.quantizer.quantize(vector.as_slice());
            let packed_query = compressed_query.pack();
            let packed_i8 = bytemuck::cast_slice::<u8, i8>(&packed_query);
            let candidates = self.inner.filtered_search(
                packed_i8,
                oversample_limit,
                |row_id| filter(row_id.into()),
            )?;

            // Phase 2: Precise asymmetric reranking (same as unfiltered search)
            let query_state = tq4_state.quantizer.prepare_query(vector.as_slice());
            let packed_dim = tq4_state.packed_dimension;
            // Reusable buffer for USearch get() — avoids per-candidate heap allocation
            let mut get_buf = vec![0i8; packed_dim];
            // Phase 2 collects (id, raw_ip, x_norm) for correct cosine conversion
            let mut results: Vec<(PrimaryId, f32, f32)> = candidates.keys
                .iter()
                .filter_map(|&id| {
                    get_buf.fill(0);
                    let found = self.inner.get(id, &mut get_buf).ok()?;
                    if found == 0 { return None; }
                    let buf_u8 = bytemuck::cast_slice::<i8, u8>(&get_buf);
                    let compressed = Tq4CompressedVector::unpack(buf_u8, tq4_state.original_dimension);
                    let pid = PrimaryId::from(id);
                    let x_norm = compressed.norm;
                    let ip = tq4_state.quantizer.inner_product(&query_state, &compressed);
                    Some((pid, ip, x_norm))
                })
                .collect();

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit.0.get());

            // Convert to Distance (same logic as unfiltered search)
            Ok(Box::new(results.into_iter().map(move |(id, raw_ip, x_norm)| {
                let distance = match self.space_type {
                    SpaceType::Cosine => {
                        let denom = query_state.query_norm * x_norm;
                        let sim = if denom > 0.0 { raw_ip / denom } else { 0.0 };
                        1.0 - sim.clamp(-1.0, 1.0)
                    }
                    SpaceType::DotProduct => -raw_ip,
                    _ => unreachable!("TQ4 only supports Cosine and DotProduct"),
                };
                Distance::try_from((distance, self.space_type.into(), vector.dim()))
                    .map(|dist| (id, dist))
            })))
        }
        None if self.quantization == ScalarKind::B1 => {
            let vector_b1 = f32_to_b1x8(vector.as_slice());
            let matches = self.inner
                .filtered_search(&vector_b1, limit.0.get(), |row_id| filter(row_id.into()))?;
            Ok(Box::new(matches.keys.into_iter().zip(matches.distances)
                .map(move |(primary_id, distance)| {
                    Distance::try_from((distance, self.space_type.try_into()?, vector.dim()))
                        .map(|dist| (primary_id.into(), dist))
                })))
        }
        None => {
            let matches = self.inner
                .filtered_search(vector.as_slice(), limit.0.get(), |row_id| filter(row_id.into()))?;
            Ok(Box::new(matches.keys.into_iter().zip(matches.distances)
                .map(move |(primary_id, distance)| {
                    Distance::try_from((distance, self.space_type.try_into()?, vector.dim()))
                        .map(|dist| (primary_id.into(), dist))
                })))
        }
    }
}
```

#### 4.3.8 Modified Remove Path

```rust
fn remove(&self, primary_id: PrimaryId) -> anyhow::Result<()> {
    // No side buffer to clean up — USearch handles everything
    Ok(self.inner.remove(primary_id.into())?)
}
```

### Phase 4: API Surface

**Goal**: Expose TQ4 as a user-selectable quantization type.

#### 4.4.1 CQL Type Mapping

In the CQL schema / configuration:

```
quantization: 'TQ4'   -- 4-bit TurboQuant with QJL (new)
```

Maps to `Quantization::TQ4` in `lib.rs`. Compatible with `SpaceType::Cosine` and
`SpaceType::DotProduct`. Returns an error for `SpaceType::Euclidean` (unsupported initially)
and `SpaceType::Hamming` (incompatible).

#### 4.4.2 HTTP API

No new endpoints. The existing search/add endpoints work unchanged — the quantization
type is a per-index configuration. The `/info` endpoint should report TQ4 support.

#### 4.4.3 OpenAPI Spec

Add `TQ4` to the `Quantization` enum in `api/openapi.json` (auto-generated from code).

### Phase 5: Testing

#### 4.5.1 Unit Tests

| Test | Location | What |
|------|----------|------|
| Codebook symmetry | `codebook.rs` | Centroids symmetric around 0, boundaries correct |
| Encode/decode roundtrip | `codebook.rs` | `decode(encode(x)) ≈ nearest_centroid(x)` |
| RHT norm preservation | `rotation.rs` | `‖forward(x)‖ = ‖x‖` for random vectors (normalized Hadamard) |
| RHT invertibility | `rotation.rs` | `inverse(forward(x)) = x` within f32 tolerance |
| RHT distributional | `rotation.rs` | Coordinates of H·D·x follow Beta(d/2,1/2) (KS test) |
| QJL unbiasedness | `qjl.rs` | `E[⟨y, Q_qjl^{-1}(Q_qjl(x))⟩] ≈ ⟨y,x⟩` over many trials |
| QJL variance bound | `qjl.rs` | Variance ≤ π/(2d) · ‖y‖² |
| TQ4 MSE bound | `quantize.rs` | `E[‖x − dequant(quant(x))‖²] ≤ 0.009` for b=4 |
| TQ4 inner product unbiasedness | `distance.rs` | `E[ip_estimate] ≈ true_ip` |
| TQ4 inner product variance | `distance.rs` | Variance matches paper bounds |
| Packed storage size | `quantize.rs` | `Tq4CompressedVector` size = d/2 + 8 bytes |

#### 4.5.2 Recall & Quality Tests (standalone, no ScyllaDB)

These tests validate TQ4 search quality by directly exercising the USearch index with
synthetic and structured datasets. No ScyllaDB or CDC required — vectors are inserted
and searched via the `ThreadedUsearchIndex` API directly.

| Test | What | Pass criterion |
|------|------|----------------|
| Recall@10 random 10K | Insert 10K random unit-norm d=768 vectors, query with 100 random queries. Compute recall@10 against brute-force f32 exact search. | recall@10 ≥ 0.90 (*) |
| Recall@10 random 50K | Same as above with 50K vectors. | recall@10 ≥ 0.92 (*) |
| Recall@1 (top-1 accuracy) | Insert 10K vectors, query with 100 queries. Check if TQ4 top-1 result matches exact top-1. | accuracy ≥ 0.85 (*) |
| TQ4 vs I8 recall | Insert same 10K dataset into TQ4 and I8 indexes, compare recall@10. TQ4 should be competitive. | TQ4 recall ≥ I8 recall × 0.95 (*) |
| Recall vs oversample | Sweep oversample_factor ∈ {1, 2, 3, 4} on 10K vectors. Verify recall monotonically improves. | recall@10 increases with oversample |
| Cosine ordering | Insert 100 vectors, search with a known query. Verify TQ4 result ordering matches F32 ground truth for top-5. | top-5 order matches exactly |
| DotProduct ordering | Same as above with DotProduct metric. | top-5 order matches exactly |
| High-dimensional recall | Insert 5K vectors at d=1536, query with 50 queries. | recall@10 ≥ 0.88 (*) |
| Symmetric vs asymmetric | Compare TQ4-to-TQ4 symmetric metric distances against asymmetric reranked distances. Asymmetric should be closer to true f32 distances. | asymmetric MSE < symmetric MSE |

(*) **Preliminary thresholds**: these numeric targets are initial estimates based on the
paper's reported distortion rates and typical HNSW recall behavior. They will be calibrated
during implementation based on actual measurements with the chosen HNSW parameters
(connectivity, expansion). Tests should be written to report the measured recall value
regardless of the pass/fail threshold, enabling data-driven threshold adjustment.

**Recall computation helper** (in test utilities):

```rust
/// Compute recall@k: fraction of true top-k results present in the retrieved top-k.
fn recall_at_k(
    retrieved: &[(PrimaryId, Distance)],
    ground_truth: &[(PrimaryId, Distance)],
    k: usize,
) -> f32 {
    let truth_set: HashSet<_> = ground_truth.iter().take(k).map(|(id, _)| id).collect();
    let found = retrieved.iter().take(k).filter(|(id, _)| truth_set.contains(id)).count();
    found as f32 / k as f32
}

/// Brute-force exact search for ground truth computation.
fn exact_search_f32(
    vectors: &[(PrimaryId, Vec<f32>)],
    query: &[f32],
    k: usize,
    metric: SpaceType,
) -> Vec<(PrimaryId, f32)>;
```

#### 4.5.3 Integration Tests (via DbBasic mock, no ScyllaDB)

These tests use the existing `setup_store_with_quantization()` helper with the `DbBasic`
mock database — the same pattern as existing F32/I8/B1 integration tests. No ScyllaDB
instance is required.

| Test | What |
|------|------|
| Index create with TQ4 | Create index with `quantization: TQ4`, verify config roundtrip |
| Add + search TQ4 | Insert vectors via mock CDC, search via HTTP, verify results returned |
| TQ4 + Cosine metric | Verify cosine similarity values are in [0, 1] range |
| TQ4 + DotProduct metric | Verify dot product search returns expected ordering |
| TQ4 + filtered search | Filtered search with TQ4 returns correct subset |
| TQ4 remove | Insert, remove, verify not found in search |
| Euclidean rejection | Creating TQ4 index with Euclidean metric returns error |

#### 4.5.4 Benchmarks

Add to `crates/benchmark/`:

| Benchmark | Measure |
|-----------|---------|
| TQ4 encode throughput | Vectors/second for quantization |
| TQ4 decode throughput | Vectors/second for dequantization |
| TQ4 search latency | p50/p95/p99 search latency vs F32/I8/B1 |
| TQ4 memory usage | RSS per 1M vectors at various dimensions |
| RHT throughput | Hadamard transforms/second |
| TQ4 recall@k | Recall vs ground truth at k=1,10,100 |

### Phase 6: Configuration

#### 4.6.1 New Config Parameters

```rust
/// TQ4-specific configuration parameters.
/// All values are hardcoded constants for now. The struct exists to make
/// future configurability straightforward — add a CQL index option and
/// populate the field from `get_index_params()` when needed.
pub struct Tq4Config {
    /// Oversample factor for HNSW candidate retrieval before TQ4 reranking.
    /// Higher = better recall, slower search.
    pub oversample_factor: f32,

    /// Seed for rotation matrix generation.
    /// Same seed ensures identical rotation across restarts.
    pub rotation_seed: u64,

    /// Seed for QJL projection matrix.
    pub qjl_seed: u64,
}

impl Default for Tq4Config {
    fn default() -> Self {
        Self {
            oversample_factor: 3.0,
            rotation_seed: 42,
            qjl_seed: 137,
        }
    }
}
```

#### 4.6.2 Configuration Flow

All TQ4 parameters use hardcoded defaults via `Tq4Config::default()`. No new CQL index
options are introduced. The `Tq4Config` struct and its `Default` impl make future
configurability a one-line change: read the value from `get_index_params()` instead of
using the default.

The existing per-index parameter pipeline (for reference):

```
CQL: CREATE CUSTOM INDEX ... WITH OPTIONS = { 'connectivity': '32', ... }
  → ScyllaDB system_schema.indexes table (stored as map<text, text>)
    → monitor_indexes::get_indexes() → get_index_params()
      → IndexMetadata { connectivity, expansion_add, expansion_search, ... }
        → IndexConfiguration
          → ThreadedUsearchIndex::new()
```

When TQ4 is selected via `quantization: 'TQ4'`, the index initialization creates a
`Tq4Config::default()` and passes it to `Tq4IndexState`. No changes to `IndexMetadata`,
`IndexConfiguration`, or `get_index_params()` are required.

#### 4.6.3 Summary

| Parameter | Source | User-configurable? | Value |
|-----------|--------|-------------------|-------|
| `quantization: TQ4` | CQL index option | Yes | — |
| `oversample_factor` | Hardcoded constant | No (future: CQL option) | 3.0 |
| `rotation_seed` | Hardcoded constant | No | 42 |
| `qjl_seed` | Hardcoded constant | No | 137 |

## 5. Performance Considerations

### 5.1 Memory Budget

| Quantization | Bytes/dim | d=768 | d=1536 | d=3072 | Compression vs F32 |
|--------------|-----------|-------|--------|--------|--------------------|
| F32 | 4.0 | 3,072 | 6,144 | 12,288 | 1× |
| F16 | 2.0 | 1,536 | 3,072 | 6,144 | 2× |
| I8 | 1.0 | 768 | 1,536 | 3,072 | 4× |
| B1 | 0.125 | 96 | 192 | 384 | 32× |
| **TQ4** | **0.5 + 8/d** | **392** | **776** | **1,544** | **~8×** |

TQ4 stores `d/2 + 8` bytes per vector directly in USearch. No side buffer, no duplicate
storage. The `+8` constant accounts for gamma (4 bytes) and norm (4 bytes) packed alongside
the 3-bit MSE indices and 1-bit QJL signs.

USearch HNSW graph overhead (neighbor lists, metadata) is additional and common to all
quantization types — it scales with the number of vectors, not dimensions.

### 5.2 Compute Costs Per Search

| Operation | Cost | When |
|-----------|------|------|
| Query rotation (RHT) | O(d log d) | Once per search |
| Query QJL projection | O(d²) | Once per search |
| Query TQ4 quantization | O(d) | Once per search |
| HNSW traversal (TQ4-to-TQ4) | O(ef·log n · d) | Once per search |
| USearch get() per candidate | O(d) | Per candidate (× oversample·k) |
| Asymmetric rerank per candidate | O(d) | Per candidate (× oversample·k) |

The QJL projection is the most expensive per-query operation at O(d²). For d=1536, this
is ~2.4M multiply-adds. With NumKong's SIMD achieving ~10 GFLOP/s single-threaded, this
takes ~0.24ms — acceptable for most workloads. The structured SRHT optimization (future
work) reduces this to O(d log d).

### 5.3 SIMD Optimization via NumKong

All hot-path operations map to NumKong kernels:

- **TQ4-to-TQ4 codebook cross-product** (HNSW metric, #1 hot path): For each coordinate pair,
  look up `CROSS_TABLE[a_j][b_j]` and accumulate. The 3-bit index extraction and table lookup
  can use SIMD gather instructions. Streaming accumulation (no heap allocation).
- **TQ4-to-TQ4 Hamming** (HNSW metric): XOR packed QJL sign bytes, then popcount.
  `u1x8::hamming(&a, &b)` — NumKong's `Hamming` trait on packed bits.
- **Codebook dot product** (asymmetric MSE term): `f32::dot(&centroids, &rotated_query)` —
  NumKong's `Dot` trait with f64 widened accumulator. Hot path during reranking.
- **Sign-float dot product** (asymmetric QJL term): unpack d sign bits to f32 ±1 values, then
  `f32::dot()`. Alternative: pack as i8 {-1,+1}, use `i8::dot()` for integer SIMD path.
- **Hadamard butterfly**: d_pad log₂(d_pad) add/subtract pairs. SIMD-friendly (independent
  lanes within each butterfly stage). Custom kernel, not a direct NumKong API, but use
  NumKong `EachScale` or element-wise ops for the diagonal sign multiply.

## 6. Future Work

### 6.1 Lower Bit-Widths (TQ2, TQ3)

The architecture is parameterized by bit-width. To add TQ2 or TQ3:
1. Add corresponding codebook constants (1-bit or 2-bit Lloyd-Max centroids).
2. Add `Quantization::TQ2` / `Quantization::TQ3` enum variants.
3. The TQ encode/decode pipeline is identical — only the codebook changes.
4. For TQ2: 1 bit MSE + 1 bit QJL = 2 bits/dim → 16× compression over F32.
5. For TQ3: 2 bits MSE + 1 bit QJL = 3 bits/dim → ~10.7× compression.

### 6.2 Euclidean Distance Support

TurboQuant_mse (without QJL) naturally supports Euclidean distance because it minimizes
reconstruction MSE. To add Euclidean support:
1. Use TurboQuant_mse codebook (all b bits for MSE, no QJL).
2. Euclidean distance from dequantized vectors: `‖q − dequant(x)‖²`.
3. The reranking step computes `‖q − x̃‖²` from codebook values directly.

### 6.3 Structured QJL Projection

Replace dense S matrix with Subsampled Randomized Hadamard Transform (SRHT):
- Storage: O(d) seeds → O(d log d) multiply (vs O(d²) dense)
- Preserves QJL theoretical guarantees
- ~100× speedup for query projection at d=1536

### 6.4 PolarQuant Variant

PolarQuant (arXiv:2502.02617) uses polar coordinates instead of random rotation. It can
be added as an alternative to TurboQuant's rotation step, potentially with better constants
for specific use cases. The same codebook and QJL infrastructure applies.

### 6.5 Query-Query Quantized Distance

When both query and database vectors are TQ4-quantized, the inner product can be estimated
from the two compressed representations directly using quantized-quantized distance formulas.
This avoids dequantization entirely and enables pure integer arithmetic paths.
