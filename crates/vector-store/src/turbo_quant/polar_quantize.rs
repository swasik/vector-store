/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! PolarQuant vector quantization: RHT + polar transform + angle codebooks + QJL.
//!
//! PolarQuant encodes a d-dimensional vector via:
//! 1. Normalize to unit norm, store ‖x‖
//! 2. RHT random rotation (coordinates become approximately Gaussian)
//! 3. Recursive polar transform → (d−1) angles + radius=1
//! 4. Quantize each angle with a level-specific 3-bit Lloyd-Max codebook
//! 5. Compute residual in Cartesian rotated space, store QJL sign bits
//!
//! The angle distribution at each level is analytically known after Gaussian
//! preconditioning (Lemma 2 in arXiv:2502.02617):
//! - Level 1: uniform on [0, 2π)
//! - Level ℓ ≥ 2: f(ψ) ∝ sin^{2^{ℓ-1}−1}(2ψ) on [0, π/2]
//!
//! Higher levels concentrate around π/4 → fewer bits suffice, but we use
//! 3 bits uniformly across all levels for storage compatibility with TQ4.
//!
//! Reference: PolarQuant paper (arXiv:2502.02617), Han et al. (AISTATS 2026).

use std::f32::consts::PI;

use crate::turbo_quant::polar::{self, cartesian_to_polar, polar_to_cartesian};
use crate::turbo_quant::qjl::QjlProjection;
use crate::turbo_quant::rotation::RotationMatrix;
use numkong::{Dot, Hamming, u1x8};

// ---------------------------------------------------------------------------
// Level-specific angle codebooks (3-bit = 8 centroids per level)
// ---------------------------------------------------------------------------

/// Maximum number of recursion levels supported (d up to 2^MAX_LEVELS).
const MAX_LEVELS: usize = 16;

/// Level-specific 3-bit angle codebooks for PolarQuant.
///
/// Each level has 8 centroids and 7 boundaries, computed via Lloyd-Max
/// on the analytical angle distribution. The codebooks are computed once
/// at quantizer construction and reused for all vectors.
#[derive(Debug, Clone)]
pub struct PolarCodebooks {
    /// Centroids per level. `centroids[level-1][0..8]`.
    centroids: Vec<[f32; 8]>,
    /// Decision boundaries per level. `boundaries[level-1][0..7]`.
    boundaries: Vec<[f32; 7]>,
    /// Number of levels.
    num_levels: usize,
}

impl PolarCodebooks {
    /// Compute Lloyd-Max codebooks for all levels up to `num_levels`.
    ///
    /// Uses numerical sampling from the analytical distribution followed
    /// by iterative Lloyd-Max refinement (1D k-means).
    pub fn new(num_levels: usize) -> Self {
        assert!(num_levels >= 1 && num_levels <= MAX_LEVELS);
        let mut centroids = Vec::with_capacity(num_levels);
        let mut boundaries = Vec::with_capacity(num_levels);

        for level in 1..=num_levels {
            let (c, b) = compute_level_codebook(level);
            centroids.push(c);
            boundaries.push(b);
        }

        Self {
            centroids,
            boundaries,
            num_levels,
        }
    }

    /// Quantize an angle at the given level to a 3-bit index (0..7).
    #[inline]
    pub fn encode(&self, level: usize, angle: f32) -> u8 {
        debug_assert!(level >= 1 && level <= self.num_levels);
        let bounds = &self.boundaries[level - 1];
        let mut idx = 0u8;
        for &b in bounds {
            idx += (angle >= b) as u8;
        }
        idx
    }

    /// Dequantize a 3-bit index back to the centroid angle at the given level.
    #[inline]
    pub fn decode(&self, level: usize, index: u8) -> f32 {
        debug_assert!(level >= 1 && level <= self.num_levels);
        self.centroids[level - 1][index as usize]
    }

    /// Number of levels.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
}

/// Compute 8 Lloyd-Max centroids and 7 boundaries for a given level.
///
/// - Level 1: uniform distribution on [0, 2π)
/// - Level ≥ 2: f(ψ) ∝ sin^{2^{ℓ-1}-1}(2ψ) on [0, π/2]
fn compute_level_codebook(level: usize) -> ([f32; 8], [f32; 7]) {
    if level == 1 {
        // Uniform on [0, 2π): equally spaced centroids
        let step = 2.0 * PI / 8.0;
        let mut centroids = [0.0f32; 8];
        let mut boundaries = [0.0f32; 7];
        for k in 0..8 {
            centroids[k] = (k as f32 + 0.5) * step;
        }
        for k in 0..7 {
            boundaries[k] = (k as f32 + 1.0) * step;
        }
        (centroids, boundaries)
    } else {
        // Distribution: f(ψ) ∝ sin^{n}(2ψ) on [0, π/2] where n = 2^{ℓ-1} - 1
        // Highly concentrated around π/4 for large n.
        // Use Lloyd-Max: iteratively refine centroids and boundaries.
        let exponent = (1u64 << (level - 1)) - 1;
        lloyd_max_codebook(exponent, 50)
    }
}

/// Run Lloyd-Max algorithm for the distribution f(ψ) ∝ sin^n(2ψ) on [0, π/2].
///
/// `exponent`: the power n in sin^n(2ψ).
/// `iterations`: number of Lloyd-Max refinement iterations.
fn lloyd_max_codebook(exponent: u64, iterations: usize) -> ([f32; 8], [f32; 7]) {
    let half_pi = PI / 2.0;
    let n_samples = 10_000;

    // Sample from the distribution via inverse CDF (rejection sampling fallback)
    let samples = sample_angle_distribution(exponent, n_samples);

    // Initialize centroids: percentile-based
    let mut centroids = [0.0f32; 8];
    for k in 0..8 {
        let idx = ((k as f64 + 0.5) / 8.0 * n_samples as f64) as usize;
        centroids[k] = samples[idx.min(n_samples - 1)];
    }

    let mut boundaries = [0.0f32; 7];

    for _ in 0..iterations {
        // Update boundaries: midpoints of consecutive centroids
        for k in 0..7 {
            boundaries[k] = (centroids[k] + centroids[k + 1]) / 2.0;
        }

        // Update centroids: conditional means within each interval
        let lo_bounds = [0.0f32]
            .iter()
            .chain(boundaries.iter())
            .copied()
            .collect::<Vec<_>>();
        let hi_bounds = boundaries
            .iter()
            .chain([half_pi].iter())
            .copied()
            .collect::<Vec<_>>();

        for k in 0..8 {
            let lo = lo_bounds[k];
            let hi = hi_bounds[k];

            // Numerical conditional mean: ∫ ψ·f(ψ) dψ / ∫ f(ψ) dψ over [lo, hi]
            let (mean, weight) = numerical_conditional_mean(exponent, lo, hi);
            if weight > 0.0 {
                centroids[k] = mean;
            } else {
                centroids[k] = (lo + hi) / 2.0;
            }
        }
    }

    // Final boundary update
    for k in 0..7 {
        boundaries[k] = (centroids[k] + centroids[k + 1]) / 2.0;
    }

    (centroids, boundaries)
}

/// Sample sorted values from the distribution f(ψ) ∝ sin^n(2ψ) on [0, π/2].
///
/// Uses the inverse CDF computed via numerical integration.
fn sample_angle_distribution(exponent: u64, n_samples: usize) -> Vec<f32> {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let n_grid = 10_000usize;
    let step = half_pi / n_grid as f64;

    // Build CDF via trapezoidal integration
    let mut cdf = Vec::with_capacity(n_grid + 1);
    let mut cumulative = 0.0f64;
    cdf.push(0.0f64);

    for i in 1..=n_grid {
        let psi = i as f64 * step;
        let psi_prev = (i - 1) as f64 * step;
        let f_prev = angle_pdf_unnormalized(exponent, psi_prev as f32) as f64;
        let f_curr = angle_pdf_unnormalized(exponent, psi as f32) as f64;
        cumulative += 0.5 * (f_prev + f_curr) * step;
        cdf.push(cumulative);
    }

    // Normalize CDF
    let total = cumulative;
    for v in &mut cdf {
        *v /= total;
    }

    // Inverse CDF sampling: uniformly spaced quantiles
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let target = (i as f64 + 0.5) / n_samples as f64;
        // Binary search for the CDF value
        let mut lo = 0usize;
        let mut hi = n_grid;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let psi = lo as f64 * step;
        samples.push(psi as f32);
    }

    samples
}

/// Unnormalized PDF: sin^n(2ψ) for ψ ∈ [0, π/2].
#[inline]
fn angle_pdf_unnormalized(exponent: u64, psi: f32) -> f32 {
    let s = (2.0 * psi).sin();
    if exponent <= 30 {
        s.powi(exponent as i32)
    } else {
        // For large exponents, use exp(n · ln(sin(2ψ))) to avoid overflow
        if s > 0.0 {
            (exponent as f32 * s.ln()).exp()
        } else {
            0.0
        }
    }
}

/// Numerical conditional mean of ψ under f(ψ) ∝ sin^n(2ψ) over [lo, hi].
///
/// Returns (mean, weight) where mean = ∫ψ·f(ψ)dψ / weight, weight = ∫f(ψ)dψ.
fn numerical_conditional_mean(exponent: u64, lo: f32, hi: f32) -> (f32, f32) {
    let n_steps = 200;
    let step = (hi - lo) / n_steps as f32;
    let mut sum_pf = 0.0f64;
    let mut sum_f = 0.0f64;

    for i in 0..=n_steps {
        let psi = lo + i as f32 * step;
        let f = angle_pdf_unnormalized(exponent, psi) as f64;
        let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
        sum_pf += w * psi as f64 * f;
        sum_f += w * f;
    }

    if sum_f > 0.0 {
        ((sum_pf / sum_f) as f32, (sum_f * step as f64) as f32)
    } else {
        ((lo + hi) / 2.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// Packed angle representation (3-bit bitstream, same layout as TQ4 MSE indices)
// ---------------------------------------------------------------------------

/// Pack angle indices into a 3-bit big-endian bitstream.
///
/// Same bit layout as TQ4's `encode_vector_3bit` for storage compatibility.
pub fn pack_angle_indices(indices: &[u8]) -> Vec<u8> {
    let packed_len = (indices.len() * 3).div_ceil(8);
    let mut packed = vec![0u8; packed_len];

    for (j, &idx) in indices.iter().enumerate() {
        debug_assert!(idx < 8);
        let bit_offset = j * 3;
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        if bit_pos <= 5 {
            packed[byte_pos] |= idx << (5 - bit_pos);
        } else {
            let first_bits = 8 - bit_pos;
            packed[byte_pos] |= idx >> (3 - first_bits);
            if byte_pos + 1 < packed.len() {
                packed[byte_pos + 1] |= idx << (8 - (3 - first_bits));
            }
        }
    }
    packed
}

/// Extract a single 3-bit index from a packed big-endian bitstream.
#[inline]
pub fn extract_angle_index(packed: &[u8], j: usize) -> u8 {
    let bit_offset = j * 3;
    let byte_pos = bit_offset / 8;
    let bit_pos = bit_offset % 8;

    if bit_pos <= 5 {
        (packed[byte_pos] >> (5 - bit_pos)) & 0x07
    } else {
        let first_bits = 8 - bit_pos;
        let hi = packed[byte_pos] & ((1 << first_bits) - 1);
        let lo = packed[byte_pos + 1] >> (8 - (3 - first_bits));
        (hi << (3 - first_bits)) | lo
    }
}

// ---------------------------------------------------------------------------
// PolarQuant compressed vector
// ---------------------------------------------------------------------------

/// Packed PolarQuant representation of a single vector.
///
/// Storage layout matches TQ4 for index interoperability:
/// `[angle_indices | qjl_signs | gamma(4 bytes LE) | norm(4 bytes LE)]`
#[derive(Debug, Clone)]
pub struct PolarCompressedVector {
    /// 3-bit angle indices, packed as big-endian bitstream (d_pad−1 angles).
    pub angle_indices: Vec<u8>,
    /// QJL sign bits, packed: bit j of byte j/8 = sign of residual coord j.
    pub qjl_signs: Vec<u8>,
    /// Residual L2 norm (γ = ‖r‖) where r = y − ŷ in rotated Cartesian space.
    pub gamma: f32,
    /// Original vector L2 norm (‖x‖).
    pub norm: f32,
}

impl PolarCompressedVector {
    /// Total storage in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.angle_indices.len() + self.qjl_signs.len() + 8
    }

    /// Pack into a contiguous byte array.
    pub fn pack(&self) -> Vec<u8> {
        let total = self.storage_bytes();
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.angle_indices);
        buf.extend_from_slice(&self.qjl_signs);
        buf.extend_from_slice(&self.gamma.to_le_bytes());
        buf.extend_from_slice(&self.norm.to_le_bytes());
        buf
    }

    /// Unpack from a contiguous byte array.
    pub fn unpack(bytes: &[u8], dimension: usize) -> Self {
        let d_pad = dimension.next_power_of_two();
        let num_angles = d_pad - 1;
        let angle_len = (num_angles * 3).div_ceil(8);
        let qjl_len = d_pad.div_ceil(8);

        let angle_indices = bytes[..angle_len].to_vec();
        let qjl_signs = bytes[angle_len..angle_len + qjl_len].to_vec();
        let gamma = f32::from_le_bytes(
            bytes[angle_len + qjl_len..angle_len + qjl_len + 4]
                .try_into()
                .expect("gamma bytes"),
        );
        let norm = f32::from_le_bytes(
            bytes[angle_len + qjl_len + 4..angle_len + qjl_len + 8]
                .try_into()
                .expect("norm bytes"),
        );

        Self {
            angle_indices,
            qjl_signs,
            gamma,
            norm,
        }
    }

    /// Expected packed size in bytes for a given dimension.
    pub fn packed_size(dimension: usize) -> usize {
        let d_pad = dimension.next_power_of_two();
        let num_angles = d_pad - 1;
        let angle_len = (num_angles * 3).div_ceil(8);
        let qjl_len = d_pad.div_ceil(8);
        angle_len + qjl_len + 8
    }
}

// ---------------------------------------------------------------------------
// PolarQuant quantizer
// ---------------------------------------------------------------------------

/// PolarQuant quantizer state, shared across all vectors in an index.
pub struct PolarQuantizer {
    rotation: RotationMatrix,
    qjl: QjlProjection,
    codebooks: PolarCodebooks,
    /// Original vector dimension d.
    dimension: usize,
    /// Padded dimension d_pad = d.next_power_of_two().
    padded_dim: usize,
    /// Number of recursion levels = log2(d_pad).
    num_levels: usize,
}

impl PolarQuantizer {
    /// Create a new PolarQuant quantizer for the given dimension and seeds.
    pub fn new(dimension: usize, rotation_seed: u64, qjl_seed: u64) -> Self {
        let rotation = RotationMatrix::from_seed(dimension, rotation_seed);
        let d_pad = rotation.padded_dim();
        let qjl = QjlProjection::from_seed(d_pad, qjl_seed);
        let num_levels = d_pad.trailing_zeros() as usize;
        let codebooks = PolarCodebooks::new(num_levels);

        Self {
            rotation,
            qjl,
            codebooks,
            dimension,
            padded_dim: d_pad,
            num_levels,
        }
    }

    /// Original vector dimension d.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Padded dimension d_pad.
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Quantize a single f32 vector to PolarQuant representation.
    ///
    /// Steps:
    /// 1. Compute and store ‖x‖
    /// 2. Normalize: x̂ = x / ‖x‖
    /// 3. Rotate: y = Π · x̂  (RHT, full padded output)
    /// 4. Polar transform: y → (angles, radius≈1)
    /// 5. Quantize each angle with level-specific codebook
    /// 6. Dequantize angles → ŷ (approximate Cartesian in rotated space)
    /// 7. Residual: r = y − ŷ, γ = ‖r‖
    /// 8. QJL: signs = sign(S · r/γ)
    pub fn quantize(&self, vector: &[f32]) -> PolarCompressedVector {
        let d_pad = self.padded_dim;

        // Step 1: Original norm
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

        if norm == 0.0 {
            let num_angles = d_pad - 1;
            return PolarCompressedVector {
                angle_indices: vec![0u8; (num_angles * 3).div_ceil(8)],
                qjl_signs: vec![0u8; d_pad.div_ceil(8)],
                gamma: 0.0,
                norm: 0.0,
            };
        }

        // Step 2: Normalize
        let inv_norm = 1.0 / norm;
        let normalized: Vec<f32> = vector.iter().map(|v| v * inv_norm).collect();

        // Step 3: Rotate via RHT (full padded output)
        let mut rotated = vec![0.0f32; d_pad];
        self.rotation.forward_padded(&normalized, &mut rotated);

        // Step 4: Polar transform
        let (angles, _radius) = cartesian_to_polar(&rotated);

        // Step 5: Quantize angles with level-specific codebooks
        let num_angles = d_pad - 1;
        let mut angle_idx_vec = Vec::with_capacity(num_angles);
        for level in 1..=self.num_levels {
            let off = polar::level_offset(d_pad, level);
            let cnt = polar::level_count(d_pad, level);
            for j in 0..cnt {
                let idx = self.codebooks.encode(level, angles[off + j]);
                angle_idx_vec.push(idx);
            }
        }
        let packed_angles = pack_angle_indices(&angle_idx_vec);

        // Step 6: Dequantize angles → approximate Cartesian
        let mut deq_angles = Vec::with_capacity(num_angles);
        for level in 1..=self.num_levels {
            let off = polar::level_offset(d_pad, level);
            let cnt = polar::level_count(d_pad, level);
            for j in 0..cnt {
                let idx = self.codebooks.encode(level, angles[off + j]);
                deq_angles.push(self.codebooks.decode(level, idx));
            }
        }
        let approx_rotated = polar_to_cartesian(&deq_angles, 1.0, d_pad);

        // Step 7: Residual
        let residual: Vec<f32> = rotated
            .iter()
            .zip(approx_rotated.iter())
            .map(|(y, y_hat)| y - y_hat)
            .collect();
        let gamma: f32 = residual.iter().map(|r| r * r).sum::<f32>().sqrt();

        // Step 8: QJL sign bits on normalized residual
        let qjl_signs = if gamma > 0.0 {
            let inv_gamma = 1.0 / gamma;
            let normalized_residual: Vec<f32> = residual.iter().map(|r| r * inv_gamma).collect();
            self.qjl.quantize(&normalized_residual)
        } else {
            vec![0u8; d_pad.div_ceil(8)]
        };

        PolarCompressedVector {
            angle_indices: packed_angles,
            qjl_signs,
            gamma,
            norm,
        }
    }

    /// Dequantize PolarQuant back to approximate f32 vector.
    ///
    /// Reconstructs: x̃ = ‖x‖ · Π⁻¹ · polar_to_cartesian(quantized_angles, 1.0)
    pub fn dequantize_f32(&self, compressed: &PolarCompressedVector) -> Vec<f32> {
        let d = self.dimension;
        let d_pad = self.padded_dim;
        let num_angles = d_pad - 1;

        // Decode angle indices
        let mut angles = Vec::with_capacity(num_angles);
        let mut flat_idx = 0usize;
        for level in 1..=self.num_levels {
            let cnt = polar::level_count(d_pad, level);
            for _ in 0..cnt {
                let idx = extract_angle_index(&compressed.angle_indices, flat_idx);
                angles.push(self.codebooks.decode(level, idx));
                flat_idx += 1;
            }
        }

        // Polar → Cartesian (unit norm reconstruction)
        let rotated_approx = polar_to_cartesian(&angles, 1.0, d_pad);

        // Inverse RHT: rotated space → original space
        let mut result = vec![0.0f32; d];
        self.rotation.inverse_padded(&rotated_approx, &mut result);

        // Rescale by original norm
        for v in &mut result {
            *v *= compressed.norm;
        }
        result
    }

    /// Dequantize angle indices back to approximate Cartesian in ROTATED space.
    ///
    /// Returns a d_pad-dimensional vector in the RHT-rotated coordinate system.
    fn dequantize_rotated(&self, compressed: &PolarCompressedVector) -> Vec<f32> {
        let d_pad = self.padded_dim;
        let num_angles = d_pad - 1;

        let mut angles = Vec::with_capacity(num_angles);
        let mut flat_idx = 0usize;
        for level in 1..=self.num_levels {
            let cnt = polar::level_count(d_pad, level);
            for _ in 0..cnt {
                let idx = extract_angle_index(&compressed.angle_indices, flat_idx);
                angles.push(self.codebooks.decode(level, idx));
                flat_idx += 1;
            }
        }

        polar_to_cartesian(&angles, 1.0, d_pad)
    }

    /// Access the rotation matrix.
    pub fn rotation(&self) -> &RotationMatrix {
        &self.rotation
    }

    /// Access the QJL projection.
    pub fn qjl(&self) -> &QjlProjection {
        &self.qjl
    }
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

/// Precomputed query state for efficient PolarQuant asymmetric distance.
pub struct PolarQueryState {
    /// Rotated query: q' = Π · q (d_pad-dimensional).
    pub rotated_query: Vec<f32>,
    /// QJL-projected query: S · q' (d_pad-dimensional).
    pub projected_query: Vec<f32>,
    /// Original query L2 norm.
    pub query_norm: f32,
}

impl PolarQuantizer {
    /// Prepare query state for PolarQuant distance computations.
    pub fn prepare_query(&self, query: &[f32]) -> PolarQueryState {
        let d_pad = self.padded_dim;

        let query_norm: f32 = query.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut rotated_query = vec![0.0f32; d_pad];
        self.rotation.forward_padded(query, &mut rotated_query);

        let projected_query = self.qjl.project_query(&rotated_query);

        PolarQueryState {
            rotated_query,
            projected_query,
            query_norm,
        }
    }

    /// Compute inner product estimate: ⟨q, x⟩ ≈ ‖x‖ · (mse_ip + qjl_ip).
    ///
    /// The MSE term dequantizes the angle representation to Cartesian and
    /// dots with the rotated query. The QJL term adds residual correction.
    pub fn inner_product(
        &self,
        query_state: &PolarQueryState,
        compressed: &PolarCompressedVector,
    ) -> f32 {
        // MSE term: dequantize angles to Cartesian, dot with rotated query
        let approx_rotated = self.dequantize_rotated(compressed);
        let mse_ip = f32::dot(&approx_rotated, &query_state.rotated_query)
            .unwrap_or(0.0) as f32;

        // QJL correction term
        let qjl_ip = self.qjl.inner_product_term(
            &compressed.qjl_signs,
            &query_state.projected_query,
            compressed.gamma,
        );

        compressed.norm * (mse_ip + qjl_ip)
    }

    /// Batch compute inner products for multiple candidates.
    pub fn batch_inner_products(
        &self,
        query_state: &PolarQueryState,
        candidates: &[PolarCompressedVector],
    ) -> Vec<f32> {
        candidates
            .iter()
            .map(|c| self.inner_product(query_state, c))
            .collect()
    }

    /// Compute cosine similarity.
    pub fn cosine_similarity(
        &self,
        query_state: &PolarQueryState,
        compressed: &PolarCompressedVector,
    ) -> f32 {
        let ip = self.inner_product(query_state, compressed);
        let denom = query_state.query_norm * compressed.norm;
        if denom > 0.0 {
            (ip / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Precomputed trig tables for fast symmetric distance (zero sin/cos at runtime)
// ---------------------------------------------------------------------------

/// Precomputed cos·cos and sin·sin cross-product tables for all levels.
///
/// For each level ℓ and each pair of 3-bit angle indices (i, j):
/// - `cc[ℓ-1][i][j] = cos(centroid_ℓ[i]) · cos(centroid_ℓ[j])`
/// - `ss[ℓ-1][i][j] = sin(centroid_ℓ[i]) · sin(centroid_ℓ[j])`
///
/// The symmetric dot product is computed via bottom-up tree recursion
/// using only table lookups, eliminating all sin/cos calls at distance time.
#[derive(Debug, Clone)]
pub struct PolarCrossTables {
    cc: Vec<[[f32; 8]; 8]>,
    ss: Vec<[[f32; 8]; 8]>,
    num_levels: usize,
}

impl PolarCrossTables {
    /// Build cross-product tables from a `PolarCodebooks`.
    pub fn new(codebooks: &PolarCodebooks) -> Self {
        let num_levels = codebooks.num_levels();
        let mut cc = Vec::with_capacity(num_levels);
        let mut ss = Vec::with_capacity(num_levels);

        for level in 1..=num_levels {
            let mut cc_table = [[0.0f32; 8]; 8];
            let mut ss_table = [[0.0f32; 8]; 8];
            for i in 0..8u8 {
                let ci = codebooks.decode(level, i);
                let (sin_i, cos_i) = ci.sin_cos();
                for j in 0..8u8 {
                    let cj = codebooks.decode(level, j);
                    let (sin_j, cos_j) = cj.sin_cos();
                    cc_table[i as usize][j as usize] = cos_i * cos_j;
                    ss_table[i as usize][j as usize] = sin_i * sin_j;
                }
            }
            cc.push(cc_table);
            ss.push(ss_table);
        }

        Self {
            cc,
            ss,
            num_levels,
        }
    }

    /// Number of levels.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
}

/// PolarQuant symmetric distance for USearch custom metric.
///
/// Computes the approximate inner product between two packed PolarQuant vectors
/// using a bottom-up tree recursion on angle indices with precomputed trig tables.
/// This avoids all sin/cos calls at distance time — only table lookups and
/// multiply-adds.
///
/// Complexity: O(d) table lookups + O(d) additions, zero trig operations.
pub fn polar_symmetric_distance(
    a: &[u8],
    b: &[u8],
    dim: usize,
    tables: &PolarCrossTables,
) -> f32 {
    let d_pad = dim; // Already padded dimension
    let num_angles = d_pad - 1;
    let angle_len = (num_angles * 3).div_ceil(8);
    let qjl_len = d_pad.div_ceil(8);
    let num_levels = tables.num_levels();

    // --- MSE term via bottom-up tree recursion ---
    // Level 1: each leaf pair contributes cc + ss = cos(angle_a - angle_b)
    let level1_count = d_pad / 2;
    let mut buf = Vec::with_capacity(level1_count);

    let a_angles = &a[..angle_len];
    let b_angles = &b[..angle_len];

    for j in 0..level1_count {
        let ai = extract_angle_index(a_angles, j) as usize;
        let bi = extract_angle_index(b_angles, j) as usize;
        buf.push(tables.cc[0][ai][bi] + tables.ss[0][ai][bi]);
    }

    // Levels 2..num_levels: combine pairs from previous level
    let mut flat_offset = level1_count; // where level-2 indices start
    let mut count = level1_count;
    for level in 2..=num_levels {
        let n = count / 2;
        for j in 0..n {
            let ai = extract_angle_index(a_angles, flat_offset + j) as usize;
            let bi = extract_angle_index(b_angles, flat_offset + j) as usize;
            buf[j] = tables.cc[level - 1][ai][bi] * buf[2 * j]
                + tables.ss[level - 1][ai][bi] * buf[2 * j + 1];
        }
        flat_offset += n;
        count = n;
    }

    let mse_term = buf[0];

    // --- QJL term: Hamming distance on sign bits ---
    let a_qjl = &a[angle_len..angle_len + qjl_len];
    let b_qjl = &b[angle_len..angle_len + qjl_len];
    let a_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(a_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let b_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(b_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let hamming_bits = u1x8::hamming(a_bits, b_bits).unwrap_or(0);

    let a_gamma = f32::from_le_bytes(
        a[angle_len + qjl_len..angle_len + qjl_len + 4]
            .try_into()
            .unwrap(),
    );
    let b_gamma = f32::from_le_bytes(
        b[angle_len + qjl_len..angle_len + qjl_len + 4]
            .try_into()
            .unwrap(),
    );
    let a_norm = f32::from_le_bytes(
        a[angle_len + qjl_len + 4..angle_len + qjl_len + 8]
            .try_into()
            .unwrap(),
    );
    let b_norm = f32::from_le_bytes(
        b[angle_len + qjl_len + 4..angle_len + qjl_len + 8]
            .try_into()
            .unwrap(),
    );

    let qjl_term =
        a_gamma * b_gamma * f32::cos(PI * hamming_bits as f32 / d_pad as f32);

    a_norm * b_norm * (mse_term + qjl_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_quant::qjl::fill_standard_normal;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn codebook_level1_uniform() {
        let cb = PolarCodebooks::new(4);
        // Level 1 centroids should be roughly equally spaced in [0, 2π)
        let step = 2.0 * PI / 8.0;
        for k in 0..8 {
            let expected = (k as f32 + 0.5) * step;
            assert!(
                (cb.centroids[0][k] - expected).abs() < 0.01,
                "Level 1 centroid {k}: expected {expected}, got {}",
                cb.centroids[0][k]
            );
        }
    }

    #[test]
    fn codebook_higher_levels_centered() {
        let cb = PolarCodebooks::new(10);
        // Higher levels should have centroids clustering around π/4
        for level in 4..=10 {
            let mean: f32 = cb.centroids[level - 1].iter().sum::<f32>() / 8.0;
            assert!(
                (mean - PI / 4.0).abs() < 0.2,
                "Level {level} mean centroid: {mean}, expected ≈ {:.3}",
                PI / 4.0
            );
        }
    }

    #[test]
    fn encode_decode_level1_roundtrip() {
        let cb = PolarCodebooks::new(4);
        for k in 0..8u8 {
            let centroid = cb.decode(1, k);
            let encoded = cb.encode(1, centroid);
            assert_eq!(encoded, k, "Level 1 encode/decode mismatch for index {k}");
        }
    }

    #[test]
    fn encode_decode_level2_roundtrip() {
        let cb = PolarCodebooks::new(4);
        for k in 0..8u8 {
            let centroid = cb.decode(2, k);
            let encoded = cb.encode(2, centroid);
            assert_eq!(encoded, k, "Level 2 encode/decode mismatch for index {k}");
        }
    }

    #[test]
    fn pack_unpack_angle_roundtrip() {
        let indices: Vec<u8> = (0..15).map(|i| i % 8).collect();
        let packed = pack_angle_indices(&indices);
        for (j, &expected) in indices.iter().enumerate() {
            let got = extract_angle_index(&packed, j);
            assert_eq!(got, expected, "Mismatch at position {j}");
        }
    }

    #[test]
    fn quantizer_pack_unpack_roundtrip() {
        let d = 768;
        let quantizer = PolarQuantizer::new(d, 42, 137);

        let mut rng = StdRng::seed_from_u64(12345);
        let mut vector = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut vector);

        let compressed = quantizer.quantize(&vector);
        let packed = compressed.pack();
        assert_eq!(packed.len(), PolarCompressedVector::packed_size(d));

        let unpacked = PolarCompressedVector::unpack(&packed, d);
        assert_eq!(compressed.angle_indices, unpacked.angle_indices);
        assert_eq!(compressed.qjl_signs, unpacked.qjl_signs);
        assert!((compressed.gamma - unpacked.gamma).abs() < 1e-7);
        assert!((compressed.norm - unpacked.norm).abs() < 1e-7);
    }

    #[test]
    fn norm_preserved_in_compressed() {
        let d = 768;
        let quantizer = PolarQuantizer::new(d, 42, 137);

        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..20 {
            let mut raw = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut raw);
            let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let compressed = quantizer.quantize(&raw);
            assert!(
                (compressed.norm - norm).abs() < 1e-5,
                "Stored norm mismatch: {:.6} vs {:.6}",
                compressed.norm,
                norm
            );
        }
    }

    #[test]
    fn zero_vector_handling() {
        let d = 128;
        let quantizer = PolarQuantizer::new(d, 42, 137);
        let zero = vec![0.0f32; d];
        let compressed = quantizer.quantize(&zero);
        assert_eq!(compressed.norm, 0.0);
        assert_eq!(compressed.gamma, 0.0);
    }

    #[test]
    fn mse_distortion_bound() {
        let d = 768;
        let quantizer = PolarQuantizer::new(d, 42, 137);
        let n_trials = 100;
        let mut total_mse = 0.0f64;

        let mut rng = StdRng::seed_from_u64(54321);
        for _ in 0..n_trials {
            let mut raw = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut raw);
            let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x: Vec<f32> = raw.iter().map(|v| v / norm).collect();

            let compressed = quantizer.quantize(&x);
            let reconstructed = quantizer.dequantize_f32(&compressed);

            let mse: f64 = x
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / d as f64;
            total_mse += mse;
        }

        let avg_mse = total_mse / n_trials as f64;
        // PolarQuant 3-bit should have reasonable MSE; allow wider bound than TQ4
        assert!(
            avg_mse < 0.05,
            "Average MSE too high: {avg_mse:.6} (expected < 0.05 for 3-bit polar codebook)"
        );
    }

    #[test]
    fn asymmetric_inner_product_unbiased() {
        let d = 256;
        let n_trials = 200;
        let mut total_err = 0.0f64;
        let mut total_abs = 0.0f64;

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..n_trials {
            let qjl_seed: u64 = rand::Rng::random(&mut rng);
            let quantizer = PolarQuantizer::new(d, 42, qjl_seed);

            let mut x = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x);
            let mut q = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut q);

            let true_ip: f32 = x.iter().zip(q.iter()).map(|(a, b)| a * b).sum();

            let compressed = quantizer.quantize(&x);
            let query_state = quantizer.prepare_query(&q);
            let pq_ip = quantizer.inner_product(&query_state, &compressed);

            total_err += (pq_ip - true_ip) as f64;
            total_abs += true_ip.abs() as f64;
        }

        let avg_err = total_err / n_trials as f64;
        let avg_abs = total_abs / n_trials as f64;
        let rel_bias = avg_err.abs() / avg_abs;

        assert!(
            rel_bias < 0.3,
            "Asymmetric IP bias too high: avg_err={avg_err:.6}, rel_bias={rel_bias:.4}"
        );
    }

    #[test]
    fn symmetric_distance_self() {
        let d = 256;
        let quantizer = PolarQuantizer::new(d, 42, 137);
        let tables = PolarCrossTables::new(&quantizer.codebooks);

        let mut rng = StdRng::seed_from_u64(42);
        let mut v = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let compressed = quantizer.quantize(&v);
        let packed = compressed.pack();

        let ip = polar_symmetric_distance(
            &packed,
            &packed,
            quantizer.padded_dim(),
            &tables,
        );
        let expected = norm * norm;
        let rel_err = (ip - expected).abs() / expected;
        assert!(
            rel_err < 0.5,
            "Self inner product should approximate ‖x‖²={expected:.4}, got {ip:.4}, rel_err={rel_err:.4}"
        );
    }
}
