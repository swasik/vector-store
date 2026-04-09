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
use crate::turbo_quant::rotation::{RotationMatrix, hadamard_transform};
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
        assert!((1..=MAX_LEVELS).contains(&num_levels));
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
        for (k, c) in centroids.iter_mut().enumerate() {
            *c = (k as f32 + 0.5) * step;
        }
        for (k, b) in boundaries.iter_mut().enumerate() {
            *b = (k as f32 + 1.0) * step;
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
    for (k, c) in centroids.iter_mut().enumerate() {
        let idx = ((k as f64 + 0.5) / 8.0 * n_samples as f64) as usize;
        *c = samples[idx.min(n_samples - 1)];
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

/// Extract 8 consecutive 3-bit indices from a packed big-endian bitstream.
///
/// Eight 3-bit values occupy exactly 24 bits = 3 bytes. This processes
/// an entire group at once, reducing per-element bit-manipulation overhead
/// in the symmetric distance hot path.
#[inline]
pub fn extract_8_angle_indices(packed: &[u8], group: usize) -> [u8; 8] {
    let byte_offset = group * 3;
    let b0 = packed[byte_offset] as u32;
    let b1 = packed[byte_offset + 1] as u32;
    let b2 = packed[byte_offset + 2] as u32;
    let bits = (b0 << 16) | (b1 << 8) | b2;
    [
        ((bits >> 21) & 0x07) as u8,
        ((bits >> 18) & 0x07) as u8,
        ((bits >> 15) & 0x07) as u8,
        ((bits >> 12) & 0x07) as u8,
        ((bits >> 9) & 0x07) as u8,
        ((bits >> 6) & 0x07) as u8,
        ((bits >> 3) & 0x07) as u8,
        (bits & 0x07) as u8,
    ]
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

/// Thread-local workspace for quantize() to avoid per-call heap allocations.
struct QuantizeWorkspace {
    rotated: Vec<f32>,
    approx: Vec<f32>,
    expand: Vec<f32>,
    angle_idx: Vec<u8>,
    deq_angles: Vec<f32>,
}

impl QuantizeWorkspace {
    fn ensure_capacity(&mut self, d_pad: usize, num_angles: usize) {
        if self.rotated.len() < d_pad {
            self.rotated.resize(d_pad, 0.0);
        }
        if self.approx.len() < d_pad {
            self.approx.resize(d_pad, 0.0);
        }
        if self.expand.capacity() < d_pad {
            self.expand.reserve(d_pad - self.expand.capacity());
        }
        if self.angle_idx.capacity() < num_angles {
            self.angle_idx
                .reserve(num_angles - self.angle_idx.capacity());
        }
        if self.deq_angles.capacity() < num_angles {
            self.deq_angles
                .reserve(num_angles - self.deq_angles.capacity());
        }
    }
}

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
    /// Precomputed sin/cos of each codebook centroid per level.
    /// `sincos_tables[level-1][idx] = (sin(centroid), cos(centroid))` for idx ∈ [0, 8).
    sincos_tables: Vec<[(f32, f32); 8]>,
}

impl PolarQuantizer {
    /// Create a new PolarQuant quantizer for the given dimension and seeds.
    pub fn new(dimension: usize, rotation_seed: u64, qjl_seed: u64) -> Self {
        let rotation = RotationMatrix::from_seed(dimension, rotation_seed);
        let d_pad = rotation.padded_dim();
        let qjl = QjlProjection::from_seed(d_pad, qjl_seed);
        let num_levels = d_pad.trailing_zeros() as usize;
        let codebooks = PolarCodebooks::new(num_levels);

        // Precompute sin/cos for all codebook centroids at each level
        let sincos_tables: Vec<[(f32, f32); 8]> = (1..=num_levels)
            .map(|level| {
                let mut table = [(0.0f32, 0.0f32); 8];
                for i in 0..8u8 {
                    let angle = codebooks.decode(level, i);
                    table[i as usize] = angle.sin_cos();
                }
                table
            })
            .collect();

        Self {
            rotation,
            qjl,
            codebooks,
            dimension,
            padded_dim: d_pad,
            num_levels,
            sincos_tables,
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
    ///
    /// Uses thread-local workspace buffers to minimize heap allocations.
    pub fn quantize(&self, vector: &[f32]) -> PolarCompressedVector {
        let d_pad = self.padded_dim;
        let num_angles = d_pad - 1;

        // Step 1: Original norm (SIMD-accelerated via NumKong)
        let norm: f32 = (f32::dot(vector, vector).unwrap_or(0.0) as f32).sqrt();

        if norm == 0.0 {
            return PolarCompressedVector {
                angle_indices: vec![0u8; (num_angles * 3).div_ceil(8)],
                qjl_signs: vec![0u8; d_pad.div_ceil(8)],
                gamma: 0.0,
                norm: 0.0,
            };
        }

        thread_local! {
            static QBUF: std::cell::RefCell<QuantizeWorkspace> = const {
                std::cell::RefCell::new(QuantizeWorkspace {
                    rotated: Vec::new(),
                    approx: Vec::new(),
                    expand: Vec::new(),
                    angle_idx: Vec::new(),
                    deq_angles: Vec::new(),
                })
            };
        }

        QBUF.with(|cell| {
            let mut ws = cell.borrow_mut();
            ws.ensure_capacity(d_pad, num_angles);

            // Step 2: Normalize directly into rotated buffer via forward_padded
            // Apply inv_norm inline during the diagonal step of forward_padded
            let inv_norm = 1.0 / norm;
            let rotated = &mut ws.rotated[..d_pad];
            // Manually inline: normalize + forward_padded in one pass
            let dim = self.rotation.dimension();
            let d_pad_rot = self.rotation.padded_dim();
            debug_assert_eq!(d_pad, d_pad_rot);
            for i in 0..dim {
                rotated[i] = (vector[i] * inv_norm) * self.rotation.signs[i];
            }
            for v in rotated[dim..d_pad].iter_mut() {
                *v = 0.0;
            }
            hadamard_transform(rotated);
            let inv_sqrt = self.rotation.inv_sqrt_d_pad;
            for v in rotated.iter_mut() {
                *v *= inv_sqrt;
            }

            // Step 4: Polar transform (uses internal allocations — hard to avoid
            // without rewriting cartesian_to_polar)
            let (angles, _radius) = cartesian_to_polar(rotated);

            // Steps 5+6: Quantize + dequantize angles (merged)
            ws.angle_idx.clear();
            ws.deq_angles.clear();
            for level in 1..=self.num_levels {
                let off = polar::level_offset(d_pad, level);
                let cnt = polar::level_count(d_pad, level);
                for j in 0..cnt {
                    let idx = self.codebooks.encode(level, angles[off + j]);
                    ws.angle_idx.push(idx);
                    ws.deq_angles.push(self.codebooks.decode(level, idx));
                }
            }
            let packed_angles = pack_angle_indices(&ws.angle_idx);

            // Step 6b: Dequantized angles → approximate Cartesian (reuse approx buffer)
            // Use dequantize_rotated_into style expansion to avoid polar_to_cartesian allocs
            {
                ws.approx[0] = 1.0f32;

                let mut level_offsets = [0usize; 32];
                let mut off = 0usize;
                for level in 1..=self.num_levels {
                    level_offsets[level - 1] = off;
                    off += d_pad / (1 << level);
                }

                // Expand from top level down using angle_idx + sincos_tables
                // Destructure to split borrows: approx (mut), angle_idx (immut), expand (mut)
                let QuantizeWorkspace {
                    approx,
                    angle_idx,
                    expand,
                    ..
                } = &mut *ws;
                for level in (1..=self.num_levels).rev() {
                    let level_offset = level_offsets[level - 1];
                    let n_a = d_pad / (1 << level);
                    let sincos = &self.sincos_tables[level - 1];

                    expand.clear();
                    if expand.capacity() < n_a * 2 {
                        expand.reserve(n_a * 2);
                    }
                    for j in 0..n_a {
                        let r = approx[j];
                        let idx = angle_idx[level_offset + j] as usize;
                        let (sin_a, cos_a) = sincos[idx];
                        expand.push(r * cos_a);
                        expand.push(r * sin_a);
                    }
                    approx[..n_a * 2].copy_from_slice(&expand[..n_a * 2]);
                }
            }

            // Step 7: Residual (compute in-place into rotated buffer)
            for i in 0..d_pad {
                ws.rotated[i] -= ws.approx[i];
            }
            let gamma: f32 =
                (f32::dot(&ws.rotated[..d_pad], &ws.rotated[..d_pad]).unwrap_or(0.0) as f32).sqrt();

            // Step 8: QJL sign bits on normalized residual
            let qjl_signs = if gamma > 0.0 {
                let inv_gamma = 1.0 / gamma;
                for v in ws.rotated[..d_pad].iter_mut() {
                    *v *= inv_gamma;
                }
                self.qjl.quantize(&ws.rotated[..d_pad])
            } else {
                vec![0u8; d_pad.div_ceil(8)]
            };

            PolarCompressedVector {
                angle_indices: packed_angles,
                qjl_signs,
                gamma,
                norm,
            }
        })
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
        let mut out = vec![0.0f32; self.padded_dim];
        self.dequantize_rotated_into(compressed, &mut out);
        out
    }

    /// Dequantize angle indices into a pre-allocated buffer (no heap allocation).
    ///
    /// Uses precomputed sin/cos tables to avoid trig calls. Reconstructs
    /// the unit-norm Cartesian vector via top-down polar expansion using
    /// double-buffering.
    fn dequantize_rotated_into(&self, compressed: &PolarCompressedVector, out: &mut [f32]) {
        let d_pad = self.padded_dim;
        debug_assert!(out.len() >= d_pad);

        // Top-down reconstruction: start with radius=1.0, expand through levels.
        // Use the output buffer itself as working space (double-buffering in-place).
        //
        // We process levels from top (num_levels) down to 1.
        // At each level, values[0..n] are expanded into values[0..2*n] using
        // the angle indices at that level.

        // Start: single value at top level = radius = 1.0
        out[0] = 1.0f32;

        // Compute level offsets into the flat angle index array
        let mut level_offsets = [0usize; 32]; // supports up to d_pad = 2^32
        let mut off = 0usize;
        for level in 1..=self.num_levels {
            level_offsets[level - 1] = off;
            off += d_pad / (1 << level);
        }

        // Expand from top level down to level 1
        // We need a second buffer for expansion since we read values[j]
        // and write to expanded[2*j] and expanded[2*j+1]
        thread_local! {
            static EXPAND_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }

        EXPAND_BUF.with(|cell| {
            let mut expand = cell.borrow_mut();

            for level in (1..=self.num_levels).rev() {
                let level_offset = level_offsets[level - 1];
                let n_angles = d_pad / (1 << level);
                let sincos = &self.sincos_tables[level - 1];

                expand.clear();
                let needed = n_angles * 2;
                if expand.capacity() < needed {
                    expand.reserve(needed);
                }

                for (j, &r) in out[..n_angles].iter().enumerate() {
                    let idx =
                        extract_angle_index(&compressed.angle_indices, level_offset + j) as usize;
                    let (sin_a, cos_a) = sincos[idx];
                    expand.push(r * cos_a);
                    expand.push(r * sin_a);
                }

                out[..n_angles * 2].copy_from_slice(&expand[..n_angles * 2]);
            }
        });
    }

    /// Access the rotation matrix.
    pub fn rotation(&self) -> &RotationMatrix {
        &self.rotation
    }

    /// Access the QJL projection.
    pub fn qjl(&self) -> &QjlProjection {
        &self.qjl
    }

    /// Compute inner product directly from packed bytes (zero-copy).
    ///
    /// Avoids the 2 heap allocations in `PolarCompressedVector::unpack()` by
    /// reading angle_indices and qjl_signs as borrowed slices from the packed
    /// byte buffer. Used in the reranking path for efficiency.
    pub fn inner_product_packed(
        &self,
        query_state: &PolarQueryState,
        packed: &[u8],
        dimension: usize,
    ) -> f32 {
        let d_pad = dimension.next_power_of_two();
        let num_angles = d_pad - 1;
        let angle_len = (num_angles * 3).div_ceil(8);
        let qjl_len = d_pad.div_ceil(8);

        let angle_indices = &packed[..angle_len];
        let qjl_signs = &packed[angle_len..angle_len + qjl_len];
        let gamma = f32::from_le_bytes(
            packed[angle_len + qjl_len..angle_len + qjl_len + 4]
                .try_into()
                .unwrap(),
        );
        let norm = f32::from_le_bytes(
            packed[angle_len + qjl_len + 4..angle_len + qjl_len + 8]
                .try_into()
                .unwrap(),
        );

        thread_local! {
            static DEQBUF2: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }

        DEQBUF2.with(|cell| {
            let mut buf = cell.borrow_mut();
            if buf.len() < d_pad {
                buf.resize(d_pad, 0.0);
            }

            // Inline dequantize_rotated_into using borrowed angle_indices
            buf[0] = 1.0f32;
            let mut level_offsets = [0usize; 32];
            let mut off = 0usize;
            for level in 1..=self.num_levels {
                level_offsets[level - 1] = off;
                off += d_pad / (1 << level);
            }

            thread_local! {
                static EXPAND2: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            EXPAND2.with(|ecell| {
                let mut expand = ecell.borrow_mut();
                for level in (1..=self.num_levels).rev() {
                    let level_offset = level_offsets[level - 1];
                    let n_angles = d_pad / (1 << level);
                    let sincos = &self.sincos_tables[level - 1];

                    expand.clear();
                    if expand.capacity() < n_angles * 2 {
                        expand.reserve(n_angles * 2);
                    }
                    for j in 0..n_angles {
                        let r = buf[j];
                        let idx = extract_angle_index(angle_indices, level_offset + j) as usize;
                        let (sin_a, cos_a) = sincos[idx];
                        expand.push(r * cos_a);
                        expand.push(r * sin_a);
                    }
                    buf[..n_angles * 2].copy_from_slice(&expand[..n_angles * 2]);
                }
            });

            let mse_ip = f32::dot(&buf[..d_pad], &query_state.rotated_query)
                .unwrap_or(0.0) as f32;

            let qjl_ip = self.qjl.inner_product_term(
                qjl_signs,
                &query_state.projected_query,
                gamma,
            );

            norm * (mse_ip + qjl_ip)
        })
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

        let query_norm: f32 = (f32::dot(query, query).unwrap_or(0.0) as f32).sqrt();

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
    ///
    /// Uses thread-local workspace to avoid per-call heap allocation.
    pub fn inner_product(
        &self,
        query_state: &PolarQueryState,
        compressed: &PolarCompressedVector,
    ) -> f32 {
        thread_local! {
            static DEQBUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }

        DEQBUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            let d_pad = self.padded_dim;
            if buf.len() < d_pad {
                buf.resize(d_pad, 0.0);
            }
            self.dequantize_rotated_into(compressed, &mut buf);

            let mse_ip = f32::dot(&buf[..d_pad], &query_state.rotated_query).unwrap_or(0.0) as f32;

            let qjl_ip = self.qjl.inner_product_term(
                &compressed.qjl_signs,
                &query_state.projected_query,
                compressed.gamma,
            );

            compressed.norm * (mse_ip + qjl_ip)
        })
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
    /// Level 1 combined table: `cc_plus_ss[i][j] = cc[0][i][j] + ss[0][i][j]`.
    /// Fuses two lookups + addition into one lookup for the hottest loop.
    cc_plus_ss_level1: [[f32; 8]; 8],
    /// Interleaved (cc, ss) per index pair for levels 2+.
    /// `ccss_interleaved[level-2][i][j] = [cc, ss]` — both values in the
    /// same cache line, eliminating one cache miss per element at levels 2+.
    ccss_interleaved: Vec<[[[f32; 2]; 8]; 8]>,
    num_levels: usize,
    /// Precomputed cos(π·k/d_pad) for k ∈ [0, d_pad], eliminates f32::cos()
    /// in the QJL Hamming term at distance time.
    cos_table: Vec<f32>,
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

        // d_pad = 2^num_levels; precompute cos(π·k/d_pad) for k = 0..=d_pad
        let d_pad = 1usize << num_levels;
        let cos_table: Vec<f32> = (0..=d_pad)
            .map(|k| (PI * k as f32 / d_pad as f32).cos())
            .collect();

        // Precompute combined cc+ss table for level 1 (the hottest loop)
        let mut cc_plus_ss_level1 = [[0.0f32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                cc_plus_ss_level1[i][j] = cc[0][i][j] + ss[0][i][j];
            }
        }

        // Precompute interleaved (cc, ss) tables for levels 2+
        let mut ccss_interleaved = Vec::with_capacity(num_levels.saturating_sub(1));
        for level_idx in 1..num_levels {
            let mut table = [[[0.0f32; 2]; 8]; 8];
            for i in 0..8 {
                for j in 0..8 {
                    table[i][j] = [cc[level_idx][i][j], ss[level_idx][i][j]];
                }
            }
            ccss_interleaved.push(table);
        }

        Self {
            cc,
            ss,
            cc_plus_ss_level1,
            ccss_interleaved,
            num_levels,
            cos_table,
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
/// Optimizations:
/// - Thread-local buffer eliminates heap allocation per distance call
/// - Precomputed cos lookup table eliminates f32::cos() for QJL Hamming term
/// - Batch 8-at-a-time 3-bit extraction reduces bit-manipulation overhead
/// - Fused cc+ss table for level 1 (hottest loop) eliminates one lookup + add per element
///
/// Complexity: O(d) table lookups + O(d) additions, zero trig operations.
pub fn polar_symmetric_distance(a: &[u8], b: &[u8], dim: usize, tables: &PolarCrossTables) -> f32 {
    let d_pad = dim; // Already padded dimension
    let num_angles = d_pad - 1;
    let angle_len = (num_angles * 3).div_ceil(8);
    let qjl_len = d_pad.div_ceil(8);
    let num_levels = tables.num_levels();

    // --- MSE term via bottom-up tree recursion ---
    let level1_count = d_pad / 2;

    // Thread-local buffer avoids heap allocation on every distance call.
    // Use UnsafeCell to eliminate RefCell borrow overhead in the hot path.
    thread_local! {
        static BUF: std::cell::UnsafeCell<Vec<f32>> = const { std::cell::UnsafeCell::new(Vec::new()) };
    }

    let mse_term = BUF.with(|cell| {
        // SAFETY: This function is not reentrant and the thread-local
        // guarantees single-threaded access. No references escape the closure.
        let buf = unsafe { &mut *cell.get() };
        if buf.len() < level1_count {
            buf.resize(level1_count, 0.0);
        }

        let a_angles = &a[..angle_len];
        let b_angles = &b[..angle_len];

        // Level 1: batch 8-at-a-time extraction with fused cc+ss table
        let ccss = &tables.cc_plus_ss_level1;
        let full_groups = level1_count / 8;
        let remainder = level1_count % 8;

        for g in 0..full_groups {
            let a_idx = extract_8_angle_indices(a_angles, g);
            let b_idx = extract_8_angle_indices(b_angles, g);
            let base_j = g * 8;
            for k in 0..8 {
                buf[base_j + k] = ccss[a_idx[k] as usize][b_idx[k] as usize];
            }
        }
        // Handle remaining level-1 elements (< 8)
        let base = full_groups * 8;
        for j in 0..remainder {
            let ai = extract_angle_index(a_angles, base + j) as usize;
            let bi = extract_angle_index(b_angles, base + j) as usize;
            buf[base + j] = ccss[ai][bi];
        }

        // Levels 2..num_levels: combine pairs from previous level
        // Uses interleaved (cc, ss) tables for cache locality
        let mut flat_offset = level1_count;
        let mut count = level1_count;
        for level in 2..=num_levels {
            let n = count / 2;
            let ccss_l = &tables.ccss_interleaved[level - 2];

            let groups = n / 8;
            let rem = n % 8;
            let group_base = flat_offset / 8;

            for g in 0..groups {
                let a_idx = extract_8_angle_indices(a_angles, group_base + g);
                let b_idx = extract_8_angle_indices(b_angles, group_base + g);

                let base_j = g * 8;
                for k in 0..8 {
                    let j = base_j + k;
                    let pair = &ccss_l[a_idx[k] as usize][b_idx[k] as usize];
                    buf[j] = pair[0] * buf[2 * j] + pair[1] * buf[2 * j + 1];
                }
            }

            let rem_base = groups * 8;
            for j in 0..rem {
                let idx = rem_base + j;
                let ai = extract_angle_index(a_angles, flat_offset + idx) as usize;
                let bi = extract_angle_index(b_angles, flat_offset + idx) as usize;
                let pair = &ccss_l[ai][bi];
                buf[idx] = pair[0] * buf[2 * idx] + pair[1] * buf[2 * idx + 1];
            }

            flat_offset += n;
            count = n;
        }

        buf[0]
    });

    // --- QJL term: Hamming distance on sign bits ---
    let a_qjl = &a[angle_len..angle_len + qjl_len];
    let b_qjl = &b[angle_len..angle_len + qjl_len];
    let a_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(a_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let b_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(b_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let hamming_bits = u1x8::hamming(a_bits, b_bits).unwrap_or(0);

    // Read metadata (gamma, norm) from fixed-offset tail of packed buffer.
    // Layout: [...| gamma(4 LE) | norm(4 LE)]
    let meta_offset = angle_len + qjl_len;
    let a_gamma = f32::from_le_bytes([
        a[meta_offset],
        a[meta_offset + 1],
        a[meta_offset + 2],
        a[meta_offset + 3],
    ]);
    let b_gamma = f32::from_le_bytes([
        b[meta_offset],
        b[meta_offset + 1],
        b[meta_offset + 2],
        b[meta_offset + 3],
    ]);
    let a_norm = f32::from_le_bytes([
        a[meta_offset + 4],
        a[meta_offset + 5],
        a[meta_offset + 6],
        a[meta_offset + 7],
    ]);
    let b_norm = f32::from_le_bytes([
        b[meta_offset + 4],
        b[meta_offset + 5],
        b[meta_offset + 6],
        b[meta_offset + 7],
    ]);

    // Use precomputed cos table instead of f32::cos()
    let qjl_term = a_gamma * b_gamma * tables.cos_table[hamming_bits as usize];

    a_norm * b_norm * (mse_term + qjl_term)
}

/// Baseline (pre-optimization) symmetric distance for A/B benchmarking.
/// Reproduces the original algorithm: per-element extraction, separate cc/ss
/// lookups, Vec allocation per call, f32::cos for QJL term.
#[cfg(test)]
pub(crate) fn polar_symmetric_distance_baseline(
    a: &[u8],
    b: &[u8],
    dim: usize,
    tables: &PolarCrossTables,
) -> f32 {
    let d_pad = dim;
    let num_angles = d_pad - 1;
    let angle_len = (num_angles * 3).div_ceil(8);
    let qjl_len = d_pad.div_ceil(8);
    let num_levels = tables.num_levels();

    let level1_count = d_pad / 2;
    let mut buf = Vec::with_capacity(level1_count);

    let a_angles = &a[..angle_len];
    let b_angles = &b[..angle_len];

    for j in 0..level1_count {
        let ai = extract_angle_index(a_angles, j) as usize;
        let bi = extract_angle_index(b_angles, j) as usize;
        buf.push(tables.cc[0][ai][bi] + tables.ss[0][ai][bi]);
    }

    let mut flat_offset = level1_count;
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

    let qjl_term = a_gamma * b_gamma * f32::cos(PI * hamming_bits as f32 / d_pad as f32);

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
                (compressed.norm - norm).abs() < 1e-4,
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

        let ip = polar_symmetric_distance(&packed, &packed, quantizer.padded_dim(), &tables);
        let expected = norm * norm;
        let rel_err = (ip - expected).abs() / expected;
        assert!(
            rel_err < 0.5,
            "Self inner product should approximate ‖x‖²={expected:.4}, got {ip:.4}, rel_err={rel_err:.4}"
        );
    }

    #[test]
    #[ignore] // Microbenchmark, run with --ignored
    fn bench_symmetric_distance() {
        use std::hint::black_box;
        use std::time::Instant;

        let d = 1536;
        let quantizer = PolarQuantizer::new(d, 42, 137);
        let tables = PolarCrossTables::new(&quantizer.codebooks);
        let d_pad = quantizer.padded_dim();

        // Generate test vectors
        let mut rng = StdRng::seed_from_u64(42);
        let n_vecs = 200;
        let packed_vecs: Vec<Vec<u8>> = (0..n_vecs)
            .map(|_| {
                let mut v = vec![0.0f32; d];
                fill_standard_normal(&mut rng, &mut v);
                quantizer.quantize(&v).pack()
            })
            .collect();

        numkong::configure_thread();

        // Warmup
        for _ in 0..1000 {
            black_box(polar_symmetric_distance(
                &packed_vecs[0],
                &packed_vecs[1],
                d_pad,
                &tables,
            ));
        }

        // Benchmark
        let n_iters = 100_000;
        let t0 = Instant::now();
        for i in 0..n_iters {
            let a = &packed_vecs[i % n_vecs];
            let b = &packed_vecs[(i + 1) % n_vecs];
            black_box(polar_symmetric_distance(a, b, d_pad, &tables));
        }
        let elapsed = t0.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / n_iters as f64;
        let us_per_call = ns_per_call / 1000.0;
        eprintln!(
            "\n=== polar_symmetric_distance (d={d}, d_pad={d_pad}) ===\n\
             {n_iters} iterations in {:.1}ms\n\
             {ns_per_call:.0} ns/call ({us_per_call:.2} µs/call)\n\
             Throughput: {:.1}M distances/sec",
            elapsed.as_secs_f64() * 1000.0,
            1e9 / ns_per_call / 1e6,
        );

        // Baseline comparison
        for _ in 0..1000 {
            black_box(polar_symmetric_distance_baseline(
                &packed_vecs[0],
                &packed_vecs[1],
                d_pad,
                &tables,
            ));
        }
        let t0 = Instant::now();
        for i in 0..n_iters {
            let a = &packed_vecs[i % n_vecs];
            let b = &packed_vecs[(i + 1) % n_vecs];
            black_box(polar_symmetric_distance_baseline(a, b, d_pad, &tables));
        }
        let elapsed_base = t0.elapsed();
        let ns_base = elapsed_base.as_nanos() as f64 / n_iters as f64;
        let us_base = ns_base / 1000.0;
        eprintln!(
            "\n=== BASELINE polar_symmetric_distance (d={d}, d_pad={d_pad}) ===\n\
             {n_iters} iterations in {:.1}ms\n\
             {ns_base:.0} ns/call ({us_base:.2} µs/call)\n\
             Speedup: {:.2}x ({:.1}% faster)",
            elapsed_base.as_secs_f64() * 1000.0,
            ns_base / ns_per_call,
            (1.0 - ns_per_call / ns_base) * 100.0,
        );
    }
}
