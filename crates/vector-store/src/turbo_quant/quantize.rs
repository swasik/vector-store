/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! TQ4 quantize/dequantize pipeline orchestrating codebook + rotation + QJL.
//!
//! The full TQ4 encoding of a vector x ∈ ℝ^d stores:
//!   - 3-bit MSE codebook indices (ceil(3d/8) bytes)
//!   - 1-bit QJL sign bits (ceil(d/8) bytes)
//!   - Residual norm γ = ‖r‖ (4 bytes, f32 LE)
//!   - Original norm ‖x‖ (4 bytes, f32 LE)
//!     Total: d/2 + 8 bytes for d divisible by 8.

use crate::turbo_quant::codebook;
use crate::turbo_quant::qjl::QjlProjection;
use crate::turbo_quant::rotation::RotationMatrix;
use numkong::Dot;

/// Packed TQ4 representation of a single vector.
///
/// Uses an interleaved 4-bit nibble layout where each coordinate is stored as
/// a single nibble: bit 3 = QJL sign, bits 2-0 = MSE 3-bit centroid index.
/// Two nibbles per byte (high = even dim, low = odd dim).
#[derive(Debug, Clone)]
pub struct Tq4CompressedVector {
    /// Interleaved 4-bit nibbles: bit 3 = QJL sign, bits 2-0 = MSE index.
    /// High nibble (bits 7-4) = even dimension, low nibble (bits 3-0) = odd.
    /// Length: d_pad / 2.
    pub nibbles: Vec<u8>,
    /// Residual L2 norm (γ = ‖r‖).
    pub gamma: f32,
    /// Original vector L2 norm (‖x‖).
    pub norm: f32,
}

impl Tq4CompressedVector {
    /// Total storage in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.nibbles.len() + 8 // 4 bytes gamma + 4 bytes norm
    }

    /// Pack into a contiguous byte array for USearch storage.
    ///
    /// Layout: `[nibbles | gamma(4 bytes LE) | norm(4 bytes LE)]`
    pub fn pack(&self) -> Vec<u8> {
        let total = self.storage_bytes();
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.nibbles);
        buf.extend_from_slice(&self.gamma.to_le_bytes());
        buf.extend_from_slice(&self.norm.to_le_bytes());
        buf
    }

    /// Unpack from a contiguous byte array retrieved via USearch::get().
    ///
    /// `dimension` here is the original vector dimension; internally pads to
    /// next power of 2 for TQ4 encoding.
    pub fn unpack(bytes: &[u8], dimension: usize) -> Self {
        let d_pad = dimension.next_power_of_two();
        let nibble_len = d_pad / 2;

        let nibbles = bytes[..nibble_len].to_vec();
        let gamma_bytes: [u8; 4] = bytes[nibble_len..nibble_len + 4]
            .try_into()
            .expect("gamma bytes");
        let norm_bytes: [u8; 4] = bytes[nibble_len + 4..nibble_len + 8]
            .try_into()
            .expect("norm bytes");

        Self {
            nibbles,
            gamma: f32::from_le_bytes(gamma_bytes),
            norm: f32::from_le_bytes(norm_bytes),
        }
    }

    /// Expected packed size in bytes for a given dimension.
    ///
    /// Internally pads to next power of 2 for TQ4 encoding.
    pub fn packed_size(dimension: usize) -> usize {
        let d_pad = dimension.next_power_of_two();
        // d_pad/2 nibble bytes + 4 gamma + 4 norm
        d_pad / 2 + 8
    }
}

/// TQ4 quantizer state, shared across all vectors in an index.
///
/// Contains the rotation matrix (RHT) and QJL projection, both
/// regenerated deterministically from seeds on every startup.
pub struct Tq4Quantizer {
    rotation: RotationMatrix,
    qjl: QjlProjection,
    /// Original vector dimension d.
    dimension: usize,
    /// Padded dimension d_pad = d.next_power_of_two().
    padded_dim: usize,
    /// 1/√d for codebook scaling.
    inv_sqrt_d: f32,
}

impl Tq4Quantizer {
    /// Create a new TQ4 quantizer for the given dimension and seed pair.
    ///
    /// The rotation and QJL matrices are generated deterministically;
    /// same seeds always produce the same quantizer state.
    pub fn new(dimension: usize, rotation_seed: u64, qjl_seed: u64) -> Self {
        let rotation = RotationMatrix::from_seed(dimension, rotation_seed);
        let d_pad = rotation.padded_dim();
        // QJL operates in padded space (d_pad dimensions) for norm preservation.
        let qjl = QjlProjection::from_seed(d_pad, qjl_seed);
        // Codebook scaling uses 1/√d (original dimension).
        // Even though forward_padded outputs d_pad coordinates, the energy is
        // concentrated in the first d components (zero-padded input), so the
        // effective per-coordinate variance aligns closer to 1/d than 1/d_pad.
        let inv_sqrt_d = 1.0 / (dimension as f32).sqrt();

        Self {
            rotation,
            qjl,
            dimension,
            padded_dim: d_pad,
            inv_sqrt_d,
        }
    }

    /// Original vector dimension d.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Padded dimension d_pad = d.next_power_of_two().
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// 1/√d scaling factor.
    pub fn inv_sqrt_d(&self) -> f32 {
        self.inv_sqrt_d
    }

    /// Quantize a single f32 vector to TQ4 representation.
    ///
    /// Steps:
    /// 1. Compute and store ‖x‖
    /// 2. Normalize to unit norm: x̂ = x / ‖x‖
    /// 3. Rotate: y = Π · x̂
    /// 4. Scalar quantize each y_j with 3-bit codebook
    /// 5. Compute residual r = y − ỹ, γ = ‖r‖
    /// 6. QJL: qjl_signs = sign(S · (r/γ))
    /// 7. Interleave MSE indices + QJL signs into 4-bit nibbles
    pub fn quantize(&self, vector: &[f32]) -> Tq4CompressedVector {
        let d = self.dimension;
        let d_pad = self.padded_dim;
        debug_assert_eq!(vector.len(), d);

        // Step 1: Original norm (SIMD-accelerated via NumKong)
        let norm: f32 = (f32::dot(vector, vector).unwrap_or(0.0) as f32).sqrt();

        // Short-circuit for zero vector
        if norm == 0.0 {
            return Tq4CompressedVector {
                nibbles: vec![0u8; d_pad / 2],
                gamma: 0.0,
                norm: 0.0,
            };
        }

        // Step 2: Normalize to unit norm
        let inv_norm = 1.0 / norm;
        let normalized: Vec<f32> = vector.iter().map(|v| v * inv_norm).collect();

        // Step 3: Rotate via RHT (Fix B: full padded output, no truncation)
        let mut rotated = vec![0.0f32; d_pad];
        self.rotation.forward_padded(&normalized, &mut rotated);

        // Step 4: Scalar quantize to raw 3-bit indices (one byte per coordinate)
        let mse_raw = codebook::encode_vector_3bit_raw(&rotated, self.inv_sqrt_d);

        // Step 5: Dequantize MSE and compute residual
        let dequantized = codebook::decode_vector_3bit_raw(&mse_raw, self.inv_sqrt_d);
        let residual: Vec<f32> = rotated
            .iter()
            .zip(dequantized.iter())
            .map(|(y, y_hat)| y - y_hat)
            .collect();
        let gamma: f32 = (f32::dot(&residual, &residual).unwrap_or(0.0) as f32).sqrt();

        // Step 6: QJL sign bits on normalized residual
        let qjl_signs = if gamma > 0.0 {
            let inv_gamma = 1.0 / gamma;
            let normalized_residual: Vec<f32> = residual.iter().map(|r| r * inv_gamma).collect();
            self.qjl.quantize(&normalized_residual)
        } else {
            vec![0u8; d_pad.div_ceil(8)]
        };

        // Step 7: Interleave into 4-bit nibbles (bit 3 = QJL sign, bits 2-0 = MSE)
        let nibbles = codebook::interleave_nibbles(&mse_raw, &qjl_signs, d_pad);

        Tq4CompressedVector {
            nibbles,
            gamma,
            norm,
        }
    }

    /// Dequantize TQ4 back to approximate f32 vector (for testing/debugging).
    ///
    /// Reconstructs: x̃ = ‖x‖ · Π⁻¹ · ỹ
    /// where ỹ is the codebook reconstruction extracted from nibbles.
    /// The QJL residual correction is not applied (MSE-only reconstruction).
    pub fn dequantize_f32(&self, compressed: &Tq4CompressedVector) -> Vec<f32> {
        let d = self.dimension;
        let d_pad = self.padded_dim;

        // Extract MSE indices from interleaved nibbles and decode to centroids
        let mut rotated_approx = vec![0.0f32; d_pad];
        for byte_idx in 0..compressed.nibbles.len() {
            let dim_even = byte_idx * 2;
            let dim_odd = byte_idx * 2 + 1;
            let hi = (compressed.nibbles[byte_idx] >> 4) & 0x07;
            let lo = compressed.nibbles[byte_idx] & 0x07;
            rotated_approx[dim_even] = codebook::decode_scalar_3bit(hi, self.inv_sqrt_d);
            rotated_approx[dim_odd] = codebook::decode_scalar_3bit(lo, self.inv_sqrt_d);
        }

        // Inverse rotate back to original space (d_pad → d)
        let mut result = vec![0.0f32; d];
        self.rotation.inverse_padded(&rotated_approx, &mut result);

        // Rescale by original norm
        for v in &mut result {
            *v *= compressed.norm;
        }

        result
    }

    /// Access the rotation matrix (for prepare_query).
    pub fn rotation(&self) -> &RotationMatrix {
        &self.rotation
    }

    /// Access the QJL projection (for prepare_query).
    pub fn qjl(&self) -> &QjlProjection {
        &self.qjl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_quant::qjl::fill_standard_normal;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn pack_unpack_roundtrip() {
        let d = 768;
        let quantizer = Tq4Quantizer::new(d, 42, 137);

        let mut rng = StdRng::seed_from_u64(12345);
        let mut vector = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut vector);
        let compressed = quantizer.quantize(&vector);

        let packed = compressed.pack();
        assert_eq!(packed.len(), Tq4CompressedVector::packed_size(d));

        let unpacked = Tq4CompressedVector::unpack(&packed, d);
        assert_eq!(compressed.nibbles, unpacked.nibbles);
        assert!((compressed.gamma - unpacked.gamma).abs() < 1e-7);
        assert!((compressed.norm - unpacked.norm).abs() < 1e-7);
    }

    #[test]
    fn packed_size_formula() {
        for d in [768usize, 1024, 1536, 3072] {
            let d_pad = d.next_power_of_two();
            let expected = d_pad / 2 + 8;
            assert_eq!(Tq4CompressedVector::packed_size(d), expected);
        }
    }

    #[test]
    fn mse_distortion_bound() {
        let d = 768;
        let quantizer = Tq4Quantizer::new(d, 42, 137);
        let n_trials = 100;
        let mut total_mse = 0.0f64;

        let mut rng = StdRng::seed_from_u64(54321);
        for _ in 0..n_trials {
            // Random unit-norm vector
            let mut raw = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut raw);
            let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x: Vec<f32> = raw.iter().map(|v| v / norm).collect();

            let compressed = quantizer.quantize(&x);
            let reconstructed = quantizer.dequantize_f32(&compressed);

            // MSE computation (on unit-norm input, so output is also roughly unit-norm)
            let mse: f64 = x
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / d as f64;
            total_mse += mse;
        }

        let avg_mse = total_mse / n_trials as f64;
        // 3-bit Lloyd-Max MSE ≈ 0.009 per coordinate.
        // The dequantize (MSE-only, no QJL correction) should be below ~0.02.
        assert!(
            avg_mse < 0.03,
            "Average MSE too high: {avg_mse:.6} (expected < 0.03 for 3-bit codebook)"
        );
    }

    #[test]
    fn norm_preserved_in_compressed() {
        let d = 768;
        let quantizer = Tq4Quantizer::new(d, 42, 137);

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
        let quantizer = Tq4Quantizer::new(d, 42, 137);

        let zero = vec![0.0f32; d];
        let compressed = quantizer.quantize(&zero);
        assert_eq!(compressed.norm, 0.0);
        assert_eq!(compressed.gamma, 0.0);
    }
}
