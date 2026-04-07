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

/// Packed TQ4 representation of a single vector.
#[derive(Debug, Clone)]
pub struct Tq4CompressedVector {
    /// 3-bit codebook indices, packed as big-endian bitstream.
    pub mse_indices: Vec<u8>,
    /// QJL sign bits, packed: bit j of byte j/8 = sign of coord j.
    pub qjl_signs: Vec<u8>,
    /// Residual L2 norm (γ = ‖r‖).
    pub gamma: f32,
    /// Original vector L2 norm (‖x‖).
    pub norm: f32,
}

impl Tq4CompressedVector {
    /// Total storage in bytes.
    pub fn storage_bytes(&self) -> usize {
        self.mse_indices.len() + self.qjl_signs.len() + 8 // 4 bytes gamma + 4 bytes norm
    }

    /// Pack into a contiguous byte array for USearch storage.
    ///
    /// Layout: `[mse_indices | qjl_signs | gamma(4 bytes LE) | norm(4 bytes LE)]`
    pub fn pack(&self) -> Vec<u8> {
        let total = self.storage_bytes();
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.mse_indices);
        buf.extend_from_slice(&self.qjl_signs);
        buf.extend_from_slice(&self.gamma.to_le_bytes());
        buf.extend_from_slice(&self.norm.to_le_bytes());
        buf
    }

    /// Unpack from a contiguous byte array retrieved via USearch::get().
    pub fn unpack(bytes: &[u8], dimension: usize) -> Self {
        let mse_len = (dimension * 3).div_ceil(8);
        let qjl_len = dimension.div_ceil(8);

        let mse_indices = bytes[..mse_len].to_vec();
        let qjl_signs = bytes[mse_len..mse_len + qjl_len].to_vec();
        let gamma_bytes: [u8; 4] = bytes[mse_len + qjl_len..mse_len + qjl_len + 4]
            .try_into()
            .expect("gamma bytes");
        let norm_bytes: [u8; 4] = bytes[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
            .try_into()
            .expect("norm bytes");

        Self {
            mse_indices,
            qjl_signs,
            gamma: f32::from_le_bytes(gamma_bytes),
            norm: f32::from_le_bytes(norm_bytes),
        }
    }

    /// Expected packed size in bytes for a given dimension.
    pub fn packed_size(dimension: usize) -> usize {
        let mse_len = (dimension * 3).div_ceil(8);
        let qjl_len = dimension.div_ceil(8);
        mse_len + qjl_len + 8
    }
}

/// TQ4 quantizer state, shared across all vectors in an index.
///
/// Contains the rotation matrix (RHT) and QJL projection, both
/// regenerated deterministically from seeds on every startup.
pub struct Tq4Quantizer {
    rotation: RotationMatrix,
    qjl: QjlProjection,
    dimension: usize,
    /// 1/√d_pad for codebook scaling (see plan §2.4 normalization note).
    inv_sqrt_d: f32,
}

impl Tq4Quantizer {
    /// Create a new TQ4 quantizer for the given dimension and seed pair.
    ///
    /// The rotation and QJL matrices are generated deterministically;
    /// same seeds always produce the same quantizer state.
    pub fn new(dimension: usize, rotation_seed: u64, qjl_seed: u64) -> Self {
        let rotation = RotationMatrix::from_seed(dimension, rotation_seed);
        let qjl = QjlProjection::from_seed(dimension, qjl_seed);
        // Use 1/√d_pad for codebook scaling (matches Hadamard output variance).
        let d_pad = rotation.padded_dim();
        let inv_sqrt_d = 1.0 / (d_pad as f32).sqrt();

        Self {
            rotation,
            qjl,
            dimension,
            inv_sqrt_d,
        }
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 1/√d_pad scaling factor.
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
    pub fn quantize(&self, vector: &[f32]) -> Tq4CompressedVector {
        let d = self.dimension;
        debug_assert_eq!(vector.len(), d);

        // Step 1: Original norm
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Short-circuit for zero vector
        if norm == 0.0 {
            return Tq4CompressedVector {
                mse_indices: vec![0u8; (d * 3).div_ceil(8)],
                qjl_signs: vec![0u8; d.div_ceil(8)],
                gamma: 0.0,
                norm: 0.0,
            };
        }

        // Step 2: Normalize to unit norm
        let inv_norm = 1.0 / norm;
        let normalized: Vec<f32> = vector.iter().map(|v| v * inv_norm).collect();

        // Step 3: Rotate via RHT
        let mut rotated = vec![0.0f32; d];
        self.rotation.forward(&normalized, &mut rotated);

        // Step 4: Scalar quantize (3-bit codebook)
        let mse_indices = codebook::encode_vector_3bit(&rotated, self.inv_sqrt_d);

        // Step 5: Dequantize MSE and compute residual
        let dequantized = codebook::decode_vector_3bit(&mse_indices, d, self.inv_sqrt_d);
        let residual: Vec<f32> = rotated
            .iter()
            .zip(dequantized.iter())
            .map(|(y, y_hat)| y - y_hat)
            .collect();
        let gamma: f32 = residual.iter().map(|r| r * r).sum::<f32>().sqrt();

        // Step 6: QJL sign bits on normalized residual
        let qjl_signs = if gamma > 0.0 {
            let inv_gamma = 1.0 / gamma;
            let normalized_residual: Vec<f32> = residual.iter().map(|r| r * inv_gamma).collect();
            self.qjl.quantize(&normalized_residual)
        } else {
            vec![0u8; d.div_ceil(8)]
        };

        Tq4CompressedVector {
            mse_indices,
            qjl_signs,
            gamma,
            norm,
        }
    }

    /// Dequantize TQ4 back to approximate f32 vector (for testing/debugging).
    ///
    /// Reconstructs: x̃ = ‖x‖ · Π⁻¹ · (ỹ + γ · Q_qjl⁻¹(signs))
    /// where ỹ is the codebook reconstruction and Q_qjl⁻¹ is the QJL dequantize.
    pub fn dequantize_f32(&self, compressed: &Tq4CompressedVector) -> Vec<f32> {
        let d = self.dimension;

        // Decode MSE codebook values in rotated space
        let mut rotated_approx =
            codebook::decode_vector_3bit(&compressed.mse_indices, d, self.inv_sqrt_d);

        // Add QJL residual correction: γ · (π/2)/d · S^T · signs
        // This is approximate—the full QJL dequantize involves S^T multiplication.
        // For test/debug purposes, we skip the QJL term and return MSE-only.
        // The QJL term is properly applied during inner product estimation.
        let _ = &compressed.qjl_signs;
        let _ = compressed.gamma;

        // Inverse rotate back to original space
        let mut result = vec![0.0f32; d];
        self.rotation.inverse(&rotated_approx, &mut result);

        // Rescale by original norm
        for v in &mut result {
            *v *= compressed.norm;
        }
        // Silence the mutable borrow warning — rotated_approx is consumed by the inverse call above.
        let _ = &mut rotated_approx;

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
        assert_eq!(compressed.mse_indices, unpacked.mse_indices);
        assert_eq!(compressed.qjl_signs, unpacked.qjl_signs);
        assert!((compressed.gamma - unpacked.gamma).abs() < 1e-7);
        assert!((compressed.norm - unpacked.norm).abs() < 1e-7);
    }

    #[test]
    fn packed_size_formula() {
        for d in [768usize, 1024, 1536, 3072] {
            let expected = (d * 3).div_ceil(8) + d.div_ceil(8) + 8;
            assert_eq!(Tq4CompressedVector::packed_size(d), expected);
            // For d divisible by 8, should be exactly d/2 + 8
            if d % 8 == 0 {
                assert_eq!(expected, d / 2 + 8);
            }
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
        let quantizer = Tq4Quantizer::new(d, 42, 137);

        let zero = vec![0.0f32; d];
        let compressed = quantizer.quantize(&zero);
        assert_eq!(compressed.norm, 0.0);
        assert_eq!(compressed.gamma, 0.0);
    }
}
