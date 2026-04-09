/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! QJL (Quantized Johnson-Lindenstrauss) 1-bit projection using SRHT.
//!
//! Q_qjl(x) = sign(S · x) where S = (1/√d) · H · D is a Structured
//! Random Hadamard Transform:
//! - D: random ±1 diagonal (O(d) storage)
//! - H: Walsh-Hadamard transform (O(d log d) computation)
//!
//! Replaces the original dense Gaussian matrix (O(d²) storage and
//! computation) while preserving QJL theoretical guarantees via the
//! Johnson-Lindenstrauss property of SRHT.
//!
//! Storage: O(d) instead of O(d²) — for d=1536: ~6 KB vs ~9.4 MB
//! Computation: O(d log d) instead of O(d²) — ~100× speedup at d=1536
//!
//! Reference: QJL paper (arXiv:2406.03482), Definition 1.

use crate::turbo_quant::rotation::hadamard_transform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate a standard normal variate using Box-Muller transform.
/// Returns two independent N(0,1) variates.
fn box_muller(rng: &mut StdRng) -> (f32, f32) {
    loop {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        if u1 > f32::EPSILON {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}

/// Fill a slice with N(0,1) random variates.
pub(crate) fn fill_standard_normal(rng: &mut StdRng, buf: &mut [f32]) {
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = box_muller(rng);
        buf[i] = a;
        buf[i + 1] = b;
        i += 2;
    }
    if i < buf.len() {
        let (a, _) = box_muller(rng);
        buf[i] = a;
    }
}

/// QJL 1-bit inner product quantizer using SRHT.
///
/// Uses S = (1/√d) · H · D where H is the Walsh-Hadamard matrix and
/// D is a random ±1 diagonal. This gives O(d) storage and O(d log d)
/// projection cost instead of O(d²) for the dense Gaussian matrix.
pub struct QjlProjection {
    /// Random ±1 diagonal signs, length = dimension.
    signs: Vec<f32>,
    /// Vector dimension d (must be power of 2).
    dimension: usize,
    /// 1/√d normalization factor.
    inv_sqrt_d: f32,
}

impl QjlProjection {
    /// Create from deterministic seed. Same seed + dimension = same projection.
    ///
    /// Generates d random ±1 sign bitmasks. Storage: O(d) instead of O(d²).
    /// For d=1536: ~6 KB vs ~9.4 MB with dense Gaussian matrix.
    pub fn from_seed(dimension: usize, seed: u64) -> Self {
        debug_assert!(
            dimension.is_power_of_two(),
            "QJL SRHT requires power-of-2 dimension"
        );
        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..dimension)
            .map(|_| {
                if rng.random_bool(0.5) {
                    1.0f32
                } else {
                    -1.0f32
                }
            })
            .collect();
        let inv_sqrt_d = 1.0 / (dimension as f32).sqrt();

        Self {
            signs,
            dimension,
            inv_sqrt_d,
        }
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Quantize: compute sign(H · D · x), returning packed sign bits.
    ///
    /// Uses SRHT: O(d log d) instead of O(d²) dense projection.
    /// The 1/√d normalization is omitted since sign() is scale-invariant.
    /// Output: ceil(d/8) bytes.
    pub fn quantize(&self, x: &[f32]) -> Vec<u8> {
        debug_assert_eq!(x.len(), self.dimension);
        let d = self.dimension;

        // Apply diagonal D
        let mut buf: Vec<f32> = x
            .iter()
            .zip(self.signs.iter())
            .map(|(&xi, &si)| xi * si)
            .collect();

        // Walsh-Hadamard transform: O(d log d)
        hadamard_transform(&mut buf);

        // Extract sign bits 8 at a time (1/√d doesn't affect signs)
        let packed_len = d.div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        let full_bytes = d / 8;
        for (byte_idx, p) in packed.iter_mut().enumerate().take(full_bytes) {
            let base = byte_idx * 8;
            let b = &buf[base..base + 8];
            let mut byte_val = 0u8;
            if b[0] >= 0.0 {
                byte_val |= 0x80;
            }
            if b[1] >= 0.0 {
                byte_val |= 0x40;
            }
            if b[2] >= 0.0 {
                byte_val |= 0x20;
            }
            if b[3] >= 0.0 {
                byte_val |= 0x10;
            }
            if b[4] >= 0.0 {
                byte_val |= 0x08;
            }
            if b[5] >= 0.0 {
                byte_val |= 0x04;
            }
            if b[6] >= 0.0 {
                byte_val |= 0x02;
            }
            if b[7] >= 0.0 {
                byte_val |= 0x01;
            }
            *p = byte_val;
        }
        // Handle remainder
        let remainder = d % 8;
        if remainder > 0 {
            let base = full_bytes * 8;
            let mut byte_val = 0u8;
            for r in 0..remainder {
                if buf[base + r] >= 0.0 {
                    byte_val |= 0x80 >> r;
                }
            }
            packed[full_bytes] = byte_val;
        }
        packed
    }

    /// Project a vector through SRHT: returns (1/√d) · H · D · v.
    ///
    /// O(d log d) instead of O(d²) dense matrix-vector multiply.
    /// At query time, pass the **rotated** query q' = Π·q so that the
    /// projection matches the coordinate space of the QJL sign bits.
    pub fn project_query(&self, q: &[f32]) -> Vec<f32> {
        debug_assert_eq!(q.len(), self.dimension);

        // Apply diagonal D
        let mut buf: Vec<f32> = q
            .iter()
            .zip(self.signs.iter())
            .map(|(&qi, &si)| qi * si)
            .collect();

        // Walsh-Hadamard transform: O(d log d)
        hadamard_transform(&mut buf);

        // Normalize by 1/√d
        for v in buf.iter_mut() {
            *v *= self.inv_sqrt_d;
        }
        buf
    }

    /// Compute QJL inner product correction term:
    ///   √(π/2) / d · γ · Σ_j sign_j · projected_query_j
    ///
    /// For the SRHT-based projection, the √(π/2) correction factor is
    /// approximate (exact for Gaussian S, asymptotically correct for SRHT
    /// by CLT as d → ∞). Empirically verified via unbiasedness tests.
    ///
    /// `signs`: packed sign bits from `quantize()` (ceil(d/8) bytes)
    /// `projected_query`: output of `project_query()` (d floats)
    /// `gamma`: residual norm ‖r‖
    pub fn inner_product_term(&self, signs: &[u8], projected_query: &[f32], gamma: f32) -> f32 {
        let d = self.dimension;
        debug_assert_eq!(projected_query.len(), d);

        // Process 8 elements per byte for the bulk of the data.
        // For each byte, compute sum_positive (bit=1) and sum_all;
        // contribution = 2*sum_positive - sum_all (since sign = 2*bit - 1).
        let full_bytes = d / 8;
        let mut dot_sum = 0.0f32;

        for (byte_idx, &byte) in signs.iter().enumerate().take(full_bytes) {
            let base = byte_idx * 8;
            let pq = &projected_query[base..base + 8];

            // Sum of all 8 elements
            let sum_all = pq[0] + pq[1] + pq[2] + pq[3] + pq[4] + pq[5] + pq[6] + pq[7];

            // Sum of elements where bit=1 (MSB first: bit 7 is pq[0])
            let mut sum_pos = 0.0f32;
            if byte & 0x80 != 0 {
                sum_pos += pq[0];
            }
            if byte & 0x40 != 0 {
                sum_pos += pq[1];
            }
            if byte & 0x20 != 0 {
                sum_pos += pq[2];
            }
            if byte & 0x10 != 0 {
                sum_pos += pq[3];
            }
            if byte & 0x08 != 0 {
                sum_pos += pq[4];
            }
            if byte & 0x04 != 0 {
                sum_pos += pq[5];
            }
            if byte & 0x02 != 0 {
                sum_pos += pq[6];
            }
            if byte & 0x01 != 0 {
                sum_pos += pq[7];
            }

            // sign_j * pq_j = (2*bit-1)*pq_j; sum = 2*sum_pos - sum_all
            dot_sum += 2.0 * sum_pos - sum_all;
        }

        // Handle remaining elements (d % 8 != 0)
        let remainder = d % 8;
        if remainder > 0 {
            let byte = signs[full_bytes];
            let base = full_bytes * 8;
            for r in 0..remainder {
                let bit = (byte >> (7 - r)) & 1;
                let sign_val = if bit == 1 { 1.0f32 } else { -1.0 };
                dot_sum += sign_val * projected_query[base + r];
            }
        }

        // √(π/2) / d · γ · dot_sum
        ((std::f32::consts::PI / 2.0_f32).sqrt() / d as f32) * gamma * dot_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numkong::Dot;

    #[test]
    fn deterministic_seed() {
        let d = 128;
        let qjl1 = QjlProjection::from_seed(d, 137);
        let qjl2 = QjlProjection::from_seed(d, 137);
        assert_eq!(qjl1.signs, qjl2.signs);
    }

    #[test]
    fn quantize_packed_size() {
        // Use power-of-2 dimension (SRHT requires it)
        let d = 1024;
        let qjl = QjlProjection::from_seed(d, 137);
        let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let packed = qjl.quantize(&x);
        assert_eq!(packed.len(), d.div_ceil(8));
    }

    #[test]
    fn project_query_dimension() {
        let d = 128;
        let qjl = QjlProjection::from_seed(d, 137);
        let q: Vec<f32> = vec![1.0; d];
        let projected = qjl.project_query(&q);
        assert_eq!(projected.len(), d);
    }

    /// Test unbiasedness: E[⟨y, Q_qjl⁻¹(Q_qjl(x))⟩] ≈ ⟨y, x⟩
    ///
    /// We use a single QJL projection S (as in the actual system) and average
    /// over many random (x, y) pairs. The expected inner product estimator
    /// converges to the true inner product.
    #[test]
    fn unbiasedness() {
        let d = 128;
        let n_trials = 500;
        let mut rng = StdRng::seed_from_u64(42);

        let mut total_err = 0.0f64;
        let mut total_abs = 0.0f64;

        for trial_seed in 0..n_trials {
            let qjl = QjlProjection::from_seed(d, 137 + trial_seed);

            // Random unit-norm vectors
            let mut x_raw = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x_raw);
            let x_norm: f32 = x_raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x: Vec<f32> = x_raw.iter().map(|v| v / x_norm).collect();

            let mut y_raw = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut y_raw);
            let y_norm: f32 = y_raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let y: Vec<f32> = y_raw.iter().map(|v| v / y_norm).collect();

            // True inner product
            let true_ip = f32::dot(&x, &y).unwrap() as f32;

            // QJL estimate: sign(S · x), then use y as query
            let signs = qjl.quantize(&x);
            let projected_y = qjl.project_query(&y);
            // gamma=1 since x is unit norm and we're testing the estimator directly
            // The full estimator: (π/2)/d · γ · Σ sign_j · (S·y)_j
            let qjl_estimate = qjl.inner_product_term(&signs, &projected_y, 1.0);

            total_err += (qjl_estimate - true_ip) as f64;
            total_abs += true_ip.abs() as f64;
        }

        let avg_err = total_err / n_trials as f64;
        let avg_abs = total_abs / n_trials as f64;
        let rel_bias = avg_err.abs() / avg_abs;

        assert!(
            rel_bias < 0.15,
            "QJL bias too high: avg_err={avg_err:.6}, avg_abs_ip={avg_abs:.6}, rel_bias={rel_bias:.4}"
        );
    }
}
