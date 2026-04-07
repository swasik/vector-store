/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! Randomized Hadamard Transform (RHT) for fast, reproducible random rotation.
//!
//! The RHT maps any unit-norm vector to a vector whose coordinates follow
//! approximately Beta(d/2, 1/2) distribution, as required by TurboQuant.
//!
//! Π = (1/√d_pad) · H_{d_pad} · D
//!
//! - H_{d_pad}: Walsh-Hadamard matrix via O(d_pad log d_pad) butterfly
//! - D: diagonal of random ±1 signs from seeded PRNG
//! - 1/√d_pad normalization ensures ‖Π·x‖ = ‖x‖
//!
//! When d is not a power of 2, the input is zero-padded to d_pad and the
//! result is truncated back to d coordinates. The extra coordinates are
//! discarded and do not affect storage or compression.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Randomized Hadamard Transform state.
///
/// Shared across all vectors in an index; regenerated deterministically
/// from `seed` on every restart.
pub struct RotationMatrix {
    /// Diagonal D: +1.0 or -1.0 per coordinate, length = padded_dim.
    signs: Vec<f32>,
    /// Original vector dimension d.
    dimension: usize,
    /// Next power of 2 ≥ dimension (e.g., 1024 for d=768).
    padded_dim: usize,
    /// 1/√padded_dim, applied after the unnormalized butterfly.
    inv_sqrt_d_pad: f32,
}

impl RotationMatrix {
    /// Create from a deterministic seed. Same seed + dimension = same rotation.
    pub fn from_seed(dimension: usize, seed: u64) -> Self {
        let padded_dim = dimension.next_power_of_two();
        let inv_sqrt_d_pad = 1.0 / (padded_dim as f32).sqrt();

        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();

        Self {
            signs,
            dimension,
            padded_dim,
            inv_sqrt_d_pad,
        }
    }

    /// Original vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Padded dimension (power of 2).
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Forward transform: y = (1/√d_pad) · H · D · x
    ///
    /// Input `x` has length `dimension`. The result (also length `dimension`)
    /// is written back into the provided output slice. The padded-dimension
    /// working buffer is allocated internally.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dimension);
        debug_assert_eq!(out.len(), self.dimension);

        // Zero-padded working buffer
        let mut buf = vec![0.0f32; self.padded_dim];
        // Apply diagonal D: buf[i] = signs[i] * x[i] for i < d, rest stays 0
        for i in 0..self.dimension {
            buf[i] = self.signs[i] * x[i];
        }

        // Unnormalized Walsh-Hadamard butterfly
        hadamard_transform(&mut buf);

        // Normalize by 1/√d_pad and truncate to first d coordinates
        for i in 0..self.dimension {
            out[i] = buf[i] * self.inv_sqrt_d_pad;
        }
    }

    /// Forward transform in-place (allocates a working buffer internally).
    pub fn forward_in_place(&self, data: &mut [f32]) {
        debug_assert_eq!(data.len(), self.dimension);
        let mut buf = vec![0.0f32; self.padded_dim];
        for i in 0..self.dimension {
            buf[i] = self.signs[i] * data[i];
        }
        hadamard_transform(&mut buf);
        for i in 0..self.dimension {
            data[i] = buf[i] * self.inv_sqrt_d_pad;
        }
    }

    /// Forward transform without truncation: y = (1/√d_pad) · H · D · x
    ///
    /// Input `x` has length `dimension`. The full `padded_dim` output is
    /// written into `out` (no truncation). This preserves the L2 norm exactly.
    pub fn forward_padded(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dimension);
        debug_assert_eq!(out.len(), self.padded_dim);

        let mut buf = vec![0.0f32; self.padded_dim];
        for i in 0..self.dimension {
            buf[i] = self.signs[i] * x[i];
        }

        hadamard_transform(&mut buf);

        for i in 0..self.padded_dim {
            out[i] = buf[i] * self.inv_sqrt_d_pad;
        }
    }

    /// Inverse of `forward_padded`: D · (1/√d_pad) · H · y
    ///
    /// Input `y` has length `padded_dim`. Output has length `dimension`
    /// (the extra padded coordinates are discarded after inverse Hadamard).
    pub fn inverse_padded(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.padded_dim);
        debug_assert_eq!(out.len(), self.dimension);

        let mut buf = y.to_vec();
        hadamard_transform(&mut buf);

        for i in 0..self.dimension {
            out[i] = buf[i] * self.inv_sqrt_d_pad * self.signs[i];
        }
    }

    /// Inverse transform: x = D · (1/√d_pad) · H · y
    ///
    /// Since H is symmetric and orthogonal (H = H^T, H·H = d_pad·I),
    /// and D is self-inverse (D·D = I), the inverse is: D · H · y / d_pad.
    /// Equivalently: apply (1/√d_pad)·H then multiply by D (element-wise).
    pub fn inverse(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.dimension);
        debug_assert_eq!(out.len(), self.dimension);

        let mut buf = vec![0.0f32; self.padded_dim];
        buf[..self.dimension].copy_from_slice(y);

        // Apply normalized Hadamard
        hadamard_transform(&mut buf);

        // Scale by 1/√d_pad and multiply by D (self-inverse diagonal)
        for i in 0..self.dimension {
            out[i] = buf[i] * self.inv_sqrt_d_pad * self.signs[i];
        }
    }

    /// Inverse transform in-place.
    pub fn inverse_in_place(&self, data: &mut [f32]) {
        debug_assert_eq!(data.len(), self.dimension);
        let mut buf = vec![0.0f32; self.padded_dim];
        buf[..self.dimension].copy_from_slice(data);
        hadamard_transform(&mut buf);
        for i in 0..self.dimension {
            data[i] = buf[i] * self.inv_sqrt_d_pad * self.signs[i];
        }
    }
}

/// In-place unnormalized Walsh-Hadamard transform via iterative butterfly.
///
/// Operates on a power-of-2 length slice. Each stage butterflies pairs of
/// elements at distance `half`: `(a, b) → (a + b, a − b)`. Total cost:
/// O(n log n) additions/subtractions where n = data.len().
///
/// The result is the *unnormalized* Hadamard transform; multiply by
/// `1/√n` after calling to obtain an orthonormal transform.
fn hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "Hadamard requires power-of-2 length");

    let mut half = 1;
    while half < n {
        for i in (0..n).step_by(half * 2) {
            for j in i..i + half {
                let a = data[j];
                let b = data[j + half];
                data[j] = a + b;
                data[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_preservation() {
        // Use power-of-2 dimension for exact norm preservation
        let d = 1024;
        let rot = RotationMatrix::from_seed(d, 42);

        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10 {
            let x: Vec<f32> = (0..d).map(|_| rng.random::<f32>() - 0.5).collect();
            let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

            let mut y = vec![0.0f32; d];
            rot.forward(&x, &mut y);
            let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

            let rel_err = (norm_x - norm_y).abs() / norm_x;
            assert!(
                rel_err < 1e-5,
                "Norm not preserved: ‖x‖={norm_x}, ‖y‖={norm_y}, rel_err={rel_err}"
            );
        }
    }

    /// Non-power-of-2 dimensions: norm may decrease due to truncation.
    /// Verify it's reasonably close (within d/d_pad).
    #[test]
    fn norm_approximate_non_pow2() {
        let d = 768;
        let rot = RotationMatrix::from_seed(d, 42);
        let d_pad = rot.padded_dim(); // 1024

        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10 {
            let x: Vec<f32> = (0..d).map(|_| rng.random::<f32>() - 0.5).collect();
            let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

            let mut y = vec![0.0f32; d];
            rot.forward(&x, &mut y);
            let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

            // Expected energy ratio: ≈ d/d_pad = 768/1024 = 0.75, ‖y‖ ≈ √(d/d_pad) · ‖x‖
            let expected_ratio = (d as f32 / d_pad as f32).sqrt();
            let actual_ratio = norm_y / norm_x;
            let err = (actual_ratio - expected_ratio).abs() / expected_ratio;
            assert!(
                err < 0.15,
                "Non-pow2 norm ratio off: expected ≈{expected_ratio}, got {actual_ratio}, err={err}"
            );
        }
    }

    #[test]
    fn invertibility() {
        // Use power-of-2 dimension for exact invertibility
        let d = 1024;
        let rot = RotationMatrix::from_seed(d, 42);

        let mut rng = StdRng::seed_from_u64(54321);
        for _ in 0..10 {
            let x: Vec<f32> = (0..d).map(|_| rng.random::<f32>() - 0.5).collect();

            let mut y = vec![0.0f32; d];
            rot.forward(&x, &mut y);

            let mut x_recovered = vec![0.0f32; d];
            rot.inverse(&y, &mut x_recovered);

            let max_err: f32 = x
                .iter()
                .zip(x_recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max_err < 1e-4, "Inverse failed: max_err={max_err}");
        }
    }

    #[test]
    fn deterministic_seed() {
        let d = 256;
        let rot1 = RotationMatrix::from_seed(d, 42);
        let rot2 = RotationMatrix::from_seed(d, 42);

        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();

        let mut y1 = vec![0.0f32; d];
        let mut y2 = vec![0.0f32; d];
        rot1.forward(&x, &mut y1);
        rot2.forward(&x, &mut y2);

        assert_eq!(y1, y2, "Same seed should produce identical results");
    }

    #[test]
    fn different_seeds_differ() {
        let d = 256;
        let rot1 = RotationMatrix::from_seed(d, 42);
        let rot2 = RotationMatrix::from_seed(d, 99);

        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.01).collect();

        let mut y1 = vec![0.0f32; d];
        let mut y2 = vec![0.0f32; d];
        rot1.forward(&x, &mut y1);
        rot2.forward(&x, &mut y2);

        assert_ne!(y1, y2, "Different seeds should produce different results");
    }

    #[test]
    fn power_of_two_dimension() {
        // d_pad == d when dimension is already power-of-2
        let d = 1024;
        let rot = RotationMatrix::from_seed(d, 42);
        assert_eq!(rot.padded_dim(), d);

        let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let mut y = vec![0.0f32; d];
        rot.forward(&x, &mut y);

        let mut recovered = vec![0.0f32; d];
        rot.inverse(&y, &mut recovered);

        let max_err: f32 = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "Roundtrip failed: max_err={max_err}");
    }

    /// Verify that rotated unit-norm vectors have coordinates with mean ≈ 0
    /// and variance ≈ 1/d_pad, consistent with the Beta(d/2, 1/2) → N(0, 1/d) convergence.
    #[test]
    fn distributional_properties() {
        let d: usize = 768;
        let d_pad = d.next_power_of_two(); // 1024
        let rot = RotationMatrix::from_seed(d, 42);

        let mut rng = StdRng::seed_from_u64(99999);
        let n_vectors = 200;
        let mut all_means = Vec::new();
        let mut all_vars = Vec::new();

        for _ in 0..n_vectors {
            // Generate random unit-norm vector
            let raw: Vec<f32> = (0..d).map(|_| rng.random::<f32>() - 0.5).collect();
            let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            let x: Vec<f32> = raw.iter().map(|v| v / norm).collect();

            let mut y = vec![0.0f32; d];
            rot.forward(&x, &mut y);

            let mean: f32 = y.iter().sum::<f32>() / d as f32;
            let var: f32 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;

            all_means.push(mean);
            all_vars.push(var);
        }

        let avg_mean: f32 = all_means.iter().sum::<f32>() / n_vectors as f32;
        let avg_var: f32 = all_vars.iter().sum::<f32>() / n_vectors as f32;
        let expected_var = 1.0 / d_pad as f32;

        assert!(
            avg_mean.abs() < 0.01,
            "Average mean should be ≈0, got {avg_mean}"
        );
        assert!(
            (avg_var - expected_var).abs() < expected_var * 0.3,
            "Average variance should be ≈1/d_pad={expected_var}, got {avg_var}"
        );
    }
}
