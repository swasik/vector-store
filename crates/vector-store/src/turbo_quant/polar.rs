/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! Recursive polar transformation for PolarQuant vector quantization.
//!
//! Converts a d-dimensional Cartesian vector into polar coordinates
//! (d−1 angles + 1 radius) using a recursive binary factorization.
//! After RHT preconditioning the coordinates are approximately Gaussian,
//! which makes the angles at each recursion level follow a known
//! sin^{2^{ℓ-1}-1}(2ψ) distribution concentrated around π/4.
//!
//! Reference: PolarQuant paper (arXiv:2502.02617), Section 3.1, Definition 1.

use std::f32::consts::PI;

/// Convert a d-dimensional Cartesian vector to polar representation.
///
/// The recursive polar transform groups pairs of values and computes:
/// - Level 1: ψ_j = atan2(x_{2j}, x_{2j-1}) for j = 0..d/2  (d/2 angles in [0, 2π))
/// - Level ℓ ≥ 2: ψ_j = atan2(‖right_half‖, ‖left_half‖)     (d/2^ℓ angles in [0, π/2])
///
/// Returns (angles, radius) where:
/// - `angles`: flat Vec of all d-1 angles, ordered level-1 first, then level-2, etc.
/// - `radius`: the final L2 norm (single scalar).
///
/// Requires d to be a power of 2.
pub fn cartesian_to_polar(x: &[f32]) -> (Vec<f32>, f32) {
    let d = x.len();
    debug_assert!(d.is_power_of_two(), "PolarQuant requires power-of-2 dimension");
    debug_assert!(d >= 2, "Dimension must be at least 2");

    let num_levels = d.trailing_zeros() as usize; // log2(d)
    let total_angles = d - 1;
    let mut angles = Vec::with_capacity(total_angles);

    // Level 1: pair consecutive coordinates, compute atan2 and norms
    let mut radii = Vec::with_capacity(d / 2);
    for j in 0..d / 2 {
        let x0 = x[2 * j];
        let x1 = x[2 * j + 1];
        // atan2(y, x) gives angle in (-π, π]; shift to [0, 2π)
        let angle = x1.atan2(x0);
        let angle = if angle < 0.0 { angle + 2.0 * PI } else { angle };
        angles.push(angle);
        radii.push((x0 * x0 + x1 * x1).sqrt());
    }

    // Levels 2..log2(d): recursively pair norms
    for _level in 2..=num_levels {
        let n = radii.len();
        let mut new_radii = Vec::with_capacity(n / 2);
        for j in 0..n / 2 {
            let r_left = radii[2 * j];
            let r_right = radii[2 * j + 1];
            // atan2(right, left) → angle in [0, π/2] for non-negative radii
            let angle = r_right.atan2(r_left);
            angles.push(angle);
            new_radii.push((r_left * r_left + r_right * r_right).sqrt());
        }
        radii = new_radii;
    }

    debug_assert_eq!(angles.len(), total_angles);
    debug_assert_eq!(radii.len(), 1);
    (angles, radii[0])
}

/// Convert polar representation back to Cartesian coordinates.
///
/// Inverse of `cartesian_to_polar`. Takes the flat angle vector and radius,
/// reconstructs the d-dimensional Cartesian vector via recursive expansion.
pub fn polar_to_cartesian(angles: &[f32], radius: f32, d: usize) -> Vec<f32> {
    debug_assert!(d.is_power_of_two());
    debug_assert_eq!(angles.len(), d - 1);

    let num_levels = d.trailing_zeros() as usize;

    // Start from the top level: single radius
    let mut values = vec![radius];

    // Expand from top level down to level 1
    // Process levels in reverse: level log2(d) .. 2 .. 1
    // Angles are stored level-1 first. Compute level offsets.
    let mut level_offsets = Vec::with_capacity(num_levels);
    let mut offset = 0usize;
    for level in 1..=num_levels {
        level_offsets.push(offset);
        offset += d / (1 << level);
    }

    // Expand from the top level (last in angle array) down to level 1
    for level in (1..=num_levels).rev() {
        let level_offset = level_offsets[level - 1];
        let n_angles = d / (1 << level);
        let mut expanded = Vec::with_capacity(n_angles * 2);

        for j in 0..n_angles {
            let r = values[j];
            let angle = angles[level_offset + j];

            if level == 1 {
                // Level 1: angle ∈ [0, 2π); reconstruct (x, y) = r·(cos θ, sin θ)
                expanded.push(r * angle.cos());
                expanded.push(r * angle.sin());
            } else {
                // Level ≥ 2: angle ∈ [0, π/2]; reconstruct (left, right) = r·(cos θ, sin θ)
                expanded.push(r * angle.cos());
                expanded.push(r * angle.sin());
            }
        }
        values = expanded;
    }

    debug_assert_eq!(values.len(), d);
    values
}

/// Compute the angle offset into the flat angle array for a given level.
///
/// Level 1 starts at offset 0, level 2 at d/2, level 3 at d/2 + d/4, etc.
#[inline]
pub fn level_offset(d: usize, level: usize) -> usize {
    debug_assert!(level >= 1);
    // Sum of d/2^k for k = 1..level-1
    // = d · (1 - 1/2^{level-1}) = d - d/2^{level-1}
    if level == 1 {
        0
    } else {
        d - d / (1 << (level - 1))
    }
}

/// Number of angles at a given level.
#[inline]
pub fn level_count(d: usize, level: usize) -> usize {
    d / (1 << level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_quant::qjl::fill_standard_normal;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn roundtrip_identity() {
        let d = 16;
        let x: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let (angles, radius) = cartesian_to_polar(&x);
        let reconstructed = polar_to_cartesian(&angles, radius, d);

        let max_err: f32 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-5,
            "Roundtrip error too large: {max_err}"
        );
    }

    #[test]
    fn roundtrip_random() {
        let d = 1024;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            let mut x = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x);

            let (angles, radius) = cartesian_to_polar(&x);
            assert_eq!(angles.len(), d - 1);

            let reconstructed = polar_to_cartesian(&angles, radius, d);
            let max_err: f32 = x
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-4,
                "Roundtrip error too large: {max_err}"
            );
        }
    }

    #[test]
    fn radius_equals_norm() {
        let d = 256;
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..20 {
            let mut x = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x);
            let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let (_, radius) = cartesian_to_polar(&x);
            let rel_err = (radius - norm).abs() / norm;
            assert!(
                rel_err < 1e-5,
                "Radius should equal L2 norm: radius={radius}, norm={norm}"
            );
        }
    }

    #[test]
    fn level_1_angles_in_range() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(42);
        let mut x = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut x);
        let (angles, _) = cartesian_to_polar(&x);

        // Level 1: first d/2 angles should be in [0, 2π)
        for &a in &angles[..d / 2] {
            assert!(
                (0.0..2.0 * PI).contains(&a),
                "Level 1 angle out of range: {a}"
            );
        }
    }

    #[test]
    fn higher_level_angles_in_range() {
        let d = 64;
        let mut rng = StdRng::seed_from_u64(42);
        let mut x = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut x);
        let (angles, _) = cartesian_to_polar(&x);

        // Levels ≥ 2: angles should be in [0, π/2]
        for &a in &angles[d / 2..] {
            assert!(
                (-0.001..=PI / 2.0 + 0.001).contains(&a),
                "Higher level angle out of range: {a}"
            );
        }
    }

    #[test]
    fn level_offset_and_count() {
        let d = 256;
        let num_levels = (d as f32).log2() as usize;

        let mut total = 0;
        for level in 1..=num_levels {
            let off = level_offset(d, level);
            let cnt = level_count(d, level);
            assert_eq!(off, total, "Offset mismatch at level {level}");
            total += cnt;
        }
        assert_eq!(total, d - 1, "Total angles should be d-1");
    }

    #[test]
    fn unit_vector_radius_one() {
        let d = 128;
        let mut rng = StdRng::seed_from_u64(42);
        let mut x = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut x);
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut x {
            *v /= norm;
        }
        let (_, radius) = cartesian_to_polar(&x);
        assert!(
            (radius - 1.0).abs() < 1e-5,
            "Unit vector radius should be 1.0, got {radius}"
        );
    }
}
