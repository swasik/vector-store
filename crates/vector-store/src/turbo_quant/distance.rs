/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! TQ4 distance computation for HNSW metric and asymmetric reranking.
//!
//! Two distance modes:
//! 1. **Symmetric** (TQ4-to-TQ4): Used by USearch custom metric during HNSW
//!    graph traversal. Computes approximate inner product from two packed TQ4
//!    representations using codebook cross-product table + Hamming QJL term.
//! 2. **Asymmetric** (f32 query vs TQ4): Used for final reranking. Provides
//!    the unbiased TurboQuant inner product estimator at full precision.

use crate::turbo_quant::codebook::{self, CENTROIDS_3BIT};
use crate::turbo_quant::quantize::{Tq4CompressedVector, Tq4Quantizer};
use numkong::{Hamming, u1x8};

/// Precomputed query state for efficient asymmetric TQ4 distance computation.
///
/// Computed once per search query, reused across all candidates during reranking.
pub struct Tq4QueryState {
    /// Rotated query: q' = Π · q (d-dimensional).
    pub rotated_query: Vec<f32>,
    /// QJL-projected query: S · q' (d-dimensional).
    /// S is applied to the rotated query (same space as residual sign bits).
    pub projected_query: Vec<f32>,
    /// Original query L2 norm.
    pub query_norm: f32,
}

impl Tq4Quantizer {
    /// Prepare query state for TQ4 distance computations.
    ///
    /// O(d log d) for rotation + O(d²) for QJL projection.
    /// Called once per search, amortized across all candidates.
    pub fn prepare_query(&self, query: &[f32]) -> Tq4QueryState {
        let d = self.dimension();

        // Query norm
        let query_norm: f32 = query.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Rotate query (NOT normalized — raw query is rotated so that
        // the MSE dot product preserves the ‖q‖ factor)
        let mut rotated_query = vec![0.0f32; d];
        self.rotation().forward(query, &mut rotated_query);

        // Project rotated query through QJL matrix: S · q'
        let projected_query = self.qjl().project_query(&rotated_query);

        Tq4QueryState {
            rotated_query,
            projected_query,
            query_norm,
        }
    }

    /// Compute inner product estimate between prepared query and a TQ4 vector.
    ///
    /// Returns: ⟨q, x⟩ ≈ ‖x‖ · (mse_ip + qjl_ip)
    /// where mse_ip uses codebook centroids dotted with rotated query,
    /// and qjl_ip uses sign bits dotted with projected query.
    pub fn inner_product(
        &self,
        query_state: &Tq4QueryState,
        compressed: &Tq4CompressedVector,
    ) -> f32 {
        let d = self.dimension();
        let inv_sqrt_d = self.inv_sqrt_d();

        // MSE term: Σ_j centroid[idx_j] · rotated_query[j]
        let mut mse_ip = 0.0f32;
        for j in 0..d {
            let idx = codebook::extract_3bit_index(&compressed.mse_indices, j);
            let centroid_val = CENTROIDS_3BIT[idx as usize] * inv_sqrt_d;
            mse_ip += centroid_val * query_state.rotated_query[j];
        }

        // QJL correction term: (π/2)/d · γ · Σ_j sign_j · projected_query_j
        let qjl_ip = self.qjl().inner_product_term(
            &compressed.qjl_signs,
            &query_state.projected_query,
            compressed.gamma,
        );

        // Full inner product estimate: ‖x‖ · (MSE + QJL)
        compressed.norm * (mse_ip + qjl_ip)
    }

    /// Compute cosine similarity from TQ4 inner product.
    ///
    /// cosine = ⟨q, x⟩ / (‖q‖ · ‖x‖)
    pub fn cosine_similarity(
        &self,
        query_state: &Tq4QueryState,
        compressed: &Tq4CompressedVector,
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

/// TQ4-to-TQ4 symmetric distance for USearch custom metric.
///
/// Computes approximate inner product from two packed TQ4 representations:
///   ⟨a, b⟩ ≈ norm_a · norm_b · (mse_term + qjl_term)
///
/// - MSE term: cross-product table lookup over 3-bit index pairs
/// - QJL term: Hamming distance on sign bits → cosine correction
///
/// # Arguments
/// - `a`, `b`: packed TQ4 byte arrays (layout: [mse | qjl | gamma | norm])
/// - `dim`: original vector dimension d
/// - `cross_table`: precomputed 8×8 cross-product table
///
/// Returns the approximate inner product (not a distance).
pub fn tq4_symmetric_distance(a: &[u8], b: &[u8], dim: usize, cross_table: &[[f32; 8]; 8]) -> f32 {
    let mse_len = (dim * 3).div_ceil(8);
    let qjl_len = dim.div_ceil(8);

    let a_mse = &a[..mse_len];
    let b_mse = &b[..mse_len];
    let a_qjl = &a[mse_len..mse_len + qjl_len];
    let b_qjl = &b[mse_len..mse_len + qjl_len];
    let a_gamma = f32::from_le_bytes(
        a[mse_len + qjl_len..mse_len + qjl_len + 4]
            .try_into()
            .unwrap(),
    );
    let b_gamma = f32::from_le_bytes(
        b[mse_len + qjl_len..mse_len + qjl_len + 4]
            .try_into()
            .unwrap(),
    );
    let a_norm = f32::from_le_bytes(
        a[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
            .try_into()
            .unwrap(),
    );
    let b_norm = f32::from_le_bytes(
        b[mse_len + qjl_len + 4..mse_len + qjl_len + 8]
            .try_into()
            .unwrap(),
    );

    // MSE term: sum of cross_table[a_j][b_j] for all j
    let mse_term = accumulate_cross_products(a_mse, b_mse, dim, cross_table);

    // QJL term: Hamming distance between sign bits → cosine of angle
    // hamming_bits = number of differing bits between a_qjl and b_qjl
    let a_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(a_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let b_bits: &[u1x8] =
        unsafe { std::slice::from_raw_parts(b_qjl.as_ptr().cast::<u1x8>(), qjl_len) };
    let hamming_bits = u1x8::hamming(a_bits, b_bits).unwrap_or(0);

    // cos(π · hamming / d) is the sign-based inner product estimator
    let qjl_term =
        a_gamma * b_gamma * f32::cos(std::f32::consts::PI * hamming_bits as f32 / dim as f32);

    a_norm * b_norm * (mse_term + qjl_term)
}

/// Accumulate cross-product lookups over pairs of 3-bit indices.
///
/// For each coordinate j ∈ [0, dim): extract 3-bit indices from a and b,
/// look up cross_table[a_j][b_j], and accumulate. Zero heap allocation.
fn accumulate_cross_products(
    a_mse: &[u8],
    b_mse: &[u8],
    dim: usize,
    cross_table: &[[f32; 8]; 8],
) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..dim {
        let a_idx = codebook::extract_3bit_index(a_mse, j) as usize;
        let b_idx = codebook::extract_3bit_index(b_mse, j) as usize;
        sum += cross_table[a_idx][b_idx];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_quant::codebook::cross_product_table_3bit;
    use crate::turbo_quant::qjl::fill_standard_normal;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn symmetric_distance_self() {
        let d = 768;
        let quantizer = Tq4Quantizer::new(d, 42, 137);
        let inv_sqrt_d = quantizer.inv_sqrt_d();
        let cross_table = cross_product_table_3bit(inv_sqrt_d);

        let mut rng = StdRng::seed_from_u64(42);
        let mut v = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let compressed = quantizer.quantize(&v);
        let packed = compressed.pack();

        // Self-distance should approximate ‖x‖²
        let ip = tq4_symmetric_distance(&packed, &packed, d, &cross_table);
        let expected = norm * norm;
        let rel_err = (ip - expected).abs() / expected;
        assert!(
            rel_err < 0.5,
            "Self inner product should approximate ‖x‖²={expected:.4}, got {ip:.4}, rel_err={rel_err:.4}"
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
            // Use different QJL seeds for each trial to average over randomness
            let qjl_seed: u64 = rng.random();
            let quantizer = Tq4Quantizer::new(d, 42, qjl_seed);

            // Random vectors
            let mut x = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x);
            let mut q = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut q);

            // True inner product
            let true_ip: f32 = x.iter().zip(q.iter()).map(|(a, b)| a * b).sum();

            // TQ4 estimate
            let compressed = quantizer.quantize(&x);
            let query_state = quantizer.prepare_query(&q);
            let tq4_ip = quantizer.inner_product(&query_state, &compressed);

            total_err += (tq4_ip - true_ip) as f64;
            total_abs += true_ip.abs() as f64;
        }

        let avg_err = total_err / n_trials as f64;
        let avg_abs = total_abs / n_trials as f64;
        let rel_bias = avg_err.abs() / avg_abs;

        assert!(
            rel_bias < 0.2,
            "Asymmetric IP bias too high: avg_err={avg_err:.6}, rel_bias={rel_bias:.4}"
        );
    }

    #[test]
    fn cosine_similarity_range() {
        let d = 256;
        let quantizer = Tq4Quantizer::new(d, 42, 137);

        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..50 {
            let mut x = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut x);
            let mut q = vec![0.0f32; d];
            fill_standard_normal(&mut rng, &mut q);

            let compressed = quantizer.quantize(&x);
            let query_state = quantizer.prepare_query(&q);
            let cos = quantizer.cosine_similarity(&query_state, &compressed);

            assert!(
                (-1.0..=1.0).contains(&cos),
                "Cosine similarity out of range: {cos}"
            );
        }
    }
}
