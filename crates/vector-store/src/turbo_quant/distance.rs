/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! TQ4 distance computation for HNSW metric and asymmetric reranking.
//!
//! Uses the interleaved 4-bit nibble layout where each coordinate is stored
//! as a single nibble: bit 3 = QJL sign, bits 2-0 = MSE centroid index.
//! Two nibbles per byte (high nibble = even dimension, low nibble = odd).
//!
//! Two distance modes:
//! 1. **Symmetric** (TQ4-to-TQ4): Used by USearch custom metric during HNSW
//!    graph traversal. Computes approximate inner product from two packed TQ4
//!    representations using codebook cross-product table + Hamming QJL term.
//! 2. **Asymmetric** (f32 query vs TQ4): Used for final reranking. Provides
//!    the unbiased TurboQuant inner product estimator at full precision.

use crate::turbo_quant::codebook::CENTROIDS_3BIT;
use crate::turbo_quant::quantize::{Tq4CompressedVector, Tq4Quantizer};
use numkong::Dot;

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
        let d_pad = self.padded_dim();

        // Query norm (SIMD-accelerated via NumKong)
        let query_norm: f32 = (f32::dot(query, query).unwrap_or(0.0) as f32).sqrt();

        // Rotate query (NOT normalized — raw query is rotated so that
        // the MSE dot product preserves the ‖q‖ factor)
        // Fix B: full padded output for norm-preserving distance computation.
        let mut rotated_query = vec![0.0f32; d_pad];
        self.rotation().forward_padded(query, &mut rotated_query);

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
    ///
    /// Fused single pass over nibbles: extracts MSE indices and QJL signs
    /// together, zero heap allocation.
    pub fn inner_product(
        &self,
        query_state: &Tq4QueryState,
        compressed: &Tq4CompressedVector,
    ) -> f32 {
        let d_pad = self.padded_dim();
        let inv_sqrt_d = self.inv_sqrt_d();
        let rq = &query_state.rotated_query;
        let pq = &query_state.projected_query;

        // Precompute scaled centroids (8 entries, fits in registers)
        let mut scaled = [0.0f32; 8];
        for i in 0..8 {
            scaled[i] = CENTROIDS_3BIT[i] * inv_sqrt_d;
        }

        // Fused MSE + QJL in a single pass over nibbles
        let mut mse_ip = 0.0f32;
        let mut qjl_dot = 0.0f32;
        for (byte_idx, &nib) in compressed.nibbles.iter().enumerate() {
            let de = byte_idx * 2;
            let do_ = de + 1;
            let hi = ((nib >> 4) & 0x07) as usize;
            let lo = (nib & 0x07) as usize;
            mse_ip += scaled[hi] * rq[de] + scaled[lo] * rq[do_];
            let se: f32 = if (nib & 0x80) != 0 { 1.0 } else { -1.0 };
            let so: f32 = if (nib & 0x08) != 0 { 1.0 } else { -1.0 };
            qjl_dot += se * pq[de] + so * pq[do_];
        }

        let qjl_ip =
            ((std::f32::consts::PI / 2.0_f32).sqrt() / d_pad as f32) * compressed.gamma * qjl_dot;

        compressed.norm * (mse_ip + qjl_ip)
    }

    /// Compute inner product estimate directly from a packed TQ4 byte slice.
    ///
    /// Fused single pass: extracts MSE centroids and QJL signs from nibbles
    /// in one loop with zero heap allocation and no thread-local buffers.
    pub fn inner_product_packed(
        &self,
        query_state: &Tq4QueryState,
        packed: &[u8],
        dimension: usize,
    ) -> f32 {
        let d_pad = dimension.next_power_of_two();
        debug_assert_eq!(d_pad, self.padded_dim());

        let nibble_len = d_pad / 2;
        let nibbles = &packed[..nibble_len];
        let gamma = f32::from_le_bytes(packed[nibble_len..nibble_len + 4].try_into().unwrap());
        let norm = f32::from_le_bytes(packed[nibble_len + 4..nibble_len + 8].try_into().unwrap());

        let inv_sqrt_d = self.inv_sqrt_d();
        let rq = &query_state.rotated_query;
        let pq = &query_state.projected_query;

        let mut scaled = [0.0f32; 8];
        for i in 0..8 {
            scaled[i] = CENTROIDS_3BIT[i] * inv_sqrt_d;
        }

        let mut mse_ip = 0.0f32;
        let mut qjl_dot = 0.0f32;
        for (byte_idx, &nib) in nibbles.iter().enumerate() {
            let de = byte_idx * 2;
            let do_ = de + 1;
            let hi = ((nib >> 4) & 0x07) as usize;
            let lo = (nib & 0x07) as usize;
            mse_ip += scaled[hi] * rq[de] + scaled[lo] * rq[do_];
            let se: f32 = if (nib & 0x80) != 0 { 1.0 } else { -1.0 };
            let so: f32 = if (nib & 0x08) != 0 { 1.0 } else { -1.0 };
            qjl_dot += se * pq[de] + so * pq[do_];
        }

        let qjl_ip = ((std::f32::consts::PI / 2.0_f32).sqrt() / d_pad as f32) * gamma * qjl_dot;

        norm * (mse_ip + qjl_ip)
    }

    /// Batch compute inner product estimates for multiple TQ4 candidates.
    ///
    /// Fused single pass per candidate: extracts MSE and QJL from nibbles
    /// in one loop with zero heap allocation per candidate.
    pub fn batch_inner_products(
        &self,
        query_state: &Tq4QueryState,
        candidates: &[Tq4CompressedVector],
    ) -> Vec<f32> {
        let d_pad = self.padded_dim();
        let inv_sqrt_d = self.inv_sqrt_d();
        let rq = &query_state.rotated_query;
        let pq = &query_state.projected_query;
        let qjl_scale = (std::f32::consts::PI / 2.0_f32).sqrt() / d_pad as f32;

        let mut scaled = [0.0f32; 8];
        for i in 0..8 {
            scaled[i] = CENTROIDS_3BIT[i] * inv_sqrt_d;
        }

        let mut results = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let mut mse_ip = 0.0f32;
            let mut qjl_dot = 0.0f32;
            for (byte_idx, &nib) in candidate.nibbles.iter().enumerate() {
                let de = byte_idx * 2;
                let do_ = de + 1;
                let hi = ((nib >> 4) & 0x07) as usize;
                let lo = (nib & 0x07) as usize;
                mse_ip += scaled[hi] * rq[de] + scaled[lo] * rq[do_];
                let se: f32 = if (nib & 0x80) != 0 { 1.0 } else { -1.0 };
                let so: f32 = if (nib & 0x08) != 0 { 1.0 } else { -1.0 };
                qjl_dot += se * pq[de] + so * pq[do_];
            }

            let qjl_ip = qjl_scale * candidate.gamma * qjl_dot;
            results.push(candidate.norm * (mse_ip + qjl_ip));
        }

        results
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

/// Precompute cos(π·k/d) for k ∈ [0, d].
///
/// Eliminates the `f32::cos()` call from every symmetric distance evaluation.
/// The table has d+1 entries and fits comfortably in L1 cache (~8 KB for d=2048).
pub fn precompute_cos_table(dim: usize) -> Vec<f32> {
    (0..=dim)
        .map(|k| f32::cos(std::f32::consts::PI * k as f32 / dim as f32))
        .collect()
}

/// TQ4-to-TQ4 symmetric distance for USearch custom metric.
///
/// Computes approximate inner product from two packed TQ4 representations
/// using the interleaved nibble layout:
///   ⟨a, b⟩ ≈ norm_a · norm_b · (mse_term + qjl_term)
///
/// - MSE term: cross-product table lookup over 3-bit index pairs (nibble & 0x07)
/// - QJL term: Hamming on interleaved sign bits (nibble bit 3) → cosine correction
///
/// # Arguments
/// - `a`, `b`: packed TQ4 byte arrays (layout: [nibbles | gamma | norm])
/// - `dim`: padded dimension d_pad
/// - `cross_table`: precomputed 8×8 cross-product table
/// - `cos_table`: precomputed cos(π·k/d) for k ∈ [0, d] (from `precompute_cos_table`)
///
/// Returns the approximate inner product (not a distance).
pub fn tq4_symmetric_distance(
    a: &[u8],
    b: &[u8],
    dim: usize,
    cross_table: &[[f32; 8]; 8],
    cos_table: &[f32],
) -> f32 {
    let nibble_len = dim / 2;

    let a_nibbles = &a[..nibble_len];
    let b_nibbles = &b[..nibble_len];
    let a_gamma = f32::from_le_bytes(a[nibble_len..nibble_len + 4].try_into().unwrap());
    let b_gamma = f32::from_le_bytes(b[nibble_len..nibble_len + 4].try_into().unwrap());
    let a_norm = f32::from_le_bytes(a[nibble_len + 4..nibble_len + 8].try_into().unwrap());
    let b_norm = f32::from_le_bytes(b[nibble_len + 4..nibble_len + 8].try_into().unwrap());

    // MSE term: cross-product table lookup from nibble indices
    let mse_term = accumulate_cross_products_nibble(a_nibbles, b_nibbles, cross_table);

    // QJL term: Hamming on interleaved sign bits (bit 3 of each nibble)
    // 0x88 mask selects bits 7 and 3 — the QJL sign bits of the two nibbles per byte.
    let mut hamming_bits: u32 = 0;
    for k in 0..nibble_len {
        let xor = a_nibbles[k] ^ b_nibbles[k];
        hamming_bits += (xor & 0x88).count_ones();
    }

    // cos(π · hamming / d) via precomputed table lookup (zero trig at runtime)
    let qjl_term = a_gamma * b_gamma * cos_table[hamming_bits as usize];

    a_norm * b_norm * (mse_term + qjl_term)
}

/// Accumulate cross-product lookups over pairs of nibble-packed MSE indices.
///
/// Each byte contains two 4-bit nibbles; bits 2-0 of each nibble are the
/// 3-bit MSE index. Two table lookups per byte, zero heap allocation.
fn accumulate_cross_products_nibble(
    a_nibbles: &[u8],
    b_nibbles: &[u8],
    cross_table: &[[f32; 8]; 8],
) -> f32 {
    let mut sum = 0.0f32;
    for k in 0..a_nibbles.len() {
        let a_byte = a_nibbles[k];
        let b_byte = b_nibbles[k];
        let a_hi = ((a_byte >> 4) & 0x07) as usize;
        let a_lo = (a_byte & 0x07) as usize;
        let b_hi = ((b_byte >> 4) & 0x07) as usize;
        let b_lo = (b_byte & 0x07) as usize;
        sum += cross_table[a_hi][b_hi] + cross_table[a_lo][b_lo];
    }
    sum
}

/// Asymmetric float-vs-TQ4 inner product from packed bytes. Zero allocation.
///
/// Computes ⟨q, x⟩ where q is a full-precision float query (represented by its
/// pre-rotated and pre-projected forms) and x is a packed TQ4 vector.
///
/// Uses the interleaved nibble layout: each byte holds two 4-bit nibbles
/// (bit 3 = QJL sign, bits 2-0 = MSE index).
///
/// # Arguments
/// - `packed`: a packed TQ4 byte array (layout: [nibbles | gamma | norm])
/// - `dim`: padded dimension d_pad
/// - `inv_sqrt_d`: 1/√d for codebook scaling
/// - `rotated_query`: Π · q (d_pad floats)
/// - `projected_query`: S · (Π · q) (d_pad floats)
#[allow(dead_code)]
pub fn tq4_asymmetric_distance_packed(
    packed: &[u8],
    dim: usize,
    inv_sqrt_d: f32,
    rotated_query: &[f32],
    projected_query: &[f32],
) -> f32 {
    let nibble_len = dim / 2;

    let nibbles = &packed[..nibble_len];
    let gamma = f32::from_le_bytes(packed[nibble_len..nibble_len + 4].try_into().unwrap());
    let norm = f32::from_le_bytes(packed[nibble_len + 4..nibble_len + 8].try_into().unwrap());

    // MSE term: extract indices from nibbles and dot with rotated query
    let mut mse_ip = 0.0f32;
    for (byte_idx, &nib) in nibbles.iter().enumerate() {
        let dim_even = byte_idx * 2;
        let dim_odd = byte_idx * 2 + 1;
        let hi = ((nib >> 4) & 0x07) as usize;
        let lo = (nib & 0x07) as usize;
        mse_ip += CENTROIDS_3BIT[hi] * inv_sqrt_d * rotated_query[dim_even];
        mse_ip += CENTROIDS_3BIT[lo] * inv_sqrt_d * rotated_query[dim_odd];
    }

    // QJL correction term: extract sign bits from nibbles inline
    let mut dot_sum = 0.0f32;
    for (byte_idx, &nib) in nibbles.iter().enumerate() {
        let dim_even = byte_idx * 2;
        let dim_odd = byte_idx * 2 + 1;
        let hi_nibble = nib >> 4;
        let lo_nibble = nib & 0x0F;
        let sign_even = if (hi_nibble & 0x08) != 0 {
            1.0f32
        } else {
            -1.0
        };
        let sign_odd = if (lo_nibble & 0x08) != 0 {
            1.0f32
        } else {
            -1.0
        };
        dot_sum += sign_even * projected_query[dim_even];
        dot_sum += sign_odd * projected_query[dim_odd];
    }
    let qjl_ip = ((std::f32::consts::PI / 2.0_f32).sqrt() / dim as f32) * gamma * dot_sum;

    norm * (mse_ip + qjl_ip)
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
        let d_pad = quantizer.padded_dim();
        let inv_sqrt_d = quantizer.inv_sqrt_d();
        let cross_table = cross_product_table_3bit(inv_sqrt_d);
        let cos_table = precompute_cos_table(d_pad);

        let mut rng = StdRng::seed_from_u64(42);
        let mut v = vec![0.0f32; d];
        fill_standard_normal(&mut rng, &mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let compressed = quantizer.quantize(&v);
        let packed = compressed.pack();

        // Self-distance should approximate ‖x‖²
        let ip = tq4_symmetric_distance(&packed, &packed, d_pad, &cross_table, &cos_table);
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

    #[test]
    #[ignore] // Microbenchmark, run with --ignored
    fn bench_tq4_symmetric_distance() {
        use std::hint::black_box;
        use std::time::Instant;

        let d = 1536;
        let quantizer = Tq4Quantizer::new(d, 42, 137);
        let d_pad = quantizer.padded_dim();
        let inv_sqrt_d = quantizer.inv_sqrt_d();
        let cross_table = cross_product_table_3bit(inv_sqrt_d);
        let cos_table = precompute_cos_table(d_pad);

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
            black_box(tq4_symmetric_distance(
                &packed_vecs[0],
                &packed_vecs[1],
                d_pad,
                &cross_table,
                &cos_table,
            ));
        }

        // Benchmark
        let n_iters = 100_000;
        let t0 = Instant::now();
        for i in 0..n_iters {
            let a = &packed_vecs[i % n_vecs];
            let b = &packed_vecs[(i + 1) % n_vecs];
            black_box(tq4_symmetric_distance(
                a,
                b,
                d_pad,
                &cross_table,
                &cos_table,
            ));
        }
        let elapsed = t0.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / n_iters as f64;
        let us_per_call = ns_per_call / 1000.0;
        eprintln!(
            "\n=== tq4_symmetric_distance (d={d}, d_pad={d_pad}) ===\n\
             {n_iters} iterations in {:.1}ms\n\
             {ns_per_call:.0} ns/call ({us_per_call:.2} µs/call)\n\
             Throughput: {:.1}M distances/sec",
            elapsed.as_secs_f64() * 1000.0,
            1e9 / ns_per_call / 1e6,
        );
    }
}
