/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! Precomputed Lloyd-Max codebooks for TurboQuant scalar quantization.
//!
//! The 3-bit codebook (8 centroids) is optimized for the Beta(d/2, 1/2)
//! distribution that arises from randomized Hadamard rotation of unit-norm
//! vectors. For moderate-to-high dimensions (d ≥ 64), this distribution
//! converges to N(0, 1/d), and centroids are stored as multiples of 1/√d.
//!
//! Reference: TurboQuant paper (arXiv:2504.19874), Section 3.1.

/// 3-bit Lloyd-Max centroids for N(0, 1) distribution (8 centroids).
/// These are the raw centroids before scaling by `1/√d`.
/// Ordered from most negative to most positive for easy binary search.
pub const CENTROIDS_3BIT: [f32; 8] = [
    -1.2240, -0.7914, -0.4528, -0.1528, 0.1528, 0.4528, 0.7914, 1.2240,
];

/// Decision boundaries between 3-bit centroids (midpoints).
/// 7 boundaries for 8 centroids: value < boundary[i] → index i, else i+1.
pub const BOUNDARIES_3BIT: [f32; 7] = [-1.0077, -0.6221, -0.3028, 0.0, 0.3028, 0.6221, 1.0077];

/// Encode a single scalar coordinate to a 3-bit centroid index (0..7).
///
/// The value is first scaled by `inv_sqrt_d` to convert from the raw rotated
/// coordinate (which has variance ≈ 1/d) to the unit-variance space of the
/// codebook. Uses branchless comparisons against precomputed boundaries.
#[inline]
pub fn encode_scalar_3bit(value: f32, inv_sqrt_d: f32) -> u8 {
    let scaled = value / inv_sqrt_d;
    // Branchless binary search: sum the number of boundaries the value exceeds
    let mut idx = 0u8;
    for &boundary in &BOUNDARIES_3BIT {
        idx += (scaled >= boundary) as u8;
    }
    idx
}

/// Decode a 3-bit centroid index back to the centroid value.
#[inline]
pub fn decode_scalar_3bit(index: u8, inv_sqrt_d: f32) -> f32 {
    CENTROIDS_3BIT[index as usize] * inv_sqrt_d
}

/// Batch-encode d coordinates to packed 3-bit indices.
///
/// Output layout: big-endian bitstream. Eight consecutive 3-bit indices
/// are packed into 3 bytes:
/// ```text
/// Byte 0: [i₀₂ i₀₁ i₀₀ | i₁₂ i₁₁ i₁₀ | i₂₂ i₂₁]
/// Byte 1: [i₂₀ | i₃₂ i₃₁ i₃₀ | i₄₂ i₄₁ i₄₀ | i₅₂]
/// Byte 2: [i₅₁ i₅₀ | i₆₂ i₆₁ i₆₀ | i₇₂ i₇₁ i₇₀]
/// ```
pub fn encode_vector_3bit(rotated: &[f32], inv_sqrt_d: f32) -> Vec<u8> {
    let d = rotated.len();
    let packed_len = (d * 3).div_ceil(8);
    let mut packed = vec![0u8; packed_len];

    for (j, &val) in rotated.iter().enumerate() {
        let idx = encode_scalar_3bit(val, inv_sqrt_d);
        let bit_offset = j * 3;
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        // idx is 3 bits (0..7), write them starting at bit_pos within byte_pos
        if bit_pos <= 5 {
            // Fits entirely within one byte
            packed[byte_pos] |= idx << (5 - bit_pos);
        } else {
            // Spans two bytes
            let first_bits = 8 - bit_pos;
            packed[byte_pos] |= idx >> (3 - first_bits);
            packed[byte_pos + 1] |= idx << (8 - (3 - first_bits));
        }
    }
    packed
}

/// Extract a single 3-bit index from a packed big-endian bitstream.
#[inline]
pub fn extract_3bit_index(packed: &[u8], j: usize) -> u8 {
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
/// an entire group in one operation, avoiding per-element bit arithmetic
/// in the hot loop of cross-product accumulation and MSE dot products.
#[inline]
pub fn extract_8_3bit_indices(packed: &[u8], group: usize) -> [u8; 8] {
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

/// Batch-decode packed 3-bit indices to f32 centroid values.
pub fn decode_vector_3bit(packed: &[u8], d: usize, inv_sqrt_d: f32) -> Vec<f32> {
    (0..d)
        .map(|j| {
            let idx = extract_3bit_index(packed, j);
            decode_scalar_3bit(idx, inv_sqrt_d)
        })
        .collect()
}

/// Precomputed 8×8 cross-product table for TQ4-to-TQ4 symmetric distance.
///
/// `table[i][j] = CENTROIDS_3BIT[i] * inv_sqrt_d * CENTROIDS_3BIT[j] * inv_sqrt_d`
/// This avoids repeated multiplications in the HNSW metric hot path.
pub fn cross_product_table_3bit(inv_sqrt_d: f32) -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            table[i][j] = CENTROIDS_3BIT[i] * inv_sqrt_d * CENTROIDS_3BIT[j] * inv_sqrt_d;
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Interleaved 4-bit nibble layout (3-bit MSE index + 1-bit QJL sign)
// ---------------------------------------------------------------------------

/// Encode a vector to raw 3-bit centroid indices (one byte per coordinate).
///
/// Returns values 0-7, one per coordinate. Used as input for the interleaved
/// nibble packing where each 4-bit nibble stores MSE index (3 bits) + QJL
/// sign (1 bit).
pub fn encode_vector_3bit_raw(rotated: &[f32], inv_sqrt_d: f32) -> Vec<u8> {
    rotated
        .iter()
        .map(|&val| encode_scalar_3bit(val, inv_sqrt_d))
        .collect()
}

/// Decode raw 3-bit centroid indices to f32 centroid values.
pub fn decode_vector_3bit_raw(indices: &[u8], inv_sqrt_d: f32) -> Vec<f32> {
    indices
        .iter()
        .map(|&idx| decode_scalar_3bit(idx, inv_sqrt_d))
        .collect()
}

/// Interleave MSE indices and QJL sign bits into packed 4-bit nibbles.
///
/// Each nibble stores: bit 3 = QJL sign, bits 2-0 = MSE centroid index (0-7).
/// Two nibbles per byte: high nibble (bits 7-4) = even dimension,
/// low nibble (bits 3-0) = odd dimension.
///
/// This interleaved format enables trivial extraction (shift + mask) and
/// produces the same total packed size as the separate 3-bit + 1-bit layout.
///
/// # Arguments
/// - `mse_raw`: raw MSE centroid indices (0-7), length d_pad
/// - `qjl_signs`: packed QJL sign bits (MSB-first), length d_pad/8
///
/// # Returns
/// Interleaved nibble bytes, length d_pad/2.
pub fn interleave_nibbles(mse_raw: &[u8], qjl_signs: &[u8], d_pad: usize) -> Vec<u8> {
    debug_assert_eq!(mse_raw.len(), d_pad);
    debug_assert_eq!(qjl_signs.len(), d_pad.div_ceil(8));

    let nibble_len = d_pad / 2;
    let mut nibbles = vec![0u8; nibble_len];

    for (byte_idx, nibble) in nibbles.iter_mut().enumerate() {
        let dim_even = byte_idx * 2;
        let dim_odd = byte_idx * 2 + 1;

        let qjl_even = (qjl_signs[dim_even / 8] >> (7 - (dim_even % 8))) & 1;
        let qjl_odd = (qjl_signs[dim_odd / 8] >> (7 - (dim_odd % 8))) & 1;

        let hi = (qjl_even << 3) | (mse_raw[dim_even] & 0x07);
        let lo = (qjl_odd << 3) | (mse_raw[dim_odd] & 0x07);
        *nibble = (hi << 4) | lo;
    }

    nibbles
}

/// Extract contiguous QJL sign bits from interleaved nibble bytes.
///
/// Returns packed QJL signs in the same MSB-first format as `qjl.quantize()`,
/// suitable for passing to `QjlProjection::inner_product_term()`.
pub fn extract_qjl_from_nibbles(nibbles: &[u8], d_pad: usize) -> Vec<u8> {
    let qjl_len = d_pad.div_ceil(8);
    let mut qjl_signs = vec![0u8; qjl_len];

    for (byte_idx, &nib) in nibbles.iter().enumerate() {
        let dim_even = byte_idx * 2;
        let dim_odd = byte_idx * 2 + 1;

        // QJL sign is bit 3 of each nibble → bit 7 and bit 3 of the byte
        let qjl_even = (nib >> 7) & 1;
        let qjl_odd = (nib >> 3) & 1;

        if qjl_even != 0 {
            qjl_signs[dim_even / 8] |= 1 << (7 - (dim_even % 8));
        }
        if qjl_odd != 0 {
            qjl_signs[dim_odd / 8] |= 1 << (7 - (dim_odd % 8));
        }
    }

    qjl_signs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroids_are_symmetric() {
        for i in 0..4 {
            let pos = CENTROIDS_3BIT[7 - i];
            let neg = CENTROIDS_3BIT[i];
            assert!(
                (pos + neg).abs() < 1e-6,
                "Centroids not symmetric: {pos} vs {neg}"
            );
        }
    }

    #[test]
    fn boundaries_are_symmetric() {
        for i in 0..3 {
            let pos = BOUNDARIES_3BIT[6 - i];
            let neg = BOUNDARIES_3BIT[i];
            assert!(
                (pos + neg).abs() < 1e-6,
                "Boundaries not symmetric: {pos} vs {neg}"
            );
        }
        assert!(
            BOUNDARIES_3BIT[3].abs() < 1e-6,
            "Middle boundary should be ~0"
        );
    }

    #[test]
    fn encode_decode_roundtrip() {
        let inv_sqrt_d = 1.0; // d=1 for simplicity
        for idx in 0..8u8 {
            let centroid = decode_scalar_3bit(idx, inv_sqrt_d);
            let encoded = encode_scalar_3bit(centroid, inv_sqrt_d);
            assert_eq!(
                encoded, idx,
                "Encode/decode roundtrip failed for index {idx}"
            );
        }
    }

    #[test]
    fn packing_roundtrip() {
        // Test all possible 3-bit index patterns in groups of 8
        let inv_sqrt_d = 1.0 / (768.0_f32).sqrt();
        for pattern in 0..=255u8 {
            let indices: Vec<u8> = (0..8).map(|i| (pattern >> i) & 0x07).collect();
            let values: Vec<f32> = indices
                .iter()
                .map(|&idx| CENTROIDS_3BIT[idx as usize] * inv_sqrt_d)
                .collect();
            let packed = encode_vector_3bit(&values, inv_sqrt_d);
            for (j, &expected_idx) in indices.iter().enumerate() {
                let extracted = extract_3bit_index(&packed, j);
                assert_eq!(
                    extracted, expected_idx,
                    "Packing roundtrip failed at position {j} for pattern {pattern:#010b}"
                );
            }
        }
    }

    #[test]
    fn vector_encode_decode_roundtrip() {
        let d = 768;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        // Generate values at each centroid
        let values: Vec<f32> = (0..d).map(|i| CENTROIDS_3BIT[i % 8] * inv_sqrt_d).collect();
        let packed = encode_vector_3bit(&values, inv_sqrt_d);
        let decoded = decode_vector_3bit(&packed, d, inv_sqrt_d);
        for (j, (orig, dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (orig - dec).abs() < 1e-6,
                "Mismatch at dim {j}: orig={orig}, decoded={dec}"
            );
        }
    }

    #[test]
    fn packed_size_correct() {
        for d in [768, 1024, 1536, 3072] {
            let inv_sqrt_d = 1.0 / (d as f32).sqrt();
            let values: Vec<f32> = vec![0.0; d];
            let packed = encode_vector_3bit(&values, inv_sqrt_d);
            assert_eq!(
                packed.len(),
                (d * 3).div_ceil(8),
                "Wrong packed size for d={d}"
            );
        }
    }

    #[test]
    fn cross_product_table_values() {
        let d = 768;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let table = cross_product_table_3bit(inv_sqrt_d);
        for i in 0..8 {
            for j in 0..8 {
                let expected = CENTROIDS_3BIT[i] * inv_sqrt_d * CENTROIDS_3BIT[j] * inv_sqrt_d;
                assert!(
                    (table[i][j] - expected).abs() < 1e-10,
                    "Cross table mismatch at [{i}][{j}]"
                );
            }
        }
    }

    #[test]
    fn nibble_interleave_roundtrip() {
        let d = 128;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let values: Vec<f32> = (0..d).map(|i| CENTROIDS_3BIT[i % 8] * inv_sqrt_d).collect();
        let mse_raw = encode_vector_3bit_raw(&values, inv_sqrt_d);

        // Synthetic QJL signs: alternating pattern
        let qjl_signs: Vec<u8> = (0..d / 8)
            .map(|i| if i % 2 == 0 { 0xAA } else { 0x55 })
            .collect();

        let nibbles = interleave_nibbles(&mse_raw, &qjl_signs, d);
        assert_eq!(nibbles.len(), d / 2);

        // Verify MSE indices survive the roundtrip
        for byte_idx in 0..nibbles.len() {
            let dim_even = byte_idx * 2;
            let dim_odd = byte_idx * 2 + 1;
            let hi = (nibbles[byte_idx] >> 4) & 0x07;
            let lo = nibbles[byte_idx] & 0x07;
            assert_eq!(hi, mse_raw[dim_even], "MSE mismatch at dim {dim_even}");
            assert_eq!(lo, mse_raw[dim_odd], "MSE mismatch at dim {dim_odd}");
        }

        // Verify QJL roundtrip
        let extracted_qjl = extract_qjl_from_nibbles(&nibbles, d);
        assert_eq!(extracted_qjl, qjl_signs, "QJL roundtrip failed");
    }

    #[test]
    fn raw_encode_decode_roundtrip() {
        let d = 768;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let values: Vec<f32> = (0..d).map(|i| CENTROIDS_3BIT[i % 8] * inv_sqrt_d).collect();
        let raw = encode_vector_3bit_raw(&values, inv_sqrt_d);
        let decoded = decode_vector_3bit_raw(&raw, inv_sqrt_d);
        for (j, (orig, dec)) in values.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (orig - dec).abs() < 1e-6,
                "Mismatch at dim {j}: orig={orig}, decoded={dec}"
            );
        }
    }
}
