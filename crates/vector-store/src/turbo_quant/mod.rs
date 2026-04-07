/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

//! TurboQuant 4-bit vector quantization with QJL residual correction.
//!
//! This module implements the TQ4 quantization scheme from the TurboQuant paper
//! (arXiv:2504.19874). It compresses high-dimensional vectors to ~4 bits per
//! coordinate (d/2 + 8 bytes total) while providing unbiased inner product
//! estimates via a two-stage process:
//!
//! 1. **MSE-optimal scalar quantization** (3 bits): Rotate via Randomized
//!    Hadamard Transform, then apply precomputed Lloyd-Max codebook per coord.
//! 2. **QJL residual correction** (1 bit): 1-bit sign projection of the
//!    quantization residual for inner product bias correction.
//!
//! # Usage
//!
//! ```ignore
//! use crate::turbo_quant::{Tq4Quantizer, Tq4CompressedVector, Tq4QueryState, Tq4Config};
//!
//! let config = Tq4Config::default();
//! let quantizer = Tq4Quantizer::new(768, config.rotation_seed, config.qjl_seed);
//!
//! // Quantize a vector
//! let compressed = quantizer.quantize(&vector);
//! let packed = compressed.pack();  // → Vec<u8> for USearch storage
//!
//! // Search: prepare query once, evaluate many candidates
//! let query_state = quantizer.prepare_query(&query);
//! let ip = quantizer.inner_product(&query_state, &compressed);
//! ```

// Methods are part of the public TQ4 API but not all callers exist yet.
#![allow(dead_code)]

pub mod codebook;
pub mod distance;
pub mod qjl;
pub mod quantize;
pub mod rotation;

#[allow(unused_imports)]
pub use distance::Tq4QueryState;
#[allow(unused_imports)]
pub use quantize::{Tq4CompressedVector, Tq4Quantizer};

/// TQ4-specific configuration parameters.
///
/// All values are hardcoded constants for now. The struct exists to make
/// future configurability straightforward — add a CQL index option and
/// populate the field from `get_index_params()` when needed.
#[derive(Debug, Clone)]
pub struct Tq4Config {
    /// Oversample factor for HNSW candidate retrieval before TQ4 reranking.
    /// Higher values improve recall at the cost of search latency.
    pub oversample_factor: f32,

    /// Seed for rotation matrix generation.
    /// Same seed ensures identical rotation across restarts.
    pub rotation_seed: u64,

    /// Seed for QJL projection matrix.
    pub qjl_seed: u64,
}

impl Default for Tq4Config {
    fn default() -> Self {
        Self {
            oversample_factor: 3.0,
            rotation_seed: 42,
            qjl_seed: 137,
        }
    }
}
