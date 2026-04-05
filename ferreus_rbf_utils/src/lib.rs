/////////////////////////////////////////////////////////////////////////////////////////////
//
// Re-exports kernel utilities, constants, and helper functions used across the ferreus_rbf crates.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Utilities for the [`ferreus_rbf`] crate and its Python bindings
mod constants;
mod kernel_helpers;
mod non_rbf_kernels;
mod rbf_kernels;
mod traits;
mod utils;

/// Implemented Kernels for use in the [`ferreus_rbf`] crate and its Python bindings,
/// and the [`ferreus_bbfmm`] crate's Python bindings.
pub mod kernels {
    pub use super::non_rbf_kernels::*;
    pub use super::rbf_kernels::*;
}

pub use {
    kernel_helpers::KernelParams,
    traits::KernelFromParams,
    utils::{
        FmmTree, KernelType, argmax, argmin, argsort, cartesian_product, get_a_matrix,
        get_a_matrix_symmetric_solver, get_distance, get_pointarray_extents, kernel_phi, max,
        select_mat_rows,
    },
};
