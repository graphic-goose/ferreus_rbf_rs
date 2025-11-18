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
mod rbf_kernels;
mod non_rbf_kernels;
mod traits;
mod utils;
mod kernel_helpers;

/// Implemented Kernels for use in the [`ferreus_rbf`] crate and its Python bindings,
/// and the [`ferreus_bbfmm`] crate's Python bindings.
pub mod kernels {
    pub use super::rbf_kernels::*;
    pub use super::non_rbf_kernels::*;
}

pub use {
    kernel_helpers::KernelParams, 
    utils::{
        FmmTree, KernelType, get_a_matrix, get_a_matrix_symmetric_solver,
        argmax, argmin, argsort, cartesian_product, get_distance, get_pointarray_extents,
        max, select_mat_rows, kernel_phi,
    },
    traits::KernelFromParams,
};
