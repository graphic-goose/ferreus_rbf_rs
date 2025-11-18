/////////////////////////////////////////////////////////////////////////////////////////////
//
// Exposes the public API for the Black Box Fast Multipole Method (BBFMM) crate.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Black Box Fast Multipole Method (BBFMM)
//!
//! This crate is a parallel implementation of the `Black Box Fast Multipole Method` in Rust.
//!
//! BBFMM is a kernel-independent, hierarchical algorithm for rapidly evaluating
//! all pairwise interactions in a collection of particles.
//!
//! While originally developed for radial basis function (RBF) interpolation problems,
//! `ferreus_bbfmm` has been generalised to support a broad range of FMM use cases where
//! the kernel is smooth (i.e. non-oscillatory).
//! 
//! # Features:
//! - 1D (binary tree), 2D (quadtree) and 3D (octree)
//! - Optimised low-rank M2L interactions that leverage symmetries and compression
//! - Both adaptive and non-adaptive tree structures
//! - Multiple right-hand sides
//!
//! # Example: Fast Matrix-Vector Product
//!
//! ```
//! use ferreus_bbfmm::{FmmTree, KernelFunction};
//! use faer::{Mat, RowRef};
//! use rand::{Rng, SeedableRng};
//! use rand::rngs::StdRng;
//!
//! // Define a kernel that implements the KernelFunction trait
//! pub struct LinearRbfKernel;
//!
//! impl KernelFunction for LinearRbfKernel {
//!     #[inline(always)]
//!     fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
//!         let mut dist = 0.0;
//!         for (t, s) in target.iter().zip(source.iter()) {
//!             let diff = t - s;
//!             dist += diff * diff;
//!         }
//!         dist.sqrt()
//!        
//!         -dist
//!     }
//! }
//! 
//! let kernel = LinearRbfKernel;
//! 
//! // Generate random source points in 3D
//! let num_points = 10000;
//! let dim = 3;
//! let mut rng = StdRng::seed_from_u64(42);
//! let num_rhs = 2;
//!
//! let source_points = Mat::from_fn(num_points, dim, |_, _| rng.random_range(-1.0..1.0));
//! let weights = Mat::from_fn(num_points, num_rhs, |_, _| rng.random_range(0.0..1.0));
//!
//! // Interpolation order defines the number of Chebyshev nodes in each dimension
//! // used in the far-field approximation
//! // A higher interpolation order is more accurate, but takes longer to compute
//! let interpolation_order = 7;
//!
//! // Create an adaptive tree
//! let adaptive_tree = true;
//!
//! // No need to store empty leaves for fast matrix-vector product
//! let sparse_tree = true;
//!
//! // Create a new tree
//! let mut tree = FmmTree::new(
//!     source_points.clone(),
//!     interpolation_order,
//!     kernel,
//!     adaptive_tree,
//!     sparse_tree,
//!     None,
//!     None,
//! );
//!
//! // Set the weights - this performs an upward pass through the tree
//! // and sets the multipole coefficients
//! tree.set_weights(&weights.as_ref());
//!
//! // Evaluate at the source points
//! let target_points = source_points.clone();
//!
//! // Perform a downward pass to set the local coefficients, then perform a leaf evaluation
//! tree.evaluate(&weights.as_ref(), &target_points);
//!
//! println!("Evaluated values at source locations: {:?}", tree.target_values);
//! ```
//!
//! # Example: RBF Evaluator
//!
//! ```
//! use ferreus_bbfmm::{FmmTree, FmmParams, M2LCompressionType, KernelFunction};
//! use faer::{Mat, RowRef};
//! use rand::{Rng, SeedableRng};
//! use rand::rngs::StdRng;
//!
//! // Define a kernel that implements the KernelFunction trait
//! pub struct LinearRbfKernel;
//!
//! impl KernelFunction for LinearRbfKernel {
//!     #[inline(always)]
//!     fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
//!         let mut dist = 0.0;
//!         for (t, s) in target.iter().zip(source.iter()) {
//!             let diff = t - s;
//!             dist += diff * diff;
//!         }
//!         dist.sqrt()
//!        
//!         -dist
//!     }
//! }
//! 
//! let kernel = LinearRbfKernel;
//! 
//! // Generate random source points in 3D
//! let num_points = 10000;
//! let dim = 3;
//! let mut rng = StdRng::seed_from_u64(42);
//! let num_rhs = 1;
//!
//! let source_points = Mat::from_fn(num_points, dim, |_, _| rng.random_range(-1.0..1.0));
//! let weights = Mat::from_fn(num_points, num_rhs, |_, _| rng.random_range(0.0..1.0));
//!
//! let interpolation_order = 7;
//!
//! // Creating an adaptive tree for the evaluator uses less memory
//! let adaptive_tree = true;
//!
//! // Store empty leaves for general RBF evaluation
//! let sparse_tree = false;
//!
//! // For the evaluator we may wish to evaluate over a larger region than the source points cover
//! let extents = vec![-2.0, -2.0, -2.0, 2.0, 2.0, 2.0f64];
//!
//! // Optionally define some tuning parameters
//! let params = FmmParams{
//!     max_points_per_cell: 256,
//!     compression_type: M2LCompressionType::ACA,
//!     epsilon: 10f64.powi(-(interpolation_order as i32)),
//!     eval_chunk_size: 1024,
//! };
//! 
//! // Create a new tree
//! let mut tree = FmmTree::new(
//!     source_points.clone(),
//!     interpolation_order,
//!     kernel,
//!     adaptive_tree,
//!     sparse_tree,
//!     Some(extents),
//!     Some(params),
//! );
//!
//! // Set the weights - this performs an upward pass through the tree
//! // and sets the multipole coefficients
//! tree.set_weights(&weights.as_ref());
//!
//! // For implicit modelling where a 'surface following' method of generating an isosurface
//! // is used, the evaluator may be called many times. In this case it's more efficient to
//! // perform a single downward pass to set all the local coefficients, then call the evaluator
//! // on the relevant leaves for each evaluation
//! tree.set_local_coefficients(&weights.as_ref());
//!
//! // Create some arbritrary target points
//! let num_target_points = 100;
//! let target_points = Mat::from_fn(num_target_points, dim, |_, _| rng.random_range(-2.0..2.0));
//!
//! // Perform a leaf evaluation
//! tree.evaluate_leaves(&weights.as_ref(), &target_points);
//!
//! println!("Evaluated values at target locations: {:?}", tree.target_values);
//!
//! // Create some more target points
//! let num_target_points = 1000;
//! let target_points = Mat::from_fn(num_target_points, dim, |_, _| rng.random_range(-2.0..2.0));
//!
//! // Perform another leaf evaluation
//! tree.evaluate_leaves(&weights.as_ref(), &target_points);
//!
//! println!("Evaluated values at target locations: {:?}", tree.target_values);
//! ```
//! 
//! # References
//!
//! 1. Fong, W., & Darve, E. (2009).
//!    *[The black-box fast multipole method.](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf)*
//!    *Journal of Computational Physics*, **228**(23), 8712–8725.  
//!
//! 2. Messner, M., Bramas, B., Coulaud, O., & Darve, E. (2012).
//!     *[Optimized M2L kernels for the Chebyshev interpolation-based fast multipole method.](https://arxiv.org/pdf/1210.7292)*  
//!
//! 3. Pouransari, H., & Darve, E. (2015).
//!     *[Optimizing the adaptive fast multipole method for fractal sets.](https://doi.org/10.1137/140962681)*
//!     *SIAM Journal on Scientific Computing*, **37**, A1040–A1066.


mod aca;
mod bbfmm;
mod chebyshev;
mod linear_tree;
mod morton;
mod morton_constants;
mod utils;
mod traits;

#[doc(inline)]
pub use {
    bbfmm::{FmmError, FmmParams, FmmTree, M2LCompressionType},
    traits::KernelFunction,
};
