/////////////////////////////////////////////////////////////////////////////////////////////
//
// Exposes the public API and high-level documentation for fast global RBF interpolation.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Fast global Radial Basis Function (RBF) interpolation.
//!
//! Radial Basis Function (RBF) interpolation is a powerful but computationally
//! expensive technique. Direct solvers (e.g. LU factorisation) require **O(N²)**
//! memory and **O(N³)** operations, which quickly becomes impractical beyond
//! ~10,000 points on a typical machine.
//!
//! This crate provides a scalable alternative by combining two key techniques:
//!
//! - **Domain Decomposition** - following `1`, used as a preconditioner within a
//!   Flexible Generalised Minimal Residual (FGMRES) iterative solver.
//! - **The Fast Multipole Method (FMM)** - via the [`ferreus_bbfmm`] crate, used as a
//!   fast evaluator to reduce per-iteration cost.
//!
//! Together, these methods reduce the overall complexity to **O(N log N)**,
//! enabling efficient interpolation on datasets with millions of points in
//! up to three dimensions.
//! 
//! Check out the examples directory in the repository for more examples of usage.
//!
//! # Features
//! - Supports 1D, 2D, and 3D input domains
//! - Scales efficiently to datasets with over 1,000,000 input source points
//! - Optional global trend transforms to capture large-scale patterns in the data
//! - Provides fast 3D isosurface extraction using a surface-following,
//!   non-adaptive Surface Nets method
//! - Built on [`faer`](https://docs.rs/faer/latest/faer/) for linear algebra, avoiding complex build dependencies
//!
//! # Examples
//!
//! ```
//! use ferreus_rbf::{
//!     RBFInterpolator,
//!     interpolant_config::{
//!         InterpolantSettings, 
//!         RBFKernelType,
//!         FittingAccuracy, 
//!         FittingAccuracyType
//!     },
//!     generate_random_points,
//!     RBFTestFunctions,
//! };
//! use ferreus_rbf_utils;
//!
//! // Generate some random data in the unit square
//! let dimensions = 2;
//! let num_points = 100;
//! let source_points = generate_random_points(num_points, dimensions, Some(42));
//!
//! // Assign some values to the source points using Franke's function
//! let source_values = RBFTestFunctions::franke_2d(&source_points);
//! 
//! // Define an absolute fitting accuracy tolerance
//! let fitting_accuracy = FittingAccuracy {
//!     tolerance: 0.01,
//!     tolerance_type: FittingAccuracyType::Absolute,
//! };
//!
//! // Create an InterpolantSettings instance
//! let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear)
//!     .fitting_accuracy(fitting_accuracy)
//!     .build();
//!
//! // Setup and solve the RBF
//! let mut rbfi = RBFInterpolator::builder(source_points, source_values, interpolant_settings).build();
//! 
//! // Evaluate the RBF at the input source locations
//! let fitted = rbfi.evaluate_at_source(true);
//! 
//! // Test that the interpolated values match the input source values to the requested tolerance.
//! let max_diff: f64 = rbfi
//!     .point_values
//!     .col(0)
//!     .iter()
//!     .zip(fitted.col(0).iter())
//!     .fold(0.0, |acc, (a, b)| acc.max((a - b).abs()));
//! 
//! assert!(max_diff < fitting_accuracy.tolerance)
//! ```
//!
//! # References
//! 1.  R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis
//!     function interpolation equations: domain decomposition methods. SIAM J. Sci.
//!     Comput., 22(5):1717–1740 (electronic), 2000.
//! 2.  Haase, G., Martin, D., Schiffmann, P., Offner, G. (2018). A Domain Decomposition
//!     Multilevel Preconditioner for Interpolation with Radial Basis Functions.
//!     In: Lirkov, I., Margenov, S. (eds) Large-Scale Scientific Computing. LSSC 2017.
//! 3.  Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab. World Scientific Publishing Co.
//! 4.  J. B. Cherrie. Fast Evaluation of Radial Basis Functions: Theory and Application.
//!     PhD thesis, University of Canterbury, 2000.
pub mod interpolant_config;

mod common;

mod rbf;

mod domain;

mod polynomials;

mod preconditioning;

mod rtree;

mod kdtree;

mod linalg;

mod surfacing;

mod iterative_solvers;

mod global_trend;

pub mod progress;

pub mod config;

mod rbf_test_functions;

pub use {
    common::{
        create_evaluation_grid, pad_and_snap_extents,
        generate_random_points, point_arrays_to_csv, csv_to_point_arrays,
    },
    global_trend::GlobalTrend,
    rbf::{
        RBFInterpolator, RBFInterpolatorBuilder,
        ModelIOError, Coefficients,
    },
    rbf_test_functions::RBFTestFunctions,
    surfacing::surfacing_io::save_obj,
};
