/////////////////////////////////////////////////////////////////////////////////////////////
//
// Declares configuration types for domain decomposition, FMM compression, and solver options.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Declares configuration types for domain decomposition, FMM compression, and solver options.
use ferreus_bbfmm::{self, M2LCompressionType};
use serde::{Deserialize, Serialize};
use crate::interpolant_config::RBFKernelType;

/// Parameters controlling construction of the **domain decomposition hierarchy**.
/// 
/// `ferreus_rbf` employs a *domain decomposition preconditioner* to accelerate
/// convergence of the iterative RBF solver. The algorithm recursively partitions
/// the input point cloud into a hierarchy of overlapping subdomains, within which
/// local RBF systems are solved directly and combined to form a global preconditioner.
/// 
/// This struct defines the key thresholds and ratios governing how that
/// hierarchy is generated - for example, the number of points permitted per
/// leaf domain, how much overlap occurs between neighboring subdomains, and
/// the scale at which coarse levels are formed.
/// 
/// ### Intended Usage
/// This configuration is part of the public API mainly for **developers and
/// advanced users** who wish to experiment with or tune the decomposition
/// process. For example, increasing subdomain overlap and coarse ratio can
/// improve convergence, but at the cost of higher memory usage.
/// In general, the default values have been selected to provide
/// a robust trade-off between memory usage and solver performance across
/// a wide range of problem sizes.
/// 
/// ### Default Values
/// - `leaf_threshold`: `1024`
/// - `overlap_quota`: `0.5`
/// - `coarse_ratio`: `0.125`
/// - `coarse_threshold`: `4096`
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct DDMParams {
    /// Target maximum number of points (internal + overlapping)
    /// within a leaf domain.
    pub leaf_threshold: usize,

    /// Overlap fraction. Larger fraction will add more overlapping
    /// points to each leaf domain.
    pub overlap_quota: f64,

    /// Fraction of **internal** points per leaf promoted to the next
    /// coarser level.
    pub coarse_ratio: f64,

    /// Maximum number of points in the coarsest level.
    pub coarse_threshold: usize,
}

impl Default for DDMParams {
    fn default() -> Self {
        DDMParams {
            leaf_threshold: 1024,
            overlap_quota: 0.5,
            coarse_ratio: 0.125,
            coarse_threshold: 4096,
        }
    }
}

/// Supported M2L operator compression methods.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum FmmCompressionType {
    /// No compression is applied.
    None,

    /// A truncated Singular Value Decompositio (SVD) is 
    /// performed on the M2L operators.
    SVD,

    /// Adaptive cross approximation (ACA) is performed on the M2L
    /// operators, followed by SVD recompression.
    ACA,
}

impl From<FmmCompressionType> for M2LCompressionType {
    fn from(value: FmmCompressionType) -> M2LCompressionType {
        match value {
            FmmCompressionType::None => M2LCompressionType::None,
            FmmCompressionType::SVD => M2LCompressionType::SVD,
            FmmCompressionType::ACA => M2LCompressionType::ACA,
        }
    }
}

#[doc = include_str!("../docs/params.md")]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Params {
    /// Iterative solver to use when fitting the RBF system.
    pub solver_type: Solvers,

    /// Parameters controlling domain decomposition preconditioning.
    pub ddm_params: DDMParams,

    /// Parameters controlling the fast multipole method (FMM).
    pub fmm_params: FmmParams,

    /// Threshold below which the system is solved directly
    /// rather than using iterative methods.
    pub naive_solve_threshold: usize,

    /// Whether to enforce uniqueness checks on the input dataset.
    pub test_unique: bool,
}

impl Params {
    /// Returns a new [`ParamsBuilder`] with defaults appropriate for
    /// the specified kernel type.
    pub fn builder(kernel_type: RBFKernelType) -> ParamsBuilder {
        ParamsBuilder::new(kernel_type)
    }
}

/// A convenience builder for constructing a [`Params`] instance
/// with parameters tailored to the selected kernel type.
/// 
/// The builder should be called via the [`Params::builder`] method.
///
/// See [`Params`] for details on each field.
#[derive(Debug, Clone)]
pub struct ParamsBuilder {
    pub solver_type: Solvers,
    pub ddm_params: DDMParams,
    pub fmm_params: FmmParams,
    pub naive_solve_threshold: usize,
    pub test_unique: bool,
}

impl ParamsBuilder {
    /// Creates a new builder with defaults appropriate for the given kernel type.
    fn new(kernel_type: RBFKernelType) -> Self {
        Self {
            solver_type: Solvers::FGMRES,
            ddm_params: DDMParams::default(),
            fmm_params: FmmParams::new_defaults(kernel_type),
            naive_solve_threshold: 4096,
            test_unique: true,
        }
    }

    /// Sets the solver type.
    pub fn solver_type(mut self, solver_type: Solvers) -> Self {
        self.solver_type = solver_type;
        self
    }

    /// Sets the domain decomposition parameters.
    pub fn ddm_params(mut self, ddm_params: DDMParams) -> Self {
        self.ddm_params = ddm_params;
        self
    }

    /// Sets the FMM parameters.
    pub fn fmm_params(mut self, fmm_params: FmmParams) -> Self {
        self.fmm_params = fmm_params;
        self
    }

    /// Sets the threshold for switching to direct solves.
    pub fn naive_solve_threshold(mut self, naive_solve_threshold: usize) -> Self {
        self.naive_solve_threshold = naive_solve_threshold;
        self
    }

    /// Enables or disables uniqueness checks.
    pub fn test_unique(mut self, test_unique: bool) -> Self {
        self.test_unique = test_unique;
        self
    }

    /// Builds and returns a [`Params`] instance.
    pub fn build(self) -> Params {
        Params {
            solver_type: self.solver_type,
            ddm_params: self.ddm_params,
            fmm_params: self.fmm_params,
            naive_solve_threshold: self.naive_solve_threshold,
            test_unique: self.test_unique,
        }
    }
}

/// Returns the default FMM interpolation order for the given kernel type.
///
/// These defaults are based on empirical accuracy/performance trade-offs:
/// - `Linear`: 7
/// - `ThinPlateSpline`: 9
/// - `Cubic`: 11
/// - All others: 7
fn get_default_fmm_interpolation_order(kernel_type: RBFKernelType) -> usize {
    match kernel_type {
        RBFKernelType::Linear => 7,
        RBFKernelType::ThinPlateSpline => 9,
        RBFKernelType::Cubic => 11,
        _ => 7,
    }
}

#[doc = include_str!("../docs/fmm_params.md")]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct FmmParams {
    /// Number of Chebyshev interpolation nodes per dimension.
    pub interpolation_order: usize,

    /// Maximum number of points per cell before it is subdivided.
    pub max_points_per_cell: usize,

    /// Whether to compress multipole-to-local (M2L) operators using ACA
    /// with truncated SVD.
    pub compression_type: FmmCompressionType,

    /// Tolerance threshold for M2L compression.
    pub epsilon: f64,

    /// Number of target points to evaluate in each chunk.
    pub eval_chunk_size: usize,
}

impl From<FmmParams> for ferreus_bbfmm::FmmParams {
    fn from(v: FmmParams) -> Self {
        ferreus_bbfmm::FmmParams {
            max_points_per_cell: v.max_points_per_cell,
            compression_type: v.compression_type.into(),
            epsilon: v.epsilon,
            eval_chunk_size: v.eval_chunk_size,
        }
    }
}

impl FmmParams {
    /// Returns a new [`FmmParams`] populated with defaults appropriate for
    /// the given kernel type.
    pub fn new_defaults(kernel_type: RBFKernelType) -> Self {
        let default_interpolation_order = get_default_fmm_interpolation_order(kernel_type);
        Self {
            interpolation_order: default_interpolation_order,
            max_points_per_cell: 256,
            compression_type: FmmCompressionType::ACA,
            epsilon: 10f64.powi(-(default_interpolation_order as i32)),
            eval_chunk_size: 1024,
        }
    }
}

/// Enum for the available iterative solvers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Solvers {
    /// Domain Decomposition solver.
    DDM,

    /// Flexible generalised minimal residual method (FGMRES) solver.
    FGMRES,
}

impl Default for Solvers {
    fn default() -> Self {
        Solvers::FGMRES
    }
}
