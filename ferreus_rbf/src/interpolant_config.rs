/////////////////////////////////////////////////////////////////////////////////////////////
//
// Specifies kernel, drift, and fitting accuracy options for configuring RBF interpolants.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Specifies kernel, drift, and fitting accuracy options for configuring RBF interpolants.
use ferreus_rbf_utils::{KernelParams, KernelType};
use serde::{Deserialize, Serialize};

/// The implemented orders for the spheroidal kernel.
///
/// See [`RBFKernelType`] for more information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpheroidalOrder {
    Three,
    Five,
    Seven,
    Nine,
}

#[doc = include_str!("../docs/drift.md")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Drift {
    None,
    Constant,
    Linear,
    Quadratic,
}

#[doc = include_str!("../docs/kernel_types.md")]
#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq)]
pub enum RBFKernelType {
    Linear,  // 0
    ThinPlateSpline,  // 1
    Cubic,  // 2
    Spheroidal,  // 3
    WendlandsC2,  // 4
    Spherical,  // 5
    Exponential,  // 6
    Gaussian,  // 7
    Cubic2,  // 8
    InverseMultiquadratic  // 9
}

/// Returns the minimum required [`Drift`] for the provided [`RBFKernelType`]
pub fn get_min_drift(kernel: RBFKernelType) -> Drift {
    match kernel {
        RBFKernelType::Linear => Drift::Constant,
        RBFKernelType::ThinPlateSpline => Drift::Linear,
        RBFKernelType::Cubic => Drift::Linear,
        RBFKernelType::Spheroidal => Drift::None,
        _ => Drift::None
    }
}

/// Defines whether to use relative or absolute stopping criteria for the solver.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FittingAccuracyType {
    /// The mismatch must be reduced by this factor compared to the initial mismatch.
    Relative,

    /// The mismatch must be less than this fixed amount in the same units as your
    /// data values.
    Absolute,
}

/// Defines how closely the interpolated RBF solution should match the input data.
///
/// # Overview
/// When solving an RBF system, the algorithm iteratively refines the coefficients
/// until the predicted values at the data locations are sufficiently close to the
/// given sample values. `FittingAccuracy` tells the solver *when to stop refining*.
///
/// # Tolerance
/// - `tolerance` sets the acceptable mismatch between the model and the input data.
///   Smaller values mean the solution will track the data more tightly,
///   but may require more iterations and time to compute.
///
/// # Tolerance type
/// - 'tolerance_type' sets the type of stopping criteria.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FittingAccuracy {
    pub tolerance: f64,
    pub tolerance_type: FittingAccuracyType,
}

impl Default for FittingAccuracy {
    fn default() -> Self {
        FittingAccuracy {
            tolerance: 1E-6,
            tolerance_type: FittingAccuracyType::Relative,
        }
    }
}

/// A convenience builder for constructing a [`InterpolantSettings`] instance
/// with parameters tailored to the selected kernel type.
///
/// The builder should be called via the [`InterpolantSettings::builder`] method.
///
/// See [`InterpolantSettings`] for details on each field.
#[derive(Debug, Clone, Copy)]
pub struct InterpolantSettingsBuilder {
    pub kernel_type: RBFKernelType,
    pub var_contrib: f64,
    pub spheroidal_order: SpheroidalOrder,
    pub drift: Drift,
    pub nugget: f64,
    pub base_range: f64,
    pub total_sill: f64,
    pub fitting_accuracy: FittingAccuracy,
}

impl InterpolantSettingsBuilder {
    /// Creates a new instance of the [`InterpolantSettingsBuilder`].
    fn new(kernel_type: RBFKernelType) -> Self {
        Self {
            kernel_type: kernel_type,
            var_contrib: 1.0,
            spheroidal_order: SpheroidalOrder::Three,
            drift: get_min_drift(kernel_type),
            nugget: 0.0,
            base_range: 1.0,
            total_sill: 1.0,
            fitting_accuracy: FittingAccuracy::default(),
        }
    }

    /// Sets the kernel contribution. 
    pub fn var_contrib(mut self, var_contrib: f64) -> Self {
        self.var_contrib = var_contrib;
        self
    }

    /// Sets the spheroidal order.
    pub fn spheroidal_order(mut self, spheroidal_order: SpheroidalOrder) -> Self {
        self.spheroidal_order = spheroidal_order;
        self
    }

    /// Sets the drift term.
    pub fn drift(mut self, drift: Drift) -> Self {
        self.drift = drift;
        self
    }

    /// Sets the nugget (smoothing) value.
    pub fn nugget(mut self, nugget: f64) -> Self {
        self.nugget = nugget;
        self
    }

    /// Sets the base range. Only used by spheroidal kernels.
    pub fn base_range(mut self, base_range: f64) -> Self {
        self.base_range = base_range;
        self
    }

    /// Sets the total sill. Only used by spheroidal kernels.
    pub fn total_sill(mut self, total_sill: f64) -> Self {
        self.total_sill = total_sill;
        self
    }

    pub fn fitting_accuracy(mut self, fitting_accuracy: FittingAccuracy) -> Self {
        self.fitting_accuracy = fitting_accuracy;
        self
    }

    /// Builds and returns an instance of [`InterpolantSettings`] from the values
    /// defined in the builder.
    pub fn build(self) -> InterpolantSettings {
        InterpolantSettings {
            kernel_type: self.kernel_type,
            var_contrib: self.var_contrib,
            spheroidal_order: self.spheroidal_order,
            drift: self.drift,
            nugget: self.nugget,
            base_range: self.base_range,
            total_sill: self.total_sill,
            basis_size: 0,
            polynomial_degree: -1,
            fitting_accuracy: self.fitting_accuracy,
        }
    }
}

#[doc = include_str!("../docs/interpolant_settings.md")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InterpolantSettings {
    /// The RBF kernel to use for interpolation.
    pub kernel_type: RBFKernelType,

    /// The kernel contribution. 
    pub var_contrib: f64,

    /// The spheroidal oder.
    pub spheroidal_order: SpheroidalOrder,

    /// The polynomial drift term added to the RBF system.
    pub drift: Drift,

    /// Optional smoothing parameter. A value of `0.0` (default) enforces an exact
    /// fit to all input data. Larger values soften the fit, which can reduce
    /// sensitivity to noisy data.
    pub nugget: f64,

    /// Controls how quickly the interpolant decays with distance from each point. Smaller
    /// values restrict influence to a local neighborhood, while larger values produce
    /// smoother, broader effects.  
    ///
    /// Typically chosen based on the spacing of your data.  
    /// Only used in spheroidal kernels.
    pub base_range: f64,

    /// Sets the overall strength of influence each point exerts. Higher values give
    /// points more weight and stronger local effects. Lower values yield smoother,
    /// less pronounced variation.  
    ///
    /// Works in combination with `base_range` and the kernel degree (α).  
    /// Only used in spheroidal kernels.
    pub total_sill: f64,

    /// Number of additional polynomial basis columns automatically calculated
    /// for the RBF system.
    pub basis_size: usize,

    /// Degree of the polynomial drift automatically calculated for the RBF system.
    pub polynomial_degree: i32,

    /// Desired fitting accuracy and tolerance criteria.
    pub fitting_accuracy: FittingAccuracy,
}

impl InterpolantSettings {
    /// Returns a new [`InterpolantSettingsBuilder`] for the given kernel type.
    pub fn builder(kernel_type: RBFKernelType) -> InterpolantSettingsBuilder {
        InterpolantSettingsBuilder::new(kernel_type)
    }

    // Sets the basis size of the problem based on the drift and dimensionality.
    #[doc(hidden)]
    pub fn set_basis_size(&mut self, dimensions: &usize) {
        let poly_degree = match &self.drift {
            Drift::None => -1,
            Drift::Constant => 0,
            Drift::Linear => 1,
            Drift::Quadratic => 2,
        };

        // The minimum required polynomial degree for each kernel.
        // Spheroidal is strictly positive definite, so doesn't need polynomials.
        let min_degree = match &self.kernel_type {
            RBFKernelType::Linear => 0,
            RBFKernelType::ThinPlateSpline => 1,
            RBFKernelType::Cubic => 1,
            RBFKernelType::Spheroidal => -1,
            _ => -1
        };

        match poly_degree >= min_degree {
            true => {
                let k = &poly_degree + 1;
                if poly_degree < 0 {
                    self.basis_size = 0;
                } else {
                    if dimensions == &1 {
                        self.basis_size = k as usize;
                    } else if dimensions == &2 {
                        self.basis_size = (k * (k + 1) / 2) as usize;
                    } else if dimensions == &3 {
                        self.basis_size = (k * (k + 1) * (k + 2) / 6) as usize;
                    }
                }
                self.polynomial_degree = poly_degree;
            }
            false => panic!("Min degree for kernel: {}", min_degree),
        }
    }
}

impl From<InterpolantSettings> for KernelParams {
    /// Converts a [`InterpolantSettings`] instance into a
    /// [`ferreus_rbf_utils::KernelParams`].
    ///
    /// This allows `.into()` or `KernelParams::from(...)` to be used
    /// directly when passing settings into lower-level utility functions.
    fn from(v: InterpolantSettings) -> Self {
        KernelParams {
            kernel_type: {
                match v.kernel_type {
                    RBFKernelType::Linear => KernelType::LinearRbf,
                    RBFKernelType::ThinPlateSpline => KernelType::ThinPlateSplineRbf,
                    RBFKernelType::Cubic => KernelType::CubicRbf,
                    RBFKernelType::Spheroidal => match v.spheroidal_order {
                        SpheroidalOrder::Three => KernelType::Spheroidal3Rbf,
                        SpheroidalOrder::Five => KernelType::Spheroidal5Rbf,
                        SpheroidalOrder::Seven => KernelType::Spheroidal7Rbf,
                        SpheroidalOrder::Nine => KernelType::Spheroidal9Rbf,
                    },
                    RBFKernelType::WendlandsC2 => KernelType::WendlandsC2Rbf,
                    RBFKernelType::Spherical => KernelType::SphericalRbf,
                    RBFKernelType::Exponential => KernelType::ExponentialRbf,
                    RBFKernelType::Gaussian => KernelType::GaussianRbf,
                    RBFKernelType::Cubic2 => KernelType::Cubic2Rbf,
                    RBFKernelType::InverseMultiquadratic => KernelType::InverseMultiquadraticRbf
                }
            },
            base_range: v.base_range,
            total_sill: v.total_sill,
            var_contrib: v.var_contrib,
        }
    }
}
