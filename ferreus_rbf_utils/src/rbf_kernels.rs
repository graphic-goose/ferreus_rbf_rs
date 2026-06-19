/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements the concrete RBF kernel functions and their faer-compatible evaluations.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    KernelFromParams, KernelParams,
    constants::{
        SPHEROIDAL_CONSTANTS_FIVE, SPHEROIDAL_CONSTANTS_NINE, SPHEROIDAL_CONSTANTS_SEVEN,
        SPHEROIDAL_CONSTANTS_THREE, SpheroidalConstants,
    },
    utils::{fill_diff_and_distance_sq, scale_in_place},
};
use faer::RowRef;
use ferreus_bbfmm::KernelFunction;
use std::marker::PhantomData;

/// Linear RBF kernel with `phi(r) = -r`.
#[derive(Clone, Debug, Copy)]
pub struct LinearRbfKernel {
    /// Variance contribution for this kernel.
    pub var_contrib: f64,
}

impl LinearRbfKernel {
    /// Creates a new LinearRbfKernel with the specified variance contribution.
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        -r * self.var_contrib
    }
}

impl KernelFunction for LinearRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        target: RowRef<f64>,
        source: RowRef<f64>,
        gradient_out: &mut [f64],
    ) -> Option<f64> {
        let r2 = fill_diff_and_distance_sq(target, source, gradient_out);
        if r2 <= f64::EPSILON {
            gradient_out.fill(0.0);
            return Some(-r2.sqrt() * self.var_contrib);
        }

        let r = r2.sqrt();
        scale_in_place(gradient_out, -self.var_contrib / r);
        Some(-r * self.var_contrib)
    }
}

impl KernelFromParams for LinearRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Thin plate spline RBF kernel with `phi(r) = r^2 log r`.
#[derive(Clone, Debug, Copy)]
pub struct ThinPlateSplineRbfKernel {
    /// Variance contribution for this kernel.
    pub var_contrib: f64,
}

impl ThinPlateSplineRbfKernel {
    /// Creates a new ThinPlateSplineRbfKernel with the specified variance contribution.
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r.abs() < f64::EPSILON {
            true => 0.0,
            false => self.var_contrib * r.powi(2) * r.ln(),
        }
    }
}

impl KernelFunction for ThinPlateSplineRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        target: RowRef<f64>,
        source: RowRef<f64>,
        gradient_out: &mut [f64],
    ) -> Option<f64> {
        let r2 = fill_diff_and_distance_sq(target, source, gradient_out);

        if r2 <= f64::EPSILON {
            gradient_out.fill(0.0);
            return Some(0.0);
        }

        let r = r2.sqrt();
        let ln_r = r.ln();
        let factor = self.var_contrib * (2.0 * ln_r + 1.0);
        scale_in_place(gradient_out, factor);
        Some(self.var_contrib * r2 * ln_r)
    }
}

impl KernelFromParams for ThinPlateSplineRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Cubic RBF kernel with `phi(r) = r^3`.
#[derive(Clone, Debug, Copy)]
pub struct CubicRbfKernel {
    /// Variance contribution for this kernel.
    pub var_contrib: f64,
}

impl CubicRbfKernel {
    /// Creates a new CubicRbfKernel with the specified variance contribution.
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        self.var_contrib * r.powi(3)
    }
}

impl KernelFunction for CubicRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        target: RowRef<f64>,
        source: RowRef<f64>,
        gradient_out: &mut [f64],
    ) -> Option<f64> {
        let r2 = fill_diff_and_distance_sq(target, source, gradient_out);

        if r2 <= f64::EPSILON {
            gradient_out.fill(0.0);
            return Some(0.0);
        }

        let r = r2.sqrt();
        let factor = 3.0 * self.var_contrib * r;
        scale_in_place(gradient_out, factor);
        Some(self.var_contrib * r2 * r)
    }
}

impl KernelFromParams for CubicRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Compile-time specification for a spheroidal kernel order and its tuned constants.
pub trait SpheroidalSpec {
    const POW: i32; // 1 for order=3, 2 for order=5, 3 for order=7, 4 for order=9
    fn constants() -> &'static SpheroidalConstants;
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Order3;

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Order5;

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Order7;

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Order9;

impl SpheroidalSpec for Order3 {
    const POW: i32 = 1;
    #[inline(always)]
    fn constants() -> &'static SpheroidalConstants {
        &SPHEROIDAL_CONSTANTS_THREE
    }
}
impl SpheroidalSpec for Order5 {
    const POW: i32 = 2;
    #[inline(always)]
    fn constants() -> &'static SpheroidalConstants {
        &SPHEROIDAL_CONSTANTS_FIVE
    }
}
impl SpheroidalSpec for Order7 {
    const POW: i32 = 3;
    #[inline(always)]
    fn constants() -> &'static SpheroidalConstants {
        &SPHEROIDAL_CONSTANTS_SEVEN
    }
}
impl SpheroidalSpec for Order9 {
    const POW: i32 = 4;
    #[inline(always)]
    fn constants() -> &'static SpheroidalConstants {
        &SPHEROIDAL_CONSTANTS_NINE
    }
}

/// Spheroidal RBF kernel, parameterised by a compile-time order specification.
#[derive(Clone, Debug, Copy)]
pub struct SpheroidalRbfKernel<S: SpheroidalSpec> {
    // user inputs
    pub base_range: f64,
    pub total_sill: f64,
    pub var_contrib: f64,

    // derived (computed once)
    s2: f64,         // s^2
    ip2: f64,        // (inflexion_point)^2
    near_slope: f64, // total_sill * linear_slope * s
    far_coef: f64,   // total_sill * inv_y_intercept
    _spec: core::marker::PhantomData<S>,
}

impl<S: SpheroidalSpec> SpheroidalRbfKernel<S> {
    #[inline(always)]
    pub fn new(base_range: f64, total_sill: f64, var_contrib: f64) -> Self {
        let c = S::constants();
        let s = c.range_scaling / base_range;
        Self {
            base_range,
            total_sill,
            var_contrib,
            s2: s * s,
            ip2: c.inflexion_point * c.inflexion_point,
            near_slope: total_sill * c.linear_slope * s,
            far_coef: total_sill * c.inv_y_intercept,
            _spec: PhantomData,
        }
    }

    #[inline(always)]
    pub fn eval_r2(&self, r2: f64) -> f64 {
        let sr2 = self.s2 * r2;
        let result = if sr2 <= self.ip2 {
            // near: total_sill - near_slope * r
            let r = r2.sqrt();
            self.total_sill - self.near_slope * r
        } else {
            // far: far_coef / (t^POW * sqrt(t)),  t = 1 + (s r)^2
            let t = 1.0 + sr2;
            self.far_coef / (t.powi(S::POW) * t.sqrt())
        };
        self.var_contrib * result
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        let r2 = r * r;
        self.eval_r2(r2)
    }
}

impl<S: SpheroidalSpec> KernelFunction for SpheroidalRbfKernel<S> {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r2 = get_distance_sq(target, source);
        self.eval_r2(r2)
    }

    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        target: RowRef<f64>,
        source: RowRef<f64>,
        gradient_out: &mut [f64],
    ) -> Option<f64> {
        let r2 = fill_diff_and_distance_sq(target, source, gradient_out);

        if r2 <= f64::EPSILON {
            gradient_out.fill(0.0);
            return Some(self.eval_r2(r2));
        }

        let sr2 = self.s2 * r2;
        if sr2 <= self.ip2 {
            let inv_r = 1.0 / r2.sqrt();
            let factor = -self.var_contrib * self.near_slope * inv_r;
            scale_in_place(gradient_out, factor);
            return Some(self.eval_r2(r2));
        }

        let t = 1.0 + sr2;
        let p = S::POW as f64 + 0.5;
        let denom = t.powf(p + 1.0);
        let factor = -2.0 * p * self.s2 * self.var_contrib * self.far_coef / denom;
        scale_in_place(gradient_out, factor);
        Some(self.eval_r2(r2))
    }
}

impl<S: SpheroidalSpec> KernelFromParams for SpheroidalRbfKernel<S> {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.base_range, p.total_sill, p.var_contrib)
    }
}

/// Order-3 spheroidal RBF kernel type alias.
pub type Spheroidal3RbfKernel = SpheroidalRbfKernel<Order3>;
/// Order-5 spheroidal RBF kernel type alias.
pub type Spheroidal5RbfKernel = SpheroidalRbfKernel<Order5>;
/// Order-7 spheroidal RBF kernel type alias.
pub type Spheroidal7RbfKernel = SpheroidalRbfKernel<Order7>;
/// Order-9 spheroidal RBF kernel type alias.
pub type Spheroidal9RbfKernel = SpheroidalRbfKernel<Order9>;

/// WendlandsC2 RBF kernel.
#[derive(Clone, Debug, Copy)]
pub struct WendlandsC2RbfKernel {
    /// Variance contribution for this kernel.
    pub var_contrib: f64,
}

impl WendlandsC2RbfKernel {
    /// Creates a new WendlandsC2RbfKernel with the specified variance contribution.
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r < 1.0 {
            true => {
                let v = 1.0 - r;
                self.var_contrib * (v * v * v * v) * (4.0 * r + 1.0)
            }
            false => 0.0,
        }
    }
}

impl KernelFunction for WendlandsC2RbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for WendlandsC2RbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Spherical RBF kernel.
#[derive(Clone, Debug, Copy)]
pub struct SphericalRbfKernel {
    pub var_contrib: f64,
}

impl SphericalRbfKernel {
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r < 1.0 {
            true => {
                self.var_contrib * (1.0 - r * (1.5 - 0.5 * r * r))
            }
            false => 0.0,
        }
    }
}

impl KernelFunction for SphericalRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for SphericalRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Exponential RBF kernel.
#[derive(Clone, Debug, Copy)]
pub struct ExponentialRbfKernel {
    pub var_contrib: f64,
}

impl ExponentialRbfKernel {
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        self.var_contrib * (-3.0 * r).exp()
    }
}

impl KernelFunction for ExponentialRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for ExponentialRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Gaussian RBF kernel.
#[derive(Clone, Debug, Copy)]
pub struct GaussianRbfKernel {
    pub var_contrib: f64,
}

impl GaussianRbfKernel {
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        self.var_contrib * (-3.0 * r * r).exp()
    }
}

impl KernelFunction for GaussianRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for GaussianRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Cubic RBF kernel as defined by Chiles, Delfiner (1999).
#[derive(Clone, Debug, Copy)]
pub struct Cubic2RbfKernel {
    pub var_contrib: f64,
}

impl Cubic2RbfKernel {
    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r < 1.0 {
            true => {
                let d2 = r * r;
                let d3 = d2 * r;
                let d5 = d3 * d2;
                let d7 = d5 * d2;
                self.var_contrib * (1.0 - 7.0 * d2 + 8.75 * d3 - 3.5 * d5 + 0.75 * d7)
            }
            false => 0.0,
        }
    }
}

impl KernelFunction for Cubic2RbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for Cubic2RbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Inverse Multiquadratic RBF kernel.
///
/// Kernel decay is scaled to ~align with other kernels.
#[derive(Clone, Debug, Copy)]
pub struct InverseMultiquadraticRbfKernel {
    pub var_contrib: f64,
}

impl InverseMultiquadraticRbfKernel {
    const KM_SQ: f64 = 42.25;  // 6.5 ^ 2

    #[inline(always)]
    pub fn new(var_contrib: f64) -> Self {
        Self { var_contrib }
    }

    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        self.var_contrib / (1.0 + r * r * Self::KM_SQ).sqrt()
    }
}

impl KernelFunction for InverseMultiquadraticRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for InverseMultiquadraticRbfKernel {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.var_contrib)
    }
}

/// Returns the squared Euclidean distance between two points.
#[inline(always)]
pub fn get_distance_sq(target: RowRef<f64>, source: RowRef<f64>) -> f64 {
    let mut dist = 0.0;
    for (t, s) in target.iter().zip(source.iter()) {
        let diff = t - s;
        dist += diff * diff;
    }
    dist
}