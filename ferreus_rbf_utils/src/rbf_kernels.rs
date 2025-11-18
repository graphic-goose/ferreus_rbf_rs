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
};
use faer::RowRef;
use ferreus_bbfmm::KernelFunction;
use std::marker::PhantomData;

/// Linear RBF kernel with `phi(r) = -r`.
#[derive(Clone, Debug, Copy)]
pub struct LinearRbfKernel;

impl LinearRbfKernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        -r
    }
}

impl KernelFunction for LinearRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for LinearRbfKernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        LinearRbfKernel
    }
}

/// Thin plate spline RBF kernel with `phi(r) = r^2 log r`.
#[derive(Clone, Debug, Copy)]
pub struct ThinPlateSplineRbfKernel;

impl ThinPlateSplineRbfKernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r.abs() < f64::EPSILON {
            true => 0.0,
            false => r.powi(2) * r.ln(),
        }
    }
}

impl KernelFunction for ThinPlateSplineRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for ThinPlateSplineRbfKernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        ThinPlateSplineRbfKernel
    }
}

/// Cubic RBF kernel with `phi(r) = r^3`.
#[derive(Clone, Debug, Copy)]
pub struct CubicRbfKernel;

impl CubicRbfKernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        r.powi(3)
    }
}

impl KernelFunction for CubicRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }
}

impl KernelFromParams for CubicRbfKernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        CubicRbfKernel
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

    // derived (computed once)
    s2: f64,         // s^2
    ip2: f64,        // (inflexion_point)^2
    near_slope: f64, // total_sill * linear_slope * s
    far_coef: f64,   // total_sill * inv_y_intercept
    _spec: core::marker::PhantomData<S>,
}

impl<S: SpheroidalSpec> SpheroidalRbfKernel<S> {
    #[inline(always)]
    pub fn new(base_range: f64, total_sill: f64) -> Self {
        let c = S::constants();
        let s = c.range_scaling / base_range;
        Self {
            base_range,
            total_sill,
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
        if sr2 <= self.ip2 {
            // near: total_sill - near_slope * r
            let r = r2.sqrt();
            self.total_sill - self.near_slope * r
        } else {
            // far: far_coef / (t^POW * sqrt(t)),  t = 1 + (s r)^2
            let t = 1.0 + sr2;
            self.far_coef / (t.powi(S::POW) * t.sqrt())
        }
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
}

impl<S: SpheroidalSpec> KernelFromParams for SpheroidalRbfKernel<S> {
    #[inline(always)]
    fn from_params(p: &KernelParams) -> Self {
        Self::new(p.base_range, p.total_sill)
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
