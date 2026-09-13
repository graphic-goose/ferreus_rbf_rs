/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements non-RBF kernel functions and their faer-compatible evaluations.
//
// Created on: 16 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    KernelFromParams, KernelParams,
    utils::{fill_diff_and_distance_sq, scale_in_place},
};
use faer::RowRef;
use ferreus_bbfmm::{GradientScale, KernelFunction};

/// Laplacian-like kernel with `phi(r) = 1 / r` away from the origin.
#[derive(Clone, Debug, Copy)]
pub struct LaplacianKernel;

impl LaplacianKernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r.abs() < f64::EPSILON {
            true => 0.0,
            false => 1.0 / r,
        }
    }
}

impl KernelFunction for LaplacianKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    /// `evaluate` is `phi(distance_sq(..).sqrt())`, so taking the root here
    /// reproduces it exactly.
    #[inline(always)]
    fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
        Some(self.phi(r2.sqrt()))
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

        let inv_r = 1.0 / r2.sqrt();
        let inv_r3 = inv_r * inv_r * inv_r;
        scale_in_place(gradient_out, -inv_r3);
        Some(inv_r)
    }

    #[inline(always)]
    fn value_and_gradient_from_distance_sq(&self, r2: f64) -> Option<(f64, GradientScale)> {
        if r2 <= f64::EPSILON {
            return Some((0.0, GradientScale::Zero));
        }
        let inv_r = 1.0 / r2.sqrt();
        let inv_r3 = inv_r * inv_r * inv_r;
        Some((inv_r, GradientScale::Scale(-inv_r3)))
    }
}

impl KernelFromParams for LaplacianKernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        LaplacianKernel
    }
}

/// Inverse-square kernel with `phi(r) = 1 / r^2` away from the origin.
#[derive(Clone, Debug, Copy)]
pub struct OneOverR2Kernel;

impl OneOverR2Kernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r.abs() < f64::EPSILON {
            true => 0.0,
            false => 1.0 / r.powi(2),
        }
    }
}

impl KernelFunction for OneOverR2Kernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    /// `evaluate` is `phi(distance_sq(..).sqrt())`, so taking the root here
    /// reproduces it exactly.
    #[inline(always)]
    fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
        Some(self.phi(r2.sqrt()))
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

        let inv_r4 = 1.0 / (r2 * r2);
        for g in gradient_out.iter_mut() {
            *g *= -2.0 * inv_r4;
        }
        Some(1.0 / r2)
    }

    #[inline(always)]
    fn value_and_gradient_from_distance_sq(&self, r2: f64) -> Option<(f64, GradientScale)> {
        if r2 <= f64::EPSILON {
            return Some((0.0, GradientScale::Zero));
        }
        let inv_r4 = 1.0 / (r2 * r2);
        Some((1.0 / r2, GradientScale::Scale(-2.0 * inv_r4)))
    }
}

impl KernelFromParams for OneOverR2Kernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        OneOverR2Kernel
    }
}

/// Inverse-quartic kernel with `phi(r) = 1 / r^4` away from the origin.
#[derive(Clone, Debug, Copy)]
pub struct OneOverR4Kernel;

impl OneOverR4Kernel {
    #[inline(always)]
    pub fn phi(&self, r: f64) -> f64 {
        match r.abs() < f64::EPSILON {
            true => 0.0,
            false => 1.0 / r.powi(4),
        }
    }
}

impl KernelFunction for OneOverR4Kernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let r = crate::get_distance(target, source);
        self.phi(r)
    }

    /// `evaluate` is `phi(distance_sq(..).sqrt())`, so taking the root here
    /// reproduces it exactly.
    #[inline(always)]
    fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
        Some(self.phi(r2.sqrt()))
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

        let inv_r6 = 1.0 / (r2 * r2 * r2);
        scale_in_place(gradient_out, -4.0 * inv_r6);
        Some(1.0 / (r2 * r2))
    }

    #[inline(always)]
    fn value_and_gradient_from_distance_sq(&self, r2: f64) -> Option<(f64, GradientScale)> {
        if r2 <= f64::EPSILON {
            return Some((0.0, GradientScale::Zero));
        }
        let inv_r6 = 1.0 / (r2 * r2 * r2);
        Some((1.0 / (r2 * r2), GradientScale::Scale(-4.0 * inv_r6)))
    }
}

impl KernelFromParams for OneOverR4Kernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        OneOverR4Kernel
    }
}
