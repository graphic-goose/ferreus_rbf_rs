/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements non-RBF kernel functions and their faer-compatible evaluations.
//
// Created on: 16 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{KernelFromParams, KernelParams};
use faer::RowRef;
use ferreus_bbfmm::KernelFunction;

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
}

impl KernelFromParams for OneOverR4Kernel {
    #[inline(always)]
    fn from_params(_: &KernelParams) -> Self {
        OneOverR4Kernel
    }
}
