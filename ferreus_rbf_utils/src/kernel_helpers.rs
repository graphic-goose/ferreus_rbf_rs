/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides parameter and builder types for configuring RBF kernels.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use serde::{Deserialize, Serialize};
use crate::utils::KernelType;

/// Defines the [`KernelType`] to use, along with parameter
/// values for spheroidal kernels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KernelParams {
    /// KernelType enum variant to use.
    pub kernel_type: KernelType,

    /// Controls how quickly the interpolant decays with distance from each point. 
    /// Smaller values restrict influence to a local neighborhood, while larger values
    /// produce smoother, broader effects.
    /// 
    /// Typically chosen based on the spacing of your data.
    /// Only used in spheroidal kernels.
    pub base_range: f64,

    /// Sets the overall strength of influence each point exerts. Higher values give
    /// points more weight and stronger local effects. Lower values yield smoother, 
    /// less pronounced variation.
    ///
    /// Works in combination with base_range and the kernel degree.
    /// Only used in spheroidal kernels.
    pub total_sill: f64,
}

impl KernelParams {
    /// Begins building a [`KernelParams`] instance for the given kernel type.
    pub fn builder(kernel_type: KernelType) -> KernelParamsBuilder {
        KernelParamsBuilder {
            kernel_type,
            base_range: 1.0,
            total_sill: 1.0,
        }
    }
}

/// Builder for [`KernelParams`] that provides sensible defaults.
#[derive(Debug, Clone, Copy)]
pub struct KernelParamsBuilder {
    kernel_type: KernelType,
    base_range: f64,
    total_sill: f64,
}

impl KernelParamsBuilder {
    /// Sets the `base_range` parameter on the builder.
    pub fn base_range(mut self, v: f64) -> Self {
        self.base_range = v;
        self
    }

    /// Sets the `total_sill` parameter on the builder.
    pub fn total_sill(mut self, v: f64) -> Self {
        self.total_sill = v;
        self
    }

    /// Finalises the builder into a [`KernelParams`] value.
    pub fn build(self) -> KernelParams {
        assert!(self.base_range > 0.0);
        assert!(self.total_sill <= self.base_range);
        KernelParams {
            kernel_type: self.kernel_type,
            base_range: self.base_range,
            total_sill: self.total_sill,
        }
    }
}
