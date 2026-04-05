/////////////////////////////////////////////////////////////////////////////////////////////
//
// Declares the kernel evaluation trait used by BBFMM for black-box kernel functions.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::RowRef;

/// Evaluates a kernel function between a target and source point.
///
/// Implementors define how the kernel is computed given two
/// [`faer::RowRef<f64>`](https://docs.rs/faer/latest/faer/row/type.RowRef.html)
/// arguments representing the target and source locations. This interface
/// supports 1D–3D inputs without requiring separate traits.
pub trait KernelFunction {
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64;

    /// Optionally evaluates both value and gradient in a single call.
    ///
    /// Returns `None` if the kernel does not support gradients.
    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        _target: RowRef<f64>,
        _source: RowRef<f64>,
        _gradient_out: &mut [f64],
    ) -> Option<f64> {
        None
    }
}
