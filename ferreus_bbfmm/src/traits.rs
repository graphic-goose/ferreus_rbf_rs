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

    /// Optionally evaluates the kernel from an already-computed squared distance.
    ///
    /// Every kernel used here depends on the two points only through their
    /// separation, so the dense near-field sums can compute a whole batch of
    /// squared distances in one vectorisable pass and then call this, instead of
    /// walking a strided [`RowRef`] pair per interaction.
    ///
    /// Returns `None` if the kernel cannot be expressed this way, which keeps
    /// callers on the scalar [`KernelFunction::evaluate`] path.
    ///
    /// # Contract
    ///
    /// For any pair of points whose squared distance is `r2`, this must return
    /// **exactly** what [`KernelFunction::evaluate`] returns for that pair,
    /// bit-for-bit, including intermediate rounding. In particular a kernel
    /// whose `evaluate` works from the squared distance must not be re-expressed
    /// in terms of `r`, because `r2.sqrt() * r2.sqrt()` does not reproduce `r2`.
    /// Callers are entitled to substitute one for the other freely.
    #[inline(always)]
    fn evaluate_from_distance_sq(&self, _r2: f64) -> Option<f64> {
        None
    }

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

    /// Optionally returns the value and gradient scaling for an already-computed
    /// squared distance, the gradient counterpart of
    /// [`KernelFunction::evaluate_from_distance_sq`].
    ///
    /// Every gradient kernel here has the same shape: the gradient is the
    /// component-wise difference `target - source` multiplied by a factor that
    /// depends only on the separation. Exposing that factor lets the near-field
    /// sums compute a batch of differences and squared distances in one
    /// vectorisable pass.
    ///
    /// Returns `None` if the kernel has no gradient, or cannot be expressed this
    /// way, which keeps callers on the scalar
    /// [`KernelFunction::evaluate_value_gradient`] path.
    ///
    /// # Contract
    ///
    /// For any pair whose squared distance is `r2`, the returned value must equal
    /// what `evaluate_value_gradient` returns, and applying the returned
    /// [`GradientScale`] to `target - source` must reproduce its `gradient_out`,
    /// both bit-for-bit. Note that [`GradientScale::Zero`] is not the same as
    /// `Scale(0.0)`: see its documentation.
    #[inline(always)]
    fn value_and_gradient_from_distance_sq(&self, _r2: f64) -> Option<(f64, GradientScale)> {
        None
    }
}

/// How the component-wise difference `target - source` becomes the gradient.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GradientScale {
    /// Every gradient component is exactly `+0.0`.
    ///
    /// This is what the scalar kernels write when the two points coincide, and it
    /// is deliberately distinct from `Scale(0.0)`: multiplying a negative
    /// difference by zero yields `-0.0`, whose sign bit survives into the
    /// accumulated gradient and can flip a later sum from `+0.0` to `-0.0`.
    Zero,

    /// Multiply each component of the difference by this factor.
    Scale(f64),
}

impl GradientScale {
    /// Applies the scaling to one difference component.
    #[inline(always)]
    pub fn apply(self, difference: f64) -> f64 {
        match self {
            GradientScale::Zero => 0.0,
            GradientScale::Scale(factor) => difference * factor,
        }
    }
}
