/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides utility routines for bounding box computation, row selection, and dense kernel matrices.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::KernelFunction;
use faer::{Mat, MatRef};

/// Tile width for the batched kernel-matrix fill. One tile of squared distances
/// lives on the stack between the two passes below.
const R2_TILE: usize = 64;

/// Fills one column of a dense kernel matrix: the kernel between every target and
/// a single source.
///
/// Splitting this into a squared-distance pass and a kernel-value pass lets the
/// first one vectorise unconditionally - it is pure arithmetic over contiguous
/// per-axis coordinates - instead of being held back by whatever the radial
/// profile does. `target_axes[d]` is the d-th coordinate of every target, which
/// is contiguous because `Mat` is column-major.
///
/// Produces bit-identical values to `kernel.evaluate(target_i, source)`: the
/// squared distance is summed in the same axis order as
/// `ferreus_rbf_utils::distance_sq`, and
/// [`KernelFunction::evaluate_from_distance_sq`] is contractually required to
/// match `evaluate` for that squared distance.
#[inline(always)]
fn fill_kernel_column_impl<K>(
    kernel_function: &K,
    target_axes: &[&[f64]; 3],
    source: &[f64; 3],
    dims: usize,
    out: &mut [f64],
) where
    K: KernelFunction,
{
    let m = out.len();
    let mut r2 = [0.0f64; R2_TILE];
    let mut base = 0usize;

    while base < m {
        let len = R2_TILE.min(m - base);
        let r2 = &mut r2[..len];

        match dims {
            1 => {
                let a = &target_axes[0][base..base + len];
                for (slot, &ax) in r2.iter_mut().zip(a) {
                    let d0 = ax - source[0];
                    *slot = d0 * d0;
                }
            }
            2 => {
                let a = &target_axes[0][base..base + len];
                let b = &target_axes[1][base..base + len];
                for ((slot, &ax), &bx) in r2.iter_mut().zip(a).zip(b) {
                    let d0 = ax - source[0];
                    let d1 = bx - source[1];
                    *slot = d0 * d0 + d1 * d1;
                }
            }
            _ => {
                let a = &target_axes[0][base..base + len];
                let b = &target_axes[1][base..base + len];
                let c = &target_axes[2][base..base + len];
                for (((slot, &ax), &bx), &cx) in r2.iter_mut().zip(a).zip(b).zip(c) {
                    let d0 = ax - source[0];
                    let d1 = bx - source[1];
                    let d2 = cx - source[2];
                    *slot = d0 * d0 + d1 * d1 + d2 * d2;
                }
            }
        }

        for (slot, &d2) in out[base..base + len].iter_mut().zip(r2.iter()) {
            *slot = kernel_function
                .evaluate_from_distance_sq(d2)
                .expect("batched path is only taken when the kernel supports it");
        }

        base += len;
    }
}

/// Accumulates `sum_s kernel(target, source_s) * weights[s]` onto `acc`.
///
/// Same two-pass split as [`fill_kernel_column`], and the running sum is folded
/// in ascending source order starting from the `acc` passed in, so this performs
/// exactly the sequence of additions the scalar loop does - including beginning
/// from whatever the destination already held. The result is bit-identical.
#[inline(always)]
fn accumulate_weighted_kernel_impl<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    mut acc: f64,
) -> f64
where
    K: KernelFunction,
{
    let n = weights.len();
    let mut values = [0.0f64; R2_TILE];
    let mut base = 0usize;

    while base < n {
        let len = R2_TILE.min(n - base);
        let values = &mut values[..len];

        // Squared distances, then kernel values. Pure arithmetic over contiguous
        // per-axis source coordinates, so this vectorises.
        match dims {
            1 => {
                let a = &source_axes[0][base..base + len];
                for (slot, &ax) in values.iter_mut().zip(a) {
                    let d0 = target[0] - ax;
                    *slot = d0 * d0;
                }
            }
            2 => {
                let a = &source_axes[0][base..base + len];
                let b = &source_axes[1][base..base + len];
                for ((slot, &ax), &bx) in values.iter_mut().zip(a).zip(b) {
                    let d0 = target[0] - ax;
                    let d1 = target[1] - bx;
                    *slot = d0 * d0 + d1 * d1;
                }
            }
            _ => {
                let a = &source_axes[0][base..base + len];
                let b = &source_axes[1][base..base + len];
                let c = &source_axes[2][base..base + len];
                for (((slot, &ax), &bx), &cx) in values.iter_mut().zip(a).zip(b).zip(c) {
                    let d0 = target[0] - ax;
                    let d1 = target[1] - bx;
                    let d2 = target[2] - cx;
                    *slot = d0 * d0 + d1 * d1 + d2 * d2;
                }
            }
        }

        let w = &weights[base..base + len];
        for (slot, &wk) in values.iter_mut().zip(w) {
            let v = kernel_function
                .evaluate_from_distance_sq(*slot)
                .expect("batched path is only taken when the kernel supports it");
            *slot = v * wk;
        }

        // Fold in source order: the same addition sequence as the scalar loop.
        for &v in values.iter() {
            acc += v;
        }

        base += len;
    }

    acc
}

// ---------------------------------------------------------------------------
// Runtime SIMD dispatch
//
// These crates are published, so the default build targets baseline x86-64 and
// every hand-written loop above compiles to SSE2 - two f64 lanes. faer's own
// kernels avoid that by dispatching on the CPU at run time, and the two tile
// loops here are worth the same treatment: measured on a Zen 4 part, widening
// them is worth about 1.2x on evaluation and isosurfacing.
//
// `avx512f` and `avx2` only. `fma` is deliberately NOT enabled: contracting
// `a * b + c` into a single fused op removes an intermediate rounding and would
// change results. Wider lanes alone cannot, because each lane performs the same
// IEEE-754 operation as the scalar code, and LLVM will not vectorise the
// in-order accumulation fold without fast-math, which is off.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fill_kernel_column_avx512<K>(
    kernel_function: &K,
    target_axes: &[&[f64]; 3],
    source: &[f64; 3],
    dims: usize,
    out: &mut [f64],
) where
    K: KernelFunction,
{
    fill_kernel_column_impl(kernel_function, target_axes, source, dims, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn fill_kernel_column_avx<K>(
    kernel_function: &K,
    target_axes: &[&[f64]; 3],
    source: &[f64; 3],
    dims: usize,
    out: &mut [f64],
) where
    K: KernelFunction,
{
    fill_kernel_column_impl(kernel_function, target_axes, source, dims, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_weighted_kernel_avx512<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: f64,
) -> f64
where
    K: KernelFunction,
{
    accumulate_weighted_kernel_impl(kernel_function, target, source_axes, weights, dims, acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn accumulate_weighted_kernel_avx<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: f64,
) -> f64
where
    K: KernelFunction,
{
    accumulate_weighted_kernel_impl(kernel_function, target, source_axes, weights, dims, acc)
}

/// Widest instruction set this CPU supports, resolved once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SimdLevel {
    Baseline,
    #[cfg(target_arch = "x86_64")]
    Avx,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

#[cfg(target_arch = "x86_64")]
fn detect_simd_level() -> SimdLevel {
    if std::arch::is_x86_feature_detected!("avx512f") {
        SimdLevel::Avx512
    } else if std::arch::is_x86_feature_detected!("avx") {
        SimdLevel::Avx
    } else {
        SimdLevel::Baseline
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_simd_level() -> SimdLevel {
    SimdLevel::Baseline
}

/// Cached so the feature probe is paid once per process rather than per cell.
pub(crate) fn simd_level() -> SimdLevel {
    use std::sync::OnceLock;
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(detect_simd_level)
}

/// See [`fill_kernel_column_impl`]. Dispatches to the widest available lanes;
/// every variant is the same source and produces bit-identical results.
#[inline]
fn fill_kernel_column<K>(
    kernel_function: &K,
    target_axes: &[&[f64]; 3],
    source: &[f64; 3],
    dims: usize,
    out: &mut [f64],
) where
    K: KernelFunction,
{
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe {
            fill_kernel_column_avx512(kernel_function, target_axes, source, dims, out)
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx => unsafe {
            fill_kernel_column_avx(kernel_function, target_axes, source, dims, out)
        },
        SimdLevel::Baseline => {
            fill_kernel_column_impl(kernel_function, target_axes, source, dims, out)
        }
    }
}

/// See [`accumulate_weighted_kernel_impl`]. Dispatches to the widest available
/// lanes; every variant is the same source and produces bit-identical results.
#[inline]
pub(crate) fn accumulate_weighted_kernel<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: f64,
) -> f64
where
    K: KernelFunction,
{
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe {
            accumulate_weighted_kernel_avx512(
                kernel_function,
                target,
                source_axes,
                weights,
                dims,
                acc,
            )
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx => unsafe {
            accumulate_weighted_kernel_avx(kernel_function, target, source_axes, weights, dims, acc)
        },
        SimdLevel::Baseline => accumulate_weighted_kernel_impl(
            kernel_function,
            target,
            source_axes,
            weights,
            dims,
            acc,
        ),
    }
}

/// The four running sums a gradient-mode near-field interaction feeds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueAndGradient {
    pub value: f64,
    pub gradient: [f64; 3],
}

/// Gradient-mode counterpart of [`accumulate_weighted_kernel_impl`].
///
/// Accumulates `sum_s value_s * w_s` and, per axis, `sum_s grad_s[axis] * w_s`,
/// where `grad_s` is `(target - source)` scaled by the kernel's gradient factor.
///
/// The four sums are independent, and each sees its own terms in ascending source
/// order starting from the value passed in, which is exactly what the scalar loop
/// does - so the result is bit-identical. Only the first `dims` gradient
/// components are touched, matching the scalar path's handling of 1-2 dimensions.
#[inline(always)]
fn accumulate_weighted_kernel_gradients_impl<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    mut acc: ValueAndGradient,
) -> ValueAndGradient
where
    K: KernelFunction,
{
    let n = weights.len();
    let mut diff = [[0.0f64; R2_TILE]; 3];
    let mut values = [0.0f64; R2_TILE];
    let mut base = 0usize;

    while base < n {
        let len = R2_TILE.min(n - base);

        // Differences and squared distances: pure arithmetic over contiguous
        // per-axis source coordinates, so this vectorises. The axis order of the
        // sum matches `ferreus_rbf_utils::fill_diff_and_distance_sq`.
        for d in 0..dims {
            let axis = &source_axes[d][base..base + len];
            for (slot, &sx) in diff[d][..len].iter_mut().zip(axis) {
                *slot = target[d] - sx;
            }
        }
        {
            let (d0, rest) = diff.split_at_mut(1);
            let (d1, d2) = rest.split_at_mut(1);
            for k in 0..len {
                let mut r2 = d0[0][k] * d0[0][k];
                if dims > 1 {
                    r2 += d1[0][k] * d1[0][k];
                }
                if dims > 2 {
                    r2 += d2[0][k] * d2[0][k];
                }
                values[k] = r2;
            }
        }

        // Kernel value and gradient scaling per source, then scale the stored
        // differences in place to become gradient components.
        for k in 0..len {
            let (value, scale) = kernel_function
                .value_and_gradient_from_distance_sq(values[k])
                .expect("batched gradient path is only taken when the kernel supports it");
            values[k] = value;
            for d in 0..dims {
                diff[d][k] = scale.apply(diff[d][k]);
            }
        }

        // Fold each sum in source order.
        let w = &weights[base..base + len];
        for (k, &wk) in w.iter().enumerate() {
            acc.value += values[k] * wk;
        }
        for d in 0..dims {
            for (k, &wk) in w.iter().enumerate() {
                acc.gradient[d] += diff[d][k] * wk;
            }
        }

        base += len;
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_weighted_kernel_gradients_avx512<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: ValueAndGradient,
) -> ValueAndGradient
where
    K: KernelFunction,
{
    accumulate_weighted_kernel_gradients_impl(
        kernel_function,
        target,
        source_axes,
        weights,
        dims,
        acc,
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn accumulate_weighted_kernel_gradients_avx<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: ValueAndGradient,
) -> ValueAndGradient
where
    K: KernelFunction,
{
    accumulate_weighted_kernel_gradients_impl(
        kernel_function,
        target,
        source_axes,
        weights,
        dims,
        acc,
    )
}

/// See [`accumulate_weighted_kernel_gradients_impl`]. Dispatches to the widest
/// available lanes; every variant is bit-identical.
#[inline]
pub(crate) fn accumulate_weighted_kernel_gradients<K>(
    kernel_function: &K,
    target: &[f64; 3],
    source_axes: &[&[f64]; 3],
    weights: &[f64],
    dims: usize,
    acc: ValueAndGradient,
) -> ValueAndGradient
where
    K: KernelFunction,
{
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => unsafe {
            accumulate_weighted_kernel_gradients_avx512(
                kernel_function,
                target,
                source_axes,
                weights,
                dims,
                acc,
            )
        },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx => unsafe {
            accumulate_weighted_kernel_gradients_avx(
                kernel_function,
                target,
                source_axes,
                weights,
                dims,
                acc,
            )
        },
        SimdLevel::Baseline => accumulate_weighted_kernel_gradients_impl(
            kernel_function,
            target,
            source_axes,
            weights,
            dims,
            acc,
        ),
    }
}

/// Whether the batched gradient near-field path can be used.
#[inline(always)]
pub(crate) fn supports_batched_gradients<K>(dims: usize, kernel_function: &K) -> bool
where
    K: KernelFunction,
{
    (1..=3).contains(&dims)
        && kernel_function
            .value_and_gradient_from_distance_sq(1.0)
            .is_some()
}

/// Whether the batched near-field path can be used: 1-3 dimensions and a kernel
/// that can be evaluated from a squared distance.
#[inline(always)]
pub(crate) fn supports_batched_kernel<K>(dims: usize, kernel_function: &K) -> bool
where
    K: KernelFunction,
{
    (1..=3).contains(&dims) && kernel_function.evaluate_from_distance_sq(0.0).is_some()
}

/// Gathers the per-axis target coordinate slices, and the batched path's
/// precondition: 1-3 dimensions and a kernel that can work from a squared
/// distance.
#[inline(always)]
fn batched_fill_inputs<'a, K>(
    target_points: &'a Mat<f64>,
    kernel_function: &K,
) -> Option<[&'a [f64]; 3]>
where
    K: KernelFunction,
{
    let dims = target_points.shape().1;
    if !(1..=3).contains(&dims) || kernel_function.evaluate_from_distance_sq(0.0).is_none() {
        return None;
    }
    let empty: &[f64] = &[];
    let mut axes: [&[f64]; 3] = [empty; 3];
    for d in 0..dims {
        axes[d] = target_points.col_as_slice(d);
    }
    Some(axes)
}

/// Computes the axis aligned bounding box (AABB) extents of a matrix of points.
///
/// Returns a flat vector containing the minimum and maximum values along each column (dimension)
/// of the input matrix. The result is arranged as:
///
/// `[min_0, min_1, ..., min_n, max_0, max_1, ..., max_n]`
///
/// where `n` is the number of columns in the matrix.
#[inline(always)]
pub fn get_pointarray_extents<T>(points: &Mat<T>) -> Vec<T>
where
    T: PartialOrd + Clone,
{
    let ncols = points.shape().1;

    // Initialize extents with min and max values for each column.
    // The first half of the vector stores mins, the second half stores maxs.
    let mut extents: Vec<T> = vec![points.get(0, 0).clone(); 2 * ncols];

    // Initialize the mins and maxs
    for col in 0..ncols {
        extents[col] = points.get(0, col).clone(); // Min value for each column
        extents[col + ncols] = points.get(0, col).clone(); // Max value for each column
    }

    // Iterate over the rows
    for row in points.row_iter() {
        for (col, item) in row.iter().enumerate() {
            // Update min value for the column
            if item < &extents[col] {
                extents[col] = item.clone();
            }
            // Update max value for the column
            if item > &extents[col + ncols] {
                extents[col + ncols] = item.clone();
            }
        }
    }

    extents
}

#[inline(always)]
pub fn select_mat_rows(existing_mat: MatRef<f64>, row_indices: &[usize]) -> Mat<f64> {
    Mat::from_fn(row_indices.len(), existing_mat.ncols(), |i, j| {
        existing_mat.get(row_indices[i], j).clone()
    })
}

#[inline(always)]
pub fn get_a_matrix<K>(
    target_points: &Mat<f64>,
    source_points: &Mat<f64>,
    kernel_function: &K,
) -> Mat<f64>
where
    K: KernelFunction,
{
    let m = target_points.shape().0;
    let n = source_points.shape().0;
    let dims = target_points.shape().1;

    let mut a_matrix = Mat::<f64>::zeros(m, n);

    if let Some(target_axes) = batched_fill_inputs(target_points, kernel_function) {
        for j in 0..n {
            let row = source_points.row(j);
            let mut source = [0.0f64; 3];
            for d in 0..dims {
                source[d] = row[d];
            }
            fill_kernel_column(
                kernel_function,
                &target_axes,
                &source,
                dims,
                a_matrix.col_as_slice_mut(j),
            );
        }
        return a_matrix;
    }

    for j in 0..n {
        let source = source_points.row(j);

        for i in 0..m {
            let target = target_points.row(i);

            a_matrix[(i, j)] = kernel_function.evaluate(target, source);
        }
    }

    a_matrix
}

#[inline(always)]
pub fn get_a_matrix_subset<K>(
    target_points: &Mat<f64>,
    source_points: &Mat<f64>,
    kernel_function: &K,
    rows_start: &usize,
    rows_end: &usize,
    columns_start: &usize,
    columns_end: &usize,
) -> Mat<f64>
where
    K: KernelFunction,
{
    let m = rows_end - rows_start;
    let n = columns_end - columns_start;
    let dims = target_points.shape().1;

    let mut a_matrix = Mat::<f64>::zeros(m, n);

    if let Some(all_axes) = batched_fill_inputs(target_points, kernel_function) {
        let empty: &[f64] = &[];
        let mut target_axes: [&[f64]; 3] = [empty; 3];
        for d in 0..dims {
            target_axes[d] = &all_axes[d][*rows_start..*rows_end];
        }
        for j in 0..n {
            let row = source_points.row(columns_start + j);
            let mut source = [0.0f64; 3];
            for d in 0..dims {
                source[d] = row[d];
            }
            fill_kernel_column(
                kernel_function,
                &target_axes,
                &source,
                dims,
                a_matrix.col_as_slice_mut(j),
            );
        }
        return a_matrix;
    }

    for j in 0..n {
        let source = source_points.row(columns_start + j);

        for i in 0..m {
            let target = target_points.row(rows_start + i);

            a_matrix[(i, j)] = kernel_function.evaluate(target, source);
        }
    }

    a_matrix
}

/// Generates the cartesian product of a slice of values repeated `num_columns` times.
#[inline(always)]
pub fn cartesian_product<T>(values: &[T], num_columns: usize) -> Mat<T>
where
    T: Clone,
{
    let base = values.len();
    let total_rows = base.pow(num_columns as u32);

    Mat::from_fn(total_rows, num_columns, |i, j| {
        let index = (i / base.pow((num_columns - j - 1) as u32)) % base;
        values[index].clone()
    })
}

/// Returns the indices that would sort the input slice.
#[inline(always)]
pub fn argsort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| {
        data[i]
            .partial_cmp(&data[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

#[cfg(test)]
mod batched_kernel_matrix_tests {
    use super::*;
    use faer::RowRef;

    /// A kernel that opts in to the batched path, exercising it end to end.
    struct RadialKernel;
    impl KernelFunction for RadialKernel {
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut r2 = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let d = t - s;
                r2 += d * d;
            }
            -r2.sqrt()
        }
        fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
            Some(-r2.sqrt())
        }
    }

    /// A kernel that does NOT opt in, so the scalar fallback must still be taken.
    struct ScalarOnlyKernel;
    impl KernelFunction for ScalarOnlyKernel {
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut r2 = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let d = t - s;
                r2 += d * d;
            }
            -r2.sqrt()
        }
    }

    fn reference<K: KernelFunction>(
        target_points: &Mat<f64>,
        source_points: &Mat<f64>,
        kernel: &K,
    ) -> Mat<f64> {
        let m = target_points.shape().0;
        let n = source_points.shape().0;
        let mut a = Mat::<f64>::zeros(m, n);
        for j in 0..n {
            let source = source_points.row(j);
            for i in 0..m {
                a[(i, j)] = kernel.evaluate(target_points.row(i), source);
            }
        }
        a
    }

    fn points(n: usize, dims: usize, seed: f64) -> Mat<f64> {
        Mat::<f64>::from_fn(n, dims, |i, j| {
            ((i * 53 + j * 29) as f64 + seed).sin() * 4.0 + seed
        })
    }

    #[test]
    fn batched_get_a_matrix_is_bit_identical() {
        for dims in 1usize..=3 {
            // Sizes either side of the tile width, so partial tiles are covered.
            for &m in &[1usize, 7, 63, 64, 65, 129, 200] {
                for &n in &[1usize, 5, 64, 70] {
                    let t = points(m, dims, 0.0);
                    let s = points(n, dims, 1.5);

                    let got = get_a_matrix(&t, &s, &RadialKernel);
                    let want = reference(&t, &s, &RadialKernel);
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(
                                got[(i, j)].to_bits(),
                                want[(i, j)].to_bits(),
                                "batched get_a_matrix differs at ({i},{j}) dims={dims} m={m} n={n}"
                            );
                        }
                    }

                    // The opt-out kernel must go down the scalar path and agree too.
                    let got_scalar = get_a_matrix(&t, &s, &ScalarOnlyKernel);
                    let want_scalar = reference(&t, &s, &ScalarOnlyKernel);
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(got_scalar[(i, j)].to_bits(), want_scalar[(i, j)].to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batched_get_a_matrix_subset_is_bit_identical() {
        for dims in 1usize..=3 {
            let t = points(180, dims, 0.0);
            let s = points(90, dims, 1.5);
            for &(r0, r1) in &[(0usize, 180usize), (3, 70), (64, 129), (100, 101)] {
                for &(c0, c1) in &[(0usize, 90usize), (5, 9), (10, 74)] {
                    let got = get_a_matrix_subset(&t, &s, &RadialKernel, &r0, &r1, &c0, &c1);
                    for i in 0..(r1 - r0) {
                        for j in 0..(c1 - c0) {
                            let want = RadialKernel.evaluate(t.row(r0 + i), s.row(c0 + j));
                            assert_eq!(
                                got[(i, j)].to_bits(),
                                want.to_bits(),
                                "subset differs at ({i},{j}) dims={dims} rows={r0}..{r1} cols={c0}..{c1}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod batched_accumulation_tests {
    use super::*;
    use faer::RowRef;

    /// Deliberately awkward radial profile: a branch plus a transcendental, so the
    /// batched pass cannot be trivially reassociated by the optimiser.
    struct ProbeKernel;
    impl ProbeKernel {
        #[inline(always)]
        fn phi(r: f64) -> f64 {
            if r < 1e-12 {
                0.0
            } else {
                r * r * r.ln() - 0.5 * r
            }
        }
    }
    impl KernelFunction for ProbeKernel {
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut r2 = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let d = t - s;
                r2 += d * d;
            }
            Self::phi(r2.sqrt())
        }
        fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
            Some(Self::phi(r2.sqrt()))
        }
    }

    /// The scalar loop the batched helper replaces, written out verbatim.
    fn reference_accumulate<K: KernelFunction>(
        kernel: &K,
        target: RowRef<f64>,
        sources: &Mat<f64>,
        weights: &[f64],
        start: f64,
    ) -> f64 {
        let mut acc = start;
        for (s, &w) in weights.iter().enumerate() {
            acc += kernel.evaluate(target, sources.row(s)) * w;
        }
        acc
    }

    #[test]
    fn accumulate_weighted_kernel_matches_scalar_loop_bit_for_bit() {
        let mut cases = 0usize;

        for dims in 1usize..=3 {
            // Spans sub-tile, exact-tile, tile+1 and several-tile source counts.
            for &n in &[1usize, 2, 7, 63, 64, 65, 127, 128, 129, 343, 500] {
                let sources = Mat::<f64>::from_fn(n, dims, |i, j| {
                    ((i * 47 + j * 19) as f64).sin() * 2.0 + 0.1
                });
                // Mixed-sign weights of widely differing magnitude, so any change
                // in summation order shows up in the low bits.
                let weights: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.91).sin() * 10f64.powi((i % 7) as i32 - 3))
                    .collect();

                let axes_owned: Vec<&[f64]> = (0..dims).map(|d| sources.col_as_slice(d)).collect();
                let empty: &[f64] = &[];
                let mut source_axes: [&[f64]; 3] = [empty; 3];
                for d in 0..dims {
                    source_axes[d] = axes_owned[d];
                }

                for &start in &[0.0f64, 1.0, -12345.678, 1e-14, 1e12] {
                    let target_mat = Mat::<f64>::from_fn(1, dims, |_, j| 0.37 + j as f64 * 0.61);
                    let target_row = target_mat.row(0);
                    let mut target = [0.0f64; 3];
                    for d in 0..dims {
                        target[d] = target_mat[(0, d)];
                    }

                    let got = accumulate_weighted_kernel(
                        &ProbeKernel,
                        &target,
                        &source_axes,
                        &weights,
                        dims,
                        start,
                    );
                    let want =
                        reference_accumulate(&ProbeKernel, target_row, &sources, &weights, start);

                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "accumulate_weighted_kernel differs for dims={dims} n={n} start={start}: \
                         {got} vs {want}"
                    );
                    cases += 1;
                }
            }
        }

        assert!(cases > 100, "expected a broad sweep, checked {cases}");
    }

    #[test]
    fn supports_batched_kernel_gates_on_dimension_and_kernel() {
        struct NoBatch;
        impl KernelFunction for NoBatch {
            fn evaluate(&self, _t: RowRef<f64>, _s: RowRef<f64>) -> f64 {
                0.0
            }
        }
        for d in 1..=3 {
            assert!(supports_batched_kernel(d, &ProbeKernel));
            assert!(!supports_batched_kernel(d, &NoBatch));
        }
        assert!(!supports_batched_kernel(0, &ProbeKernel));
        assert!(!supports_batched_kernel(4, &ProbeKernel));
    }
}

#[cfg(test)]
mod simd_dispatch_tests {
    use super::*;
    use faer::RowRef;

    struct ProbeKernel;
    impl KernelFunction for ProbeKernel {
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut r2 = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let d = t - s;
                r2 += d * d;
            }
            -r2.sqrt()
        }
        fn evaluate_from_distance_sq(&self, r2: f64) -> Option<f64> {
            Some(-r2.sqrt())
        }
    }

    /// Every instruction-set variant must agree bit-for-bit with the baseline.
    /// Widening lanes cannot change a per-element IEEE result, and the
    /// accumulation fold stays in order, so any difference here means a variant
    /// picked up contraction or reassociation and the dispatch is unsound.
    #[test]
    fn all_simd_variants_agree_bit_for_bit() {
        let mut compared = 0usize;

        for dims in 1usize..=3 {
            for &n in &[1usize, 3, 31, 63, 64, 65, 127, 200, 343] {
                let sources = Mat::<f64>::from_fn(n, dims, |i, j| {
                    ((i * 41 + j * 23) as f64).sin() * 3.0 + 0.25
                });
                let weights: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 1.31).cos() * 10f64.powi((i % 5) as i32 - 2))
                    .collect();
                let empty: &[f64] = &[];
                let mut axes: [&[f64]; 3] = [empty; 3];
                for d in 0..dims {
                    axes[d] = sources.col_as_slice(d);
                }
                let mut target = [0.0f64; 3];
                for d in 0..dims {
                    target[d] = 0.41 + d as f64 * 0.83;
                }

                for &start in &[0.0f64, -7654.321, 1e11] {
                    let baseline = accumulate_weighted_kernel_impl(
                        &ProbeKernel,
                        &target,
                        &axes,
                        &weights,
                        dims,
                        start,
                    );

                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::arch::is_x86_feature_detected!("avx") {
                            let got = unsafe {
                                accumulate_weighted_kernel_avx(
                                    &ProbeKernel,
                                    &target,
                                    &axes,
                                    &weights,
                                    dims,
                                    start,
                                )
                            };
                            assert_eq!(
                                got.to_bits(),
                                baseline.to_bits(),
                                "avx variant differs: dims={dims} n={n} start={start}"
                            );
                            compared += 1;
                        }
                        if std::arch::is_x86_feature_detected!("avx512f") {
                            let got = unsafe {
                                accumulate_weighted_kernel_avx512(
                                    &ProbeKernel,
                                    &target,
                                    &axes,
                                    &weights,
                                    dims,
                                    start,
                                )
                            };
                            assert_eq!(
                                got.to_bits(),
                                baseline.to_bits(),
                                "avx512 variant differs: dims={dims} n={n} start={start}"
                            );
                            compared += 1;
                        }
                    }

                    // Whatever the dispatcher picked must also agree.
                    let dispatched = accumulate_weighted_kernel(
                        &ProbeKernel,
                        &target,
                        &axes,
                        &weights,
                        dims,
                        start,
                    );
                    assert_eq!(dispatched.to_bits(), baseline.to_bits());
                    compared += 1;
                }

                // And the matrix-fill variants.
                let mut base_col = vec![0.0f64; n];
                fill_kernel_column_impl(&ProbeKernel, &axes, &target, dims, &mut base_col);
                let mut disp_col = vec![0.0f64; n];
                fill_kernel_column(&ProbeKernel, &axes, &target, dims, &mut disp_col);
                for i in 0..n {
                    assert_eq!(
                        disp_col[i].to_bits(),
                        base_col[i].to_bits(),
                        "fill_kernel_column dispatch differs at {i}: dims={dims} n={n}"
                    );
                }
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("avx512f") {
                    let mut c = vec![0.0f64; n];
                    unsafe {
                        fill_kernel_column_avx512(&ProbeKernel, &axes, &target, dims, &mut c)
                    };
                    for i in 0..n {
                        assert_eq!(c[i].to_bits(), base_col[i].to_bits());
                    }
                }
            }
        }

        assert!(
            compared > 100,
            "expected a broad sweep, compared {compared}"
        );
        eprintln!("simd level in use: {:?}", simd_level());
    }
}

#[cfg(test)]
mod batched_gradient_tests {
    use super::*;
    use crate::traits::GradientScale;
    use faer::RowRef;

    /// Branchy profile with a coincident-point case, so both `GradientScale`
    /// variants are exercised.
    struct ProbeGradKernel;
    impl KernelFunction for ProbeGradKernel {
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut r2 = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let d = t - s;
                r2 += d * d;
            }
            -r2.sqrt()
        }
        fn evaluate_value_gradient(
            &self,
            target: RowRef<f64>,
            source: RowRef<f64>,
            gradient_out: &mut [f64],
        ) -> Option<f64> {
            let mut r2 = 0.0;
            for (d, (t, s)) in gradient_out
                .iter_mut()
                .zip(target.iter().zip(source.iter()))
            {
                let diff = t - s;
                *d = diff;
                r2 += diff * diff;
            }
            if r2 <= f64::EPSILON {
                gradient_out.fill(0.0);
                return Some(-r2.sqrt());
            }
            let r = r2.sqrt();
            for g in gradient_out.iter_mut() {
                *g *= -1.0 / r;
            }
            Some(-r)
        }
        fn value_and_gradient_from_distance_sq(&self, r2: f64) -> Option<(f64, GradientScale)> {
            if r2 <= f64::EPSILON {
                return Some((-r2.sqrt(), GradientScale::Zero));
            }
            let r = r2.sqrt();
            Some((-r, GradientScale::Scale(-1.0 / r)))
        }
    }

    /// The scalar gradient loop the batched helper replaces, written out verbatim.
    fn reference<K: KernelFunction>(
        kernel: &K,
        target: RowRef<f64>,
        sources: &Mat<f64>,
        weights: &[f64],
        dims: usize,
        start: ValueAndGradient,
    ) -> ValueAndGradient {
        let mut acc = start;
        for (s, &w) in weights.iter().enumerate() {
            let mut grad_buf = [0.0f64; 3];
            let value = kernel
                .evaluate_value_gradient(target, sources.row(s), &mut grad_buf[..dims])
                .unwrap();
            acc.value += value * w;
            for d in 0..dims {
                acc.gradient[d] += grad_buf[d] * w;
            }
        }
        acc
    }

    #[test]
    fn batched_gradients_match_scalar_loop_bit_for_bit() {
        let mut cases = 0usize;

        for dims in 1usize..=3 {
            for &n in &[1usize, 7, 63, 64, 65, 129, 343] {
                // Include an exactly-coincident source so the Zero branch fires,
                // and negative differences so a -0.0 would be visible.
                let sources = Mat::<f64>::from_fn(n, dims, |i, j| {
                    if i == 3 {
                        0.41 + j as f64 * 0.83
                    } else {
                        ((i * 43 + j * 17) as f64).sin() * 3.0
                    }
                });
                let weights: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.77).sin() * 10f64.powi((i % 5) as i32 - 2))
                    .collect();
                let empty: &[f64] = &[];
                let mut axes: [&[f64]; 3] = [empty; 3];
                for d in 0..dims {
                    axes[d] = sources.col_as_slice(d);
                }
                let target_mat = Mat::<f64>::from_fn(1, dims, |_, j| 0.41 + j as f64 * 0.83);
                let mut target = [0.0f64; 3];
                for d in 0..dims {
                    target[d] = target_mat[(0, d)];
                }

                for &start in &[0.0f64, -0.0, 913.25, -1e9] {
                    let seed = ValueAndGradient {
                        value: start,
                        gradient: [start, -start, start * 0.5],
                    };

                    let got = accumulate_weighted_kernel_gradients(
                        &ProbeGradKernel,
                        &target,
                        &axes,
                        &weights,
                        dims,
                        seed,
                    );
                    let want = reference(
                        &ProbeGradKernel,
                        target_mat.row(0),
                        &sources,
                        &weights,
                        dims,
                        seed,
                    );

                    assert_eq!(
                        got.value.to_bits(),
                        want.value.to_bits(),
                        "value differs: dims={dims} n={n} start={start}"
                    );
                    for d in 0..dims {
                        assert_eq!(
                            got.gradient[d].to_bits(),
                            want.gradient[d].to_bits(),
                            "gradient {d} differs: dims={dims} n={n} start={start}: {} vs {}",
                            got.gradient[d],
                            want.gradient[d]
                        );
                    }
                    // Components beyond `dims` must be left exactly as supplied.
                    for d in dims..3 {
                        assert_eq!(got.gradient[d].to_bits(), seed.gradient[d].to_bits());
                    }
                    cases += 1;
                }
            }
        }

        assert!(cases > 50, "expected a broad sweep, checked {cases}");
    }

    /// All instruction-set variants must agree, for the same reason as the
    /// values-only path.
    #[test]
    fn batched_gradient_simd_variants_agree() {
        let dims = 3usize;
        let n = 200usize;
        let sources = Mat::<f64>::from_fn(n, dims, |i, j| ((i * 37 + j * 13) as f64).cos() * 2.5);
        let weights: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.5).sin()).collect();
        let empty: &[f64] = &[];
        let mut axes: [&[f64]; 3] = [empty; 3];
        for d in 0..dims {
            axes[d] = sources.col_as_slice(d);
        }
        let target = [0.13f64, -0.42, 0.91];
        let seed = ValueAndGradient {
            value: 3.5,
            gradient: [1.0, -2.0, 0.25],
        };

        let baseline = accumulate_weighted_kernel_gradients_impl(
            &ProbeGradKernel,
            &target,
            &axes,
            &weights,
            dims,
            seed,
        );

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                let got = unsafe {
                    accumulate_weighted_kernel_gradients_avx(
                        &ProbeGradKernel,
                        &target,
                        &axes,
                        &weights,
                        dims,
                        seed,
                    )
                };
                assert_eq!(got.value.to_bits(), baseline.value.to_bits());
                for d in 0..3 {
                    assert_eq!(got.gradient[d].to_bits(), baseline.gradient[d].to_bits());
                }
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                let got = unsafe {
                    accumulate_weighted_kernel_gradients_avx512(
                        &ProbeGradKernel,
                        &target,
                        &axes,
                        &weights,
                        dims,
                        seed,
                    )
                };
                assert_eq!(got.value.to_bits(), baseline.value.to_bits());
                for d in 0..3 {
                    assert_eq!(got.gradient[d].to_bits(), baseline.gradient[d].to_bits());
                }
            }
        }
    }
}
