/////////////////////////////////////////////////////////////////////////////////////////////
//
// Supplies general-purpose utilities for matrices, distances, FMM trees, and kernel helpers.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{KernelFromParams, KernelParams};
use faer::{Mat, MatRef, RowRef};
use ferreus_bbfmm::{FmmParams, FmmTree as TypedFmmTree, KernelFunction};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// Returns an owned `Mat<T>` from a subset of row indices.
///
/// # Examples
///
/// ```
/// use faer::mat;
/// use ferreus_rbf_utils::select_mat_rows;
///
/// let matrix = mat![
///     [0.0, 1.0],
///     [1.0, 1.0],
///     [2.0, 2.0],
///     [3.0, 3.0f64],
/// ];
///
/// let wanted_rows = vec![0usize, 2];
///
/// let sub_matrix = select_mat_rows(&matrix, &wanted_rows);
///
/// assert_eq!(
///     sub_matrix,
///     mat![
///         [0.0, 1.0],
///         [2.0, 2.0f64],    
///     ]
/// );
/// ```
#[inline(always)]
pub fn select_mat_rows<T>(existing_mat: MatRef<T>, row_indices: &Vec<usize>) -> Mat<T>
where
    T: Clone,
{
    Mat::from_fn(row_indices.len(), existing_mat.ncols(), |i, j| {
        existing_mat.get(row_indices[i], j).clone()
    })
}

/// Generates the cartesian product of a slice of values repeated `num_columns` times.
///
/// # Examples
///
/// ```
/// use faer::mat;
/// use ferreus_rbf_utils::cartesian_product;
///
/// let values = vec![0, 1];
///
/// let result = cartesian_product(&values, 2);
///
/// assert_eq!(
///     result,
///     mat![
///         [0, 0],
///         [0, 1],
///         [1, 0],
///         [1, 1],
///     ]
/// );
/// ```
#[inline(always)]
pub fn cartesian_product<T>(values: &[T], num_columns: usize) -> Mat<T>
where
    T: Clone + Debug + Default,
{
    let base = values.len();
    let total_rows = base.pow(num_columns as u32);

    Mat::from_fn(total_rows, num_columns, |i, j| {
        let index = (i / base.pow((num_columns - j - 1) as u32)) % base;
        values[index].clone()
    })
}

/// Returns the indices that would sort the input slice.
///
/// # Examples
///
/// ```
/// use ferreus_rbf_utils::argsort;
///
/// let data = [30, 10, 20];
///
/// let sorted_indices = argsort(&data);
///
/// assert_eq!(sorted_indices, vec![1, 2, 0]);
/// ```
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

/// Returns the index of the minimum (optionally weighted) value.
#[inline(always)]
pub fn argmin<T>(data: &[T], weights: &Option<&[T]>) -> usize
where
    T: Copy + PartialOrd + Default + std::ops::Add<Output = T>,
{
    assert!(!data.is_empty(), "Data slice cannot be empty");

    // Initialize min_value and min_index using the first element
    let mut min_index = 0;
    let mut min_value = data[0];
    if let Some(w) = weights {
        min_value = min_value + w[0];
    }

    // Iterate through the rest of the elements
    for (idx, &value) in data.iter().enumerate().skip(1) {
        let mut current_value = value;
        if let Some(w) = weights {
            current_value = current_value + w[idx];
        }

        if current_value < min_value {
            min_value = current_value;
            min_index = idx;
        }
    }

    min_index
}

/// Returns the index of the maximum (optionally weighted) value.
#[inline(always)]
pub fn argmax<T>(data: &[T], weights: &Option<&[T]>) -> usize
where
    T: Copy + PartialOrd + Default + std::ops::AddAssign,
{
    let mut max_index = 0;
    let mut max_value = T::default(); // Default value for comparison

    for (idx, &value) in data.iter().enumerate() {
        let mut current_value = value;
        // Use weight from weights slice if it's provided.
        if let Some(w) = weights {
            let weight_value = w[idx];
            current_value += weight_value;
        }

        // Update the max value and index if the current weighted value is greater
        if current_value > max_value {
            max_value = current_value;
            max_index = idx;
        }
    }

    max_index
}

/// Computes the axis aligned bounding box (AABB) extents of a matrix of points.
///
/// Returns a flat vector containing the minimum and maximum values along each column (dimension)
/// of the input matrix. The result is arranged as:
///
/// `[min_0, min_1, ..., min_n, max_0, max_1, ..., max_n]`
///
/// where `n` is the number of columns in the matrix.
///
/// # Examples
///
/// ```
/// use faer::mat;
/// use ferreus_rbf_utils::get_pointarray_extents;
///
/// let points = mat![
///     [1.0, 2.0],
///     [3.0, -1.0],
///     [0.5, 4.0f64]
/// ];
/// let extents = get_pointarray_extents(points.as_ref());
/// assert_eq!(extents, vec![0.5, -1.0, 3.0, 4.0]);
/// ```
#[inline(always)]
pub fn get_pointarray_extents<T>(points: MatRef<T>) -> Vec<T>
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

/// Squared euclidean distance between two points.
///
/// Specialised for the 1-3 dimensional cases the FMM actually runs. The generic
/// loop below cannot be unrolled, because `RowRef` carries its length at run
/// time, which leaves every kernel evaluation on a scalar two-load-per-axis
/// path. The specialised arms sum in the same order as the generic loop
/// (`0.0 + d0*d0` is exact for any `d0`), so results are bit-for-bit identical.
#[inline(always)]
pub(crate) fn distance_sq(target: RowRef<f64>, source: RowRef<f64>) -> f64 {
    match target.ncols().min(source.ncols()) {
        1 => {
            let d0 = target[0] - source[0];
            d0 * d0
        }
        2 => {
            let d0 = target[0] - source[0];
            let d1 = target[1] - source[1];
            d0 * d0 + d1 * d1
        }
        3 => {
            let d0 = target[0] - source[0];
            let d1 = target[1] - source[1];
            let d2 = target[2] - source[2];
            d0 * d0 + d1 * d1 + d2 * d2
        }
        _ => {
            let mut dist = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let diff = t - s;
                dist += diff * diff;
            }
            dist
        }
    }
}

/// Writes the component-wise difference into `diff_out` and returns the squared
/// distance. Specialised for 1-3 dimensions for the same reason as
/// [`distance_sq`], and likewise bit-for-bit identical to the generic loop.
#[inline(always)]
pub(crate) fn fill_diff_and_distance_sq(
    target: RowRef<f64>,
    source: RowRef<f64>,
    diff_out: &mut [f64],
) -> f64 {
    let n = diff_out.len().min(target.ncols()).min(source.ncols());
    match n {
        1 => {
            let d0 = target[0] - source[0];
            diff_out[0] = d0;
            d0 * d0
        }
        2 => {
            let d0 = target[0] - source[0];
            let d1 = target[1] - source[1];
            diff_out[0] = d0;
            diff_out[1] = d1;
            d0 * d0 + d1 * d1
        }
        3 => {
            let d0 = target[0] - source[0];
            let d1 = target[1] - source[1];
            let d2 = target[2] - source[2];
            diff_out[0] = d0;
            diff_out[1] = d1;
            diff_out[2] = d2;
            d0 * d0 + d1 * d1 + d2 * d2
        }
        _ => {
            let mut dist = 0.0;
            for (d, (t, s)) in diff_out.iter_mut().zip(target.iter().zip(source.iter())) {
                let diff = t - s;
                *d = diff;
                dist += diff * diff;
            }
            dist
        }
    }
}

#[inline(always)]
pub(crate) fn scale_in_place(values: &mut [f64], factor: f64) {
    for value in values.iter_mut() {
        *value *= factor;
    }
}

/// Calculates the euclidean distance between two points.
///
/// # Examples
///
/// ```
/// use faer::mat;
/// use ferreus_rbf_utils::get_distance;
///
/// let points = mat![
///     [1.0, 2.0],
///     [4.0, 6.0],
/// ];
///
/// let target = points.row(0);
/// let source = points.row(1);
///
/// let dist = get_distance(target, source);
///
/// assert_eq!(dist, 5.0);
/// ```
#[inline(always)]
pub fn get_distance(target: RowRef<f64>, source: RowRef<f64>) -> f64 {
    distance_sq(target, source).sqrt()
}

/// Builds a dense kernel matrix using a typed kernel function.
#[inline(always)]
pub fn get_a_matrix_typed<K>(
    target_points: &Mat<f64>,
    source_points: &Mat<f64>,
    kernel_function: &K,
) -> Mat<f64>
where
    K: KernelFunction,
{
    let m = target_points.shape().0;
    let n = source_points.shape().0;

    let mut a_matrix = Mat::<f64>::zeros(m, n);

    for j in 0..n {
        let source = source_points.row(j);

        for i in 0..m {
            let target = target_points.row(i);

            a_matrix[(i, j)] = kernel_function.evaluate(target, source);
        }
    }

    a_matrix
}

/// Builds a symmetric kernel matrix using a typed kernel function, adding a nugget on the diagonal.
#[inline(always)]
pub fn get_a_matrix_symmetric_solver_typed<K>(
    target_points: &Mat<f64>,
    source_points: &Mat<f64>,
    kernel_function: &K,
    nugget: &f64,
) -> Mat<f64>
where
    K: KernelFunction,
{
    let m = target_points.nrows();
    let n = source_points.nrows();

    let mut a_matrix = Mat::<f64>::zeros(m, n);

    for j in 0..n {
        let source_row = source_points.row(j);

        for i in j..m {
            let target_row = target_points.row(i);
            let mut k_val = kernel_function.evaluate(target_row, source_row);

            // Add nugget to the diagonal
            if i == j {
                k_val += nugget;
            }

            // Write both symmetric entries
            a_matrix[(i, j)] = k_val;
            a_matrix[(j, i)] = k_val;
        }
    }

    a_matrix
}

/// Returns the maximum value in a slice.
#[inline(always)]
pub fn max<T>(data: &[T]) -> T
where
    T: Copy + PartialOrd + Default + std::ops::AddAssign,
{
    let mut max_value = T::default();

    for &value in data.iter() {
        let current_value = value;

        if current_value > max_value {
            max_value = current_value;
        }
    }

    max_value
}

// K-free dispatcher generated from the kernel registry below.
// Assumes each kernel type implements `KernelFromParams::from_params(&KernelParams) -> K`.
macro_rules! for_each_kernel {
    ( registry = [ $( ($V:ident, $Kty:path) ),* $(,)? ] ) => {

        /// Runtime kernel selector built from the kernel registry
        #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
        pub enum KernelType {
            $( $V, )*
        }

        /// Runtime-erased wrapper so callers don't need to be generic over [`KernelType`].
        #[derive(Debug)]
        pub enum FmmTree {
            $( $V(TypedFmmTree<$Kty>), )*
        }

        impl FmmTree {
            /// Constructs a new erased `FmmTree` from points and parameters by
            /// instantiating the appropriate typed tree for `kernel_type`.
            #[allow(clippy::too_many_arguments)]
            #[inline]
            pub fn new(
                source_points: Arc<Mat<f64>>,
                interpolation_order: usize,
                kernel_params: KernelParams,
                adaptive_tree: bool,
                sparse: bool,
                extents: Option<Vec<f64>>,
                params: Option<FmmParams>,
            ) -> Self {
                let kernel_type = kernel_params.kernel_type;

                match kernel_type {
                    $(
                        KernelType::$V => {
                            // Build kernel K from shared KernelParams
                            let k: $Kty = <$Kty as KernelFromParams>::from_params(&kernel_params);
                            let tree = TypedFmmTree::new(
                                source_points,
                                interpolation_order,
                                k,
                                adaptive_tree,
                                sparse,
                                extents,
                                params,
                            );
                            FmmTree::$V(tree)
                        }
                    ),*
                }
            }

            /// Sets the source weights for the underlying FMM tree.
            #[inline]
            pub fn set_weights(&mut self, w: faer::MatRef<'_, f64>) {
                match self {
                    $( Self::$V(t) => t.set_weights(w), )*
                }
            }

            /// Sets local expansion coefficients for the underlying FMM tree.
            #[inline]
            pub fn set_local_coefficients(&mut self, w: faer::MatRef<'_, f64>) {
                match self {
                    $( Self::$V(t) => t.set_local_coefficients(w), )*
                }
            }

            /// Evaluates the FMM at the supplied target points.
            #[inline]
            pub fn evaluate(
                &mut self,
                w: faer::MatRef<'_, f64>,
                x: faer::MatRef<f64>,
            ) -> Result<Mat<f64>, ferreus_bbfmm::FmmError> {
                match self {
                    $( Self::$V(t) => t.evaluate(w, x), )*
                }
            }

            /// Evaluates the FMM and gradients at the supplied target points.
            #[inline]
            pub fn evaluate_with_gradients(
                &mut self,
                w: faer::MatRef<'_, f64>,
                x: faer::MatRef<f64>,
            ) -> Result<(Mat<f64>, Mat<f64>), ferreus_bbfmm::FmmError> {
                match self {
                    $( Self::$V(t) => t.evaluate_with_gradients(w, x), )*
                }
            }

            /// Evaluates only the leaf-level contributions at the supplied target points.
            #[inline]
            pub fn evaluate_leaves(
                &mut self,
                w: faer::MatRef<'_, f64>,
                x: faer::MatRef<f64>,
            ) -> Result<Mat<f64>, ferreus_bbfmm::FmmError> {
                match self {
                    $( Self::$V(t) => t.evaluate_leaves(w, x), )*
                }
            }

            /// Evaluates only the leaf-level contributions and gradients at the supplied target points.
            #[inline]
            pub fn evaluate_leaves_with_gradients(
                &mut self,
                w: faer::MatRef<'_, f64>,
                x: faer::MatRef<f64>,
            ) -> Result<(Mat<f64>, Mat<f64>), ferreus_bbfmm::FmmError> {
                match self {
                    $( Self::$V(t) => t.evaluate_leaves_with_gradients(w, x), )*
                }
            }

            /// Returns the source points used to build the FMM tree.
            #[inline]
            pub fn source_points(&self) -> &faer::Mat<f64> {
                match self {
                    $( Self::$V(t) => &t.source_points, )*
                }
            }
        }

        /// Builds a dense kernel matrix for the selected [`KernelType`].
        #[inline(always)]
        pub fn get_a_matrix(
            target_points: &faer::Mat<f64>,
            source_points: &faer::Mat<f64>,
            params: crate::KernelParams,
        ) -> Mat<f64> {
            match params.kernel_type {
                $(
                    KernelType::$V => {
                        // Convert uniform params -> concrete kernel type
                        let k = <$Kty as crate::KernelFromParams>::from_params(&params);
                        // Call the generic; type `K` is inferred as `$Kty`
                        crate::utils::get_a_matrix_typed(target_points, source_points, &k)
                    }
                ),*
            }
        }

        /// Builds a symmetric kernel matrix with a nugget term on the diagonal.
        #[inline(always)]
        pub fn get_a_matrix_symmetric_solver(
            target_points: &Mat<f64>,
            source_points: &Mat<f64>,
            params: &crate::KernelParams,
            nugget: &f64,
        ) -> Mat<f64> {
            match params.kernel_type {
                $(
                    KernelType::$V => {
                        // Convert uniform params -> concrete kernel type
                        let k = <$Kty as crate::KernelFromParams>::from_params(&params);
                        // Call the generic; type `K` is inferred as `$Kty`
                        crate::utils::get_a_matrix_symmetric_solver_typed(
                            target_points,
                            source_points,
                            &k,
                            &nugget
                        )
                    }
                ),*
            }
        }

        /// Evaluates the selected kernel function at distance `r`.
        #[inline(always)]
        pub fn kernel_phi(
            r: f64,
            params: &crate::KernelParams,
        ) -> f64 {
            match params.kernel_type {
                $(
                    KernelType::$V => {
                        let k = <$Kty as crate::KernelFromParams>::from_params(&params);
                        k.phi(r)
                    }
                ), *
            }
        }
    };
}

for_each_kernel! {
    registry = [
        (LinearRbf,          crate::kernels::LinearRbfKernel),
        (ThinPlateSplineRbf, crate::kernels::ThinPlateSplineRbfKernel),
        (CubicRbf,           crate::kernels::CubicRbfKernel),
        (Spheroidal3Rbf,     crate::kernels::Spheroidal3RbfKernel),
        (Spheroidal5Rbf,     crate::kernels::Spheroidal5RbfKernel),
        (Spheroidal7Rbf,     crate::kernels::Spheroidal7RbfKernel),
        (Spheroidal9Rbf,     crate::kernels::Spheroidal9RbfKernel),
        (WendlandsC2Rbf,           crate::kernels::WendlandsC2RbfKernel),
        (SphericalRbf,             crate::kernels::SphericalRbfKernel),
        (ExponentialRbf,           crate::kernels::ExponentialRbfKernel),
        (GaussianRbf,              crate::kernels::GaussianRbfKernel),
        (Cubic2Rbf,             crate::kernels::Cubic2RbfKernel),
        (InverseMultiquadraticRbf, crate::kernels::InverseMultiquadraticRbfKernel),
        (Laplacian,          crate::kernels::LaplacianKernel),
        (OneOverR2,          crate::kernels::OneOverR2Kernel),
        (OneOverR4,          crate::kernels::OneOverR4Kernel),
    ]
}

#[cfg(test)]
mod distance_specialisation_tests {
    use super::{distance_sq, fill_diff_and_distance_sq};
    use faer::Mat;

    /// `distance_sq` exactly as it was written before being specialised for 1-3
    /// dimensions, so the specialisation can be held to bit-for-bit equality.
    fn reference_distance_sq(target: faer::RowRef<f64>, source: faer::RowRef<f64>) -> f64 {
        let mut dist = 0.0;
        for (t, s) in target.iter().zip(source.iter()) {
            let diff = t - s;
            dist += diff * diff;
        }
        dist
    }

    fn reference_fill_diff(
        target: faer::RowRef<f64>,
        source: faer::RowRef<f64>,
        diff_out: &mut [f64],
    ) -> f64 {
        let mut dist = 0.0;
        for (d, (t, s)) in diff_out.iter_mut().zip(target.iter().zip(source.iter())) {
            let diff = t - s;
            *d = diff;
            dist += diff * diff;
        }
        dist
    }

    /// Spans ordinary values, near-coincident points (where the subtraction
    /// cancels), zeros, signed zeros, and very large and very small magnitudes.
    fn probe_values() -> Vec<f64> {
        vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            1e-300,
            -1e-300,
            1e300,
            -1e300,
            f64::MIN_POSITIVE,
            1.0 + f64::EPSILON,
            1.0 - f64::EPSILON / 2.0,
            123456.789,
            -987654.321,
            std::f64::consts::PI,
        ]
    }

    #[test]
    fn distance_helpers_are_bit_identical_to_reference() {
        let probes = probe_values();
        let mut compared = 0usize;

        for dims in 1usize..=4 {
            // Exhaustive over the probe set for dim 1-2, sampled for 3-4.
            let step = if dims <= 2 { 1 } else { 2 };
            for a in (0..probes.len()).step_by(step) {
                for b in (0..probes.len()).step_by(step) {
                    let target =
                        Mat::<f64>::from_fn(1, dims, |_, j| probes[(a + j) % probes.len()]);
                    let source =
                        Mat::<f64>::from_fn(1, dims, |_, j| probes[(b + j * 3) % probes.len()]);
                    let (t, s) = (target.row(0), source.row(0));

                    let got = distance_sq(t, s);
                    let want = reference_distance_sq(t, s);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "distance_sq differs for dims={dims} a={a} b={b}: {got} vs {want}"
                    );
                    compared += 1;

                    let mut got_diff = vec![0.0f64; dims];
                    let mut want_diff = vec![0.0f64; dims];
                    let got_d = fill_diff_and_distance_sq(t, s, &mut got_diff);
                    let want_d = reference_fill_diff(t, s, &mut want_diff);
                    assert_eq!(
                        got_d.to_bits(),
                        want_d.to_bits(),
                        "fill_diff_and_distance_sq distance differs for dims={dims}"
                    );
                    for k in 0..dims {
                        assert_eq!(
                            got_diff[k].to_bits(),
                            want_diff[k].to_bits(),
                            "fill_diff_and_distance_sq component {k} differs for dims={dims}"
                        );
                    }
                    compared += 1;
                }
            }
        }

        assert!(
            compared > 500,
            "expected a broad sweep, compared {compared}"
        );
    }

    /// Near-coincident points are the case the specialisation could plausibly
    /// break, since the subtraction cancels almost completely.
    #[test]
    fn distance_helpers_bit_identical_for_near_coincident_points() {
        for dims in 1usize..=3 {
            for scale in [1e-16f64, 1e-12, 1e-8, 1e-4] {
                for i in 0..64 {
                    let base = 0.3 + i as f64 * 0.01;
                    let target = Mat::<f64>::from_fn(1, dims, |_, j| base + j as f64 * 0.7);
                    let source = Mat::<f64>::from_fn(1, dims, |_, j| {
                        base + j as f64 * 0.7 + scale * ((i + j) as f64).sin()
                    });
                    let (t, s) = (target.row(0), source.row(0));
                    assert_eq!(
                        distance_sq(t, s).to_bits(),
                        reference_distance_sq(t, s).to_bits(),
                        "distance_sq differs at scale {scale} dims {dims}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod evaluate_from_distance_sq_tests {
    use crate::kernels::*;
    use faer::Mat;
    use ferreus_bbfmm::KernelFunction;

    /// Holds a kernel to the contract on
    /// [`KernelFunction::evaluate_from_distance_sq`]: for any point pair, feeding
    /// the pair's squared distance to it must reproduce `evaluate` bit-for-bit.
    fn assert_contract<K: KernelFunction>(kernel: &K, name: &str) -> usize {
        let mut checked = 0usize;

        // Separations spanning coincident points, the denormal range, the scales
        // where a piecewise kernel switches branch, and the far field.
        let mut separations: Vec<f64> = vec![0.0, f64::MIN_POSITIVE, 1e-300, 1e-30];
        // Dense geometric sweep, so every internal branch boundary is crossed
        // from both sides rather than stepped over.
        let mut x = 1e-12f64;
        while x < 1e6 {
            separations.push(x);
            x *= 1.07;
        }

        for dims in 1usize..=3 {
            for &sep in &separations {
                // Spread the separation across the axes so r2 is a genuine sum.
                let target = Mat::<f64>::from_fn(1, dims, |_, j| 0.3 + j as f64 * 1.7);
                let source = Mat::<f64>::from_fn(1, dims, |_, j| {
                    0.3 + j as f64 * 1.7 + sep / (dims as f64).sqrt()
                });
                let (t, s) = (target.row(0), source.row(0));

                let direct = kernel.evaluate(t, s);
                let r2 = crate::utils::distance_sq(t, s);
                let from_r2 = kernel
                    .evaluate_from_distance_sq(r2)
                    .unwrap_or_else(|| panic!("{name} returned None"));

                assert_eq!(
                    from_r2.to_bits(),
                    direct.to_bits(),
                    "{name}: evaluate_from_distance_sq({r2:e}) = {from_r2} but \
                     evaluate = {direct} (dims={dims}, separation={sep:e})"
                );
                checked += 1;
            }
        }
        checked
    }

    #[test]
    fn every_kernel_matches_evaluate_bit_for_bit() {
        let mut total = 0usize;

        total += assert_contract(&LinearRbfKernel, "LinearRbf");
        total += assert_contract(&ThinPlateSplineRbfKernel, "ThinPlateSplineRbf");
        total += assert_contract(&CubicRbfKernel, "CubicRbf");
        total += assert_contract(&WendlandsC2RbfKernel, "WendlandsC2Rbf");
        total += assert_contract(&SphericalRbfKernel, "SphericalRbf");
        total += assert_contract(&ExponentialRbfKernel, "ExponentialRbf");
        total += assert_contract(&GaussianRbfKernel, "GaussianRbf");
        total += assert_contract(&Cubic2RbfKernel, "Cubic2Rbf");
        total += assert_contract(&InverseMultiquadraticRbfKernel, "InverseMultiquadraticRbf");
        total += assert_contract(&LaplacianKernel, "Laplacian");
        total += assert_contract(&OneOverR2Kernel, "OneOverR2");
        total += assert_contract(&OneOverR4Kernel, "OneOverR4");

        // The spheroidal family evaluates from r2 directly and has a near/far
        // branch, so sweep several ranges and sills as well as all four orders.
        for &base_range in &[0.05f64, 1.0, 7.5, 250.0] {
            for &sill in &[0.5f64, 1.0, 12.0] {
                total += assert_contract(
                    &Spheroidal3RbfKernel::new(base_range, sill),
                    "Spheroidal3Rbf",
                );
                total += assert_contract(
                    &Spheroidal5RbfKernel::new(base_range, sill),
                    "Spheroidal5Rbf",
                );
                total += assert_contract(
                    &Spheroidal7RbfKernel::new(base_range, sill),
                    "Spheroidal7Rbf",
                );
                total += assert_contract(
                    &Spheroidal9RbfKernel::new(base_range, sill),
                    "Spheroidal9Rbf",
                );
            }
        }

        assert!(total > 20_000, "expected a broad sweep, checked {total}");
    }
}

#[cfg(test)]
mod value_and_gradient_from_distance_sq_tests {
    use crate::kernels::*;
    use faer::Mat;
    use ferreus_bbfmm::KernelFunction;

    /// Holds a kernel to the contract on
    /// [`KernelFunction::value_and_gradient_from_distance_sq`]: the value must
    /// match `evaluate_value_gradient`, and applying the returned scaling to
    /// `target - source` must reproduce its `gradient_out`, bit-for-bit.
    fn assert_contract<K: KernelFunction>(kernel: &K, name: &str) -> usize {
        let mut checked = 0usize;

        let mut separations: Vec<f64> = vec![0.0, f64::MIN_POSITIVE, 1e-300, 1e-30, 1e-9];
        let mut x = 1e-12f64;
        while x < 1e6 {
            separations.push(x);
            x *= 1.09;
        }

        for dims in 1usize..=3 {
            for &sep in &separations {
                // Both signs of the difference, so the -0.0 case is exercised:
                // the coincident-point branch must write +0.0 even when the
                // difference is negative.
                for &sign in &[1.0f64, -1.0] {
                    let target = Mat::<f64>::from_fn(1, dims, |_, j| 0.3 + j as f64 * 1.7);
                    let source = Mat::<f64>::from_fn(1, dims, |_, j| {
                        0.3 + j as f64 * 1.7 + sign * sep / (dims as f64).sqrt()
                    });
                    let (t, s) = (target.row(0), source.row(0));

                    let mut want_grad = vec![0.0f64; dims];
                    let want_value = match kernel.evaluate_value_gradient(t, s, &mut want_grad) {
                        Some(v) => v,
                        None => return checked, // kernel has no gradient at all
                    };

                    let r2 = crate::utils::distance_sq(t, s);
                    let (got_value, scale) = kernel
                        .value_and_gradient_from_distance_sq(r2)
                        .unwrap_or_else(|| panic!("{name} returned None but has gradients"));

                    assert_eq!(
                        got_value.to_bits(),
                        want_value.to_bits(),
                        "{name}: value mismatch at r2={r2:e} (dims={dims}, sep={sep:e}, sign={sign})"
                    );

                    for d in 0..dims {
                        let difference = t[d] - s[d];
                        let got = scale.apply(difference);
                        assert_eq!(
                            got.to_bits(),
                            want_grad[d].to_bits(),
                            "{name}: gradient component {d} mismatch at r2={r2:e} \
                             (dims={dims}, sep={sep:e}, sign={sign}): {got} vs {}",
                            want_grad[d]
                        );
                    }
                    checked += 1;
                }
            }
        }
        checked
    }

    #[test]
    fn every_gradient_kernel_matches_evaluate_value_gradient_bit_for_bit() {
        let mut total = 0usize;

        total += assert_contract(&LinearRbfKernel, "LinearRbf");
        total += assert_contract(&ThinPlateSplineRbfKernel, "ThinPlateSplineRbf");
        total += assert_contract(&CubicRbfKernel, "CubicRbf");
        total += assert_contract(&LaplacianKernel, "Laplacian");
        total += assert_contract(&OneOverR2Kernel, "OneOverR2");
        total += assert_contract(&OneOverR4Kernel, "OneOverR4");

        for &base_range in &[0.05f64, 1.0, 7.5, 250.0] {
            for &sill in &[0.5f64, 1.0, 12.0] {
                total += assert_contract(&Spheroidal3RbfKernel::new(base_range, sill), "Sph3");
                total += assert_contract(&Spheroidal5RbfKernel::new(base_range, sill), "Sph5");
                total += assert_contract(&Spheroidal7RbfKernel::new(base_range, sill), "Sph7");
                total += assert_contract(&Spheroidal9RbfKernel::new(base_range, sill), "Sph9");
            }
        }

        assert!(total > 20_000, "expected a broad sweep, checked {total}");
    }

    /// Kernels without gradients must keep returning `None`, so callers stay on
    /// the scalar path rather than silently getting a wrong gradient.
    #[test]
    fn non_gradient_kernels_return_none() {
        assert!(
            WendlandsC2RbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
        assert!(
            SphericalRbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
        assert!(
            ExponentialRbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
        assert!(
            GaussianRbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
        assert!(
            Cubic2RbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
        assert!(
            InverseMultiquadraticRbfKernel
                .value_and_gradient_from_distance_sq(1.0)
                .is_none()
        );
    }
}
