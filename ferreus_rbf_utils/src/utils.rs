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
use faer::{Mat, MatMut, MatRef, RowRef};
use ferreus_bbfmm::{EvaluationTargets, FmmParams, FmmTree as TypedFmmTree, KernelFunction};
use rayon::prelude::*;
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
/// let sub_matrix = select_mat_rows(matrix.as_ref(), &wanted_rows);
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

#[inline(always)]
pub(crate) fn distance_sq(target: RowRef<f64>, source: RowRef<f64>) -> f64 {
    let mut dist = 0.0;
    for (t, s) in target.iter().zip(source.iter()) {
        let diff = t - s;
        dist += diff * diff;
    }
    dist
}

#[inline(always)]
pub(crate) fn fill_diff_and_distance_sq(
    target: RowRef<f64>,
    source: RowRef<f64>,
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

/// Direct summation over parallel target chunks without a target-by-source matrix.
fn evaluate_direct_typed<K: KernelFunction + KernelFromParams + Sync, const WITH_GRADS: bool>(
    target_points: MatRef<f64>,
    source_points: MatRef<f64>,
    weights: MatRef<f64>,
    params: &KernelParams,
    batch_size: usize,
) -> Result<(Mat<f64>, Option<Mat<f64>>), ferreus_bbfmm::FmmError> {
    assert!(
        batch_size > 0,
        "Direct evaluation batch size must be positive"
    );
    let kernel = K::from_params(params);
    let dims = source_points.ncols();
    assert!((1..=3).contains(&dims), "Unsupported source dimensionality");
    assert_eq!(
        target_points.ncols(),
        dims,
        "Target/source dimensionality mismatch"
    );
    assert_eq!(
        weights.nrows(),
        source_points.nrows(),
        "Source/weight count mismatch"
    );
    let mut values = Mat::zeros(target_points.nrows(), weights.ncols());
    let mut gradients =
        WITH_GRADS.then(|| Mat::zeros(target_points.nrows(), weights.ncols() * dims));
    // Each worker owns disjoint output rows. Keep source sums sequential so the
    // result is independent of worker count, with no reductions or locks.
    let evaluate_chunk = |target_points: MatRef<f64>,
                          mut values: MatMut<f64>,
                          mut gradients: Option<MatMut<f64>>| {
        let mut gradient = [0.0; 3];
        for i in 0..target_points.nrows() {
            let target = target_points.row(i);
            for j in 0..source_points.nrows() {
                let source = source_points.row(j);
                let value = if WITH_GRADS {
                    kernel
                        .evaluate_value_gradient(target, source, &mut gradient[..dims])
                        .ok_or(ferreus_bbfmm::FmmError::KernelDoesNotSupportGradients)?
                } else {
                    kernel.evaluate(target, source)
                };
                for rhs in 0..weights.ncols() {
                    let weight = weights[(j, rhs)];
                    values[(i, rhs)] += weight * value;
                    if let Some(grads) = gradients.as_mut() {
                        for d in 0..dims {
                            grads[(i, rhs * dims + d)] += weight * gradient[d];
                        }
                    }
                }
            }
        }
        Ok::<_, ferreus_bbfmm::FmmError>(())
    };
    // Amortize scheduling over chunks and avoid it altogether for small queries.
    let work = target_points
        .nrows()
        .saturating_mul(source_points.nrows())
        .saturating_mul(weights.ncols());
    if target_points.nrows() > batch_size && work >= 32_768 {
        if let Some(grads) = gradients.as_mut() {
            target_points
                .par_row_chunks(batch_size)
                .zip(values.par_row_chunks_mut(batch_size))
                .zip(grads.par_row_chunks_mut(batch_size))
                .try_for_each(|((target_points, values), gradients)| {
                    evaluate_chunk(target_points, values, Some(gradients))
                })?;
        } else {
            target_points
                .par_row_chunks(batch_size)
                .zip(values.par_row_chunks_mut(batch_size))
                .try_for_each(|(target_points, values)| {
                    evaluate_chunk(target_points, values, None)
                })?;
        }
    } else {
        for start in (0..target_points.nrows()).step_by(batch_size) {
            let count = batch_size.min(target_points.nrows() - start);
            evaluate_chunk(
                target_points.subrows(start, count),
                values.as_mut().subrows_mut(start, count),
                gradients
                    .as_mut()
                    .map(|grads| grads.as_mut().subrows_mut(start, count)),
            )?;
        }
    }
    Ok((values, gradients))
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
            #[allow(clippy::too_many_arguments)]
            #[inline]
            /// Constructs a new erased `FmmTree` from points and parameters by
            /// instantiating the appropriate typed tree for `kernel_type`.
            pub fn new(
                source_points: Arc<Mat<f64>>,
                interpolation_order: usize,
                kernel_params: KernelParams,
                sparse: bool,
                extents: Option<Vec<f64>>,
                params: Option<FmmParams>,
            ) -> Self {
                Self::new_with_targets(
                    source_points, interpolation_order, kernel_params,
                   sparse, extents, params, None
                )
            }

            /// Constructs a new erased `FmmTree` from points and parameters by
            /// instantiating the appropriate typed tree for `kernel_type`.
            /// This variant uses target points to refine the tree.
            #[allow(clippy::too_many_arguments)]
            #[inline]
            pub fn new_with_targets(
                source_points: Arc<Mat<f64>>,
                interpolation_order: usize,
                kernel_params: KernelParams,
                sparse: bool,
                extents: Option<Vec<f64>>,
                params: Option<FmmParams>,
                targets: Option<EvaluationTargets<'_>>
            ) -> Self {
                let kernel_type = kernel_params.kernel_type;

                match kernel_type {
                    $(
                        KernelType::$V => {
                            // Build kernel K from shared KernelParams
                            let k: $Kty = <$Kty as KernelFromParams>::from_params(&kernel_params);
                            let tree = TypedFmmTree::new_with_targets(
                                source_points,
                                interpolation_order,
                                k,
                                sparse,
                                extents,
                                params,
                                targets,
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
            pub fn set_local_coefficients(&mut self) {
                match self {
                    $( Self::$V(t) => t.set_local_coefficients(), )*
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

            /// Configured maximum target chunk size.
            pub fn target_chunk_size(&self) -> usize {
                match self { $( Self::$V(t) => t.target_chunk_size(), )* }
            }

            /// Evaluate a regular grid without a full coordinate matrix.
            pub fn evaluate_grid(&mut self, w: faer::MatRef<'_, f64>, grid: &ferreus_bbfmm::TargetGrid)
                -> Result<Mat<f64>, ferreus_bbfmm::FmmError>
            {
                match self { $( Self::$V(t) => t.evaluate_grid(w, grid), )* }
            }

            /// Evaluate a regular grid without a full coordinate matrix.
            pub fn evaluate_grid_with_gradients(&mut self, w: faer::MatRef<'_, f64>, grid: &ferreus_bbfmm::TargetGrid)
                -> Result<(Mat<f64>, Mat<f64>), ferreus_bbfmm::FmmError>
            {
                match self { $( Self::$V(t) => t.evaluate_grid_with_gradients(w, grid), )* }
            }

            /// Evaluate a regular grid without a full coordinate matrix.
            pub fn evaluate_grid_leaves(&mut self, w: faer::MatRef<'_, f64>, grid: &ferreus_bbfmm::TargetGrid)
                -> Result<Mat<f64>, ferreus_bbfmm::FmmError>
            {
                match self { $( Self::$V(t) => t.evaluate_grid_leaves(w, grid), )* }
            }

            /// Evaluate a regular grid without a full coordinate matrix.
            pub fn evaluate_grid_leaves_with_gradients(&mut self, w: faer::MatRef<'_, f64>, grid: &ferreus_bbfmm::TargetGrid)
                -> Result<(Mat<f64>, Mat<f64>), ferreus_bbfmm::FmmError>
            {
                match self { $( Self::$V(t) => t.evaluate_grid_leaves_with_gradients(w, grid), )* }
            }

            /// Returns the source points used to build the FMM tree.
            #[inline]
            pub fn source_points(&self) -> &faer::Mat<f64> {
                match self {
                    $( Self::$V(t) => &t.source_points, )*
                }
            }
        }

        /// Evaluates weighted kernel sums directly, optionally returning gradients.
        /// Selects the concrete kernel once; the typed helper owns all batching
        /// and parallelism. `batch_size` must be positive. Gradient columns are
        /// grouped by right-hand side, then coordinate.
        pub fn evaluate_direct(
            target_points: MatRef<f64>,
            source_points: MatRef<f64>,
            weights: MatRef<f64>,
            params: &KernelParams,
            with_gradients: bool,
            batch_size: usize,
        ) -> Result<(Mat<f64>, Option<Mat<f64>>), ferreus_bbfmm::FmmError> {
            match params.kernel_type {
                $(
                    KernelType::$V => {
                        if with_gradients {
                            crate::utils::evaluate_direct_typed::<$Kty, true>(
                                target_points, source_points, weights, params, batch_size,
                            )
                        } else {
                            crate::utils::evaluate_direct_typed::<$Kty, false>(
                                target_points, source_points, weights, params, batch_size,
                            )
                        }
                    }
                ),*
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
