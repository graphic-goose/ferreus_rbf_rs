/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides utility routines for bounding box computation, row selection, and dense kernel matrices.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::Mat;
use crate::KernelFunction;

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
        extents[col] = points.get(0, col).clone();               // Min value for each column
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
pub fn select_mat_rows(existing_mat: &Mat<f64>, row_indices: &Vec<usize>) -> Mat<f64> 
{
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

    let mut a_matrix = Mat::<f64>::zeros(m, n);

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
    indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}
