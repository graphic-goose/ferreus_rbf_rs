/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements Adaptive Cross Approximation (ACA) for low-rank matrix compression.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, RowRef, mat::AsMatRef};

/// Adaptive Cross Approximation (ACA) with partial pivoting
///
/// # Arguments
/// * `num_rows` - Number of rows in the target matrix
/// * `num_columns` - Number of columns in the target matrix
/// * `matrix_subset_generator` - Closure to compute a submatrix (row range, col range)
/// * `epsilon` - Desired accuracy (Frobenius norm relative tolerance)
///
/// # Returns
/// A tuple of low-rank factors `(U, V)` such that A ≈ U * V^T
pub fn aca_partial_pivoting<F>(
    num_rows: usize,
    num_columns: usize,
    matrix_subset_generator: F,
    epsilon: &f64,
) -> (Mat<f64>, Mat<f64>)
where
    F: Fn(&usize, &usize, &usize, &usize) -> Mat<f64>,
{   
    // Track unused rows and columns with binary flags (1 = unused, 0 = used)
    let mut unused_rows = vec![1; num_rows];
    let mut unused_columns = vec![1; num_columns];

    // Iteration cap: can't exceed min dimension
    let max_iterations = std::cmp::min(num_rows, num_columns);

    // Relative tolerance on squared residual norm
    let tolerance = epsilon.powi(2);

    // U and V will store the rank-1 updates: A ≈ ∑ u_k v_k^T
    let mut u = Mat::<f64>::zeros(num_rows, max_iterations);
    let mut v = Mat::<f64>::zeros(num_columns, max_iterations);

    // Accumulated residual norm estimate
    let mut residual_norm = 0.0;

    // Start ACA from row 0
    let mut i = 0 as usize;
    let mut j: usize;

    // Inner product estimate of previously added terms
    let mut sum_k = 0.0;

    // Iteration index = current rank
    let mut k = 0;

    for _iteration in 0..max_iterations {
        // Extract current row i across all columns
        let mut v_column_i = matrix_subset_generator(&i, &(i + 1), &0, &num_columns);

        // Mark row i as used
        unused_rows[i] = 0;

        // Subtract all previously computed low-rank contributions
        if k > 0 {
            // v_column_i -= u.submatrix(i, 0, 1, k) * v.submatrix(0, 0, num_rows, k).transpose();
            v_column_i -= u.submatrix(i, 0, 1, k) * v.submatrix(0, 0, num_columns, k).transpose();
        }

        // Choose pivot column j with largest absolute residual in row i
        j = argmax_masked(&v_column_i.row(0), &unused_columns);

        // Normalize v_k so v_k[j] = 1
        let pivot = 1.0 / v_column_i.row(0)[j];

        v_column_i *= pivot;

        // Extract current column j across all rows
        let mut u_column_j = matrix_subset_generator(&0, &num_rows, &j, &(j + 1));

        // Mark column j as used
        unused_columns[j] = 0;

        // Subtract previous low-rank contributions from column
        if k > 0 {
            u_column_j -=
                (v.submatrix(j, 0, 1, k) * u.submatrix(0, 0, num_rows, k).transpose()).transpose();
        }

        // Choose pivot row i with largest absolute residual in column j
        i = argmax_masked(&u_column_j.col(0).transpose(), &unused_rows);

        // Estimate cross terms from previous updates: sum_k = 2 * sum_{i<k} <u_i, u_k> <v_i, v_k>
        if k > 0 {
            if k == 1 {
                // Optimized case for rank-1
                let part_1 = u.col(0).transpose() * &u_column_j.col(0);
                let part_2 = v.col(0).transpose() * &v_column_i.row(0).transpose();
                sum_k = part_1 * part_2;
            } else {
                // General case: sum of dot products between u_k and previous u_i times v_k and v_i
                let part1 = u.submatrix(0, 0, num_rows, k).transpose() * &u_column_j;
                // let part2 = v.submatrix(0, 0, num_rows, k).transpose() * &v_column_i.transpose();
                let part2 = v.submatrix(0, 0, num_columns, k).transpose() * &v_column_i.transpose();
                let part3 = &part1.transpose() * &part2;

                // Frobenius-like inner product
                sum_k = part3.as_mat_ref().sum();
            }
        }

        // Estimate Frobenius norm increase from u_k * v_k^T term
        let norm_u_v_2 = (u_column_j.col(0).transpose() * u_column_j.col(0))
            * (v_column_i.row(0) * v_column_i.row(0).transpose());

        // Update residual estimate
        residual_norm += norm_u_v_2 + 2.0 * sum_k;

        // Store new rank-1 pair into U and V
        u.col_mut(k).copy_from(&u_column_j.col(0));
        v.col_mut(k).copy_from(&v_column_i.row(0).transpose());

        k += 1;

        // Stopping criteria is:
        //   ||u_k||2 ||v_k||2 <= tolerance ||A_k||F
        if norm_u_v_2 <= tolerance * residual_norm {
            break;
        }
    }

    // Return truncated U, V with only first k columns
    (u.subcols(0, k).to_owned(), v.subcols(0, k).to_owned())
}

/// Find the index of the maximum absolute value in `data` masked by a binary `mask`.
///
/// # Arguments
/// * `data` - A row vector of f64 values.
/// * `mask` - A slice of binary values (0 or 1) indicating which elements to consider.
///
/// # Returns
/// Index of the element with the largest absolute value among the unmasked elements.
fn argmax_masked(data: &RowRef<f64>, mask: &[u8]) -> usize {
    let mut max_index = 0;
    let mut max_value = 0.0;

    for (idx, &value) in data.iter().enumerate() {
        // Only include elements that are unmasked (mask == 1), and use absolute value
        let weighted_value = value.abs() * mask[idx] as f64;

        if weighted_value > max_value {
            max_value = weighted_value;
            max_index = idx;
        }
    }

    max_index
}

/// Recompress the low-rank ACA factors using QR + SVD.
/// This yields a more stable and potentially lower-rank representation.
///
/// # Arguments
/// * `u_aca` - Matrix of left factors from ACA (shape m × k)
/// * `v_aca` - Matrix of right factors from ACA (shape n × k)
/// * `epsilon` - Relative truncation tolerance for recompression
///
/// # Returns
/// Recompressed low-rank factorization `(U, V)` such that A ≈ U * V^T
pub fn recompress_aca(u_aca: &Mat<f64>, v_aca: &Mat<f64>, epsilon: &f64) -> (Mat<f64>, Mat<f64>) {
    // QR decomposition of ACA factors
    let u_qr = u_aca.qr();
    let qu = u_qr.compute_thin_Q(); // m × k orthonormal basis
    let ru = u_qr.thin_R(); // k × k upper triangular

    let v_qr = v_aca.qr();
    let qv = v_qr.compute_thin_Q(); // n × k orthonormal basis
    let rv = v_qr.thin_R(); // k × k upper triangular

    // SVD of inner core product R_u * R_v^T
    let ur_vrt = &ru * &rv.transpose(); // shape: k × k

    let svd = ur_vrt.svd().unwrap();
    let ur = svd.U(); // Left singular vectors (k × k)
    let sr = svd.S(); // Singular values (vector of length k)
    let vrt = svd.V().transpose(); // Right singular vectors transposed (k × k)

    // Determine new target rank based on cumulative sum of squares
    let sigma_vec: Vec<f64> = sr.column_vector().iter().cloned().collect();
    let new_rank = calculate_singular_values_cutoff(sigma_vec, &epsilon);

    // Recompress: U = Q_u * U_r * diag(sigma), V = V_r^T * Q_v^T
    let u = qu * (ur.subcols(0, new_rank) * sr.column_vector().subrows(0, new_rank).as_diagonal());
    let vt = vrt.subrows(0, new_rank) * qv.transpose();

    (u, vt)
}

/// Determine optimal cutoff rank for truncated SVD using cumulative sum of squares.
///
/// # Arguments
/// * `sigma` - Vector of singular values.
/// * `epsilon` - Desired relative approximation error.
///
/// # Returns
/// Minimum rank `r` such that residual Frobenius norm is ≤ epsilon * total norm.
pub fn calculate_singular_values_cutoff(sigma: Vec<f64>, epsilon: &f64) -> usize {
    // Compute backward cumulative sum of squared singular values
    let cumulative_sum_sqr = inverse_cumulative_sum_of_squares(&sigma);

    // Compute allowable residual threshold
    let eps_qr = cumulative_sum_sqr[0] * epsilon * epsilon;

    // Find smallest index where residual falls below threshold
    let cutoff = cumulative_sum_sqr
        .iter()
        .position(|&x| x < eps_qr)
        .unwrap_or(cumulative_sum_sqr.len());

    cutoff
}

/// Compute reverse cumulative sum of squared singular values.
/// i.e., result[i] = sum_{j=i}^{n-1} sigma[j]^2
///
/// # Arguments
/// * `sigma` - Vector of singular values.
///
/// # Returns
/// Vector of cumulative squared sum in reverse.
fn inverse_cumulative_sum_of_squares(sigma: &Vec<f64>) -> Vec<f64> {
    // Accumulate sum of squares in reverse order
    let cumulative_sum_squared: Vec<f64> = sigma
        .iter()
        .rev()
        .scan(0.0, |acc, &x| {
            *acc += x * x;
            Some(*acc)
        })
        .collect();

    // Reverse again to restore original order of indices
    cumulative_sum_squared.into_iter().rev().collect()
}
