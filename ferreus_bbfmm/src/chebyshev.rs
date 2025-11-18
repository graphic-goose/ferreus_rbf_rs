/////////////////////////////////////////////////////////////////////////////////////////////
//
// Builds Chebyshev interpolation operators and compressed M2L transfer operators for BBFMM.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    aca,
    bbfmm::{M2LCompressionType, PrecomputeOperators},
    KernelFunction,
    utils,
};
use faer::{Mat, RowRef, row::generic::Row};
use itertools::{Itertools, iproduct};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashMap;

// # References
// [1] W. Fong, E. Darve, The black-box fast multipole method, Journal of Computational Physics 228 (23) (2009) 8712-8725.
// [2] M. Messner, B. Bramas, O. Coulaud, E. Darve, Optimized M2L Kernels for the Chebyshev Interpolation based Fast Multipole Method (2012).

/// Generates Chebyshev nodes between -1 and 1 of T_n(x) for the given interpolation order.
fn generate_chebyshev_nodes(interpolation_order: &usize) -> Vec<f64> {
    (0..*interpolation_order)
        .rev()
        .map(|i| {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / *interpolation_order as f64;
            theta.cos()
        })
        .collect()
}

/// Calculates Tn(x), the chebyshev polynomials of the first kind,
/// for k between 0 and interpolation order -1 inclusive.
///
/// Uses the recurrence relation:
///  T_0(x) = 1 \n
///  T_1(x) = x \n
///  T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x) for n > 1
fn evaluate_chebyshev_polynomials(
    num_columns: &usize,
    num_rows: &usize,
    nodes: &Vec<f64>,
) -> Mat<f64> {
    let mut tn_x = Mat::<f64>::zeros(*num_rows as usize, *num_columns as usize);

    for i in 0..*num_rows {
        for j in 0..*num_columns {
            let value = if j == 0 {
                1.0
            } else if j == 1 {
                nodes[i]
            } else {
                2.0 * nodes[i] * tn_x.get(i, j - 1) - tn_x.get(i, j - 2)
            };
            tn_x[(i, j)] = value;
        }
    }

    tn_x
}

/// Applies the linear transformation to normalize Chebyshev evaluations into interpolation basis.
/// Returns S(x) matrix used for scaling/interpolating.
fn calculate_sn(
    tn_x: Mat<f64>,
    polynomial_nodes: &Mat<f64>,
    interpolation_order: &usize,
) -> Mat<f64> {
    let mut sn = tn_x * &polynomial_nodes.transpose();

    sn.col_iter_mut().for_each(|col| {
        col.iter_mut().for_each(|element| {
            *element = (*element * 2.0 - 1.0) / *interpolation_order as f64;
        });
    });
    sn
}

/// Calculates the Chebyshev weights to transfer from the Chebyshev nodes of a
/// parent cell to the Chebyshev nodes of its children.
fn get_cheb_transfer_from_parent_to_children(
    interpolation_order: &usize,
    cheb_nodes: &Vec<f64>,
    polynomial_nodes: &Mat<f64>,
) -> Mat<f64> {
    let num_child_cheb_nodes = 2 * *interpolation_order as usize;

    // Initialise the one dimensional transfer array.
    // So, using an interpolation order of 3 as an example in one dimension, we go from
    // parent_cheb_nodes = [0.866, 6.12E-17, -0.866] to
    // child_cheb_nodes = [-0.067, -0.5, -0.933, 0.933, 0.5, 0.067]
    let child_cheb_nodes = (0..num_child_cheb_nodes)
        .map(|i| {
            let child = if i < *interpolation_order {
                // Calculate the cheb nodes for the first child (left side) in one dimension.
                cheb_nodes[i] - 1.0
            } else {
                // Calculate the cheb nodes for the second child (right side) in one dimension.
                cheb_nodes[i - *interpolation_order] + 1.0
            };
            child * 0.5
        })
        .collect();

    // Evaluate the Chebyshev polynomials at the child Chebyshev nodes.
    let tn_x = evaluate_chebyshev_polynomials(
        &interpolation_order,
        &num_child_cheb_nodes,
        &child_cheb_nodes,
    );

    // Calculate the sum of the Chebyshev polynomials.
    calculate_sn(tn_x, &polynomial_nodes, &interpolation_order)
}

/// Returns the relative Morton offsets in an N-dimensional hypercube.
pub fn calculate_relative_offsets_morton(dim: &usize) -> Mat<usize> {
    let num_children = 2usize.pow(*dim as u32);

    let output = Mat::<usize>::from_fn(num_children, *dim, |i, j| match (i & (1 << j)) != 0 {
        true => 1,
        false => 0,
    });

    output
}

/// Calculates the Chebyshev weights to transfer from the Chebyshev nodes of the
/// children of a parent cell to the Chebyshev nodes of the parent cell.
fn get_m2m_transfer_matrices(
    interpolation_order: &usize,
    cheb_nodes: &Vec<f64>,
    polynomial_nodes: &Mat<f64>,
    dimensions: &usize,
    num_child_cells: &usize,
) -> Vec<Mat<f64>> {
    // Get the transfer weights to go from cheb nodes of parent to cheb nodes of children.
    let sn = get_cheb_transfer_from_parent_to_children(
        &interpolation_order,
        &cheb_nodes,
        &polynomial_nodes,
    );

    // Split the transfer weights array in half row wise.
    let child_transfers = sn.split_at_row(*interpolation_order);

    // Get an array of the unique transfer vectors for well separated cells based on Morton order.
    let m2m_transfer_vector_array: Mat<usize> = calculate_relative_offsets_morton(&dimensions);

    // Loop for each child cell
    (0..*num_child_cells)
        .map(|i| {
            // Tensor product of transfers in each dimension.
            let l2l_transfer = m2m_transfer_vector_array
                .row(i)
                .iter()
                .map(|&j| {
                    if j == 0 {
                        &child_transfers.0
                    } else {
                        &child_transfers.1
                    }
                })
                .fold(None, |acc: Option<Mat<f64>>, mat| {
                    Some(match acc {
                        Some(prev) => prev.kron(mat.as_ref()),
                        None => mat.to_owned(),
                    })
                })
                .unwrap();

            l2l_transfer.transpose().to_owned()
        })
        .collect()
}

/// Generates all possible M2L vectors in [-3,3]^d and a set of unique reference vectors from which
/// all other interactions can be realised by way of symmetry permutations.
/// Reference vectors are defined in [2] for the 3D case, as:
///  t1, t2, t3 >= 0 &
///  t1 >= t2 >= t3 >= 0
///
///  For 2D and 3D this equates to:
///      2D                        3D
///    [2, 0]                  [2, 0, 0]
///    [2, 1]                  [2, 1, 0]
///    [2, 2]                  [2, 2, 0]
///    [3, 0]                  [3, 0, 0]
///    [3, 1]                  [3, 1, 0]
///    [3, 2]                  [3, 2, 0]
///    [3, 3]                  [3, 3, 0]
///                            [2, 1, 1]
///                            [2, 2, 1]
///                            [3, 1, 1]
///                            [3, 2, 1]
///                            [3, 3, 1]
///                            [2, 2, 2]
///                            [3, 2, 2]
///                            [3, 3, 2]
///                            [3, 3, 3]
fn get_m2l_vectors(dimensions: &usize) -> (Mat<i32>, Mat<i32>) {
    let vector_range: Vec<i32> = (-3..4).into_iter().collect();
    let all_m2l_vectors = utils::cartesian_product::<i32>(&vector_range, *dimensions);
    let mut reference_m2l_vectors: Vec<i32> = Vec::new();

    let base_vector_range: Vec<i32> = (0..4).into_iter().collect();
    let base_vectors = utils::cartesian_product::<i32>(&base_vector_range, *dimensions);

    base_vectors.row_iter().for_each(|row| {
        if row[0] >= 2 {
            let mut valid = true;
            for i in 1..*dimensions {
                if row[i] > row[i - 1] {
                    valid = false;
                    break;
                }
            }
            if valid {
                reference_m2l_vectors.extend(row.iter());
            }
        }
    });

    let num_rows = reference_m2l_vectors.len() / dimensions;

    let reference_m2l_vectors = Mat::from_fn(num_rows, *dimensions, |i, j| {
        reference_m2l_vectors[i * *dimensions + j]
    });

    (all_m2l_vectors, reference_m2l_vectors)
}

/// Maps from multi-index to rows and columns of K matrix, as per section 3.1 of [2].
fn map_multi_index_to_k(alpha: &Vec<usize>, interpolation_order: &usize) -> usize {
    let alpha_length = alpha.len();
    let m_alpha: usize;

    if alpha_length == 1 {
        m_alpha = alpha[0] - 1;
    } else if alpha_length == 2 {
        m_alpha = (alpha[0] - 1) * interpolation_order + (alpha[1] - 1);
    } else {
        m_alpha = (alpha[0] - 1) * interpolation_order.pow(2)
            + (alpha[1] - 1) * interpolation_order
            + (alpha[2] - 1);
    }

    m_alpha
}

/// Permutes the multi-index for the axial symmetry case, as per equation 21 in [2].
fn permute_multi_index_axial(
    alpha: &RowRef<usize>,
    transfer_vector: RowRef<i32>,
    interpolation_order: &usize,
) -> Vec<usize> {
    let mut pi_a_t: Vec<usize> = Vec::new();

    alpha.iter().enumerate().for_each(|(idx, a_i)| {
        let value: usize;
        if transfer_vector[idx] < 0 {
            value = interpolation_order - (a_i - 1);
        } else {
            value = *a_i;
        }
        pi_a_t.push(value);
    });

    pi_a_t
}

/// Permutes the multi-index for the diagonal symmetry case, as per equation 22 in [2].
fn permute_multi_index_diagonal(alpha: &RowRef<usize>, sort_indices: &[usize]) -> Vec<usize> {
    // Reorder alpha based on the sorted indices
    sort_indices.iter().map(|&i| alpha[i]).collect()
}

/// Computes permutation indices that map a permuted multi-index to row and column indices in the K matrix
/// as per equations 26 and 27 from [2].
fn calculate_diagonal_axial_permutation_indices(
    transfer_vector: Option<RowRef<i32>>,
    sort_indices: Option<&[usize]>,
    interpolation_order: &usize,
    multi_indices: &Mat<usize>,
    diagonal: bool,
) -> Vec<usize> {
    let size = multi_indices.shape().0;
    let mut permutation_indices: Vec<usize> = vec![0; size];
    let mut alpha: RowRef<usize>;
    let mut alpha_prime: Vec<usize>;
    let mut i: usize;

    for j in 0..size {
        alpha = multi_indices.row(j);

        if diagonal {
            alpha_prime = permute_multi_index_diagonal(&alpha, &sort_indices.unwrap());
        } else {
            alpha_prime =
                permute_multi_index_axial(&alpha, transfer_vector.unwrap(), &interpolation_order)
        }

        i = map_multi_index_to_k(&alpha_prime, &interpolation_order);

        permutation_indices[i] = j;
    }

    permutation_indices
}

/// Finds matching rows in `all_m2l_vectors` that correspond to each
/// axis flip permutations.
fn lookup_axial_permutation_cases(
    axis_sign_permutations: &Mat<i32>,
    all_m2l_vectors: &Mat<i32>,
) -> Vec<usize> {
    let mut matching_indices: Vec<usize> = Vec::new();

    for vector in all_m2l_vectors.row_iter() {
        let flip_axes = Row::from_iter(vector.iter().map(|axis| if *axis < 0 { -1 } else { 1 }));

        for (idx, permutation) in axis_sign_permutations.row_iter().enumerate() {
            if flip_axes == permutation {
                matching_indices.push(idx);
                break;
            }
        }
    }

    matching_indices
}

/// Finds matching rows in `all_m2l_vectors` that correspond to each
/// diagonal flip permutations.
fn lookup_diagonal_permutation_cases(
    axis_order_permutations: &Vec<Vec<usize>>,
    all_m2l_vectors: &Mat<i32>,
) -> Vec<usize> {
    let m2l_vector_sorted_axes: Vec<Vec<usize>> = all_m2l_vectors
        .row_iter()
        .map(|row| {
            let abs_row_vec: Vec<i32> = row.iter().map(|axis| axis.abs() * -1).collect();
            utils::argsort(&abs_row_vec)
        })
        .collect();

    let mut matching_indices: Vec<usize> = Vec::new();

    for sort_axes in m2l_vector_sorted_axes.iter() {
        for (idx, permutation) in axis_order_permutations.iter().enumerate() {
            if sort_axes == permutation {
                matching_indices.push(idx);
                break;
            }
        }
    }

    matching_indices
}

/// Combines axial + diagonal permutation cases to index into precomputed reference operators.
fn lookup_combined_permutation_cases(
    combined_permutation_cases: &Vec<Vec<usize>>,
    permutation_combinations: &Vec<Vec<usize>>,
) -> Vec<usize> {
    let mut matching_indices: Vec<usize> = Vec::new();

    for case in combined_permutation_cases.iter() {
        for (idx, combination) in permutation_combinations.iter().enumerate() {
            if combination == case {
                matching_indices.push(idx);
                break;
            }
        }
    }

    matching_indices
}

/// Finds the indices of the relevant reference M2L vector for all possible M2L vectors
/// as per equations 26 and 27 in [2].
fn lookup_reference_vectors(
    all_m2l_vectors: &Mat<i32>,
    reference_m2l_vectors: &Mat<i32>,
) -> Vec<usize> {
    let mut matching_indices: Vec<usize> = vec![0; all_m2l_vectors.shape().0];

    // Sort the absolute values of each row in both arrays
    let mut sorted_reference_vectors = reference_m2l_vectors.clone();

    sorted_reference_vectors.row_iter_mut().for_each(|row| {
        let mut sorted_row_vec = Vec::from_iter(row.cloned().iter().cloned());
        sorted_row_vec.sort();
        row.iter_mut()
            .enumerate()
            .for_each(|(idx, val)| *val = sorted_row_vec[idx]);
    });

    for (global_idx, vector) in all_m2l_vectors.row_iter().enumerate() {
        let transfer_vec = {
            let mut sorted_vec: Vec<i32> = vector.iter().map(|i| i.abs()).collect();
            sorted_vec.sort();

            Row::from_iter(sorted_vec)
        };

        // Find the index of the first matching row in the reference array for each row in the transfer array
        for (idx, reference) in sorted_reference_vectors.row_iter().enumerate() {
            if transfer_vec == reference {
                matching_indices[global_idx] = idx;
                break;
            }
        }
    }

    matching_indices
}

/// Precomputes the full set of permutation lookup tables to use in exploiting symmetries in M2L operators.
fn get_permutation_lookups(
    dimensions: &usize,
    interpolation_order: &usize,
    all_m2l_vectors: &Mat<i32>,
    reference_m2l_vectors: &Mat<i32>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<usize>, Vec<usize>) {
    let dimension_range = 0..*dimensions;
    let len_dim_range = dimension_range.len();

    let axis_order_permutations: Vec<Vec<usize>> = dimension_range
        .into_iter()
        .permutations(len_dim_range)
        .unique()
        .collect();

    let axis_sign_permutations =
        utils::cartesian_product::<i32>(&vec![-1, 1], *dimensions);

    let num_axis_order_permutations: usize = axis_order_permutations.len();
    let num_axis_sign_permutations = axis_sign_permutations.shape().0;

    let multi_indices_range: Vec<usize> = (1..interpolation_order + 1).collect();

    let multi_indices =
        utils::cartesian_product::<usize>(&multi_indices_range, *dimensions);

    let unique_diagonal_permutation_indices: Vec<Vec<usize>> = axis_order_permutations
        .iter()
        .map(|i| {
            calculate_diagonal_axial_permutation_indices(
                None,
                Some(i),
                &interpolation_order,
                &multi_indices,
                true,
            )
        })
        .collect();

    let unique_axial_permutation_indices: Vec<Vec<usize>> = axis_sign_permutations
        .row_iter()
        .map(|i| {
            calculate_diagonal_axial_permutation_indices(
                Some(i),
                None,
                &interpolation_order,
                &multi_indices,
                false,
            )
        })
        .collect();

    let permutation_combinations: Vec<Vec<usize>> = iproduct!(
        0..num_axis_sign_permutations,
        0..num_axis_order_permutations
    )
    .into_iter()
    .map(|(a, b)| vec![a, b])
    .collect();

    let combined_permutation_indices: Vec<Vec<usize>> = permutation_combinations
        .iter()
        .map(|permutation| {
            let axial_permutation = &unique_axial_permutation_indices[permutation[0]];
            let diagonal_permutation = &unique_diagonal_permutation_indices[permutation[1]];
            let combo: Vec<usize> = diagonal_permutation
                .iter()
                .map(|i| axial_permutation[*i])
                .collect();
            combo
        })
        .collect();

    let inverse_permutations: Vec<Vec<usize>> = combined_permutation_indices
        .iter()
        .map(|permutation| utils::argsort(&permutation))
        .collect();

    let axial_permutation_cases =
        lookup_axial_permutation_cases(&axis_sign_permutations, all_m2l_vectors);

    let diagonal_permutation_cases =
        lookup_diagonal_permutation_cases(&axis_order_permutations, all_m2l_vectors);

    let combined_permutation_cases: Vec<Vec<usize>> = axial_permutation_cases
        .iter()
        .zip(diagonal_permutation_cases.iter())
        .map(|(a, b)| vec![*a, *b])
        .collect();

    let permutation_lookups =
        lookup_combined_permutation_cases(&combined_permutation_cases, &permutation_combinations);

    let reference_vector_lookups = lookup_reference_vectors(all_m2l_vectors, reference_m2l_vectors);

    (
        combined_permutation_indices,
        inverse_permutations,
        permutation_lookups,
        reference_vector_lookups,
    )
}

/// Generates Chebyshev target points in a unit hypercube based on the interpolation order.
fn generate_chebyshev_target_points(
    nodes: &Vec<f64>,
    dimensions: &usize,
    length: &f64,
) -> Mat<f64> {
    let mut chebyshev_target_points =
        utils::cartesian_product::<f64>(&nodes, *dimensions);

    chebyshev_target_points.row_iter_mut().for_each(|row| {
        row.iter_mut().for_each(|element| *element *= 0.5 * length);
    });

    chebyshev_target_points
}

/// Generates Chebyshev source points for the observation cell in the unit
/// hypercube based on the interpolation order.
fn generate_chebyshev_source_points(
    nodes: &Vec<f64>,
    reference_m2l_vector: RowRef<i32>,
    interpolation_order: &usize,
    dimensions: &usize,
    length: &f64,
) -> Mat<f64> {
    let locations_vec: Vec<f64> = (0..*interpolation_order)
        .into_iter()
        .map(|i| i as f64)
        .collect();

    let mut chebyshev_source_points =
        utils::cartesian_product::<f64>(&locations_vec, *dimensions);

    chebyshev_source_points.row_iter_mut().for_each(|row| {
        row.iter_mut().enumerate().for_each(|(idx, item)| {
            let initial_val = *item;
            *item =
                (reference_m2l_vector[idx] as f64 + (nodes[initial_val as usize] * 0.5)) * length;
        });
    });

    chebyshev_source_points
}

/// Precomputes all necessary interpolation and translation operators for the
/// Chebyshev-based Black-Box Fast Multipole Method (BBFMM).
///
/// This function is intended to be called once during initialization and reused
/// throughout the FMM evaluation. It constructs operators used in both the
/// upward (M2M) and downward (M2L, L2L) passes of the algorithm, as well as
/// precomputing permutation and reference data to enable symmetry exploitation
/// in the far-field interactions.
///
/// # Arguments
///
/// * `interpolation_order` - The number of Chebyshev nodes per axis (usually denoted `p`).
/// * `dimensions` - Number of spatial dimensions (e.g., 1, 2, or 3).
/// * `radius` - Half-width of the root node bounding box (defines cell scale).
/// * `depth` - Maximum depth of the FMM octree.
/// * `kernel_function` - Function handle for evaluating the kernel.
/// * `use_svd_compression` - Whether to compress M2L operators using ACA and SVD recompression.
///
/// # Returns
///
/// PrecomputeOperators - Struct containing precomputed approximation operators.
pub fn precompute_approximation_operators<K: KernelFunction + Send + Sync>(
    interpolation_order: &usize,
    dimensions: &usize,
    radius: &f64,
    depth: &u64,
    kernel: &K,
    compression_type: &M2LCompressionType,
    epsilon: &f64,
) -> PrecomputeOperators {
    let num_child_cells = (2 as usize).pow(*dimensions as u32);
    let num_nodes_nd = interpolation_order.pow(*dimensions as u32);

    // Generate the one dimensional Chebyshev nodes.
    let nodes = generate_chebyshev_nodes(&interpolation_order);

    // Generate the Chebyshev nodes for the given dimensions.
    let nodes_nd = utils::cartesian_product::<f64>(&nodes, *dimensions);

    // Evaluate the Chebyshev polynomials of the first kind at the nodes.
    let polynomial_nodes =
        evaluate_chebyshev_polynomials(&interpolation_order, &interpolation_order, &nodes);

    // Get the matrices of multipole to multipole transfer coefficients (Chebyshev weights).
    let m2m_transfer_matrices = get_m2m_transfer_matrices(
        &interpolation_order,
        &nodes,
        &polynomial_nodes,
        &dimensions,
        &num_child_cells,
    );

    // Get an array of all possible M2L transfer vectors and
    // the M2L vectors for the reference domain.
    let (all_m2l_vectors, reference_m2l_vectors) = get_m2l_vectors(&dimensions);

    // Get all the permutation lookup arrays.
    let (permutation_indices, inverse_permutations, permutation_lookups, reference_vector_lookups) =
        get_permutation_lookups(
            &dimensions,
            &interpolation_order,
            &all_m2l_vectors,
            &reference_m2l_vectors,
        );

    let num_reference_cells = reference_m2l_vectors.shape().0;

    // Parallel loop to calculate the compressed M2L operators for each level in parallel.
    let level_operators: Vec<(usize, HashMap<usize, Mat<f64>>, HashMap<usize, Mat<f64>>)> = (2
        ..*depth as usize + 1)
        .into_par_iter()
        .map(|level| {
            // Length of the cell at the current level.
            let cell_length_level = *radius / (2usize.pow(level as u32 - 1)) as f64;

            // Generate Chebyshev target points.
            let target_points =
                generate_chebyshev_target_points(&nodes, &dimensions, &cell_length_level);

            // Maps for the reference M2L operators for the current level.
            let mut u_level: HashMap<usize, Mat<f64>> = HashMap::new();
            let mut vt_level: HashMap<usize, Mat<f64>> = HashMap::new();

            for i in 0..num_reference_cells {
                let source_points = generate_chebyshev_source_points(
                    &nodes,
                    reference_m2l_vectors.row(i),
                    &interpolation_order,
                    &dimensions,
                    &cell_length_level,
                );

                match compression_type {
                    M2LCompressionType::ACA => {
                        // Use Adaptive Cross Approximation (ACA) with partial pivoting
                        // followed by an SVD recompression to compress the M2L operators.
                        let num_rows = target_points.shape().0;
                        let num_columns = source_points.shape().0;

                        // Given the way the M2L operators are used in the FMM we need to parse the source and target
                        // points to the generator in the opposite order to what you'd expect.
                        // i.e. The target_points input for the function needs to be the Chebyshev source_points and
                        // the source_points input for the function needs to be the Chebyshev target_points.
                        let matrix_subset_generator =
                            |rows_start: &usize,
                             rows_end: &usize,
                             columns_start: &usize,
                             columns_end: &usize| {
                                utils::get_a_matrix_subset(
                                    &source_points,
                                    &target_points,
                                    kernel,
                                    rows_start,
                                    rows_end,
                                    columns_start,
                                    columns_end,
                                )
                            };

                        let (u, v) = aca::aca_partial_pivoting(
                            num_rows,
                            num_columns,
                            matrix_subset_generator,
                            &epsilon,
                        );

                        let (truncated_u, truncated_vt) = aca::recompress_aca(&u, &v, &epsilon);

                        u_level.insert(i, truncated_u);
                        vt_level.insert(i, truncated_vt);
                    }
                    M2LCompressionType::SVD => {
                        let a_matrix = utils::get_a_matrix(
                            &source_points,
                            &target_points,
                            kernel,
                        );

                        let svd = a_matrix.svd().unwrap();
                        let ur = svd.U(); // Left singular vectors (k × k)
                        let sr = svd.S(); // Singular values (vector of length k)
                        let vrt = svd.V().transpose(); // Right singular vectors transposed (k × k)

                        // Determine new target rank based on cumulative sum of squares
                        let sigma_vec: Vec<f64> = sr.column_vector().iter().cloned().collect();
                        let new_rank = aca::calculate_singular_values_cutoff(sigma_vec, &epsilon);

                        // Recompress: U = Q_u * U_r * diag(sigma), V = V_r^T * Q_v^T
                        let truncated_u = ur.subcols(0, new_rank).to_owned();
                        let truncated_vt = sr.column_vector().subrows(0, new_rank).as_diagonal()
                            * vrt.subrows(0, new_rank);

                        u_level.insert(i, truncated_u);
                        vt_level.insert(i, truncated_vt);
                    }
                    M2LCompressionType::None => {
                        // Create full-size M2L reference operators.
                        let k_values = utils::get_a_matrix(
                            &source_points,
                            &target_points,
                            kernel,
                        );

                        u_level.insert(i, k_values);
                    }
                }
            }

            (level, u_level, vt_level)
        })
        .collect();

    // Aggregate the results into the final HashMaps
    let mut truncated_u: HashMap<usize, HashMap<usize, Mat<f64>>> = HashMap::new();
    let mut truncated_vt: HashMap<usize, HashMap<usize, Mat<f64>>> = HashMap::new();

    for (level, u_map, vt_map) in level_operators {
        truncated_u.insert(level, u_map);
        truncated_vt.insert(level, vt_map);
    }

    PrecomputeOperators {
        num_nodes_nd,
        nodes_nd,
        polynomial_nodes,
        m2m_transfer_matrices,
        u: truncated_u,
        vt: truncated_vt,
        permutation_indices,
        inverse_permutations,
        permutation_lookups,
        reference_vector_lookups,
    }
}

/// Calculates the coefficients to map the points in a cell to the Chebyshev nodes of the cell.
/// # Arguments
///
/// * `interpolation_order` - Number of Chebyshev nodes per axis (e.g. `p`).
/// * `cell_point_locations` - Input points to interpolate (modified in place; shape `N × d`).
/// * `center` - Center coordinates of the cell in `d` dimensions.
/// * `length` - Side length of the cell (assumes cubic cells).
/// * `polynomial_nodes` - Evaluation matrix of Chebyshev polynomials (shape `p × p`).
/// * `dimensions` - Number of spatial dimensions (`d`).
///
/// # Returns
///
/// A matrix of interpolation coefficients (`N × p^d`), where each row contains
/// the tensor-product weights used to interpolate the corresponding input point
/// onto the Chebyshev basis of the cell.
pub fn get_approximation_coefficients(
    interpolation_order: &usize,
    cell_point_locations: &mut Mat<f64>,
    center: &Vec<f64>,
    length: &f64,
    polynomial_nodes: &Mat<f64>,
    dimensions: &usize,
) -> Mat<f64> {
    // Scale points to the [-1, 1]^d hypercube.
    cell_point_locations.row_iter_mut().for_each(|row| {
        row.iter_mut().enumerate().for_each(|(idx, element)| {
            *element = (*element - center[idx]) / (length * 0.5);
        });
    });

    let mut one_d_transfer_coefficients: Vec<Mat<f64>> = Vec::new();

    for d in 0..*dimensions as usize {
        let column_vec = cell_point_locations.col_as_slice(d).to_vec();
        let tn_x = evaluate_chebyshev_polynomials(
            &interpolation_order,
            &cell_point_locations.nrows(),
            &column_vec,
        );
        let sn = calculate_sn(tn_x, &polynomial_nodes, &interpolation_order);

        one_d_transfer_coefficients.push(sn);
    }

    let num_columns = interpolation_order.pow(*dimensions as u32);
    let indices: Vec<usize> = (0..*interpolation_order).into_iter().collect();

    let tensor_product_indices =
        utils::cartesian_product::<usize>(&indices, *dimensions);

    let mut transfer_coefficients = Mat::<f64>::zeros(cell_point_locations.nrows(), num_columns);

    transfer_coefficients
        .row_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(idx, element)| {
                *element = 1.0;

                let wanted_indices = tensor_product_indices.row(idx);

                for (d, coeffs) in one_d_transfer_coefficients.iter().enumerate() {
                    let poly_idx = wanted_indices[d];
                    *element *= coeffs[(i, poly_idx)];
                }
            });
        });

    transfer_coefficients
}

/// Scales Chebyshev nodes from the reference domain `[-1, 1]^d`
/// into the physical coordinates of a given FMM cell.
///
/// # Arguments
///
/// * `nodes_nd` - Tensor-product grid of Chebyshev nodes in `[-1, 1]^d` (shape `p^d × d`)
/// * `cell_center` - Coordinates of the center of the destination cell (length `d`)
/// * `cell_length` - Side length of the destination cell
///
/// # Returns
///
/// A new matrix of scaled Chebyshev node coordinates, transformed to lie within
/// the specified cell bounds (shape `p^d × d`).
pub fn scale_cheb_nodes_to_cell(
    nodes_nd: &Mat<f64>,
    cell_center: &Vec<f64>,
    cell_length: &f64,
) -> Mat<f64> {
    let nodes_nd_shape = nodes_nd.shape();

    let mut scaled_cheb_nodes = Mat::<f64>::zeros(nodes_nd_shape.0, nodes_nd_shape.1);

    nodes_nd.row_iter().enumerate().for_each(|(row_idx, row)| {
        row.iter().enumerate().for_each(|(col_idx, element)| {
            let scaled_element = cell_center[col_idx] + (cell_length * 0.5) * element;
            scaled_cheb_nodes[(row_idx, col_idx)] = scaled_element;
        });
    });

    scaled_cheb_nodes
}
