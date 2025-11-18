/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements overlapping domain structures and local solvers for the domain decomposition preconditioner.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # domain
//! 
//! Defines one overlapping subproblem within a Domain Decomposition Method (DDM) for
//! radial basis function (RBF) interpolation and preconditioning. A `Domain` is built
//! from a subset of the global node set, maintains the bookkeeping required to
//! assemble and factorise its local system, and provides a solver that maps the local
//! right-hand side to RBF (and optional polynomial) coefficients.
//! 
//! # References
//! 1.  R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis
//!     function interpolation equations: domain decomposition methods. SIAM J. Sci.
//!     Comput., 22(5):1717–1740 (electronic), 2000.
//! 2. J. B. Cherrie. Fast Evaluation of Radial Basis Functions: Theory and Application.
//!     PhD thesis, University of Canterbury, 2000.

use std::collections::HashSet;
use std::sync::Arc;

use crate::{
    common,
    interpolant_config::InterpolantSettings,
    linalg::{LltRfp, Lblt},
    polynomials,
    rbf::Coefficients,
    global_trend::GlobalTrendTransform,
};

use faer::{
    linalg::{
        matmul,
        solvers::{PartialPivLu, Solve},
    },
    reborrow::*,
    Accum, Mat, MatRef, Par, Side,
};

use ferreus_rbf_utils;

pub enum DomainSolver {
    Llt(LltRfp<f64>),
    Lblt(Lblt<f64>),
}

impl Default for DomainSolver {
    fn default() -> Self {
        DomainSolver::Llt(LltRfp::<f64>::default())
    }
}

impl DomainSolver {
    /// Try LLᵀ first. If it fails (matrix not SPD / numerically indefinite),
    /// fall back to Bunch–Kaufman LDLᵀ.
    pub fn new(a: MatRef<'_, f64>, side: Side) -> Self {
        match LltRfp::<f64>::try_new(a, side) {
            Ok(llt) => DomainSolver::Llt(llt),
            Err(_)  => DomainSolver::Lblt(Lblt::<f64>::new(a, side)),
        }
    }

    pub fn solve(&self, rhs: &Mat<f64>) -> Mat<f64> {
        match self {
            DomainSolver::Llt(s) => s.solve(rhs),
            DomainSolver::Lblt(s) => s.solve(rhs),
        }
    }

    // /// Optional: expose which factorization you ended up using.
    // pub fn kind(&self) -> &'static str {
    //     match self {
    //         DomainSolver::Llt(_) => "llt",
    //         DomainSolver::Lblt(_) => "lblt",
    //     }
    // }
}


/// Represents a single domain in the domain decomposition method (DDM).
pub struct Domain {
    /// The global indices of all points, internal plus overlapping, within the domain.
    pub overlapping_point_indices: Vec<usize>,

    /// Boolean mask matching the overlapping point indices. true means an internal
    /// point, false means a point from another, overlapping, domain.
    pub internal_points_mask: Vec<bool>,

    /// Bounding box of the domain `[xmin, xmax, ymin, ymax, ...]`
    pub extents: Vec<f64>,

    /// Whether or not to solve for and return the polynomial 'tail' coefficients.
    pub solve_for_poly: bool,

    /// Cholesky LLT solver using Rectangular Full Packed (RFP) format.
    solver: DomainSolver,

    /// LU factorisation of the special point monomials. Used for recovering the
    /// polynomial coefficients.
    special_point_factor: Option<PartialPivLu<f64>>,

    /// Lagrange polynomial basis part of the Q matrix defined by Beatson's "possible
    /// choice for Q" in [1].
    q_matrix_top: Option<Mat<f64>>,

    /// The rows of the A matrix that relate to the chosen 'special points'.
    a_special_points_rows: Option<Mat<f64>>,

    /// The local indices of chosen 'special points'.
    special_point_indices: Option<Vec<usize>>,
}

impl Domain {
    /// Construct a new domain with a given set of overlapping points.
    pub fn new(overlapping_point_indices: Vec<usize>) -> Self {
        Self {
            overlapping_point_indices,
            internal_points_mask: Vec::default(),
            extents: Vec::default(),
            solve_for_poly: false,
            solver: DomainSolver::default(),
            special_point_factor: None,
            q_matrix_top: None,
            a_special_points_rows: None,
            special_point_indices: None,
        }
    }

    /// Factorises the system matrix for this domain.
    ///
    /// This routine builds the local interpolation system `A` and computes 
    /// its factorisation.  
    ///
    /// If a polynomial basis is included, we follow Beatson’s `Q` formulation
    /// to construct an augmented system `QTAQ` that casts the system into a
    /// strictly positive definite form, which is scale independant and can be
    /// solved using Cholesky methods. This enables only the lower triangle of
    /// the factorisation to be stored, saving memory.
    ///
    /// If the monomial basis is rank-deficient (non-unisolvent), we apply
    /// Cherrie’s QR-based procedure described in `The non-unisolvent case`
    /// in section 1.2 of [2] to extract a reduced set of linearly independent
    /// columns, ensuring the augmented 'QTAQ' system is strictly
    /// positive definite. The momomial basis could become non-unisolvent if
    /// all points in the domain are on a plane for a 3D system, or line
    /// for a 2D system.
    pub fn factorise(
        &mut self,
        source_points: &Mat<f64>,
        interpolant_settings: Arc<InterpolantSettings>,
        solve_for_poly: bool,
        global_trend: &Option<GlobalTrendTransform>
    ) 
    {
        let mut lhs: Mat<f64>;
        let domain_points =
            ferreus_rbf_utils::select_mat_rows(&source_points, &self.overlapping_point_indices);

        if interpolant_settings.basis_size != 0 {
            // Scale the domain points to the [-1, 1]^d hypercube for monomial evaluation.
            let (translation_factor, scale_factor) =
                common::get_cheb_cube_scaling_factors(&domain_points);

            let monomial_points = match global_trend.is_some() {
                true => {
                    let gt = global_trend.as_ref().unwrap();
                    gt.inverse_transform_points(&domain_points)
                },
                false => domain_points.clone()
            };

            // Evaluate all candidate monomials.
            let scaled_monomials = polynomials::evaluate_monomials(
                &monomial_points,
                &interpolant_settings.polynomial_degree,
                &interpolant_settings.basis_size,
                &translation_factor,
                &scale_factor,
            );

            // QR with column pivoting to identify linearly independent monomials.
            let qrc = scaled_monomials.col_piv_qr();
            let rc = qrc.thin_R();
            let (piv_fwd, _) = qrc.P().arrays();

            // Rank threshold: treat tiny diagonal entries of rc as zero.
            let tol = 1E-10;
            let thresh = tol * rc.get(0, 0).abs();

            // Effective rank = number of |R_ii| above threshold (unisolvent size k).
            let rank = rc
                .diagonal()
                .column_vector()
                .iter()
                .filter(|val| val.abs() > thresh)
                .count();

            // Pick the k pivoted monomial columns, independent on this node set.
            let mut unisolvent_columns: Vec<usize> = piv_fwd[..rank].iter().cloned().collect();
            unisolvent_columns.sort();

            // Reduced full rank monomial matrix.
            let mut full_rank_monomials = Mat::<f64>::zeros(scaled_monomials.nrows(), rank);
            unisolvent_columns.iter().enumerate().for_each(|(i, j)| {
                full_rank_monomials
                    .col_mut(i)
                    .copy_from(scaled_monomials.col(*j));
            });

            // Apply rank-revealing QR to the transpose to select the “special points”.
            // The pivoting naturally selects points that are well separated in the
            // monomial feature space, giving a stable, unisolvent set for constructing
            // the Lagrange basis.
            let qrr = full_rank_monomials.transpose().col_piv_qr();
            let (piv_fwd, _) = qrr.P().arrays();

            let mut special_point_indices: Vec<usize> = piv_fwd[..rank].iter().cloned().collect();
            special_point_indices.sort();

            // Extract the special point monomials.
            let special_point_monomials =
                ferreus_rbf_utils::select_mat_rows(&full_rank_monomials, &special_point_indices);

            // Reorder overlapping point indices so special points come first.
            let special_points_set: HashSet<usize> =
                special_point_indices.iter().cloned().collect();

            let non_special_point_indices: Vec<usize> = (0..domain_points.nrows())
                .into_iter()
                .filter_map(|local_idx| {
                    if !special_points_set.contains(&local_idx) {
                        Some(local_idx)
                    } else {
                        None
                    }
                })
                .collect();

            let non_special_point_monomials =
                ferreus_rbf_utils::select_mat_rows(&full_rank_monomials, &non_special_point_indices);

            let global_special_point_indices: Vec<usize> = special_point_indices
                .iter()
                .map(|idx| self.overlapping_point_indices[*idx])
                .collect();

            let global_non_special_point_indices: Vec<usize> = non_special_point_indices
                .iter()
                .map(|idx| self.overlapping_point_indices[*idx])
                .collect();

            self.special_point_indices = Some((0..rank as usize).into_iter().collect());

            let mut sorted_point_indices: Vec<usize> =
                Vec::with_capacity(self.overlapping_point_indices.len());

            sorted_point_indices.extend(&global_special_point_indices);
            sorted_point_indices.extend(&global_non_special_point_indices);

            self.overlapping_point_indices = sorted_point_indices;

            // Also need to update the internal points mask to align with the new domain
            // point ordering.
            let special_point_internal_mask: Vec<bool> = special_point_indices
                .iter()
                .map(|idx| self.internal_points_mask[*idx])
                .collect();

            let non_special_internal_mask: Vec<bool> = self
                .internal_points_mask
                .iter()
                .enumerate()
                .filter_map(|(local_idx, &mask_val)| {
                    if !special_points_set.contains(&local_idx) {
                        Some(mask_val)
                    } else {
                        None
                    }
                })
                .collect();

            self.internal_points_mask.clear();
            self.internal_points_mask
                .extend(special_point_internal_mask);
            self.internal_points_mask.extend(non_special_internal_mask);

            let sorted_domain_points =
                ferreus_rbf_utils::select_mat_rows(source_points, &self.overlapping_point_indices);

            // Construct full kernel matrix.
            let a_matrix = ferreus_rbf_utils::get_a_matrix_symmetric_solver(
                &sorted_domain_points,
                &sorted_domain_points,
                &(*interpolant_settings).into(),
                &interpolant_settings.nugget,
            );

            let m = domain_points.nrows() - rank;

            // Build the Q matrix.
            // Q maps from non-special points into the polynomial constraint space,
            // using Lagrange polynomials built from the special points.
            // The Q matrix is full rank and its columns are orthogonal to the columns
            // of the standard polynomial matrix, P, that is, P^TQ = 0.
            let lagrange_coefficients =
                polynomials::get_lagrange_coefficients(&special_point_monomials);

            let q_matrix_top = -polynomials::evaluate_lagrange_polynomials(
                &non_special_point_monomials,
                &lagrange_coefficients,
            )
            .transpose();

            // Build the augmented Q^TAQ system.
            // Since Q^TAQ is just an identity matrix below the top lagrange basis rows
            // we can save some operations by splitting into block operations.
            let q = &q_matrix_top;
            let qt = q.transpose();
            let a11 = a_matrix.submatrix(0, 0, rank, rank);
            let a12 = a_matrix.submatrix(0, rank, rank, m);
            let a21 = a_matrix.submatrix(rank, 0, m, rank);
            let a22 = a_matrix.submatrix(rank, rank, m, m);

            lhs = Mat::<f64>::zeros(m, m);

            // qtaq_top_left = Q^T * (A11 * Q)
            let mut qtaq_top_left_tmp = Mat::<f64>::zeros(rank, m); // (A11 * Q)
            matmul::matmul(
                qtaq_top_left_tmp.rb_mut(),
                Accum::Replace,
                a11.rb(),
                q.rb(),
                1.0,
                Par::Seq,
            );
            matmul::matmul(
                lhs.rb_mut(),
                Accum::Add,
                qt.rb(),
                qtaq_top_left_tmp.rb(),
                1.0,
                Par::Seq,
            );

            // qtaq_top_right = Q^T * A12
            matmul::matmul(lhs.rb_mut(), Accum::Add, qt.rb(), a12.rb(), 1.0, Par::Seq);

            // qtaq_bottom_left = A21 * Q
            matmul::matmul(lhs.rb_mut(), Accum::Add, a21.rb(), q.rb(), 1.0, Par::Seq);

            // qtaq_bottom_right = A22
            lhs += a22;

            self.q_matrix_top = Some(q_matrix_top);

            if solve_for_poly {
                // If this is a coarse domain then we'll want to solve for the polynomial
                // 'tail' coefficients later, so form and factor for reuse during iterations.
                self.solve_for_poly = true;
                self.a_special_points_rows = Some(a_matrix.subrows(0, rank).to_owned());
                self.special_point_factor = Some(special_point_monomials.partial_piv_lu());
            }
        } else {
            // No polynomial term, just build A directly.
            lhs = ferreus_rbf_utils::get_a_matrix_symmetric_solver(
                &domain_points,
                &domain_points,
                &(*interpolant_settings).into(),
                &interpolant_settings.nugget,
            );
        }

        // Cholesky factorisation of the RHS matrix. Only stores the lower triangle
        // to save memory.
        self.solver = DomainSolver::new(lhs.as_ref(), Side::Lower);
    }

    /// Solve the local system for the given source values.
    ///
    /// Returns the RBF coefficients associated with this domain and optional
    /// polynomial coefficients, if enabled.
    ///
    /// For the polynomial-augmented case, the rhs is projected into the reduced
    /// space via the Q matrix before solving, and polynomial coefficients are
    /// recovered afterwards.    
    pub fn solve(&self, source_values: &MatRef<f64>) -> Coefficients {
        let num_points: usize;
        let rhs: Mat<f64>;
        let num_source_points = self.overlapping_point_indices.len();
        let mut num_special_points = 0usize;
        let num_rhs = source_values.ncols();

        // Gather rhs values for this domain.
        let domain_values = Mat::<f64>::from_fn(self.overlapping_point_indices.len(), num_rhs, |i, j| {
            *source_values.get(self.overlapping_point_indices[i], j)
        });

        if self.q_matrix_top.is_some() {
            // Polynomial case.
            num_special_points = self.special_point_indices.as_ref().unwrap().len();
            num_points = self.overlapping_point_indices.len() - num_special_points;

            // Augment rhs: rhs = Q^T d_special + d_non_special
            rhs = self.q_matrix_top.as_ref().unwrap().transpose()
                * &domain_values.subrows(0, num_special_points)
                + &domain_values.subrows(num_special_points, num_points);
        } else {
            // Standard case.
            rhs = domain_values.clone();
        }

        let mut point_coefficients = Mat::<f64>::zeros(num_source_points, num_rhs);

        let mut poly_coefficients = None;

        // Solve system.
        let gamma = self.solver.solve(&rhs);

        if self.q_matrix_top.is_some() {
            // Set lambda = Q * gamma
            let coefficients_top = self.q_matrix_top.as_ref().unwrap() * &gamma;
            point_coefficients
                .submatrix_mut(0, 0, num_special_points, rhs.ncols())
                .copy_from(coefficients_top);

            point_coefficients
                .submatrix_mut(
                    num_special_points,
                    0,
                    num_source_points - num_special_points,
                    rhs.ncols(),
                )
                .copy_from(gamma);
        } else {
            point_coefficients.copy_from(gamma);
        }

        // Recover polynomial coefficients if requested.
        if self.solve_for_poly {
            let d_special_point_values = Mat::<f64>::from_fn(num_special_points, num_rhs, |i, j| {
                *source_values.get(self.overlapping_point_indices[i], j)
            });

            // Find the polynomial from P_{k-1} interpolating to the following
            // residual function at the special points:
            //
            //   r(x) = d(x) - sum_{j=1..m} lambda_j * Phi(x, x_j)
            //
            // where d(x) is the data value, Phi is the kernel, and lambda_j are
            // the coefficients.
            let r = d_special_point_values
                - self.a_special_points_rows.as_ref().unwrap() * &point_coefficients;

            poly_coefficients = Some(self.special_point_factor.as_ref().unwrap().solve(&r));
        }

        Coefficients::new(point_coefficients, poly_coefficients)
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        common,
        interpolant_config::{InterpolantSettings, RBFKernelType},
        polynomials,
        RBFTestFunctions,
    };
    use equator::assert;
    use faer::{concat, utils::approx::*, Mat};
    use ferreus_rbf_utils::{self};

    fn generate_2d_points(num_points: usize) -> (Mat<f64>, Mat<f64>) {
        let dim = 2;
        let source_points = common::generate_random_points(num_points, dim, Some(42));
        let sourve_values = RBFTestFunctions::franke_2d(&source_points);

        (source_points, sourve_values)
    }

    fn generate_point_indices(num_points: usize) -> Vec<usize> {
        (0..num_points as usize).into_iter().collect()
    }

    fn naive_rbf_solve(
        source_points: &Mat<f64>,
        source_values: &Mat<f64>,
        interpolant_settings: Arc<InterpolantSettings>,
    ) -> Coefficients 
    {
        let lhs: Mat<f64>;
        let rhs: Mat<f64>;
        let point_coefficients: Mat<f64>;
        let mut poly_coefficients = None;

        let a_matrix = ferreus_rbf_utils::get_a_matrix_symmetric_solver(
            source_points,
            source_points,
            &(*interpolant_settings).into(),
            &interpolant_settings.nugget,
        );

        if interpolant_settings.basis_size != 0 {
            let num_poly = interpolant_settings.basis_size as usize;
            let (translation_factor, scale_factor) =
                common::get_cheb_cube_scaling_factors(&source_points);

            let poly_matrix = polynomials::evaluate_monomials(
                &source_points,
                &interpolant_settings.polynomial_degree,
                &interpolant_settings.basis_size,
                &translation_factor,
                &scale_factor,
            );

            let poly_t = poly_matrix.transpose().to_owned();

            let lhs_zeros = Mat::<f64>::zeros(num_poly, num_poly);

            lhs = concat![[a_matrix, poly_matrix], [poly_t, lhs_zeros]];

            rhs = concat![[&source_values], [Mat::<f64>::zeros(num_poly, 1)]];
        } else {
            lhs = a_matrix;
            rhs = source_values.clone();
        }

        let lu = lhs.partial_piv_lu();

        let all_coefficients = lu.solve(rhs);

        if interpolant_settings.basis_size != 0 {
            let split = all_coefficients.split_at_row(source_points.nrows());
            point_coefficients = split.0.to_owned();
            poly_coefficients = Some(split.1.to_owned());
        } else {
            point_coefficients = all_coefficients;
        }

        Coefficients::new(point_coefficients, poly_coefficients)
    }

    fn naive_rbf_evaluate(
        source_points: &Mat<f64>,
        target_points: &Mat<f64>,
        interpolant_settings: Arc<InterpolantSettings>,
        coefficients: &Coefficients,
    ) -> Mat<f64> 
    {
        let eval_a_matrix = ferreus_rbf_utils::get_a_matrix(
            target_points,
            source_points,
            (*interpolant_settings).into(),
        );

        let mut interpolated_values = eval_a_matrix * &coefficients.point_coefficients;

        if interpolant_settings.basis_size != 0 {
            let (translation_factor, scale_factor) =
                common::get_cheb_cube_scaling_factors(&source_points);

            let scaled_monomials = polynomials::evaluate_monomials(
                &target_points,
                &interpolant_settings.polynomial_degree,
                &interpolant_settings.basis_size,
                &translation_factor,
                &scale_factor,
            );

            // QR with column pivoting to identify linearly independent monomials.
            let qrc = scaled_monomials.col_piv_qr();
            let rc = qrc.thin_R();
            let (piv_fwd, _) = qrc.P().arrays();

            // Rank threshold: treat tiny diagonal entries of rc as zero.
            let tol = 1E-10;
            let thresh = tol * rc.get(0, 0).abs();

            // Effective rank = number of |R_ii| above threshold (unisolvent size k).
            let rank = rc
                .diagonal()
                .column_vector()
                .iter()
                .filter(|val| val.abs() > thresh)
                .count();

            // Pick the k pivoted monomial columns, independent on this node set.
            let mut unisolvent_columns: Vec<usize> = piv_fwd[..rank].iter().cloned().collect();
            unisolvent_columns.sort();

            // Reduced full rank monomial matrix.
            let mut full_rank_monomials = Mat::<f64>::zeros(scaled_monomials.nrows(), rank);
            unisolvent_columns.iter().enumerate().for_each(|(i, j)| {
                full_rank_monomials
                    .col_mut(i)
                    .copy_from(scaled_monomials.col(*j));
            });

            interpolated_values +=
                full_rank_monomials * coefficients.poly_coefficients.as_ref().unwrap();
        }

        interpolated_values
    }

    fn solve_domain(
        points: &Mat<f64>,
        values: &Mat<f64>,
        interpolant_settings: Arc<InterpolantSettings>,
    ) -> Coefficients 
    {
        let num_points = points.nrows();
        let point_indices = generate_point_indices(num_points);

        let mut naive_domain = Domain::new(point_indices);

        naive_domain.internal_points_mask =
            vec![true; naive_domain.overlapping_point_indices.len()];

        naive_domain.factorise(
            &points,
            interpolant_settings.clone(),
            interpolant_settings.basis_size != 0,
            &None,
        );

        let domain_coefficients = naive_domain.solve(&values.as_ref());

        let mut domain_point_coefficients = Mat::<f64>::zeros(num_points, 1);

        naive_domain
            .overlapping_point_indices
            .iter()
            .enumerate()
            .for_each(|(idx, global)| {
                domain_point_coefficients[(*global, 0)] =
                    domain_coefficients.point_coefficients[(idx, 0)];
            });

        Coefficients::new(
            domain_point_coefficients,
            domain_coefficients.poly_coefficients,
        )
    }

    fn test_domain_solver(points: &Mat<f64>, values: &Mat<f64>, interpolant_settings: InterpolantSettings) 
    {
        let interpolant_settings = Arc::new({
            let mut ks = interpolant_settings;
            ks.set_basis_size(&points.ncols());
            ks
        });

        let domain_coefficients = solve_domain(&points, &values, interpolant_settings.clone());

        let evaluated_values_at_source = naive_rbf_evaluate(
            &points,
            &points,
            interpolant_settings.clone(),
            &domain_coefficients,
        );

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (points.nrows() as f64));

        // RBF evaluated at the source points should return the original source values,
        // within a reasonable tolerance.
        assert!(&evaluated_values_at_source ~ &values);
    }

    #[test]
    fn solve_no_poly() {
        let num_points = 100;

        let (points, values) = generate_2d_points(num_points);

        let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Spheroidal).build();

        test_domain_solver(&points, &values, interpolant_settings);
    }

    #[test]
    fn solve_constant_poly() {
        let num_points = 100;

        let (points, values) = generate_2d_points(num_points);

        let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear).build();

        test_domain_solver(&points, &values, interpolant_settings);
    }

    #[test]
    fn solve_linear_poly() {
        let num_points = 100;

        let (points, values) = generate_2d_points(num_points);

        let interpolant_settings = InterpolantSettings::builder(RBFKernelType::ThinPlateSpline).build();

        test_domain_solver(&points, &values, interpolant_settings);
    }

    #[test]
    fn solve_linear_poly_non_unisolvent() {
        let num_points = 100;

        let (points, values) = generate_2d_points(num_points);

        // Force the points to be 3D and on the same plane.
        let points = concat![[points, Mat::<f64>::zeros(num_points, 1)]];

        let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear).build();

        test_domain_solver(&points, &values, interpolant_settings);
    }

    #[test]
    fn domain_solve_matches_naive() {
        let num_points = 100;

        let (points, values) = generate_2d_points(num_points);

        let interpolant_settings = InterpolantSettings::builder(RBFKernelType::ThinPlateSpline).build();

        let interpolant_settings = Arc::new({
            let mut ks = interpolant_settings;
            ks.set_basis_size(&2);
            ks
        });

        let domain_coefficients = solve_domain(&points, &values, interpolant_settings.clone());

        let naive_coefficients = naive_rbf_solve(&points, &values, interpolant_settings.clone());

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (num_points as f64));

        assert!(&domain_coefficients.point_coefficients ~ &naive_coefficients.point_coefficients);
        assert!(&domain_coefficients.poly_coefficients.unwrap() ~ &naive_coefficients.poly_coefficients.unwrap());
    }
}
