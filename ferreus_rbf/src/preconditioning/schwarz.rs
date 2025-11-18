/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements an overlapping Schwarz preconditioner built on the domain decomposition hierarchy.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # schwarz
//! 
//! Module defining an overlapping Schwarz preconditioner for use with RBF interpolation.
//!  
//! This Schwarz preconditioner can be thought of as Restricted Additive Schwarz within
//! the levels, and Multiplicative Schwarz between levels, as described in Section 4
//! of [1].
//! 
//! # References
//! 1.  R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis
//!     function interpolation equations: domain decomposition methods. SIAM J. Sci.
//!     Comput., 22(5):1717–1740 (electronic), 2000.
//! 2.  Haase, G., Martin, D., Schiffmann, P., Offner, G. (2018). A Domain Decomposition
//!     Multilevel Preconditioner for Interpolation with Radial Basis Functions. 
//!     In: Lirkov, I., Margenov, S. (eds) Large-Scale Scientific Computing. LSSC 2017.

use super::domain_decomposition::DDMTree;
use crate::interpolant_config::InterpolantSettings;
use faer::{Mat, MatMut, MatRef};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

pub fn schwarz_preconditioner<F>(
    rg: &MatRef<f64>,
    ddm_tree: &mut DDMTree,
    matvec: &F,
    interpolant_settings: &InterpolantSettings,
    ortho_poly_matrix: &Option<Mat<f64>>,
) -> Mat<f64>
where
    F: Fn(&MatRef<f64>, Option<&Vec<usize>>) -> Mat<f64> + Sync,
{
    let mut sl = Mat::<f64>::zeros(rg.nrows(), rg.ncols());

    let coarse_idx = ddm_tree.levels.len() - 1;

    let coarse_level_indices = ddm_tree.levels[coarse_idx].point_indices.clone();

    if coarse_idx > 0 {
        // Iterate from the finest level to the second coarsest level.
        for i in (0..coarse_idx).into_iter() {
            let level_point_indices = &ddm_tree.levels[i].point_indices;

            sl += solve_fine_level(
                rg - matvec(&sl.as_ref(), Some(level_point_indices)),
                ddm_tree,
                &i,
                &interpolant_settings,
                &ortho_poly_matrix,
            );

            // Use the coarse level as a smoother, but only return the poly 'tail'
            // coefficients if this iteration is the coarsest of the fine levels.
            sl += solve_coarse_level(
                rg - matvec(&sl.as_ref(), Some(&coarse_level_indices)),
                ddm_tree,
                i == coarse_idx - 1,
            );
        }
    } else {
        // Just a single coarse domain, so solve directly.
        sl += solve_coarse_level(
            rg - matvec(&sl.as_ref(), Some(&coarse_level_indices)),
            ddm_tree,
            true,
        );
    }

    sl
}

fn solve_fine_level(
    residuals: Mat<f64>,
    ddm_tree: &mut DDMTree,
    level: &usize,
    interpolant_settings: &InterpolantSettings,
    ortho_poly_matrix: &Option<Mat<f64>>,
) -> Mat<f64> 
{
    let mut s1 = Mat::<f64>::zeros(residuals.nrows(), residuals.ncols());

    let s1_ref = &s1;

    ddm_tree.levels[*level]
        .leaf_domains
        .par_iter_mut()
        .for_each(|domain| {
            let coeff = domain.solve(&residuals.as_ref());

            // SAFETY: Since we're only writing back values from each subdomain's
            // internal points and the union of all subdomain internal points is the
            // whole node set for this level, this is inherently safe, assuming the domain
            // decomposition has been done as expected.
            unsafe {
                let output_ptr = s1_ref.as_ptr() as *mut f64;
                for (local_idx, global_idx) in domain.overlapping_point_indices.iter().enumerate() {
                    if domain.internal_points_mask[local_idx] {
                        *output_ptr.add(*global_idx) = coeff.point_coefficients[(local_idx, 0)];
                    }
                }
            }
        });

    if interpolant_settings.basis_size != 0 {
        // If there's polynomials involved we need to orthogonalise the weights
        // from the subdomain solves against the global polynomial basis.
        let num_points = s1.nrows() - interpolant_settings.basis_size;
        orthogonalise(&mut s1.subrows_mut(0, num_points), &ortho_poly_matrix);
    }

    s1
}

fn orthogonalise(weights: &mut MatMut<f64>, orthonormal_matrix: &Option<Mat<f64>>) {
    let projection_coeffs = orthonormal_matrix.as_ref().unwrap().transpose() * weights.to_owned();

    *weights -= orthonormal_matrix.as_ref().unwrap() * projection_coeffs;
}

fn solve_coarse_level(residuals: Mat<f64>, ddm_tree: &mut DDMTree, add_poly: bool) -> Mat<f64> {
    let mut sc = Mat::<f64>::zeros(residuals.nrows(), residuals.ncols());

    let coarse_idx = ddm_tree.levels.len() - 1;

    let coarse_domain = &ddm_tree.levels[coarse_idx].leaf_domains[0];

    let coeffs = coarse_domain.solve(&residuals.as_ref());

    coarse_domain
        .overlapping_point_indices
        .iter()
        .enumerate()
        .for_each(|(local_idx, global_idx)| {
            sc.row_mut(*global_idx).copy_from(&coeffs.point_coefficients.row(local_idx));
        });

    if coarse_domain.solve_for_poly && add_poly {
        let poly_coeffs = coeffs.poly_coefficients.as_ref().unwrap();
        let num_poly = poly_coeffs.nrows();
        let idx_offset = residuals.nrows() - num_poly;

        sc
            .subrows_mut(idx_offset, num_poly)
            .copy_from(poly_coeffs);
    }

    sc
}
