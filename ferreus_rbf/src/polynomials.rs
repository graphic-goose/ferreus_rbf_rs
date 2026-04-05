/////////////////////////////////////////////////////////////////////////////////////////////
//
// Evaluates polynomial and Lagrange bases used for drift terms in RBF interpolation.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::common;
use faer::{Mat, MatRef};
use faer::{linalg::solvers::Solve, unzip, zip};

pub fn evaluate_monomials(
    points: MatRef<f64>,
    degree: &i32,
    basis_size: &usize,
    translation_factor: &[f64],
    scale_factor: &[f64],
) -> Mat<f64> {
    // Scale the domain points to the [-1, 1]^d hypercube for monomial evaluation.
    let mut scaled_points = points.clone().to_owned();

    common::scale_points(&mut scaled_points, &translation_factor, &scale_factor);

    let (n, d) = scaled_points.shape();
    let mut monomials = Mat::<f64>::zeros(n, *basis_size);

    // constant column
    monomials.col_mut(0).fill(1.0);

    // linear columns
    if *degree >= 1 {
        monomials
            .subcols_mut(1, d)
            .copy_from(&scaled_points.as_ref());
    }

    // quadratic columns
    if *degree == 2 {
        let start = 1 + d;
        let mut k = 0usize;

        for i in 0..d {
            let xi = scaled_points.col(i);
            for j in i..d {
                let xj = scaled_points.col(j);
                let mut dst = monomials.col_mut(start + k);

                // elementwise product of points columns
                zip!(&mut dst, &xi, &xj).for_each(|unzip!(dst, xi, xj)| {
                    *dst = xi * xj;
                });

                k += 1;
            }
        }
    }

    monomials
}

pub fn evaluate_monomial_gradients(
    points: MatRef<f64>,
    poly_coefficients: &Mat<f64>,
    degree: i32,
    translation_factor: &[f64],
    scale_factor: &[f64],
) -> Mat<f64> {
    let (n, dims) = points.shape();
    let nrhs = poly_coefficients.ncols();

    let mut scaled_points = points.to_owned();
    common::scale_points(&mut scaled_points, translation_factor, scale_factor);

    let mut grads = Mat::<f64>::zeros(n, nrhs * dims);

    if degree >= 1 {
        for rhs in 0..nrhs {
            for d in 0..dims {
                let coeff = poly_coefficients[(1 + d, rhs)] / scale_factor[d];
                grads.col_mut(rhs * dims + d).fill(coeff);
            }
        }
    }

    if degree == 2 {
        let start = 1 + dims;
        let mut k = 0usize;

        for i_dim in 0..dims {
            for j_dim in i_dim..dims {
                for rhs in 0..nrhs {
                    let c = poly_coefficients[(start + k, rhs)];
                    for row in 0..n {
                        let xi = scaled_points[(row, i_dim)];
                        let xj = scaled_points[(row, j_dim)];

                        if i_dim == j_dim {
                            grads[(row, rhs * dims + i_dim)] +=
                                c * (2.0 * xi / scale_factor[i_dim]);
                        } else {
                            grads[(row, rhs * dims + i_dim)] += c * (xj / scale_factor[i_dim]);
                            grads[(row, rhs * dims + j_dim)] += c * (xi / scale_factor[j_dim]);
                        }
                    }
                }

                k += 1;
            }
        }
    }

    grads
}


pub fn get_lagrange_coefficients(monomials: &Mat<f64>) -> Mat<f64> {
    let (nrows, ncols) = monomials.shape();
    let rhs = Mat::<f64>::identity(nrows, ncols);
    let lu = monomials.full_piv_lu();
    lu.solve(rhs)
}

pub fn evaluate_lagrange_polynomials(
    monomials: &Mat<f64>,
    lagrange_coefficients: &Mat<f64>,
) -> Mat<f64> {
    monomials * lagrange_coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat, mat};

    fn assert_mat_close(lhs: &Mat<f64>, rhs: &Mat<f64>, atol: f64, rtol: f64) {
        let err = (lhs - rhs).norm_max();
        let scale = lhs.norm_max().max(rhs.norm_max()).max(1.0);
        let tol = atol + rtol * scale;
        assert!(err <= tol, "err={err:e}, tol={tol:e}, scale={scale:e}");
    }

    fn run_case(points: Mat<f64>, degree: i32, expected: Mat<f64>) {
        let (n, d) = points.shape();
        assert_eq!(n, expected.nrows(), "row mismatch in test setup");
        let basis_size = expected.ncols();

        let translation_factor = vec![0.0; d];
        let scale_factor = vec![1.0; d];

        let monomials = evaluate_monomials(
            points.as_ref(),
            &degree,
            &basis_size,
            &translation_factor,
            &scale_factor,
        );

        assert_mat_close(&monomials, &expected, 1e-12, 1e-10);
    }

    #[test]
    fn monomials_constant_1d() {
        let points = mat![[1.0], [2.0]];
        // Basis: [1]
        let expected = mat![[1.0], [1.0]];
        run_case(points, 0, expected);
    }

    #[test]
    fn monomials_linear_1d() {
        let points = mat![[1.0], [2.0]];
        // Basis: [1, x]
        let expected = mat![[1.0, 1.0], [1.0, 2.0]];
        run_case(points, 1, expected);
    }

    #[test]
    fn monomials_quadratic_1d() {
        let points = mat![[1.0], [2.0]];
        // Basis: [1, x, x^2]
        let expected = mat![[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]];
        run_case(points, 2, expected);
    }

    #[test]
    fn monomials_constant_2d() {
        let points = mat![[1.0, 2.0], [1.0, 2.0]];
        // Basis: [1]
        let expected = mat![[1.0], [1.0]];
        run_case(points, 0, expected);
    }

    #[test]
    fn monomials_linear_2d() {
        let points = mat![[1.0, 2.0], [3.0, 4.0]];
        // Basis: [1, x, y]
        let expected = mat![[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]];
        run_case(points, 1, expected);
    }

    #[test]
    fn monomials_quadratic_2d() {
        let points = mat![[1.0, 2.0], [3.0, 4.0]];
        // Basis: [1, x, y, x^2, x*y, y^2]
        let expected = mat![
            [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
            [1.0, 3.0, 4.0, 9.0, 12.0, 16.0],
        ];
        run_case(points, 2, expected);
    }

    #[test]
    fn monomials_constant_3d() {
        let points = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        // Basis: [1]
        let expected = mat![[1.0], [1.0]];
        run_case(points, 0, expected);
    }

    #[test]
    fn monomials_linear_3d() {
        let points = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        // Basis: [1, x, y, z]
        let expected = mat![[1.0, 1.0, 2.0, 3.0], [1.0, 4.0, 5.0, 6.0]];
        run_case(points, 1, expected);
    }

    #[test]
    fn monomials_quadratic_3d() {
        let points = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        // Basis: [1, x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]
        let expected = mat![
            [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 4.0, 5.0, 6.0, 16.0, 20.0, 24.0, 25.0, 30.0, 36.0],
        ];
        run_case(points, 2, expected);
    }
}
