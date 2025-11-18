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
use faer::Mat;
use faer::{linalg::solvers::Solve, unzip, zip};

pub fn evaluate_monomials(
    points: &Mat<f64>,
    degree: &i32,
    basis_size: &usize,
    translation_factor: &Vec<f64>,
    scale_factor: &Vec<f64>,
) -> Mat<f64> {
    // Scale the domain points to the [-1, 1]^d hypercube for monomial evaluation.
    let mut scaled_points = points.clone();

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
    use equator::assert;
    use faer::{mat, utils::approx::*, Mat};

    fn run_case(points: Mat<f64>, degree: i32, expected: Mat<f64>) {
        let (n, d) = points.shape();
        assert_eq!(n, expected.nrows(), "row mismatch in test setup");
        let basis_size = expected.ncols();

        let translation_factor = vec![0.0; d];
        let scale_factor = vec![1.0; d];

        let monomials = evaluate_monomials(
            &points,
            &degree,
            &basis_size,
            &translation_factor,
            &scale_factor,
        );

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (2 as f64));
        assert!(&monomials ~ &expected);
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
            [1.0, 1.0, 2.0,  1.0,  2.0,  4.0],
            [1.0, 3.0, 4.0,  9.0, 12.0, 16.0],
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
            [1.0, 1.0, 2.0, 3.0,  1.0,  2.0,  3.0,  4.0,  6.0,  9.0],
            [1.0, 4.0, 5.0, 6.0, 16.0, 20.0, 24.0, 25.0, 30.0, 36.0],
        ];
        run_case(points, 2, expected);
    }
}
