/////////////////////////////////////////////////////////////////////////////////////////////
//
// Adds helper linear algebra routines, including RFP-based Cholesky factorisation and solves.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # linalg
//!
//! Helper linear algebra functionality.
//!
//! # References
//! 1. Gustavson et al. Rectangular Full Packed Format for Cholesky’s Algorithm, 2009.

use faer::{
    self, Accum, Conj, Mat, Par, Side,
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{cholesky::llt, matmul, triangular_solve},
    mat::*,
    perm::Perm,
    prelude::*,
    reborrow::ReborrowMut,
};

use faer_traits::{ComplexField, Conjugate, math_utils::one};

#[derive(Debug)]
pub enum FactorizationError {
    NotSpd,       // LLT failed (matrix not SPD or numerically indefinite)
}

#[allow(non_snake_case)]
pub struct LltRfp<T> {
    L: Mat<T>,
    side: Side,
}

impl<T> Default for LltRfp<T> {
    fn default() -> Self {
        Self {
            L: Mat::new(),
            side: Side::Lower,
        }
    }
}

#[allow(non_snake_case)]
impl<T: ComplexField> LltRfp<T> {
    /// Returns the LLT factorization of the input A in RFP format.
    pub fn try_new(A: MatRef<T>, side: Side) -> Result<Self, FactorizationError> {
        // Convert the A matrix into RFP format.
        let mut AR = to_rfp(&A, &side);
        // Compute the in-place Cholesky factorisation of the RFP matrix.
        cholesky_rfp_factor(&mut AR, side)?;
        Ok(Self { L: AR, side: side })
    }

    pub fn solve(&self, rhs: &Mat<T>) -> Mat<T> {
        // Solve A @ X = B in RFP format.
        let X = cholesky_rfp_solve(&self.L, &rhs, &self.side);
        X
    }
}

/// Helper struct that stores the row_start, col_start, nrows, ncols
/// for the 2-by-2 block matrix for either lower or upper triangle form.
#[derive(Debug)]
struct SubmatrixParams {
    a11: (usize, usize, usize, usize),
    a21_or_a12: (usize, usize, usize, usize),
    a22: (usize, usize, usize, usize),
}

/// Helper function to get the block matrix parameters in the
/// RFP format, based on the upper/lower and odd/even cases.
fn get_rfp_block_submatrix_params(
    side: &Side,
    even: &bool,
    n1: &usize,
    n2: &usize,
) -> SubmatrixParams {
    let (row_offset, col_offset) = match even {
        true => (1, 0),
        false => (0, 1),
    };

    match side {
        Side::Lower => SubmatrixParams {
            a11: (0 + row_offset, 0, *n1, *n1),
            a21_or_a12: (*n1 + row_offset, 0, *n2, *n1),
            a22: (0, 0 + col_offset, *n2, *n2),
        },
        Side::Upper => SubmatrixParams {
            a11: (n2 + 1, 0, *n2, *n2),
            a21_or_a12: (0, 0, *n2, *n1),
            a22: (*n2, 0, *n1, *n1),
        },
    }
}

/// Helper function to get the 2-by-2 block matrix parameters in
/// normal format, based on the upper/lower and odd/even cases.
fn get_a_block_submatrix_params(side: &Side, n1: &usize, n2: &usize) -> SubmatrixParams {
    match side {
        Side::Lower => SubmatrixParams {
            a11: (0, 0, *n1, *n1),
            a21_or_a12: (*n1, 0, *n2, *n1),
            a22: (*n1, *n1, *n2, *n2),
        },
        Side::Upper => SubmatrixParams {
            a11: (0, 0, *n2, *n2),
            a21_or_a12: (0, *n2, *n2, *n1),
            a22: (*n2, *n2, *n1, *n1),
        },
    }
}

/// Converts the lower or upper triangle of a symmetric matrix into
/// Rectangular Full Packed Format (RFPF).
///
/// This implementation is based on [1].
#[allow(non_snake_case)]
fn to_rfp<T: ComplexField>(A: &MatRef<T>, side: &Side) -> Mat<T> {
    let N = A.nrows();
    assert!(A.ncols() == A.nrows());

    let even = N % 2 == 0;

    let LDAR = match even {
        true => N + 1,
        false => N,
    };

    let n1 = (N + 1) / 2;
    let n2 = N - n1;
    let mut AR = Mat::<T>::zeros(LDAR, n1);

    // Set helper structs to define 2-by-2 block matrices
    // based on lower/upper and odd/even cases.
    let from_subs = get_a_block_submatrix_params(&side, &n1, &n2);
    let to_subs = get_rfp_block_submatrix_params(&side, &even, &n1, &n2);

    let A11 = A.submatrix(
        from_subs.a11.0,
        from_subs.a11.1,
        from_subs.a11.2,
        from_subs.a11.3,
    );

    AR.submatrix_mut(to_subs.a11.0, to_subs.a11.1, to_subs.a11.2, to_subs.a11.3)
        .copy_from_triangular_lower(match side {
            Side::Lower => A11,
            Side::Upper => A11.transpose(),
        });

    let A12_or_A21 = A.submatrix(
        from_subs.a21_or_a12.0,
        from_subs.a21_or_a12.1,
        from_subs.a21_or_a12.2,
        from_subs.a21_or_a12.3,
    );

    AR.submatrix_mut(
        to_subs.a21_or_a12.0,
        to_subs.a21_or_a12.1,
        to_subs.a21_or_a12.2,
        to_subs.a21_or_a12.3,
    )
    .copy_from(A12_or_A21);

    let A22 = A.submatrix(
        from_subs.a22.0,
        from_subs.a22.1,
        from_subs.a22.2,
        from_subs.a22.3,
    );

    AR.submatrix_mut(to_subs.a22.0, to_subs.a22.1, to_subs.a22.2, to_subs.a22.3)
        .copy_from_triangular_upper(match side {
            Side::Lower => A22.transpose(),
            Side::Upper => A22,
        });

    AR
}

/// Helper function for Cholesky factorisation of a subblock.
#[inline(always)]
#[allow(non_snake_case)]
fn factor_subblock<T: ComplexField>(
    AR: &mut Mat<T>,
    sub: (usize, usize, usize, usize),
    trans_block: bool,
    par: Par,
) -> Result<(), FactorizationError> {
    let cholesky_memory = llt::factor::cholesky_in_place_scratch::<T>(sub.2, par, default());
    let mut memory = MemBuffer::new(cholesky_memory);
    let stack = MemStack::new(&mut memory);
    let mut llt = Mat::<T>::zeros(sub.2, sub.3);

    match trans_block {
        true => llt.copy_from(AR.submatrix(sub.0, sub.1, sub.2, sub.3).transpose()),
        false => llt.copy_from_triangular_lower(AR.submatrix(sub.0, sub.1, sub.2, sub.3)),
    };

    llt::factor::cholesky_in_place(llt.rb_mut(), default(), par, stack, default()).map_err(|_| FactorizationError::NotSpd)?;

    match trans_block {
        true => AR
            .submatrix_mut(sub.0, sub.1, sub.2, sub.3)
            .copy_from_triangular_upper(llt.transpose()),
        false => AR
            .submatrix_mut(sub.0, sub.1, sub.2, sub.3)
            .copy_from_triangular_lower(llt),
    };

    Ok(())
}

/// Helper function for getting the dimensions of an RFP format matrix.
#[allow(non_snake_case)]
fn get_dims_from_rfp(LDAR: &usize, n1: &usize) -> (bool, usize) {
    let N = match *LDAR == 2 * n1 - 1 {
        true => *LDAR,
        false => *LDAR - 1,
    };
    let even = N % 2 == 0;
    let n2 = N - n1;

    (even, n2)
}

/// Computes the in-place Cholesky factorisation of a matrix in
/// Rectangular Full Packed Format (RFPF).
///
/// This implementation is based on [1].
///
/// Follows LAPACK's `DPFTRF` logic.
#[allow(non_snake_case)]
fn cholesky_rfp_factor<T: ComplexField>(
    AR: &mut Mat<T>,
    side: Side,
) -> Result<(), FactorizationError> {
    let (LDAR, n1) = AR.shape();

    let (even, n2) = get_dims_from_rfp(&LDAR, &n1);

    let par = faer::get_global_parallelism();

    // Set helper struct to define 2-by-2 block matrices
    // based on lower/upper and odd/even cases.
    let subs = get_rfp_block_submatrix_params(&side, &even, &n1, &n2);

    // Step 1: Factor:
    //  lower: L11 @ L11.T = A11
    //  upper: U11.T @ U11 = A11
    factor_subblock(AR, subs.a11, false, par)?;

    // Step 2: Solve:
    //  lower: L21 @ L11.T = A21
    //  upper: U12T @ U11 = A12
    //
    // Since faer's triangular solver doesn't allow right-side i.e. X @ A = B
    // need to transpose and shift things around a little for the lower case to
    // align with LAPACK's implementation.
    let mut X = match side {
        Side::Lower => AR
            .submatrix(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            )
            .transpose()
            .to_owned(),
        Side::Upper => AR
            .submatrix(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            )
            .to_owned(),
    };

    triangular_solve::solve_lower_triangular_in_place(
        AR.submatrix(subs.a11.0, subs.a11.1, subs.a11.2, subs.a11.3),
        X.rb_mut(),
        par,
    );

    match side {
        Side::Lower => AR
            .submatrix_mut(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            )
            .copy_from(X.transpose()),
        Side::Upper => AR
            .submatrix_mut(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            )
            .copy_from(X),
    };

    //Step 3: Update:
    //  lower: A22 := A22 - L21 @ L21.T
    //  upper: A22 := A22 - U12 @ U12.T
    let mut acc = AR
        .submatrix(subs.a22.0, subs.a22.1, subs.a22.2, subs.a22.3)
        .to_owned();

    let (a, b) = match side {
        Side::Lower => {
            let m = AR.submatrix(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            );
            (m, m.transpose())
        }
        Side::Upper => {
            let m = AR.submatrix(
                subs.a21_or_a12.0,
                subs.a21_or_a12.1,
                subs.a21_or_a12.2,
                subs.a21_or_a12.3,
            );
            (m.transpose(), m)
        }
    };

    matmul::matmul(&mut acc, Accum::Add, a, b, T::from_f64_impl(-1.0), par);

    AR.submatrix_mut(subs.a22.0, subs.a22.1, subs.a22.2, subs.a22.3)
        .copy_from_triangular_upper(acc);

    // Step 4: Factor: U22.T @ U22 = A22
    factor_subblock(AR, subs.a22, true, par)?;

    Ok(())
}

/// Cholesky solver for Rectangular Full Packed Format (RFPF).
///
/// This implementation is based on [1].
///
/// Follows LAPACK's `DPFTRS` logic.
#[allow(non_snake_case)]
fn cholesky_rfp_solve<T: ComplexField>(AR: &Mat<T>, B: &Mat<T>, side: &Side) -> Mat<T> {
    let (LDAR, n1) = AR.shape();
    let (even, n2) = get_dims_from_rfp(&LDAR, &n1);

    let (num_b_rows, num_b_cols) = B.shape();

    let par = faer::get_global_parallelism();

    let mut X = B.clone();

    let subs = get_rfp_block_submatrix_params(&side, &even, &n1, &n2);

    let k = match side {
        Side::Lower => n1,
        Side::Upper => n2,
    };

    // Forward substition:
    //  lower: L Y = B
    //  upper: U.T @ Y = B

    // Step 1:
    //  lower: L11 @ Y1 = B1
    //  upper: U11T @ Y1 = B1
    triangular_solve::solve_lower_triangular_in_place(
        AR.submatrix(subs.a11.0, subs.a11.1, subs.a11.2, subs.a11.3),
        X.submatrix_mut(0, 0, k, num_b_cols),
        par,
    );

    // Step 2:
    //  lower: B2 = B2 - L21 @ Y1
    //  upper: B2 = B2 - U12T @ Y1
    let rhs = X.submatrix(0, 0, k, num_b_cols).to_owned();
    let lhs = AR.submatrix(
        subs.a21_or_a12.0,
        subs.a21_or_a12.1,
        subs.a21_or_a12.2,
        subs.a21_or_a12.3,
    );
    matmul::matmul(
        X.submatrix_mut(k, 0, num_b_rows - k, num_b_cols),
        Accum::Add,
        match side {
            Side::Lower => lhs,
            Side::Upper => lhs.transpose(),
        },
        rhs,
        T::from_f64_impl(-1.0),
        par,
    );

    // Step 3:
    //  lower: L22 @ Y2 = B2
    //  upper: U22T @ Y2 = B2
    triangular_solve::solve_lower_triangular_in_place(
        AR.submatrix(subs.a22.0, subs.a22.1, subs.a22.2, subs.a22.3)
            .transpose(),
        X.submatrix_mut(k, 0, num_b_rows - k, num_b_cols),
        par,
    );

    // Backward substitution
    //  lower: L.T @ X = Y
    //  upper: U.T @ Y = B

    // Step 1:
    //  lower: L22.T @ X2 = Y2
    //  upper: U22 @ X2 = Y2
    triangular_solve::solve_upper_triangular_in_place(
        AR.submatrix(subs.a22.0, subs.a22.1, subs.a22.2, subs.a22.3),
        X.submatrix_mut(k, 0, num_b_rows - k, num_b_cols),
        par,
    );

    // Step 2:
    //  lower: Y1 = Y1 - L21.T @ X2
    //  upper: Y1 = Y1 - U12 @ X2
    let rhs = X.submatrix(k, 0, num_b_rows - k, num_b_cols).to_owned();
    let lhs = AR.submatrix(
        subs.a21_or_a12.0,
        subs.a21_or_a12.1,
        subs.a21_or_a12.2,
        subs.a21_or_a12.3,
    );
    matmul::matmul(
        X.submatrix_mut(0, 0, k, num_b_cols),
        Accum::Add,
        match side {
            Side::Lower => lhs.transpose(),
            Side::Upper => lhs,
        },
        rhs,
        T::from_f64_impl(-1.0),
        par,
    );

    // Step 3:
    //  lower: L11.T @ X1 = Y1
    //  upper: U11 @ X1 = Y1
    triangular_solve::solve_upper_triangular_in_place(
        AR.submatrix(subs.a11.0, subs.a11.1, subs.a11.2, subs.a11.3)
            .transpose(),
        X.submatrix_mut(0, 0, k, num_b_cols),
        par,
    );

    X
}

/// Pack LOWER triangle, column-major.
/// Returns a Mat of shape (n_tp, 1) where n_tp = n(n+1)/2.
pub fn pack_tril_colmajor<T: ComplexField>(a: MatRef<'_, T>) -> Mat<T> {
    let (m, n) = a.shape();
    assert!(m == n, "square matrix required");
    let n = n;
    let n_tp = n * (n + 1) / 2;

    let mut out = Mat::zeros(n_tp, 1);
    let mut off = 0usize;

    for j in 0..n {
        // in col j, rows i=j..n-1
        let col = a.col(j);
        let run = col.as_ref().get(j..n); // length = n - j
        out.col_mut(0).get_mut(off..off + (n - j)).copy_from(run);
        off += n - j;
    }
    out
}

/// Unpack LOWER packed column vector (shape n_tp×1) into n×n.
pub fn unpack_tril_colmajor<T: ComplexField>(p: MatRef<'_, T>, n: usize) -> Mat<T> {
    let (n_tp, one) = p.shape();
    assert!(one == 1, "packed must be column vector");
    assert!(n * (n + 1) / 2 == n_tp, "invalid packed length for n");

    let mut a = Mat::zeros(n, n);
    let mut off = 0usize;

    for j in 0..n {
        // write col j: rows j..n-1 from packed segment
        let len = n - j;
        a.col_mut(j)
            .get_mut(j..n)
            .copy_from(p.col(0).get(off..off + len));
        off += len;
    }

    a
}

#[allow(non_snake_case)]
pub struct Lblt<T> {
    L: Mat<T>,
    B_diag: Diag<T>,
    B_subdiag: Diag<T>,
    P: Perm<usize>,
}

#[allow(non_snake_case)]
impl<T: ComplexField> Lblt<T> {
    /// Returns the Bunch-Kaufman factorization of the input A.
    ///
    /// The A is interpreted as Hermitian, but only the provided side is accessed.
    pub fn new<C: Conjugate<Canonical = T>>(A: MatRef<'_, C>, side: Side) -> Self {
        assert!(A.nrows() == A.ncols());

        let n = A.nrows();

        let mut L = Mat::zeros(n, n);

        match side {
            Side::Lower => L.copy_from_triangular_lower(A),
            Side::Upper => L.copy_from_triangular_lower(A.adjoint()),
        }

        Self::new_imp(L)
    }

    fn new_imp(mut L: Mat<T>) -> Self {
        let par = faer::get_global_parallelism();

        let n = L.nrows();

        let mut diag = Diag::zeros(n);
        let mut subdiag = Diag::zeros(n);
        let mut perm_fwd = vec![0usize; n];
        let mut perm_bwd = vec![0usize; n];

        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::lblt::factor::cholesky_in_place_scratch::<usize, T>(
                n,
                par,
                default(),
            ),
        );

        let stack = MemStack::new(&mut mem);

        faer::linalg::cholesky::lblt::factor::cholesky_in_place(
            L.as_mut(),
            subdiag.as_mut(),
            &mut perm_fwd,
            &mut perm_bwd,
            par,
            stack,
            default(),
        );

        diag.copy_from(L.diagonal());
        L.diagonal_mut().fill(one());

        Self {
            L: pack_tril_colmajor(L.as_ref()),
            B_diag: diag,
            B_subdiag: subdiag,
            P: unsafe {
                Perm::new_unchecked(perm_fwd.into_boxed_slice(), perm_bwd.into_boxed_slice())
            },
        }
    }

    pub fn solve(&self, rhs: &Mat<T>) -> Mat<T> {
        let mut rhs = rhs.cloned();
        self.solve_in_place_with_conj_impl(rhs.as_mat_mut(), Conj::No);
        rhs
    }

    fn solve_in_place_with_conj_impl(&self, rhs: MatMut<'_, T>, conj: Conj) {
        let par = faer::get_global_parallelism();
        let rhs_nrows = rhs.nrows();

        let factors = unpack_tril_colmajor(self.L.as_ref(), rhs_nrows);

        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::lblt::solve::solve_in_place_scratch::<usize, T>(
                factors.nrows(),
                rhs.ncols(),
                par,
            ),
        );
        let stack = MemStack::new(&mut mem);

        faer::linalg::cholesky::lblt::solve::solve_in_place_with_conj(
            factors.as_ref(),
            self.B_diag.as_ref(),
            self.B_subdiag.as_ref(),
            conj,
            self.P.as_ref(),
            rhs,
            par,
            stack,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;
    use faer::{self, prelude::Solve, utils::approx::*};

    /// Deterministic SPD matrix: A = M M^T + alpha I.
    fn make_spd(n: usize, alpha: f64) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let x = (i as f64 + 1.0) * (j as f64 + 2.0);
                m[(i, j)] = (x.sin() + 2.0 * x.cos()) / (1.0 + (i + j + 1) as f64);
            }
        }
        let mut a = &m * m.transpose();
        // a += alpha I
        for i in 0..n {
            a[(i, i)] += alpha.max(1e-3);
        }
        a
    }

    #[test]
    fn to_rfp_shapes_even_and_odd_lower() {
        // even
        for n in [2usize, 4, 6, 8, 10, 12] {
            let a = make_spd(n, 1.0);
            let ar = to_rfp(&a.as_ref(), &Side::Lower);
            let (ldar, n1) = ar.shape();
            assert_eq!(n % 2, 0);
            assert_eq!(ldar, n + 1, "LDAR for even n should be N+1");
            assert_eq!(n1, (n + 1) / 2);
        }
        // odd
        for n in [1usize, 3, 5, 7, 9, 11] {
            let a = make_spd(n, 1.0);
            let ar = to_rfp(&a.as_ref(), &Side::Lower);
            let (ldar, n1) = ar.shape();
            assert_eq!(n % 2, 1);
            assert_eq!(ldar, n, "LDAR for odd n should be N");
            assert_eq!(n1, (n + 1) / 2);
        }
    }

    #[test]
    fn get_dims_from_rfp_consistency() {
        // even n = 6 -> LDAR = 7, n1 = (n+1)/2 = 3
        let (even, n2) = get_dims_from_rfp(&7, &3);
        assert!(even);
        assert_eq!(n2, 3);

        // odd n = 7 -> LDAR = 7, n1 = 4
        let (even2, n2b) = get_dims_from_rfp(&7, &4);
        assert!(!even2);
        assert_eq!(n2b, 3);
    }

    #[test]
    fn factor_and_solve_lower_even_matches_standard() {
        let n = 6usize;
        let a = make_spd(n, 1e-2);
        let b = Mat::<f64>::from_fn(n, 2, |i, j| (i + 1 + 3 * j) as f64 / (1.0 + i as f64));

        let side = Side::Lower;

        // RFP factor + solve (lower)
        let llt = LltRfp::<f64>::try_new(a.as_ref(), side).expect("LLᵀ should succeed for SPD");
        let x_rfp = llt.solve(&b);

        // Standard solve
        let chol = a.llt(side);
        let x_std = chol.unwrap().solve(&b);

        // faer-style comparator/tolerance
        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

        // Residual: A * X ≈ B
        assert!(&a * &x_rfp ~ b);

        // Solution agreement with the standard LL solve
        assert!(&x_rfp ~ &x_std);
    }

    #[test]
    fn factor_and_solve_lower_odd_matches_standard() {
        let n = 7usize;
        let a = make_spd(n, 1e-2);
        let b = Mat::<f64>::from_fn(n, 3, |i, j| ((i + j + 2) as f64).sin());

        let side = Side::Lower;

        let llt = LltRfp::<f64>::try_new(a.as_ref(), side).expect("LLᵀ should succeed for SPD");
        let x_rfp = llt.solve(&b);

        let chol = a.llt(side);
        let x_std = chol.unwrap().solve(&b);

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

        assert!(&a * &x_rfp ~ b);

        assert!(&x_rfp ~ &x_std);
    }

    #[test]
    fn factor_and_solve_upper_even_matches_standard() {
        let n = 6usize;
        let a = make_spd(n, 1e-2);
        let b = Mat::<f64>::from_fn(n, 2, |i, j| (i + 1 + 3 * j) as f64 / (1.0 + i as f64));

        let side = Side::Upper;

        let llt = LltRfp::<f64>::try_new(a.as_ref(), side).expect("LLᵀ should succeed for SPD");
        let x_rfp = llt.solve(&b);

        let chol = a.llt(side);
        let x_std = chol.unwrap().solve(&b);

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

        assert!(&a * &x_rfp ~ b);

        assert!(&x_rfp ~ &x_std);
    }

    #[test]
    fn factor_and_solve_upper_odd_matches_standard() {
        let n = 7usize;
        let a = make_spd(n, 1e-2);
        let b = Mat::<f64>::from_fn(n, 3, |i, j| ((i + j + 2) as f64).sin());

        let side = Side::Upper;

        let llt = LltRfp::<f64>::try_new(a.as_ref(), side).expect("LLᵀ should succeed for SPD");
        let x_rfp = llt.solve(&b);

        let chol = a.llt(side);
        let x_std = chol.unwrap().solve(&b);

        let approx_eq = CwiseMat(ApproxEq::eps() * 128.0 * (n as f64));

        assert!(&a * &x_rfp ~ b);

        assert!(&x_rfp ~ &x_std);
    }
}
