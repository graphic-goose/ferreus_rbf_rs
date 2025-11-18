/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements flexible GMRES and related iterative solvers for large RBF linear systems.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    interpolant_config::{FittingAccuracy, FittingAccuracyType},
    progress::{ProgressMsg, ProgressSink, progress_from_rel},
};
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::mat::AsMatRef;
use faer::{Mat, MatRef, Par};
use std::sync::Arc;

/// Flexible GMRES implementation using Saad's algorithm structure.
///
/// This function solves `Ax = b` using restarted FGMRES with optional right preconditioner `M`.
/// It supports restarts (outer iterations), inner Krylov basis iteration count,
/// and tracks convergence with optional residual output.
///
/// # Parameters
/// - `a`: Operator function A(x)
/// - `b`: Right-hand side vector
/// - `m`: Optional right preconditioner function M(x)
/// - `x0`: Optional initial guess
/// - `max_outer_iterations`: Maximum restart cycles
/// - `max_inner_iterations`: Krylov subspace size per restart
/// - `tolerance`: Stopping criterion
/// - `print_iters`: Print residual norms during iterations
///
/// # Returns
/// - `x`: Approximate solution
pub fn fgmres<A, M>(
    a: &A,
    b: MatRef<f64>,
    m: Option<&M>,
    x0: Option<&Mat<f64>>,
    max_outer_iterations: usize,
    max_inner_iterations: usize,
    tolerance: &FittingAccuracy,
    callback: Option<Arc<dyn ProgressSink>>,
) -> Mat<f64>
where
    A: Fn(&MatRef<f64>) -> Mat<f64>,
    M: Fn(&MatRef<f64>) -> Mat<f64>,
{
    let n = b.nrows();
    let mut x = x0.cloned().unwrap_or_else(|| Mat::zeros(n, 1));

    let mut r = b - &a(&x.as_ref());
    let beta = match tolerance.tolerance_type {
        FittingAccuracyType::Absolute => r.col(0).norm_max(),
        FittingAccuracyType::Relative => r.col(0).norm_l2(),
    };

    let mut iteration = 1usize;

    let mut v = Mat::<f64>::zeros(n, max_inner_iterations + 1);
    let mut h = Mat::<f64>::zeros(max_inner_iterations + 1, max_inner_iterations);
    let mut z = Mat::<f64>::zeros(n, max_inner_iterations);
    let mut g = Mat::<f64>::zeros(max_inner_iterations + 1, 1);
    let mut cs = Mat::<f64>::zeros(max_inner_iterations, 1);
    let mut sn = Mat::<f64>::zeros(max_inner_iterations, 1);

    for outer in 0..max_outer_iterations {
        if outer > 0 {
            v.fill(0.0);
            h.fill(0.0);
            z.fill(0.0);
            g.fill(0.0);
            cs.fill(0.0);
            sn.fill(0.0);
        }
        let r_norm = r.norm_l2();
        v.col_mut(0).copy_from(&(r.clone() / r_norm).col(0));
        g[(0, 0)] = r_norm;

        for j in 0..max_inner_iterations {
            // Apply preconditioner
            let vj = v.col(j);
            let w = match m {
                Some(mfun) => mfun(&vj.as_mat_ref().as_col_shape(1)),
                None => vj.as_mat_ref().as_col_shape(1).to_owned(),
            };
            z.col_mut(j).copy_from(&w.col(0));

            // Apply matvec operator
            let mut wj = a(&w.as_ref());

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let vi = v.col(i);
                let hij = vi
                    .iter()
                    .zip(wj.col(0).iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                h[(i, j)] = hij;
                wj -= &(vi.to_owned() * hij).as_mat();
            }

            let norm = wj.norm_l2();
            h[(j + 1, j)] = norm;

            // Apply previous Givens rotations
            if j > 0 {
                for i in 0..j {
                    let temp = cs[(i, 0)] * h[(i, j)] + sn[(i, 0)] * h[(i + 1, j)];
                    h[(i + 1, j)] = -sn[(i, 0)] * h[(i, j)] + cs[(i, 0)] * h[(i + 1, j)];
                    h[(i, j)] = temp;
                }
            }

            // Compute and apply new Givens rotation
            let (c, s, _r) = givens_rotation(h[(j, j)], h[(j + 1, j)]);

            h[(j, j)] = c * h[(j, j)] + s * h[(j + 1, j)];
            h[(j + 1, j)] = 0.0;

            let temp = c * g[(j, 0)] + s * g[(j + 1, 0)];
            g[(j + 1, 0)] = -s * g[(j, 0)] + c * g[(j + 1, 0)];
            g[(j, 0)] = temp;

            cs[(j, 0)] = c;
            sn[(j, 0)] = s;

            if norm != 0.0 {
                v.col_mut(j + 1).copy_from(&(wj / norm).col(0));
            }

            // Compute residual from rotated g
            let res_norm = match tolerance.tolerance_type {
                FittingAccuracyType::Absolute => g[(j + 1, 0)].abs(),
                FittingAccuracyType::Relative => g[(j + 1, 0)].abs() / beta,
            };

            if let Some(sink) = &callback {
                sink.emit(ProgressMsg::SolverIteration {
                    iter: iteration,
                    residual: res_norm,
                    progress: progress_from_rel(res_norm, beta, tolerance.tolerance),
                });
            }

            if res_norm < tolerance.tolerance {
                x += get_solution(&h, &g, &z, &(j + 1));
                return x;
            }

            iteration += 1;
        }

        // Restart update
        x += get_solution(&h, &g, &z, &max_inner_iterations);
        r = b - &a(&x.as_ref());

        let res_norm = match tolerance.tolerance_type {
            FittingAccuracyType::Absolute => r.norm_max(),
            FittingAccuracyType::Relative => r.norm_l2() / beta,
        };

        if res_norm < tolerance.tolerance {
            break;
        }
    }

    x
}

fn get_solution(ri: &Mat<f64>, gi: &Mat<f64>, z: &Mat<f64>, i: &usize) -> Mat<f64> {
    let gi = gi.subrows(0, *i);
    let hi = ri.submatrix(0, 0, *i, *i);

    let mut ym = gi.to_owned();
    solve_upper_triangular_in_place(hi, ym.as_mut(), Par::Seq);

    z.subcols(0, *i) * ym
}

/// Compute a Givens rotation: given scalars `f` and `g`,
/// returns (c, s, r) such that
///
///   [  c   s ] [ f ] = [ r ]
///   [ –s   c ] [ g ]   [ 0 ]
///
/// A port of LAPACK's dlartg.
pub fn givens_rotation(f: f64, g: f64) -> (f64, f64, f64) {
    // Safe minimum/maximum
    let safmin = f64::MIN_POSITIVE;
    let safmax = f64::MAX;

    // thresholds
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();

    // Trivial g=0  ->  no rotation
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    // Trivial f=0  ->  pure sine rotation
    if f == 0.0 {
        let s = g.signum();
        return (0.0, s, g.abs());
    }

    let f1 = f.abs();
    let g1 = g.abs();

    // No scaling needed
    if (rtmin..rtmax).contains(&f1) && (rtmin..rtmax).contains(&g1) {
        // r = sqrt(f^2+g^2) with sign of f
        let r = (f * f + g * g).sqrt().copysign(f);
        let c = f1 / r.abs();
        let s = g / r;
        (c, s, r)
    } else {
        // Scale to avoid under/overflow
        let u = f1.max(g1).clamp(safmin, safmax);
        let fs = f / u;
        let gs = g / u;
        let mag = (fs * fs + gs * gs).sqrt();
        let r = mag.copysign(f) * u;
        let c = fs.abs() / mag;
        let s = gs / mag;
        (c, s, r)
    }
}

pub fn schwarz_ddm_solver<A, M>(
    matvec: &A,
    rhs: MatRef<f64>,
    mut m: Option<&M>,
    max_interations: usize,
    tolerance: &FittingAccuracy,
    callback: Option<Arc<dyn ProgressSink>>,
) -> Mat<f64>
where
    A: Fn(&MatRef<f64>) -> Mat<f64>,
    M: Fn(&MatRef<f64>) -> Mat<f64>,
{
    let mut rg = rhs.clone().to_owned();

    let mut sg = Mat::<f64>::zeros(rhs.nrows(), rhs.ncols());

    let beta = match tolerance.tolerance_type {
        FittingAccuracyType::Absolute => rg.col(0).norm_max(),
        FittingAccuracyType::Relative => rg.col(0).norm_l2(),
    };

    let mut res_norm = beta.clone();

    let mut iteration = 0usize;

    if let Some(precon) = m.as_mut() {
        while res_norm > tolerance.tolerance && iteration < max_interations {
            sg += precon(&rg.col(0).as_mat_ref().as_col_shape(1));
            rg = rhs - matvec(&sg.as_ref());
            res_norm = match tolerance.tolerance_type {
                FittingAccuracyType::Absolute => rg.norm_max(),
                FittingAccuracyType::Relative => rg.norm_l2() / beta,
            };

            iteration += 1;

            if let Some(sink) = &callback {
                sink.emit(ProgressMsg::SolverIteration {
                    iter: iteration,
                    residual: res_norm,
                    progress: progress_from_rel(res_norm, beta, tolerance.tolerance),
                });
            }
        }
    }

    sg
}
