/////////////////////////////////////////////////////////////////////////////////////////////
//
// Projects user seed points onto the isosurface and maps them to lattice cells.
//
// Created on: 17 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Seed projection for surface-following extraction.
//!
//! User-provided seed points are allowed to be near the target isosurface rather than exactly on
//! it. This module moves seeds toward `f(x) = isovalue` using Newton steps along the local
//! gradient, clamps them to the extraction lattice, and returns the unique lattice cells that
//! should seed wavefront expansion.

use std::collections::HashSet;

use faer::{Mat, MatRef};

use crate::lattice::SampleLattice;

/// Projects seed points toward the target level-set and returns unique seed cell ijk values.
///
/// The input matrix must be `N x 3`. Seeds are first clamped to the lattice AABB and deduplicated
/// by their initial lattice cell. The remaining representatives are then iteratively projected
/// toward `f(x) = isovalue` using values and gradients supplied by `gradient_fn`.
pub(crate) fn get_unique_seed_point_ijks(
    seed_points: MatRef<f64>,
    gradient_fn: &mut dyn FnMut(MatRef<f64>) -> (Mat<f64>, Mat<f64>),
    lattice: &SampleLattice,
    isovalue: f64,
) -> HashSet<[i64; 3]> {
    const NITERS: usize = 30;
    const TOL: f64 = 0.01;
    const G2_MIN: f64 = 1.0e-20;

    assert_eq!(seed_points.ncols(), 3, "seed_points must be N x 3");

    let minc = lattice.extents.min_corner;
    let maxc = lattice.extents.max_corner;

    let clamp = |v: f64, lo: f64, hi: f64| v.max(lo).min(hi);

    let mut seen_seed_cells = HashSet::with_capacity(seed_points.nrows());
    let mut filtered_points = Vec::with_capacity(seed_points.nrows() * 3);

    // Clamp to lattice bounds and keep one seed representative per initial lattice cell.
    for i in 0..seed_points.nrows() {
        let p = [
            clamp(seed_points[(i, 0)], minc[0], maxc[0]),
            clamp(seed_points[(i, 1)], minc[1], maxc[1]),
            clamp(seed_points[(i, 2)], minc[2], maxc[2]),
        ];

        if seen_seed_cells.insert(lattice.world_to_ijk(p)) {
            filtered_points.extend_from_slice(&p);
        }
    }

    if filtered_points.is_empty() {
        return HashSet::new();
    }

    // Work on a mutable copy of the filtered input points.
    let mut x =
        MatRef::from_row_major_slice(filtered_points.as_slice(), filtered_points.len() / 3, 3)
            .to_owned();

    let mut active: Vec<usize> = (0..x.nrows()).collect();
    let mut active_points = Vec::with_capacity(active.len() * 3);

    // Repeatedly push each point toward the target level-set f(x) = isovalue.
    // Each iteration uses a Newton step along the local gradient direction.
    for _ in 0..NITERS {
        active_points.clear();
        active_points.reserve(active.len() * 3);
        for &i in &active {
            active_points.extend_from_slice(&[x[(i, 0)], x[(i, 1)], x[(i, 2)]]);
        }

        let (fx, g) = gradient_fn(MatRef::from_row_major_slice(
            active_points.as_slice(),
            active.len(),
            3,
        ));

        let mut any_ok = false;
        let mut next_active = Vec::with_capacity(active.len());
        for (active_idx, &i) in active.iter().enumerate() {
            let fxi = fx[(active_idx, 0)] - isovalue;
            if fxi.abs() < TOL {
                continue;
            }

            let gx = g[(active_idx, 0)];
            let gy = g[(active_idx, 1)];
            let gz = g[(active_idx, 2)];
            // Squared gradient norm: ||grad f||^2. Very small values are numerically unsafe.
            let g2 = gx * gx + gy * gy + gz * gz;

            if g2 >= G2_MIN {
                // Newton update for a level-set constraint:
                // x <- x - (f(x)-isovalue)/||grad f||^2 * grad f
                let scale = fxi / g2;
                x[(i, 0)] -= scale * gx;
                x[(i, 1)] -= scale * gy;
                x[(i, 2)] -= scale * gz;
                any_ok = true;
            }

            // Re-clamp after moving so projections never leave the lattice domain.
            x[(i, 0)] = clamp(x[(i, 0)], minc[0], maxc[0]);
            x[(i, 1)] = clamp(x[(i, 1)], minc[1], maxc[1]);
            x[(i, 2)] = clamp(x[(i, 2)], minc[2], maxc[2]);
            next_active.push(i);
        }

        active = next_active;

        if active.is_empty() || !any_ok {
            break;
        }
    }

    x.row_iter()
        .map(|row| lattice.world_to_ijk([row[0], row[1], row[2]]))
        .collect()
}

/// Evaluates isosurface values and central-difference gradients for a batch of points.
///
/// For each input point this samples the centre point plus positive and negative offsets along
/// each coordinate axis. The samples are batched into a single call to `isosurface_fn` so seed
/// projection can use gradient information even when the caller does not provide analytic
/// gradients.
pub(crate) fn central_difference_values_and_gradients<F>(
    points: MatRef<f64>,
    isosurface_fn: &mut F,
    lattice: &SampleLattice,
) -> (Mat<f64>, Mat<f64>)
where
    F: FnMut(MatRef<f64>) -> Mat<f64>,
{
    const SAMPLES_PER_POINT: usize = 7;
    let n = points.nrows();
    let h = central_difference_step(lattice);
    let mut targets = Vec::with_capacity(n * SAMPLES_PER_POINT * 3);

    for i in 0..n {
        let p = [points[(i, 0)], points[(i, 1)], points[(i, 2)]];
        targets.extend_from_slice(&p);
        for axis in 0..3 {
            let mut plus = p;
            plus[axis] += h;
            targets.extend_from_slice(&plus);

            let mut minus = p;
            minus[axis] -= h;
            targets.extend_from_slice(&minus);
        }
    }

    let sampled = isosurface_fn(MatRef::from_row_major_slice(
        targets.as_slice(),
        n * SAMPLES_PER_POINT,
        3,
    ));

    let values = Mat::from_fn(n, 1, |i, _| sampled[(i * SAMPLES_PER_POINT, 0)]);
    let gradients = Mat::from_fn(n, 3, |i, axis| {
        let base = i * SAMPLES_PER_POINT + 1 + axis * 2;
        (sampled[(base, 0)] - sampled[(base + 1, 0)]) / (2.0 * h)
    });

    (values, gradients)
}

/// Returns the finite-difference step length used for estimated seed projection gradients.
fn central_difference_step(lattice: &SampleLattice) -> f64 {
    lattice
        .spacing
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .abs()
        .max(1.0e-4)
        * 1.0e-4
}
