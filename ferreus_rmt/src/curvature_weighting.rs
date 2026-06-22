/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines curvature-weighted vertex clustering for regularised marching tetrahedra.
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Curvature-weighted placement of clustered surface vertices.
//!
//! This module follows Section 3.4 of Treece, Prager and Gee using the same
//! staged process as the paper:
//!
//! 1. For one intersected lattice edge `oa`, visit each calculation plane around
//!    that edge.
//! 2. Each calculation plane contains two neighbouring triangles:
//!       `o-a-b` and `o-a-c`.
//! 3. Use Equation (1) to estimate `theta_b` and `theta_c`.
//! 4. Use Equation (2) to compute the raw curvature angle:
//!       `alpha = |theta_b| + |theta_c|`.
//! 5. Estimate the local surface normal by adding the per-plane normal
//!    projections plus a unit vector along `oa`.
//! 6. Use Equation (3) to adjust `alpha` to `beta` for the plane orientation.
//! 7. Use Equation (4) to weight each surface intersection when forming the
//!    clustered vertex position.

use std::collections::HashMap;

use super::{
    constants::{EDGE_DELTAS, NEIGHBOUR_EDGE_PLANE_PAIRS, NEIGHBOUR_EDGE_PLANE_PHIS},
    geometry::Point,
    isosurface,
    lattice::SampleLattice,
};

const EPS: f64 = 1.0e-12;

/// Large finite replacement for `1 / tan(theta)` when `theta` is numerically zero.
const MAX_COT_THETA: f64 = 1.0e12;

/// Large finite replacement for the Equation (4) weight when `tan(beta / 2)`
/// is numerically zero.
const MAX_CURVATURE_WEIGHT: f64 = 1.0e12;

/// Estimates the Equation (4) curvature weight for one intersected lattice edge.
fn curvature_weight_for_edge(
    owner: [i64; 3],
    other: [i64; 3],
    edge_id: usize,
    evaluated: &HashMap<[i64; 3], f64>,
    lattice: &SampleLattice,
) -> Option<f64> {
    let do_ = *evaluated.get(&owner)?;
    let da = *evaluated.get(&other)?;

    if !do_.is_finite() || !da.is_finite() {
        return None;
    }

    let pairs = *NEIGHBOUR_EDGE_PLANE_PAIRS.get(edge_id)?;
    let phis = *NEIGHBOUR_EDGE_PLANE_PHIS.get(edge_id)?;

    if pairs.len() != phis.len() {
        return None;
    }

    if pairs.len() != 2 && pairs.len() != 3 {
        return None;
    }

    let o_world = lattice.ijk_to_world(owner);
    let a_world = lattice.ijk_to_world(other);

    let oa = a_world.sub(o_world);
    let oa_len = oa.norm();

    if oa_len <= EPS {
        return None;
    }

    let oa_hat = oa.unit()?;

    let mut plane_alphas = [0.0; 3];
    let mut plane_axis_dirs = [[0.0; 3]; 3];
    let mut plane_count = 0usize;

    let mut projection_sum = [0.0; 3];

    for (pair, phi_pair) in pairs.iter().zip(phis.iter()) {
        let mut triangle_perp_dirs = [[0.0; 3]; 2];
        let mut triangle_thetas = [0.0; 2];
        let mut triangle_cot_thetas = [0.0; 2];

        for side in 0..2 {
            let neighbour_edge_id = pair[side] as usize;
            let phi = phi_pair[side];

            let [di, dj, dk] = *EDGE_DELTAS.get(neighbour_edge_id)?;

            let neighbour = [
                owner[0] + di as i64,
                owner[1] + dj as i64,
                owner[2] + dk as i64,
            ];

            let db = *evaluated.get(&neighbour)?;
            if !db.is_finite() {
                return None;
            }

            let b_world = lattice.ijk_to_world(neighbour);
            let ob = b_world.sub(o_world);
            let ob_len = ob.norm();

            if ob_len <= EPS {
                return None;
            }

            let ob_perp_to_oa = ob.sub(oa_hat.scale(ob.dot(oa_hat)));
            let ob_perp_dir = ob_perp_to_oa.unit()?;

            // Equation (1).
            let denominator = (do_ - da) * ob_len;

            if denominator.abs() <= EPS {
                return None;
            }

            let ratio = ((do_ - db) * oa_len) / denominator;
            let divisor = ratio - phi.cos();

            let theta = if divisor.abs() <= EPS {
                if divisor.is_sign_negative() {
                    -std::f64::consts::FRAC_PI_2
                } else {
                    std::f64::consts::FRAC_PI_2
                }
            } else {
                (phi.sin() / divisor).atan()
            };

            let tan_theta = theta.tan();

            let cot_theta = if tan_theta.abs() <= EPS {
                MAX_COT_THETA.copysign(theta)
            } else {
                1.0 / tan_theta
            };

            triangle_perp_dirs[side] = ob_perp_dir;
            triangle_thetas[side] = theta;
            triangle_cot_thetas[side] = cot_theta;
        }

        // Equation (2).
        let alpha = triangle_thetas[0].abs() + triangle_thetas[1].abs();

        let plane_axis_dir = triangle_perp_dirs[0]
            .sub(triangle_perp_dirs[1])
            .unit()
            .unwrap_or(triangle_perp_dirs[0]);

        // Vector addition of the two projected normal contributions.
        let normal_projection = triangle_perp_dirs[0]
            .scale(triangle_cot_thetas[0])
            .add(triangle_perp_dirs[1].scale(triangle_cot_thetas[1]));

        plane_alphas[plane_count] = alpha;
        plane_axis_dirs[plane_count] = plane_axis_dir;
        plane_count += 1;

        projection_sum = projection_sum.add(normal_projection);
    }

    let projection_scale = if plane_count == 3 { 2.0 / 3.0 } else { 1.0 };

    let n_est = oa_hat.add(projection_sum.scale(projection_scale)).unit()?;

    let mut min_abs_tan_half_beta = f64::INFINITY;

    for plane_index in 0..plane_count {
        let alpha = plane_alphas[plane_index];
        let plane_axis_dir = plane_axis_dirs[plane_index].unit()?;

        let sin_gamma = n_est.dot(plane_axis_dir).abs().clamp(0.0, 1.0);
        let gamma = sin_gamma.asin();

        let cos_gamma = gamma.cos();
        let one_minus_cos_gamma_squared = 1.0 - cos_gamma * cos_gamma;

        let half_alpha = 0.5 * alpha;
        let sin_half_alpha = half_alpha.sin().abs();

        let beta = if sin_half_alpha <= EPS {
            0.0
        } else {
            let sin_half_alpha_squared = sin_half_alpha * sin_half_alpha;

            // Equation (3).
            let curvature_term = 1.0 / sin_half_alpha_squared - 1.0;

            if curvature_term < 0.0 {
                return None;
            }

            let inverse_tan_half_beta_squared = one_minus_cos_gamma_squared * curvature_term;

            if inverse_tan_half_beta_squared <= EPS {
                continue;
            }

            let tan_half_beta = 1.0 / inverse_tan_half_beta_squared.sqrt();

            2.0 * tan_half_beta.atan()
        };

        let abs_tan_half_beta = (0.5 * beta).tan().abs();

        min_abs_tan_half_beta = min_abs_tan_half_beta.min(abs_tan_half_beta);
    }

    if !min_abs_tan_half_beta.is_finite() {
        return None;
    }

    // Equation (4).
    if min_abs_tan_half_beta <= EPS {
        return Some(MAX_CURVATURE_WEIGHT);
    }

    Some((1.0 / min_abs_tan_half_beta).min(MAX_CURVATURE_WEIGHT))
}

/// Returns a curvature-weighted representative point for a cluster of edge intersections.
///
/// Each edge intersection contributes its world-space position weighted by the
/// Section 3.4 curvature estimate for that lattice edge. If the local stencil is
/// incomplete, degenerate, or numerically unsuitable, that edge falls back to
/// unit weight.
pub fn curvature_weighted_cluster_point(
    edge_endpoints: &[([i64; 3], [i64; 3])],
    evaluated: &HashMap<[i64; 3], f64>,
    lattice: &SampleLattice,
) -> Option<[f64; 3]> {
    let mut weighted_sum = [0.0; 3];
    let mut total_w = 0.0;

    for &(u, v) in edge_endpoints {
        let Some(p) = isosurface::edge_intersection_point(u, v, evaluated, lattice) else {
            continue;
        };

        let Some((owner, other, edge_id)) = isosurface::get_edge_owner(u, v) else {
            continue;
        };

        let w = curvature_weight_for_edge(owner, other, edge_id, evaluated, lattice).unwrap_or(1.0);

        weighted_sum = weighted_sum.add([p[0], p[1], p[2]].scale(w));
        total_w += w;
    }

    if total_w <= EPS {
        return None;
    }

    let inv_total_w = 1.0 / total_w;

    Some([
        weighted_sum[0] * inv_total_w,
        weighted_sum[1] * inv_total_w,
        weighted_sum[2] * inv_total_w,
    ])
}
