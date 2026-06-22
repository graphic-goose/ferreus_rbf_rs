/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides fast triangle-triangle intersection testing for detecting mesh self-intersections.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # moller
//! This module is a fairly direct port of the source code provided by Thomas Möller with [1] to
//! quickly test whether two triangles intersect. Since we're only interested in detecting
//! self-intersections within a mesh, the single-point and coplanar tests are omitted, as they
//! return false positives in this instance.
//!
//! # References
//! 1.  Möller, T. (1997). A Fast Triangle-Triangle Intersection Test. Journal of Graphics Tools,
//!     2(2), 25–30. [`https://doi.org/10.1080/10867651.1997.10487472`](https://doi.org/10.1080/10867651.1997.10487472)

use super::geometry::{Point, Triangle3};

const EPSILON: f64 = 1E-6;

// sort so that a<=b
fn sort_pair(mut a: f64, mut b: f64) -> (f64, f64) {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    (a, b)
}

fn isect(vv0: f64, vv1: f64, vv2: f64, d0: f64, d1: f64, d2: f64) -> (f64, f64) {
    let isect0 = vv0 + (vv1 - vv0) * d0 / (d0 - d1);
    let isect1 = vv0 + (vv2 - vv0) * d0 / (d0 - d2);
    (isect0, isect1)
}

fn compute_intervals(
    vv0: f64,
    vv1: f64,
    vv2: f64,
    d0: f64,
    d1: f64,
    d2: f64,
    d0d1: f64,
    d0d2: f64,
) -> Option<(f64, f64)> {
    if d0d1 > 0.0 {
        Some(isect(vv2, vv0, vv1, d2, d0, d1))
    } else if d0d2 > 0.0 {
        Some(isect(vv1, vv0, vv2, d1, d0, d2))
    } else if d1 * d2 > 0.0 || d0 != 0.0 {
        Some(isect(vv0, vv1, vv2, d0, d1, d2))
    } else if d1 != 0.0 {
        Some(isect(vv1, vv0, vv2, d1, d0, d2))
    } else if d2 != 0.0 {
        Some(isect(vv2, vv0, vv1, d2, d0, d1))
    } else {
        None
    }
}

fn dominant_axis(d: [f64; 3]) -> usize {
    let mut index = 0;
    let mut max = d[0].abs();
    if d[1].abs() > max {
        index = 1;
        max = d[1].abs();
    }
    if d[2].abs() > max {
        index = 2;
    }
    index
}

/// Returns true only for proper (non-coplanar, positive-length) intersections.
///
/// This excludes purely touching cases (single-point contact) and coplanar overlap,
/// which are usually not treated as mesh self-intersections.
pub fn tri_tri_intersect(t1: Triangle3, t2: Triangle3) -> bool {
    let [v0, v1, v2] = t1.vertices();
    let [u0, u1, u2] = t2.vertices();

    let e1 = v1.sub(v0);
    let e2 = v2.sub(v0);
    let n1 = e1.cross(e2);
    let d1 = -n1.dot(v0);

    let du0 = n1.dot(u0) + d1;
    let du1 = n1.dot(u1) + d1;
    let du2 = n1.dot(u2) + d1;

    let du0du1 = du0 * du1;
    let du0du2 = du0 * du2;
    if du0du1 > 0.0 && du0du2 > 0.0 {
        return false;
    }

    let e1 = u1.sub(u0);
    let e2 = u2.sub(u0);
    let n2 = e1.cross(e2);
    let d2 = -n2.dot(u0);

    let dv0 = n2.dot(v0) + d2;
    let dv1 = n2.dot(v1) + d2;
    let dv2 = n2.dot(v2) + d2;

    let dv0dv1 = dv0 * dv1;
    let dv0dv2 = dv0 * dv2;
    if dv0dv1 > 0.0 && dv0dv2 > 0.0 {
        return false;
    }

    // Parallel planes (including coplanar) are excluded for "proper" intersections.
    let dir = n1.cross(n2);
    if dir.dot(dir) <= EPSILON * EPSILON {
        return false;
    }

    let index = dominant_axis(dir);

    let vp0 = v0[index];
    let vp1 = v1[index];
    let vp2 = v2[index];
    let up0 = u0[index];
    let up1 = u1[index];
    let up2 = u2[index];

    let Some((mut isect10, mut isect11)) =
        compute_intervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2)
    else {
        return false;
    };
    let Some((mut isect20, mut isect21)) =
        compute_intervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2)
    else {
        return false;
    };

    (isect10, isect11) = sort_pair(isect10, isect11);
    (isect20, isect21) = sort_pair(isect20, isect21);

    let overlap_start = isect10.max(isect20);
    let overlap_end = isect11.min(isect21);
    overlap_end - overlap_start > EPSILON
}
