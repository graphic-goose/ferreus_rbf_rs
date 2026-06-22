/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines axis-aligned bounding box clipping helpers for triangle meshes.
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Axis-aligned bounding box clipping for extracted triangle meshes.
//!
//! This module clips triangle meshes against an [`AABB`] by successively
//! clipping each triangle polygon against the six box planes. Clipped polygons
//! are triangulated as fans, and newly created vertices are snapped back onto
//! nearby box boundaries to reduce downstream numerical noise.

/// Axis-aligned bounding box with three-dimensional minimum and maximum corners.
#[derive(Clone, Copy, Debug)]
pub struct AABB<T> {
    /// Minimum x, y, and z coordinates.
    pub min_corner: [T; 3],

    /// Maximum x, y, and z coordinates.
    pub max_corner: [T; 3],
}

/// One of the six clipping planes that bound an axis-aligned box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoxPlane {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

/// Returns a scale-aware geometric tolerance for an AABB.
pub(crate) fn bbox_eps(extents: AABB<f64>) -> f64 {
    let d = [
        extents.max_corner[0] - extents.min_corner[0],
        extents.max_corner[1] - extents.min_corner[1],
        extents.max_corner[2] - extents.min_corner[2],
    ];
    let diag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    1.0e-10 * diag.max(1.0)
}

/// Clips a triangle mesh to `extents`.
///
/// The input mesh is represented by a flat row-major vertex array and a flat
/// triangle-index array. The returned mesh uses newly generated vertices and
/// facets, because clipping may split triangles along box planes.
pub(crate) fn clip_mesh_to_aabb(
    vertices: Vec<f64>,
    facets: Vec<usize>,
    extents: AABB<f64>,
    eps: f64,
) -> (Vec<f64>, Vec<usize>) {
    let planes = [
        BoxPlane::XMin,
        BoxPlane::XMax,
        BoxPlane::YMin,
        BoxPlane::YMax,
        BoxPlane::ZMin,
        BoxPlane::ZMax,
    ];
    let mut clipped_vertices = Vec::new();
    let mut clipped_facets = Vec::new();

    for tri in facets.chunks_exact(3) {
        let mut polygon = tri
            .iter()
            .map(|&vid| {
                [
                    vertices[3 * vid],
                    vertices[3 * vid + 1],
                    vertices[3 * vid + 2],
                ]
            })
            .collect::<Vec<_>>();

        for plane in planes {
            polygon = clip_polygon_to_plane(&polygon, plane, extents, eps);
            if polygon.len() < 3 {
                break;
            }
        }

        if polygon.len() < 3 {
            continue;
        }

        let base = push_vertex(&mut clipped_vertices, polygon[0]);
        let mut prev = push_vertex(&mut clipped_vertices, polygon[1]);
        for &p in &polygon[2..] {
            let next = push_vertex(&mut clipped_vertices, p);
            clipped_facets.extend_from_slice(&[base, prev, next]);
            prev = next;
        }
    }

    (clipped_vertices, clipped_facets)
}

/// Returns `true` if all vertices of a facet lie inside `extents`.
pub(crate) fn facet_fully_inside_aabb(
    vertices: &[f64],
    facets: &[usize],
    tri_idx: usize,
    extents: AABB<f64>,
    eps: f64,
) -> bool {
    let base = tri_idx * 3;
    if base + 2 >= facets.len() {
        return false;
    }

    (0..3).all(|i| {
        let vid = facets[base + i];
        let p = [
            vertices[3 * vid],
            vertices[3 * vid + 1],
            vertices[3 * vid + 2],
        ];
        point_inside_aabb(p, extents, eps)
    })
}

/// Linearly interpolates between two points.
fn interpolate_points(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Appends `p` to a flat row-major vertex buffer and returns its vertex id.
fn push_vertex(vertices: &mut Vec<f64>, p: [f64; 3]) -> usize {
    let vid = vertices.len() / 3;
    vertices.extend_from_slice(p.as_slice());
    vid
}

/// Snaps coordinates that are within `eps` of a box boundary exactly onto it.
fn snap_near_bbox(mut p: [f64; 3], extents: AABB<f64>, eps: f64) -> [f64; 3] {
    if (p[0] - extents.min_corner[0]).abs() <= eps {
        p[0] = extents.min_corner[0];
    }
    if (p[0] - extents.max_corner[0]).abs() <= eps {
        p[0] = extents.max_corner[0];
    }
    if (p[1] - extents.min_corner[1]).abs() <= eps {
        p[1] = extents.min_corner[1];
    }
    if (p[1] - extents.max_corner[1]).abs() <= eps {
        p[1] = extents.max_corner[1];
    }
    if (p[2] - extents.min_corner[2]).abs() <= eps {
        p[2] = extents.min_corner[2];
    }
    if (p[2] - extents.max_corner[2]).abs() <= eps {
        p[2] = extents.max_corner[2];
    }
    p
}

/// Snaps the coordinate constrained by `plane` exactly onto that plane.
fn snap_to_plane(mut p: [f64; 3], plane: BoxPlane, extents: AABB<f64>) -> [f64; 3] {
    match plane {
        BoxPlane::XMin => p[0] = extents.min_corner[0],
        BoxPlane::XMax => p[0] = extents.max_corner[0],
        BoxPlane::YMin => p[1] = extents.min_corner[1],
        BoxPlane::YMax => p[1] = extents.max_corner[1],
        BoxPlane::ZMin => p[2] = extents.min_corner[2],
        BoxPlane::ZMax => p[2] = extents.max_corner[2],
    }
    p
}

/// Returns the interpolation weighting where segment `a`-`b` crosses `plane`.
///
/// Returns `None` when the endpoints are on the same side of the plane.
fn segment_plane_t(
    a: [f64; 3],
    b: [f64; 3],
    plane: BoxPlane,
    extents: AABB<f64>,
    eps: f64,
) -> Option<f64> {
    let (aa, bb, coord) = match plane {
        BoxPlane::XMin => (a[0], b[0], extents.min_corner[0]),
        BoxPlane::XMax => (a[0], b[0], extents.max_corner[0]),
        BoxPlane::YMin => (a[1], b[1], extents.min_corner[1]),
        BoxPlane::YMax => (a[1], b[1], extents.max_corner[1]),
        BoxPlane::ZMin => (a[2], b[2], extents.min_corner[2]),
        BoxPlane::ZMax => (a[2], b[2], extents.max_corner[2]),
    };
    let da = aa - coord;
    let db = bb - coord;
    if da.abs() <= eps {
        return Some(0.0);
    }
    if db.abs() <= eps {
        return Some(1.0);
    }
    if (da < 0.0) == (db < 0.0) {
        return None;
    }
    Some((coord - aa) / (bb - aa))
}

/// Returns `true` if `p` is on the inside half-space of `plane`.
fn point_inside_plane(p: [f64; 3], plane: BoxPlane, extents: AABB<f64>, eps: f64) -> bool {
    match plane {
        BoxPlane::XMin => p[0] >= extents.min_corner[0] - eps,
        BoxPlane::XMax => p[0] <= extents.max_corner[0] + eps,
        BoxPlane::YMin => p[1] >= extents.min_corner[1] - eps,
        BoxPlane::YMax => p[1] <= extents.max_corner[1] + eps,
        BoxPlane::ZMin => p[2] >= extents.min_corner[2] - eps,
        BoxPlane::ZMax => p[2] <= extents.max_corner[2] + eps,
    }
}

/// Returns `true` if `p` lies inside or on the AABB, with tolerance `eps`.
fn point_inside_aabb(p: [f64; 3], extents: AABB<f64>, eps: f64) -> bool {
    p[0] >= extents.min_corner[0] - eps
        && p[0] <= extents.max_corner[0] + eps
        && p[1] >= extents.min_corner[1] - eps
        && p[1] <= extents.max_corner[1] + eps
        && p[2] >= extents.min_corner[2] - eps
        && p[2] <= extents.max_corner[2] + eps
}

/// Clips a convex polygon against one AABB plane using Sutherland-Hodgman.
fn clip_polygon_to_plane(
    polygon: &[[f64; 3]],
    plane: BoxPlane,
    extents: AABB<f64>,
    eps: f64,
) -> Vec<[f64; 3]> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut clipped = Vec::new();
    let mut prev = *polygon.last().unwrap();
    let mut prev_inside = point_inside_plane(prev, plane, extents, eps);

    for &curr in polygon {
        let curr_inside = point_inside_plane(curr, plane, extents, eps);

        if curr_inside != prev_inside {
            if let Some(t) = segment_plane_t(prev, curr, plane, extents, eps) {
                clipped.push(snap_near_bbox(
                    snap_to_plane(interpolate_points(prev, curr, t), plane, extents),
                    extents,
                    eps,
                ));
            }
        }

        if curr_inside {
            clipped.push(snap_near_bbox(curr, extents, eps));
        }

        prev = curr;
        prev_inside = curr_inside;
    }

    clipped
}
