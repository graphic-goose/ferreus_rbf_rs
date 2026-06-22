/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides fast tests for detecting mesh self-intersections.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # mesh_intersections
//! This module identifies triangles involved in true mesh self-intersections.
//!
//! The detector uses an R-tree over triangle axis-aligned bounding boxes to find candidate
//! triangle pairs, then applies exact geometric tests to separate true crossings from valid
//! mesh adjacency. Shared edges, shared vertices, duplicate geometric vertices, near-coplanar
//! contact, and degenerate triangles are filtered out so that only triangles requiring cleanup
//! or diagnostics are returned.

use std::collections::HashSet;

use super::geometry::{Point, Triangle3};
use faer::{MatRef, RowRef};
use rstar::{
    RTree, RTreeObject,
    primitives::{GeomWithData, Rectangle},
};

/// Geometric tolerance used when classifying adjacency and near-coplanar contact.
pub const DEFAULT_INTERSECTION_TOLERANCE: f64 = 1.0e-8;

/// Converts a `faer` row reference into a fixed-size 3D point.
#[inline(always)]
fn row_to_point(row: RowRef<f64>) -> [f64; 3] {
    [row[0], row[1], row[2]]
}

/// Converts a facet row into its three vertex ids.
#[inline(always)]
fn facet_ids(row: RowRef<usize>) -> [usize; 3] {
    [row[0], row[1], row[2]]
}

/// Builds a geometric triangle from the vertex matrix and one facet row.
#[inline(always)]
fn triangle_points(vertices: MatRef<f64>, facet: RowRef<usize>) -> Triangle3 {
    let ids = facet_ids(facet);
    Triangle3::new(
        row_to_point(vertices.row(ids[0])),
        row_to_point(vertices.row(ids[1])),
        row_to_point(vertices.row(ids[2])),
    )
}

/// Counts exactly shared vertex ids between two facets.
#[inline(always)]
fn shared_vertex_count(a: &[usize; 3], b: &[usize; 3]) -> usize {
    a.iter().filter(|id| b.contains(id)).count()
}

/// Counts geometrically coincident vertices between two triangles.
///
/// This catches adjacency that is present geometrically but not represented by shared vertex ids,
/// which can occur before vertex deduplication or when comparing independently generated patches.
fn geometric_shared_vertex_count(a: Triangle3, b: Triangle3, tolerance: f64) -> usize {
    let mut used_b = [false; 3];
    let mut count = 0;
    for pa in a.vertices() {
        for (idx, pb) in b.vertices().iter().enumerate() {
            if !used_b[idx] && pa.close_to(*pb, tolerance) {
                used_b[idx] = true;
                count += 1;
                break;
            }
        }
    }
    count
}

/// Returns the bounding box used for broad-phase R-tree queries.
#[inline]
fn triangle_aabb(tri: Triangle3) -> Rectangle<[f64; 3]> {
    let (min_corner, max_corner) = tri.aabb_corners();
    Rectangle::from_corners(min_corner, max_corner)
}

/// Returns `true` when either triangle lies close enough to the other's plane to treat the pair
/// as near-coplanar contact rather than a true piercing intersection.
#[inline]
fn near_coplanar(a: Triangle3, b: Triangle3, tolerance: f64) -> bool {
    a.max_plane_distance_to(b).min(b.max_plane_distance_to(a)) <= tolerance
}

/// Returns the edge opposite `vertex_idx` in `tri`.
fn opposite_edge(tri: Triangle3, vertex_idx: usize) -> ([f64; 3], [f64; 3]) {
    tri.edges()[(vertex_idx + 1) % 3]
}

/// Tests whether two triangles sharing one vertex also cross through each other's interior.
///
/// A shared vertex alone is valid mesh adjacency. The pair is only a true self-intersection if
/// the edge opposite the shared vertex pierces the interior of the other triangle.
fn shared_vertex_extra_crossing(a_tri: Triangle3, b_tri: Triangle3, tolerance: f64) -> bool {
    let av = a_tri.vertices();
    let bv = b_tri.vertices();
    for i in 0..3 {
        for j in 0..3 {
            if !av[i].close_to(bv[j], tolerance) {
                continue;
            }
            let (a0, a1) = opposite_edge(a_tri, i);
            let (b0, b1) = opposite_edge(b_tri, j);
            return b_tri.segment_pierces_interior(a0, a1, tolerance)
                || a_tri.segment_pierces_interior(b0, b1, tolerance);
        }
    }
    false
}

/// Classifies a candidate triangle pair after their bounding boxes overlap.
///
/// The broad triangle intersection test reports contact as well as crossings. This function
/// rejects valid adjacency and contact-only cases before returning `true` for a genuine
/// self-intersection.
fn is_true_self_intersection(
    a_ids: &[usize; 3],
    b_ids: &[usize; 3],
    a_tri: Triangle3,
    b_tri: Triangle3,
    tolerance: f64,
) -> bool {
    if a_tri.is_degenerate(tolerance) || b_tri.is_degenerate(tolerance) {
        return false;
    }

    let shared = shared_vertex_count(a_ids, b_ids);
    if shared >= 2 {
        return false;
    }

    let intersects = a_tri.intersects_triangle(b_tri);
    if !intersects {
        return false;
    }

    if shared == 1 {
        return shared_vertex_extra_crossing(a_tri, b_tri, tolerance);
    }

    let geometric_shared = geometric_shared_vertex_count(a_tri, b_tri, tolerance);
    if geometric_shared >= 2 {
        return false;
    }
    if geometric_shared == 1 {
        return shared_vertex_extra_crossing(a_tri, b_tri, tolerance);
    }

    !near_coplanar(a_tri, b_tri, tolerance)
}

/// Given the vertices and facets of a mesh, returns triangle ids involved in true
/// self-intersections. Valid adjacency and contact-only cases are not returned.
pub fn get_intersecting_triangles(vertices: MatRef<f64>, facets: MatRef<usize>) -> Vec<usize> {
    let (nfacets, ncols) = facets.shape();
    if ncols != 3 || nfacets == 0 {
        return Vec::new();
    }

    let rectangles = facets
        .row_iter()
        .enumerate()
        .map(|(idx, f)| GeomWithData::new(triangle_aabb(triangle_points(vertices, f)), idx))
        .collect();
    let tree = RTree::bulk_load(rectangles);
    let mut ids = HashSet::new();

    for item in tree.iter() {
        let a_idx = item.data;
        let a_facet = facets.row(a_idx);
        let a_ids = facet_ids(a_facet);
        let a_tri = triangle_points(vertices, a_facet);

        for other in tree.locate_in_envelope_intersecting(&item.envelope()) {
            let b_idx = other.data;
            if b_idx <= a_idx {
                continue;
            }

            let b_facet = facets.row(b_idx);
            let b_ids = facet_ids(b_facet);
            let b_tri = triangle_points(vertices, b_facet);
            if is_true_self_intersection(
                &a_ids,
                &b_ids,
                a_tri,
                b_tri,
                DEFAULT_INTERSECTION_TOLERANCE,
            ) {
                ids.insert(a_idx);
                ids.insert(b_idx);
            }
        }
    }

    let mut ids: Vec<_> = ids.into_iter().collect();
    ids.sort_unstable();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intersecting_triangles(vertices: &[[f64; 3]], facets: &[[usize; 3]]) -> Vec<usize> {
        let verts: Vec<f64> = vertices.iter().flat_map(|p| *p).collect();
        let faces: Vec<usize> = facets.iter().flat_map(|f| *f).collect();
        get_intersecting_triangles(
            MatRef::from_row_major_slice(&verts, vertices.len(), 3),
            MatRef::from_row_major_slice(&faces, facets.len(), 3),
        )
    }

    fn assert_intersections(vertices: &[[f64; 3]], facets: &[[usize; 3]], expected: &[usize]) {
        assert_eq!(intersecting_triangles(vertices, facets), expected);
    }

    #[test]
    fn shared_vertex_only_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 3, 4]], &[]);
    }

    #[test]
    fn shared_vertex_plus_interior_crossing_returns_both_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.75, -0.25, -1.0],
            [-0.25, 0.75, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 3, 4]], &[0, 1]);
    }

    #[test]
    fn shared_edge_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 1, 3]], &[]);
    }

    #[test]
    fn coplanar_shared_edge_quad_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 2, 3]], &[]);
    }

    #[test]
    fn coplanar_shared_vertex_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 3, 4]], &[]);
    }

    #[test]
    fn nearly_coplanar_shared_edge_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0e-10],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 1, 3]], &[]);
    }

    #[test]
    fn geometric_shared_edge_with_distinct_indices_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[]);
    }

    #[test]
    fn geometric_shared_vertex_plus_interior_crossing_returns_both_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.75, -0.25, -1.0],
            [-0.25, 0.75, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[0, 1]);
    }

    #[test]
    fn shared_edge_plus_positive_area_overlap_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [0, 1, 3]], &[]);
    }

    #[test]
    fn fan_around_shared_vertex_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert_intersections(
            &vertices,
            &[[0, 1, 5], [0, 2, 5], [0, 3, 5], [0, 4, 5]],
            &[],
        );
    }

    #[test]
    fn overlapping_aabbs_without_intersection_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.2],
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[]);
    }

    #[test]
    fn degenerate_triangle_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[]);
    }

    #[test]
    fn piercing_triangle_returns_both_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -1.0],
            [0.25, 0.25, 1.0],
            [0.75, 0.25, 0.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[0, 1]);
    }

    #[test]
    fn non_adjacent_triangles_crossing_segment_return_both_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, -1.0],
            [0.8, 0.2, 1.0],
            [0.2, 0.8, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[0, 1]);
    }

    #[test]
    fn edge_interior_crossing_returns_both_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, -0.2, -1.0],
            [0.5, 0.8, 1.0],
            [0.5, -0.2, 1.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[0, 1]);
    }

    #[test]
    fn non_adjacent_coplanar_positive_area_overlap_returns_no_ids() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.0],
            [1.25, 0.25, 0.0],
            [0.25, 1.25, 0.0],
        ];
        assert_intersections(&vertices, &[[0, 1, 2], [3, 4, 5]], &[]);
    }
}
