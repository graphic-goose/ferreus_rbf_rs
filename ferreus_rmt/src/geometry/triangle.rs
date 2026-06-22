/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines generic and three-dimensional triangle geometry helpers.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Triangle primitives and geometric predicates.
//!
//! This module provides a lightweight generic [`Triangle`] container and
//! additional operations for triangles whose vertices are `[f64; 3]` points.

use super::Point;
use crate::moller;

/// Triangle with vertices `a`, `b`, and `c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Triangle<T> {
    /// First vertex.
    pub a: T,

    /// Second vertex.
    pub b: T,

    /// Third vertex.
    pub c: T,
}

impl<T> Triangle<T> {
    /// Creates a triangle from three vertices.
    pub const fn new(a: T, b: T, c: T) -> Self {
        Self { a, b, c }
    }
}

impl<T: Copy> Triangle<T> {
    /// Returns the vertices in stored order.
    pub const fn vertices(self) -> [T; 3] {
        [self.a, self.b, self.c]
    }

    /// Returns the oriented edges `(a, b)`, `(b, c)`, and `(c, a)`.
    pub const fn edges(self) -> [(T, T); 3] {
        [(self.a, self.b), (self.b, self.c), (self.c, self.a)]
    }

    /// Returns the same triangle with the opposite winding.
    pub const fn reversed(self) -> Self {
        Self {
            a: self.a,
            b: self.c,
            c: self.b,
        }
    }

    /// Applies `f` to each vertex and returns a triangle of mapped vertices.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Triangle<U> {
        Triangle {
            a: f(self.a),
            b: f(self.b),
            c: f(self.c),
        }
    }
}

impl<T: Copy + Eq> Triangle<T> {
    /// Returns `true` if `(a, b)` is one of the triangle's directed edges.
    pub fn contains_oriented_edge(self, a: T, b: T) -> bool {
        self.edges().iter().any(|&(x, y)| x == a && y == b)
    }
}

impl<T: Copy + Ord> Triangle<T> {
    /// Returns the vertices sorted in ascending order.
    pub fn sorted_vertices(self) -> [T; 3] {
        let mut ids = self.vertices();
        ids.sort_unstable();
        ids
    }
}

/// Triangle with three-dimensional floating-point vertices.
pub type Triangle3 = Triangle<[f64; 3]>;

impl Triangle<[f64; 3]> {
    /// Returns the unnormalised face normal.
    ///
    /// The length of this vector is twice the triangle area, and its direction
    /// follows the triangle winding.
    pub fn normal(self) -> [f64; 3] {
        self.b.sub(self.a).cross(self.c.sub(self.a))
    }

    /// Returns the unit face normal, or `None` for a degenerate triangle.
    pub fn unit_normal(self) -> Option<[f64; 3]> {
        self.normal().unit()
    }

    /// Returns twice the triangle area.
    pub fn area2(self) -> f64 {
        self.normal().norm()
    }

    /// Returns `true` when twice the area is no larger than `tolerance^2`.
    pub fn is_degenerate(self, tolerance: f64) -> bool {
        self.area2() <= tolerance * tolerance
    }

    /// Returns the minimum and maximum corners of the triangle's axis-aligned bounding box.
    pub fn aabb_corners(self) -> ([f64; 3], [f64; 3]) {
        let mut min_corner = [f64::INFINITY; 3];
        let mut max_corner = [f64::NEG_INFINITY; 3];
        for p in self.vertices() {
            for axis in 0..3 {
                min_corner[axis] = min_corner[axis].min(p[axis]);
                max_corner[axis] = max_corner[axis].max(p[axis]);
            }
        }
        (min_corner, max_corner)
    }

    /// Returns the signed distance from `point` to the triangle plane.
    ///
    /// Returns `None` if the triangle is degenerate and has no reliable plane.
    pub fn plane_distance(self, point: [f64; 3]) -> Option<f64> {
        self.unit_normal().map(|n| point.sub(self.a).dot(n))
    }

    /// Returns the largest absolute plane distance from this triangle to `other`'s vertices.
    ///
    /// Degenerate `self` is treated as infinitely far from `other`.
    pub fn max_plane_distance_to(self, other: Triangle3) -> f64 {
        other
            .vertices()
            .iter()
            .map(|p| self.plane_distance(*p).map_or(f64::INFINITY, f64::abs))
            .fold(0.0, f64::max)
    }

    /// Returns `true` if `point` lies strictly inside the triangle interior.
    ///
    /// Points outside the triangle plane by more than `tolerance`, and points on
    /// edges or vertices, return `false`.
    pub fn point_in_interior(self, point: [f64; 3], tolerance: f64) -> bool {
        let Some(n_hat) = self.unit_normal() else {
            return false;
        };
        if point.sub(self.a).dot(n_hat).abs() > tolerance {
            return false;
        }

        let c0 = self.b.sub(self.a).cross(point.sub(self.a)).dot(n_hat);
        let c1 = self.c.sub(self.b).cross(point.sub(self.b)).dot(n_hat);
        let c2 = self.a.sub(self.c).cross(point.sub(self.c)).dot(n_hat);
        let area_tol = tolerance * tolerance;
        (c0 > area_tol && c1 > area_tol && c2 > area_tol)
            || (c0 < -area_tol && c1 < -area_tol && c2 < -area_tol)
    }

    /// Returns `true` if the segment crosses the strict triangle interior.
    ///
    /// Segments touching only the triangle plane, edges, vertices, or endpoints
    /// are excluded.
    pub fn segment_pierces_interior(self, p0: [f64; 3], p1: [f64; 3], tolerance: f64) -> bool {
        let Some(n_hat) = self.unit_normal() else {
            return false;
        };
        let d0 = p0.sub(self.a).dot(n_hat);
        let d1 = p1.sub(self.a).dot(n_hat);
        if d0.abs() <= tolerance || d1.abs() <= tolerance || d0 * d1 >= 0.0 {
            return false;
        }

        let t = d0 / (d0 - d1);
        if t <= tolerance || t >= 1.0 - tolerance {
            return false;
        }

        self.point_in_interior(p0.lerp(p1, t), tolerance)
    }

    /// Returns `true` when this triangle intersects `other`.
    pub fn intersects_triangle(self, other: Triangle3) -> bool {
        moller::tri_tri_intersect(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_triangle_vertices_and_edges_preserve_order() {
        let tri = Triangle::new(3, 1, 2);
        assert_eq!(tri.vertices(), [3, 1, 2]);
        assert_eq!(tri.edges(), [(3, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn generic_triangle_reverses_orientation() {
        let tri = Triangle::new(3, 1, 2);
        assert_eq!(tri.reversed().vertices(), [3, 2, 1]);
    }

    #[test]
    fn generic_triangle_contains_only_directed_edges() {
        let tri = Triangle::new(3, 1, 2);
        assert!(tri.contains_oriented_edge(3, 1));
        assert!(tri.contains_oriented_edge(1, 2));
        assert!(tri.contains_oriented_edge(2, 3));
        assert!(!tri.contains_oriented_edge(1, 3));
        assert!(!tri.contains_oriented_edge(2, 1));
    }

    #[test]
    fn generic_triangle_sorts_vertex_ids() {
        let tri = Triangle::new(3, 1, 2);
        assert_eq!(tri.sorted_vertices(), [1, 2, 3]);
    }

    #[test]
    fn normal_area_and_degenerate_behaviour() {
        let tri = Triangle3::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
        assert_eq!(tri.normal(), [0.0, 0.0, 6.0]);
        assert_eq!(tri.area2(), 6.0);
        assert!(!tri.is_degenerate(1.0e-8));

        let degenerate = Triangle3::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!(degenerate.is_degenerate(1.0e-8));
    }

    #[test]
    fn aabb_corners_enclose_triangle() {
        let tri = Triangle3::new([1.0, -1.0, 3.0], [-2.0, 4.0, 0.5], [0.0, 2.0, -7.0]);
        assert_eq!(tri.aabb_corners(), ([-2.0, -1.0, -7.0], [1.0, 4.0, 3.0]));
    }

    #[test]
    fn point_in_interior_excludes_edges() {
        let tri = Triangle3::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(tri.point_in_interior([0.25, 0.25, 0.0], 1.0e-8));
        assert!(!tri.point_in_interior([0.5, 0.0, 0.0], 1.0e-8));
        assert!(!tri.point_in_interior([0.0, 0.0, 0.0], 1.0e-8));
        assert!(!tri.point_in_interior([0.25, 0.25, 1.0e-6], 1.0e-8));
    }

    #[test]
    fn segment_piercing_excludes_endpoints_and_touches() {
        let tri = Triangle3::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(tri.segment_pierces_interior([0.25, 0.25, -1.0], [0.25, 0.25, 1.0], 1.0e-8));
        assert!(!tri.segment_pierces_interior([0.25, 0.25, 0.0], [0.25, 0.25, 1.0], 1.0e-8));
        assert!(!tri.segment_pierces_interior([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0e-8));
    }
}
