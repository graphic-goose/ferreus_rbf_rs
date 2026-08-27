/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines vector-style operations for three-dimensional points.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Three-dimensional point and vector operations.
//!
//! The [`Point`] trait provides a compact set of arithmetic and geometric
//! helpers for `[f64; 3]` values used throughout the RMT implementation.

/// Vector-style operations for three-dimensional points.
pub trait Point {
    /// Returns the component-wise sum of two points or vectors.
    fn add(self, other: Self) -> Self;

    /// Returns the component-wise difference of two points or vectors.
    fn sub(self, other: Self) -> Self;

    /// Returns the component-wise product.
    fn mul(self, other: Self) -> Self;

    /// Returns the component-wise quotient.
    fn div(self, other: Self) -> Self;

    /// Scales every component by `s`.
    fn scale(self, s: f64) -> Self;

    /// Returns the dot product.
    fn dot(self, other: Self) -> f64;

    /// Returns the three-dimensional cross product.
    fn cross(self, other: Self) -> Self;

    /// Returns the squared Euclidean norm.
    fn norm2(self) -> f64;

    /// Returns the Euclidean norm.
    fn norm(self) -> f64;

    /// Returns the unit vector, or `None` if the vector is too close to zero.
    fn unit(self) -> Option<[f64; 3]>;

    /// Returns the squared Euclidean distance to `other`.
    fn distance2(self, other: Self) -> f64;

    /// Returns the Euclidean distance to `other`.
    fn distance(self, other: Self) -> f64;

    /// Returns `true` when the Euclidean distance to `other` is at most `tolerance`.
    fn close_to(self, other: Self, tolerance: f64) -> bool;

    /// Linearly interpolates from `self` to `other` with parameter `t`.
    fn lerp(self, other: Self, t: f64) -> Self;
}

/// Implements point operations for raw three-component arrays.
impl Point for [f64; 3] {
    fn add(self, other: Self) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]]
    }

    fn sub(self, other: Self) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]]
    }

    fn mul(self, other: Self) -> Self {
        [self[0] * other[0], self[1] * other[1], self[2] * other[2]]
    }

    fn div(self, other: Self) -> Self {
        [self[0] / other[0], self[1] / other[1], self[2] / other[2]]
    }

    fn scale(self, s: f64) -> [f64; 3] {
        [self[0] * s, self[1] * s, self[2] * s]
    }

    fn dot(self, other: Self) -> f64 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    fn cross(self, other: Self) -> Self {
        [
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ]
    }

    fn norm2(self) -> f64 {
        self.dot(self)
    }

    fn norm(self) -> f64 {
        self.norm2().sqrt()
    }

    fn unit(self) -> Option<[f64; 3]> {
        let n = self.norm();
        if n <= 1.0e-12 {
            None
        } else {
            Some(self.scale(1.0 / n))
        }
    }

    fn distance2(self, other: Self) -> f64 {
        self.sub(other).norm2()
    }

    fn distance(self, other: Self) -> f64 {
        self.distance2(other).sqrt()
    }

    fn close_to(self, other: Self, tolerance: f64) -> bool {
        self.distance(other) <= tolerance
    }

    fn lerp(self, other: Self, t: f64) -> Self {
        self.add(other.sub(self).scale(t))
    }
}
