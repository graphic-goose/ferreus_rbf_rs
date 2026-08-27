/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the tetrahedral sampling lattice and relevant methods.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # lattice
//! This module defines the tetrahedral sampling lattice and functions for converting between world
//! and lattice space. For convenience, the sample lattice is defined as a regular rectangular ijk
//! grid, formed from the stacking of two XY sample planes.
//!
//! From the paper:
//! "The locations of each of the sample points on neighbouring planes alternate,
//! so that two planes taken together form a rectangular lattice of sample points
//! with half the spacing of an individual plane."
//!
//! Neighbouring points in the tetrahedral lattice are separated by either 1.0 or 0.866 (sqrt(3)/2)
//! units, providing a more uniform sampling of space compared with a conventional cubic lattice.
//!
//! See the constants module for diagrams and more details about the indexing.

use super::{
    aabb_clipping::AABB,
    constants::{U, V, W},
    geometry::Point,
};
use faer::{linalg::solvers::PartialPivLu, mat, prelude::*};

/// Defines the sampling lattice, it's extents and conversion methods.
#[derive(Clone, Debug)]
pub struct SampleLattice {
    // The axis-aligned bounding box extents of the sampling lattice.
    pub extents: AABB<f64>,

    // The sample point spacing along each axis of the fine-grid lattice.
    pub spacing: [f64; 3],

    // The max ijk value.
    pub max_ijk: [i64; 3],

    // The inverted basis matrix to allow easy converting from world to lattice space.
    basis_inv: PartialPivLu<f64>,
}

impl SampleLattice {
    /// Creates a `SampleLattice` covering the given extents.
    ///
    /// `resolution` is the nominal tetrahedral sample spacing within each sampling plane.
    /// The lattice stores points on a finer skewed grid, with spacing
    /// `[resolution / 2, resolution * sqrt(2) / 2, resolution / sqrt(2)]`.
    pub fn new(resolution: f64, extents: AABB<f64>) -> Self {
        let sqrt2 = std::f64::consts::SQRT_2;
        let spacing = [
            resolution / 2.0,
            (resolution * sqrt2) / 2.0,
            resolution / sqrt2,
        ];

        let mut max_ijk = extents
            .max_corner
            .sub(extents.min_corner)
            .div(spacing)
            .map(|v| v.ceil())
            .map(|v| v as i64);

        max_ijk[0] += 1;

        // Columns are U, V, W are basis vectors of the owned parallelpiped.
        let basis = mat![
            [U[0] as f64, V[0] as f64, W[0] as f64],
            [U[1] as f64, V[1] as f64, W[1] as f64],
            [U[2] as f64, V[2] as f64, W[2] as f64],
        ];

        // Calculate the inverse of the basis matrix to use for converting from world to ijk.
        let basis_inv = basis.partial_piv_lu();

        Self {
            extents,
            spacing,
            max_ijk,
            basis_inv,
        }
    }

    /// Converts a point from lattice space to world space.
    pub fn ijk_to_world(&self, ijk: [i64; 3]) -> [f64; 3] {
        self.extents
            .min_corner
            .add(ijk.map(|v| v as f64).mul(self.spacing))
    }

    /// Converts a point from world space to lattice space.
    pub fn world_to_ijk(&self, world: [f64; 3]) -> [i64; 3] {
        let eps = 1e-9;

        // Point in continuous fine-grid coordinates.
        let p = world.sub(self.extents.min_corner).div(self.spacing);

        let point = mat![[p[0]], [p[1]], [p[2]],];

        // Solve basis * q = point.
        // q is the coordinate of the point in the U/V/W basis.
        let q = &self.basis_inv.solve(&point);

        let a = (q[(0, 0)] + eps).floor() as i64;
        let b = (q[(1, 0)] + eps).floor() as i64;
        let c = (q[(2, 0)] + eps).floor() as i64;

        // Convert the owner-cell origin back from basis coordinates
        // to fine-grid ijk coordinates.
        [
            a * U[0] as i64 + b * V[0] as i64 + c * W[0] as i64,
            a * U[1] as i64 + b * V[1] as i64 + c * W[1] as i64,
            a * U[2] as i64 + b * V[2] as i64 + c * W[2] as i64,
        ]
    }

    /// Returns whether a lattice point is within the padded extraction domain.
    pub fn extraction_ijk_inbounds(&self, ijk: [i64; 3]) -> bool {
        const OPEN_CLIP_IJK_PADDING: i64 = 2;

        ijk[0] >= -OPEN_CLIP_IJK_PADDING
            && ijk[0] <= self.max_ijk[0] + OPEN_CLIP_IJK_PADDING
            && ijk[1] >= -OPEN_CLIP_IJK_PADDING
            && ijk[1] <= self.max_ijk[1] + OPEN_CLIP_IJK_PADDING
            && ijk[2] >= -OPEN_CLIP_IJK_PADDING
            && ijk[2] <= self.max_ijk[2] + OPEN_CLIP_IJK_PADDING
    }
}
