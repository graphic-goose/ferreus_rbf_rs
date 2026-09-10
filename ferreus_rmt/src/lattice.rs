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
    aabb_clipping::{AABB, bbox_eps},
    constants::{EDGE_DELTAS, U, V, W},
    geometry::Point,
};
use faer::{linalg::solvers::DenseSolveCore, mat, Col, prelude::*};
use std::array;

const OPEN_CLIP_IJK_PADDING: i64 = 2;

/// Defines the sampling lattice, it's extents and conversion methods.
#[derive(Clone, Debug)]
pub struct SampleLattice {
    // The axis-aligned bounding box extents of the sampling lattice.
    pub extents: AABB<f64>,

    // The sample point spacing along each axis of the fine-grid lattice.
    pub spacing: [f64; 3],

    // Conservative fine-grid bounds covering the world extents.
    pub min_ijk: [i64; 3],
    pub max_ijk: [i64; 3],

    // Maps fine-grid displacement vectors into world coordinates.
    sampling_basis: Mat<f64>,

    // Maps world displacement vectors into fine-grid coordinates.
    inverse_sampling_basis: Mat<f64>,

    // Maps fine-grid row vectors into U/V/W owner-cell coordinates.
    basis_inv: Mat<f64>,

    // World-space step used for finite-difference gradients.
    pub gradient_step: f64,
}

impl SampleLattice {
    /// Creates a `SampleLattice` covering the given extents.
    ///
    /// `resolution` is the nominal tetrahedral sample spacing within each sampling plane.
    /// The lattice stores points on a finer skewed grid, with spacing
    /// `[resolution / 2, resolution * sqrt(2) / 2, resolution / sqrt(2)]`.
    pub fn new(resolution: f64, extents: AABB<f64>, sampling_transform: Option<MatRef<'_, f64>>,) -> Self {
        let sqrt2 = std::f64::consts::SQRT_2;
        let spacing = [
            resolution / 2.0,
            (resolution * sqrt2) / 2.0,
            resolution / sqrt2,
        ];

        let transform = sampling_transform
            .map(|transform| transform.to_owned())
            .unwrap_or_else(|| Mat::identity(3, 3));

        let singular_values = transform
            .singular_values()
            .expect("sampling_transform SVD failed");

        let minimum_scale = singular_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        let maximum_scale = singular_values
            .iter()
            .copied()
            .fold(0.0, f64::max);

        let inverse_transform = transform.partial_piv_lu().inverse();

        let sampling_spacing = Col::from_fn(3, |i| {
            spacing[i] * minimum_scale
        });

        let reciprocal_spacing = Col::from_fn(3, |i| {
            sampling_spacing[i].recip()
        });

        // Maps fine-grid displacements into world coordinates.
        let sampling_basis =
            sampling_spacing.as_ref().as_diagonal() * &inverse_transform;

        // Maps world displacements into fine-grid coordinates.
        let inverse_sampling_basis =
            &transform * reciprocal_spacing.as_ref().as_diagonal();

        // Each row contains a world-box corner relative to the lattice origin.
        let corners = Mat::from_fn(8, 3, |corner, axis| {
            if corner & (1 << axis) == 0 {
                0.0
            } else {
                extents.max_corner[axis] - extents.min_corner[axis]
            }
        });

        // Transform all eight world-box corners into fine-grid coordinates.
        let tranformed_corners = corners * &inverse_sampling_basis;

        let tranformed_corner_extents = get_pointarray_extents(tranformed_corners.as_ref());
        let (minimum, maximum) = tranformed_corner_extents.split_at(3);

        let min_ijk: [i64; 3] = array::from_fn(|i| minimum[i].floor() as i64);
        let mut max_ijk: [i64; 3] = array::from_fn(|i| maximum[i].ceil() as i64);

        max_ijk[0] += 1;

        // Columns are U, V, W are basis vectors of the owned parallelpiped.
        let basis = mat![
            [U[0] as f64, V[0] as f64, W[0] as f64],
            [U[1] as f64, V[1] as f64, W[1] as f64],
            [U[2] as f64, V[2] as f64, W[2] as f64],
        ];

        // Store the transposed inverse for use with row vectors.
        let basis_inv = basis.partial_piv_lu().inverse().transpose().to_owned();

        // Base the finite-difference step on the shortest physical scale.
        let minimum_world_spacing =
            spacing[0] * minimum_scale / maximum_scale;

        let gradient_step = minimum_world_spacing * 1.0e-4;

        Self {
            extents,
            spacing,
            min_ijk,
            max_ijk,
            sampling_basis,
            inverse_sampling_basis,
            basis_inv,
            gradient_step,
        }
    }

    /// Converts a point from lattice space to world space.
    pub fn ijk_to_world(&self, ijk: [i64; 3]) -> [f64; 3] {
        let offset = transform_vector(
            ijk.map(|value| value as f64),
            self.sampling_basis.as_ref(),
        );

        self.extents.min_corner.add(offset)
    }

    /// Returns a point in the normalized canonical sampling metric.
    pub fn ijk_to_sampling(&self, ijk: [i64; 3]) -> [f64; 3] {
        ijk.map(|value| value as f64).mul(self.spacing)
    }

    /// Converts a point from world space to lattice space.
    pub fn world_to_ijk(&self, world: [f64; 3]) -> [i64; 3] {
        let eps = 1e-9;

        // Point in continuous fine-grid coordinates.
        let p = transform_vector(
            world.sub(self.extents.min_corner),
            self.inverse_sampling_basis.as_ref(),
        );

        // q is the coordinate of the point in the U/V/W basis.
        let q = transform_vector(p, self.basis_inv.as_ref());

        let a = (q[0] + eps).floor() as i64;
        let b = (q[1] + eps).floor() as i64;
        let c = (q[2] + eps).floor() as i64;

        // Convert the owner-cell origin back from basis coordinates
        // to fine-grid ijk coordinates.
        [
            a * U[0] as i64 + b * V[0] as i64 + c * W[0] as i64,
            a * U[1] as i64 + b * V[1] as i64 + c * W[1] as i64,
            a * U[2] as i64 + b * V[2] as i64 + c * W[2] as i64,
        ]
    }

    /// Returns whether a lattice point is inside the padded traversal domain.
    pub fn extraction_ijk_inbounds(&self, ijk: [i64; 3]) -> bool {
        (0..3).all(|axis| {
            ijk[axis] >= self.min_ijk[axis] - OPEN_CLIP_IJK_PADDING
                && ijk[axis] <= self.max_ijk[axis] + OPEN_CLIP_IJK_PADDING
        })
    }

    /// Returns conservative world extents covering all callback evaluations.
    ///
    /// Includes seed owner cells, traversal cells, neighbouring topology
    /// samples and finite-difference offsets.
    pub fn evaluation_extents(&self) -> Vec<f64> {
        let padding: [i64; 3] = array::from_fn(|axis| {
            let owner_span = (U[axis] as i64).abs()
                + (V[axis] as i64).abs()
                + (W[axis] as i64).abs();

            let neighbour_span = EDGE_DELTAS
                .iter()
                .map(|delta| (delta[axis] as i64).abs())
                .max()
                .unwrap();

            OPEN_CLIP_IJK_PADDING + owner_span + 2 * neighbour_span
        });

        // Transform the padded fine-grid corners into world-space offsets.
        let corners = get_box_corners(
            array::from_fn(|axis| (self.min_ijk[axis] - padding[axis]) as f64),
            array::from_fn(|axis| (self.max_ijk[axis] + padding[axis]) as f64),
        );
        let corners = corners * &self.sampling_basis;
        let mut extents = get_pointarray_extents(corners.as_ref());

        // Add the lattice origin and include world seed-projection positions.
        for axis in 0..3 {
            extents[axis] = (extents[axis] + self.extents.min_corner[axis])
                .min(self.extents.min_corner[axis]);
            extents[axis + 3] = (extents[axis + 3] + self.extents.min_corner[axis])
                .max(self.extents.max_corner[axis]);
        }

        // Include finite-difference offsets and a scale-aware numerical margin.
        let margin = self.gradient_step
            + bbox_eps(AABB {
                min_corner: array::from_fn(|axis| extents[axis]),
                max_corner: array::from_fn(|axis| extents[axis + 3]),
            });

        for axis in 0..3 {
            extents[axis] -= margin;
            extents[axis + 3] += margin;
        }

        extents
    }

    /// Returns the metric used for sampling-space Newton projection in world coordinates.
    pub(crate) fn seed_projection_metric(&self) -> Mat<f64> {
        let inverse_spacing = Col::from_fn(3, |i| self.spacing[i].recip());
        let basis = inverse_spacing.as_ref().as_diagonal() * &self.sampling_basis;

        basis.transpose() * &basis
    }
}

#[inline(always)]
fn transform_vector(vector: [f64; 3], transform: MatRef<'_, f64>) -> [f64; 3] {
    std::array::from_fn(|col| {
        (0..3)
            .map(|row| vector[row] * transform[(row, col)])
            .sum()
    })
}


/// Returns the eight corners of an axis-aligned box as rows.
#[inline(always)]
fn get_box_corners(minimum: [f64; 3], maximum: [f64; 3]) -> Mat<f64> {
    Mat::from_fn(8, 3, |corner, axis| {
        [minimum[axis], maximum[axis]][(corner >> axis) & 1]
    })
}

/// Returns the world extents that an evaluator must cover for extraction.
#[inline(always)]
pub fn get_evaluation_extents(
    extents: &[f64],
    resolution: f64,
    sampling_transform: Option<MatRef<'_, f64>>,
) -> Vec<f64> {
    assert_eq!(extents.len(), 6, "extents must have length 6");

    let extents = AABB {
        min_corner: [extents[0], extents[1], extents[2]],
        max_corner: [extents[3], extents[4], extents[5]],
    };

    SampleLattice::new(resolution, extents, sampling_transform)
        .evaluation_extents()
}

#[inline(always)]
fn get_pointarray_extents<T>(points: MatRef<T>) -> Vec<T>
where
    T: PartialOrd + Clone,
{
    let ncols = points.shape().1;

    // Initialize extents with min and max values for each column.
    // The first half of the vector stores mins, the second half stores maxs.
    let mut extents: Vec<T> = vec![points.get(0, 0).clone(); 2 * ncols];

    // Initialize the mins and maxs
    for col in 0..ncols {
        extents[col] = points.get(0, col).clone(); // Min value for each column
        extents[col + ncols] = points.get(0, col).clone(); // Max value for each column
    }

    // Iterate over the rows
    for row in points.row_iter() {
        for (col, item) in row.iter().enumerate() {
            // Update min value for the column
            if item < &extents[col] {
                extents[col] = item.clone();
            }
            // Update max value for the column
            if item > &extents[col + ncols] {
                extents[col + ncols] = item.clone();
            }
        }
    }

    extents
}