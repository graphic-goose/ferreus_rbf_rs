/////////////////////////////////////////////////////////////////////////////////////////////
//
// Represents point and regular-grid targets used for BBFMM tree refinement and evaluation.
//
// Created on: 10 Oct 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{FmmError, bbfmm::Dimensions, morton};
use faer::{MatRef, RowRef};

/// Represents the target locations used for tree refinement and kernel evaluation.
///
/// Targets are either rows of an input matrix or samples of a regular grid.
/// Grid coordinates are calculated as needed, without storing a matrix of all target points.
#[derive(Clone, Copy, Debug)]
pub enum EvaluationTargets<'a> {
    /// Target point matrix of shape (N, D), where N is the number of points and D is the dimensionality.
    Points(MatRef<'a, f64>),
    /// Regular-grid targets stored as axis origins, spacing, and sample counts.
    Grid(&'a TargetGrid),
}

impl<'a> EvaluationTargets<'a> {
    /// Gets the coordinates of a target point as a row view.
    ///
    /// # Arguments
    /// * `index`: Global target index. Grid targets use C order, with the last axis varying fastest.
    /// * `scratch`: Storage used for grid coordinates. Point targets borrow the input matrix directly.
    ///
    /// # Returns
    /// * A row view containing the target coordinates, borrowing either the input matrix or `scratch`.
    ///
    /// # Panics
    /// Panics if `index` is outside the target range.
    pub fn target<'b>(&'b self, index: usize, scratch: &'b mut [f64; 3]) -> RowRef<'b, f64> {
        match self {
            Self::Points(points) => points.row(index),
            Self::Grid(grid) => {
                let dims = grid.dimensions();
                grid.write_target(index, &mut scratch[..dims]);
                RowRef::from_slice(&scratch[..dims])
            }
        }
    }

    /// Gets the dimensionality of the target locations.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Points(points) => points.ncols(),
            Self::Grid(grid) => grid.dimensions(),
        }
    }

    /// Gets a single coordinate of a target point for the given axis.
    ///
    /// Grid target indices use C order, with the last axis varying fastest.
    /// Panics if the target index or axis is out of bounds.
    pub fn coordinate(&self, target_index: usize, axis: usize) -> f64 {
        match self {
            Self::Points(points) => points[(target_index, axis)],
            Self::Grid(grid) => grid.coordinate(target_index, axis),
        }
    }

    /// Gets the total number of target points.
    pub fn point_count(&self) -> usize {
        match self {
            Self::Points(points) => points.nrows(),
            Self::Grid(grid) => grid.point_count(),
        }
    }
}

/// Represents a regular grid of target points in one, two, or three dimensions.
///
/// Stores only the origin, spacing, and number of samples on each axis.
/// Target coordinates are calculated in C order, with the last axis varying fastest.
#[derive(Clone, Debug)]
pub struct TargetGrid {
    /// Coordinate of the first sample on each axis.
    origin: Vec<f64>,
    /// Distance between consecutive samples on each axis.
    spacing: Vec<f64>,
    /// Number of samples on each axis.
    shape: Vec<usize>,
}

impl TargetGrid {
    /// Constructs a regular target grid from axis origins, spacing, and sample counts.
    ///
    /// # Arguments
    /// * `origin`: Coordinate of the first sample on each axis.
    /// * `spacing`: Non-negative spacing on each axis. Zero spacing is allowed only for axes with one sample.
    /// * `shape`: Positive number of samples on each axis.
    ///
    /// # Returns
    /// * A fully initialised [`TargetGrid`] with owned copies of the axis metadata.
    ///
    /// # Panics
    /// Panics if the arrays do not describe one to three axes, the coordinates or spacing
    /// are invalid, or the total number of target points exceeds `usize`.
    pub fn new(origin: &[f64], spacing: &[f64], shape: &[usize]) -> Self {
        Self::try_new(origin, spacing, shape).unwrap_or_else(|reason| panic!("{reason}"))
    }

    /// Constructs a regular target grid from axis origins, spacing, and sample counts.
    ///
    /// Uses the same validation as [`Self::new`], returning an error description instead of panicking
    /// if the axis metadata is invalid or the total number of target points exceeds `usize`.
    pub fn try_new(origin: &[f64], spacing: &[f64], shape: &[usize]) -> Result<Self, &'static str> {
        let dimensions = origin.len();
        if !(1..=3).contains(&dimensions) {
            return Err("expected 1–3 grid axes");
        }
        if spacing.len() != dimensions {
            return Err("spacing dimension mismatch");
        }
        if shape.len() != dimensions {
            return Err("shape dimension mismatch");
        }

        for axis in 0..dimensions {
            if !origin[axis].is_finite() {
                return Err("grid origin must be finite");
            }
            if shape[axis] == 0 {
                return Err("grid axes must contain a sample");
            }
            if !spacing[axis].is_finite()
                || spacing[axis] < 0.0
                || (shape[axis] > 1 && spacing[axis] == 0.0)
            {
                return Err("invalid grid spacing");
            }
            let last = origin[axis] + (shape[axis] - 1) as f64 * spacing[axis];
            if !last.is_finite() {
                return Err("grid coordinates must be finite");
            }
        }

        shape
            .iter()
            .try_fold(1usize, |count, &size| count.checked_mul(size))
            .ok_or("grid point count exceeds usize")?;
        Ok(Self {
            origin: origin.to_vec(),
            spacing: spacing.to_vec(),
            shape: shape.to_vec(),
        })
    }

    /// Constructs a regular target grid from bounding extents and axis spacing.
    ///
    /// # Arguments
    /// * `extents`: Bounding box `[xmin, ymin, ..., xmax, ymax, ...]` for one to three dimensions.
    /// * `spacing`: Positive spacing on each axis.
    ///
    /// # Returns
    /// * A grid beginning at the lower bounds and containing all samples up to the upper bounds.
    ///   The given spacing is preserved, so the final sample may fall short of the upper bound.
    /// * [`FmmError::InvalidTargets`] if the bounds, spacing, or resulting sample counts are invalid.
    pub fn from_spacing(extents: &[f64], spacing: &[f64]) -> Result<Self, FmmError> {
        let dims = spacing.len();
        if !(1..=3).contains(&dims) || extents.len() != 2 * dims {
            return Err(FmmError::InvalidTargets(
                "expected 1–3 axes and two bounds per axis",
            ));
        }
        let mut shape = Vec::with_capacity(dims);
        for d in 0..dims {
            let (lo, hi, step) = (extents[d], extents[d + dims], spacing[d]);
            if !lo.is_finite() || !hi.is_finite() || hi < lo || !step.is_finite() || step <= 0.0 {
                return Err(FmmError::InvalidTargets(
                    "bounds must be finite and ordered; spacing must be finite and positive",
                ));
            }
            let intervals = ((hi - lo) / step).floor();
            if !intervals.is_finite() || intervals >= (usize::MAX - 1) as f64 {
                return Err(FmmError::InvalidTargets("grid axis is too large"));
            }
            let mut count = intervals as usize + 1;
            while count > 1 && lo + (count - 1) as f64 * step > hi {
                count -= 1;
            }
            shape.push(count);
        }
        Self::try_new(&extents[..dims], spacing, &shape).map_err(FmmError::InvalidTargets)
    }

    /// Gets the number of grid axes.
    pub fn dimensions(&self) -> usize {
        self.origin.len()
    }

    /// Gets the number of samples on each grid axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Calculates the total number of grid targets as the product of the axis sample counts.
    pub fn point_count(&self) -> usize {
        self.shape
            .iter()
            .try_fold(1usize, |count, &size| count.checked_mul(size))
            .expect("grid point count exceeds usize")
    }

    /// Calculates a single coordinate of a grid target for the given axis.
    ///
    /// The target index uses C order, with the last axis varying fastest.
    /// Panics if the target index or axis is out of bounds.
    pub fn coordinate(&self, target_index: usize, axis: usize) -> f64 {
        assert!(
            target_index < self.point_count(),
            "target index out of bounds"
        );
        assert!(axis < self.dimensions(), "axis out of bounds");

        let stride: usize = self.shape[axis + 1..].iter().product();
        let axis_index = (target_index / stride) % self.shape[axis];

        self.axis_coordinate(axis, axis_index)
    }

    /// Calculates the coordinates of a grid target and writes them to the given output slice.
    ///
    /// The target index uses C order, with the last axis varying fastest.
    /// Panics if the target index is out of bounds or the output length differs from the dimensionality.
    pub fn write_target(&self, target_index: usize, output: &mut [f64]) {
        assert!(
            target_index < self.point_count(),
            "target index out of bounds"
        );
        assert_eq!(output.len(), self.dimensions(), "output dimension mismatch");

        let mut remaining = target_index;

        for axis in (0..self.dimensions()).rev() {
            let axis_index = remaining % self.shape[axis];
            remaining /= self.shape[axis];
            output[axis] = self.axis_coordinate(axis, axis_index);
        }
    }

    /// Gets the bounding extents of the sampled grid.
    ///
    /// Returns `[xmin, ymin, ..., xmax, ymax, ...]` using the first and last samples on each axis,
    /// before any tree padding or coordinate transforms.
    pub fn extents(&self) -> Vec<f64> {
        self.origin
            .iter()
            .copied()
            .chain((0..self.dimensions()).map(|d| self.axis_coordinate(d, self.shape[d] - 1)))
            .collect()
    }

    /// Creates a matrix containing a contiguous range of grid target coordinates.
    ///
    /// # Arguments
    /// * `start`: Global index of the first target in C order.
    /// * `count`: Number of consecutive targets to include.
    ///
    /// # Returns
    /// * A target point matrix of shape (count, D), where D is the dimensionality.
    ///
    /// # Panics
    /// Panics if the requested range extends beyond the grid targets.
    pub fn points(&self, start: usize, count: usize) -> faer::Mat<f64> {
        assert!(start <= self.point_count() && count <= self.point_count() - start);
        faer::Mat::from_fn(count, self.dimensions(), |i, d| {
            self.coordinate(start + i, d)
        })
    }

    /// Calculates the coordinate of a sample on the given grid axis.
    fn axis_coordinate(&self, axis: usize, axis_index: usize) -> f64 {
        self.origin[axis] + axis_index as f64 * self.spacing[axis]
    }

    /// Finds the grid sample index ranges contained in a Morton-encoded cell.
    ///
    /// Uses the same anchor calculation as explicit point assignment so that rounding
    /// at cell boundaries does not create gaps between adjacent ranges.
    /// Returns one half-open sample index range per axis.
    pub(crate) fn cell_ranges(
        &self,
        key: u64,
        tree_center: &[f64],
        tree_radius: f64,
        dimensions: &Dimensions,
    ) -> Vec<std::ops::Range<usize>> {
        assert_eq!(self.dimensions(), *dimensions as usize);
        assert_eq!(tree_center.len(), self.dimensions());

        let anchor = morton::decode_key(key, dimensions);
        let length = morton::get_side_length(tree_radius, morton::get_level(key));

        (0..self.dimensions())
            .map(|axis| {
                let displacement = tree_center[axis] - tree_radius;

                let first = |boundary: u64| {
                    self.first_axis_index(axis, |coordinate| {
                        morton::coordinate_to_anchor(coordinate, displacement, length)
                            >= boundary as f64
                    })
                };

                first(anchor[axis])..first(anchor[axis] + 1)
            })
            .collect()
    }

    /// Calculates the number of grid targets contained in a Morton-encoded cell.
    pub(crate) fn count_in_cell(
        &self,
        key: u64,
        tree_center: &[f64],
        tree_radius: f64,
        dimensions: &Dimensions,
    ) -> usize {
        self.cell_ranges(key, tree_center, tree_radius, dimensions)
            .iter()
            .map(|range| range.len())
            .product()
    }

    /// Calculates the global target index for a position within a leaf cell's grid ranges.
    ///
    /// Both the leaf-local position and the global index use C order.
    /// Ranges must come from [`Self::cell_ranges`]. Panics if the number of ranges differs
    /// from the dimensionality or the position is outside the leaf targets.
    pub(crate) fn target_index_in_ranges(
        &self,
        ranges: &[std::ops::Range<usize>],
        mut ordinal: usize,
    ) -> usize {
        assert_eq!(ranges.len(), self.dimensions());

        let count: usize = ranges.iter().map(|range| range.len()).product();
        assert!(ordinal < count, "leaf target index out of bounds");

        let (mut target_index, mut stride) = (0, 1);

        for axis in (0..self.dimensions()).rev() {
            let axis_index = ranges[axis].start + ordinal % ranges[axis].len();

            ordinal /= ranges[axis].len();
            target_index += axis_index * stride;
            stride *= self.shape[axis];
        }

        target_index
    }

    /// Finds the first sample on an axis that satisfies the given monotonic predicate.
    ///
    /// Uses binary search, assuming the predicate changes from false to true as coordinates
    /// increase. Returns the axis sample count if no sample satisfies the predicate.
    fn first_axis_index(&self, axis: usize, at_or_after: impl Fn(f64) -> bool) -> usize {
        let (mut lower, mut upper) = (0, self.shape[axis]);

        while lower < upper {
            let middle = lower + (upper - lower) / 2;

            if at_or_after(self.axis_coordinate(axis, middle)) {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }

        lower
    }
}
