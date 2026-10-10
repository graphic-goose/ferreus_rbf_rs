/////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculates target counts and subdivision criteria for adaptive BBFMM trees.
//
// Created on: 10 Oct 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{EvaluationTargets, TargetGrid, bbfmm::Dimensions, morton, morton_constants};
use faer::MatRef;
use rayon::prelude::*;
use std::collections::HashMap;

/// Stores the target representation used to count points within tree cells.
enum TargetCounts<'a> {
    /// Sorted Morton codes at the maximum tree level, with the level bits removed.
    MortonKeys(Vec<u64>),
    /// Regular-grid metadata used to count targets from axis index ranges.
    Grid(&'a TargetGrid),
}

/// Stores target counts and source distribution information used for tree refinement.
///
/// Cells with many targets are subdivided according to the concentration of nearby sources.
/// Larger target batches are allowed in empty or approximately uniform source neighbourhoods.
pub(crate) struct TargetRefinement<'a> {
    /// Point Morton codes or grid metadata used to count targets in a cell.
    targets: TargetCounts<'a>,
    /// Number of sources in occupied cells at the reference depth and their ancestors.
    source_counts: HashMap<u64, usize>,
    /// Center of the root cell bounding box.
    center: Vec<f64>,
    /// Half the side length of the root cell.
    radius: f64,
    /// Dimensionality of the source and target locations.
    dimensions: Dimensions,
    /// Base target count limit, set to the larger of the source cell limit and interpolation node count.
    near_limit: usize,
    /// Reference depth calculated from the total source count and dimensionality.
    optimal_depth: u64,
}

impl<'a> TargetRefinement<'a> {
    /// Constructs the target refinement information for the given source points and tree extents.
    ///
    /// # Arguments
    /// * `sources`: Source point matrix of shape (N, D), where N is the number of points and D is the dimensionality.
    /// * `center`: Center of the root cell bounding box.
    /// * `radius`: Half the side length of the root cell.
    /// * `dimensions`: Dimensionality of the source and target locations.
    /// * `interpolation_order`: Number of Chebyshev nodes per dimension.
    /// * `max_points_per_cell`: Source count limit used for cell subdivision.
    /// * `targets`: Point or regular-grid targets used to guide tree refinement.
    ///
    /// # Returns
    /// * A fully initialised [`TargetRefinement`] with target metadata and source counts.
    pub(crate) fn new(
        sources: MatRef<f64>,
        center: &[f64],
        radius: f64,
        dimensions: Dimensions,
        interpolation_order: usize,
        max_points_per_cell: usize,
        targets: EvaluationTargets<'a>,
    ) -> Self {
        let dims = dimensions as usize;
        let displacement: Vec<f64> = center.iter().map(|c| c - radius).collect();
        let n_points = sources.nrows() as f64;
        let optimal_depth = ((n_points.log2() / dimensions as isize as f64).ceil() as u64)
            .clamp(1, morton_constants::MAXIMUM_LEVEL);

        let targets = match targets {
            EvaluationTargets::Points(points) => {
                let level = morton_constants::MAXIMUM_LEVEL;
                let length = morton::get_side_length(radius, level);

                let mut keys: Vec<u64> = (0..points.nrows())
                    .into_par_iter()
                    .map(|i| {
                        let anchor =
                            morton::point_to_anchor(points.row(i), level, &displacement, length);
                        morton::encode_morton_point(anchor, &dimensions)
                            >> morton_constants::LEVEL_DISPLACEMENT
                    })
                    .collect();

                keys.par_sort_unstable();

                TargetCounts::MortonKeys(keys)
            }
            EvaluationTargets::Grid(grid) => TargetCounts::Grid(grid),
        };

        let length = morton::get_side_length(radius, optimal_depth);
        let mut counts = HashMap::new();

        for i in 0..sources.nrows() {
            let anchor =
                morton::point_to_anchor(sources.row(i), optimal_depth, &displacement, length);
            *counts
                .entry(morton::encode_morton_point(anchor, &dimensions))
                .or_insert(0) += 1;
        }

        let mut source_counts = counts.clone();

        for _ in 0..optimal_depth {
            let mut parents = HashMap::new();

            for (key, count) in counts {
                *parents
                    .entry(morton::get_parent(key, &dimensions).unwrap())
                    .or_insert(0) += count;
            }

            source_counts.extend(parents.iter().map(|(&key, &count)| (key, count)));
            counts = parents;
        }

        Self {
            targets,
            source_counts,
            center: center.to_vec(),
            radius,
            dimensions,
            near_limit: max_points_per_cell
                .max(interpolation_order.pow(dims as u32))
                .max(1),
            optimal_depth,
        }
    }

    /// Calculates the number of targets contained in a Morton-encoded cell.
    ///
    /// Point targets are counted by searching the sorted Morton code range.
    /// Grid targets are counted from the sample index ranges on each axis.
    pub(crate) fn target_count(&self, key: u64) -> usize {
        match &self.targets {
            TargetCounts::MortonKeys(keys) => {
                let shift = (morton_constants::MAXIMUM_LEVEL - morton::get_level(key))
                    * self.dimensions as u64;
                let prefix = key >> morton_constants::LEVEL_DISPLACEMENT;
                let begin = prefix << shift;
                let end = (prefix + 1) << shift;

                keys.partition_point(|&k| k < end) - keys.partition_point(|&k| k < begin)
            }
            TargetCounts::Grid(grid) => {
                grid.count_in_cell(key, &self.center, self.radius, &self.dimensions)
            }
        }
    }

    /// Determines whether a cell should be subdivided based on its target count and nearby sources.
    ///
    /// Uses the base target limit near concentrated sources. The limit is increased by a factor
    /// of 16 for approximately uniform source neighbourhoods at or above the reference depth,
    /// and by a factor of 32 where the neighbourhood contains no sources.
    pub(crate) fn should_split(&self, key: u64) -> bool {
        let targets = self.target_count(key);

        if targets <= self.near_limit {
            return false;
        }

        let mut peers = morton::get_neighbours(key, &self.dimensions);
        peers.push(key);

        let mut sum = 0.0;
        let mut sum_squared_counts = 0.0;

        for mut peer in peers.iter().copied() {
            while morton::get_level(peer) >= self.optimal_depth {
                peer = morton::get_parent(peer, &self.dimensions).unwrap();
            }

            let count = self.source_counts.get(&peer).copied().unwrap_or(0) as f64;
            sum += count;
            sum_squared_counts += count * count;
        }

        // Preserve a fine halo around concentrated source boxes. Empty/fairly
        // uniform neighbourhoods can use larger batches without as much direct
        // work.
        let factor = if sum == 0.0 {
            32
        } else if morton::get_level(key) <= self.optimal_depth
            && sum * sum / sum_squared_counts >= peers.len() as f64 * 0.5
        {
            16
        } else {
            1
        };

        targets > self.near_limit.saturating_mul(factor)
    }
}
