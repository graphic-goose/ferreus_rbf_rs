/////////////////////////////////////////////////////////////////////////////////////////////
//
// Wraps the `rstar` crate to build spatial R-trees for domain decomposition neighbourhood queries.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # rtree
//!
//! Wrapper module for the rstar crate.
//!
//! Allows the building of an Rtree using points and subsequent
//! querying of neighbouring/intersecting points.

use faer::MatRef;
use rstar::primitives::GeomWithData;
use rstar::RTree;

type IndexedPoint<const D: usize> = GeomWithData<[f64; D], usize>;

/// Point index retaining each point's global row index. One-dimensional data is
/// embedded in two dimensions because `rstar` does not implement 1D R-trees.
pub(crate) enum NdPointRTree {
    D1(RTree<IndexedPoint<2>>),
    D2(RTree<IndexedPoint<2>>),
    D3(RTree<IndexedPoint<3>>),
}

impl NdPointRTree {
    /// Return at most `count` points ordered deterministically by squared distance
    /// and then global row index.
    pub(crate) fn nearest(&self, point: &[f64], count: usize) -> Vec<(usize, f64)> {
        let mut result = match self {
            Self::D1(tree) => nearest_points(tree, &[point[0], 0.0], count),
            Self::D2(tree) => nearest_points(tree, &[point[0], point[1]], count),
            Self::D3(tree) => nearest_points(tree, &[point[0], point[1], point[2]], count),
        };
        result.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        result
    }

    /// Returns the global indices of points within an infinity-norm distance
    /// of the supplied point.
    pub(crate) fn within_distance_inf(
        &self,
        point: &[f64],
        distance: f64,
    ) -> Vec<usize> {
        let mut result = match self {
            Self::D1(tree) => points_in_box(
                tree,
                [point[0], 0.0],
                distance,
            ),
            Self::D2(tree) => points_in_box(
                tree,
                [point[0], point[1]],
                distance,
            ),
            Self::D3(tree) => points_in_box(
                tree,
                [point[0], point[1], point[2]],
                distance,
            ),
        };

        result.sort_unstable();
        result
    }

}

fn nearest_points<const D: usize>(
    tree: &RTree<IndexedPoint<D>>,
    point: &[f64; D],
    count: usize,
) -> Vec<(usize, f64)> {
    tree.nearest_neighbor_iter_with_distance_2(point)
        .take(count)
        .map(|(item, distance_2)| (item.data, distance_2))
        .collect()
}

pub(crate) fn build_nd_point_rtree(
    dimensions: usize,
    points: MatRef<f64>,
    indices: &[usize],
) -> NdPointRTree {
    match dimensions {
        1 => NdPointRTree::D1(RTree::bulk_load(
            indices
                .iter()
                .map(|&index| GeomWithData::new([points[(index, 0)], 0.0], index))
                .collect(),
        )),
        2 => NdPointRTree::D2(RTree::bulk_load(
            indices
                .iter()
                .map(|&index| {
                    GeomWithData::new([points[(index, 0)], points[(index, 1)]], index)
                })
                .collect(),
        )),
        3 => NdPointRTree::D3(RTree::bulk_load(
            indices
                .iter()
                .map(|&index| {
                    GeomWithData::new(
                        [points[(index, 0)], points[(index, 1)], points[(index, 2)]],
                        index,
                    )
                })
                .collect(),
        )),
        _ => panic!("unsupported point dimension {dimensions}; expected 1, 2, or 3"),
    }
}

fn points_in_box<const D: usize>(
    tree: &RTree<IndexedPoint<D>>,
    point: [f64; D],
    distance: f64,
) -> Vec<usize> {
    use rstar::AABB;

    let minimum =
        std::array::from_fn(|dimension| point[dimension] - distance);
    let maximum =
        std::array::from_fn(|dimension| point[dimension] + distance);

    let envelope = AABB::from_corners(minimum, maximum);

    tree.locate_in_envelope_intersecting(&envelope)
        .map(|point| point.data)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    #[test]
    fn nearest_is_deterministic_for_distance_ties() {
        let coordinates = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]];
        let points = Mat::from_fn(4, 2, |row, column| coordinates[row][column]);
        let tree = build_nd_point_rtree(2, points.as_ref(), &[0, 1, 2, 3]);
        assert_eq!(
            tree.nearest(&[0.0, 0.0], 3)
                .into_iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn noncontiguous_indices_remain_global() {
        let points = Mat::from_fn(5, 1, |row, _| row as f64);
        let tree = build_nd_point_rtree(1, points.as_ref(), &[1, 4]);
        assert_eq!(tree.nearest(&[3.9], 2)[0].0, 4);
    }
}
