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
//! Allows the building of an Rtree using extent rectangles and subsequent
//! querying of neighbouring/intersecting rectangles.

use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};
use std::convert::TryInto;

// rstar doesn't support 1D natively, so we've worked around
// that by treating it as a 2D problem and setting the y component
// for each rectangle added/queried to a min of 0 and max of 1.
// 1D is represented as 2D rectangles (y in [0,1])
type Rect1As2 = GeomWithData<Rectangle<[f64; 2]>, usize>;
type Rect2 = GeomWithData<Rectangle<[f64; 2]>, usize>;
type Rect3 = GeomWithData<Rectangle<[f64; 3]>, usize>;

pub enum NdRTree {
    D1(RTree<Rect1As2>), // 1D embedded in 2D
    D2(RTree<Rect2>),
    D3(RTree<Rect3>),
}

impl NdRTree {
    // Finds all neighbourings/intersecting rectangles to the query extents.
    pub fn find_neighbours(&self, domain_extents: &[f64], i: usize) -> Vec<usize> {
        match self {
            NdRTree::D1(tree) => find_neighbours_1d_as2d(tree, domain_extents, i),
            NdRTree::D2(tree) => find_neighbours::<2>(tree, domain_extents, i),
            NdRTree::D3(tree) => find_neighbours::<3>(tree, domain_extents, i),
        }
    }
}

/// For D=2 or D=3: `extents = [mins..., maxs...]` (len = 2*D)
fn rectangle_from_extents_nd<const D: usize>(extents: &[f64]) -> Rectangle<[f64; D]> {
    assert!(
        extents.len() == 2 * D,
        "expected {} extents for dimension {}, got {}",
        2 * D,
        D,
        extents.len()
    );
    let (min_slice, max_slice) = extents.split_at(D);
    let mins: [f64; D] = min_slice.try_into().expect("min slice length mismatch");
    let maxs: [f64; D] = max_slice.try_into().expect("max slice length mismatch");
    Rectangle::from_corners(mins, maxs)
}

/// 1D embedded as 2D with y in [0,1]
fn rectangle_from_extents_1d_as2d(extents: &[f64]) -> Rectangle<[f64; 2]> {
    assert!(extents.len() == 2, "1D expects [min_x, max_x]");
    Rectangle::from_corners([extents[0], 0.0], [extents[1], 1.0])
}

/// A wrapper that holds a AABB‐rectangle and usize index.
type IndexedRect<const D: usize> = GeomWithData<Rectangle<[f64; D]>, usize>;

/// Build up an RTree of IndexedRect<D> so that each leaf knows its domain‐index.
fn bulk_load_indexed_nd<const D: usize>(
    items: Vec<IndexedRect<D>>,
) -> RTree<IndexedRect<D>> {
    RTree::bulk_load(items)
}

fn find_neighbours<const D: usize>(
    tree: &RTree<IndexedRect<D>>,
    domain_extents: &[f64],
    i: usize,
) -> Vec<usize> {
    let (min_slice, max_slice) = domain_extents.split_at(D);
    let mins: [f64; D] = min_slice.try_into().unwrap();
    let maxs: [f64; D] = max_slice.try_into().unwrap();
    let envelope = AABB::from_corners(mins, maxs);
    tree.locate_in_envelope_intersecting(&envelope)
        .map(|item| item.data)
        .filter(|&idx| idx != i)
        .collect()
}

fn find_neighbours_1d_as2d(
    tree: &RTree<Rect1As2>,
    domain_extents: &[f64],
    i: usize,
) -> Vec<usize> {
    debug_assert_eq!(domain_extents.len(), 2); // [min_x, max_x]
    let envelope = AABB::from_corners([domain_extents[0], 0.0], [domain_extents[1], 1.0]);
    tree.locate_in_envelope_intersecting(&envelope)
        .map(|item| item.data)
        .filter(|&idx| idx != i)
        .collect()
}

/// Build an NdRTree from an iterator over (index, extents) where
/// `extents = [mins..., maxs...]` of length 2*dimensions.
/// For `dimensions == 1`, intervals are embedded as 2D rectangles with y∈[0,1].
pub fn build_nd_rtree_from_extents<'a, I>(dimensions: usize, items: I) -> NdRTree
where
    I: IntoIterator<Item = (usize, &'a [f64])>,
{
    match dimensions {
        1 => {
            let rects = items
                .into_iter()
                .map(|(idx, ext)| {
                    let rect = rectangle_from_extents_1d_as2d(ext);
                    GeomWithData::new(rect, idx)
                })
                .collect::<Vec<_>>();
            NdRTree::D1(bulk_load_indexed_nd::<2>(rects))
        }
        2 => {
            let rects = items
                .into_iter()
                .map(|(idx, ext)| {
                    let rect = rectangle_from_extents_nd::<2>(ext);
                    GeomWithData::new(rect, idx)
                })
                .collect::<Vec<_>>();
            NdRTree::D2(bulk_load_indexed_nd::<2>(rects))
        }
        3 => {
            let rects = items
                .into_iter()
                .map(|(idx, ext)| {
                    let rect = rectangle_from_extents_nd::<3>(ext);
                    GeomWithData::new(rect, idx)
                })
                .collect::<Vec<_>>();
            NdRTree::D3(bulk_load_indexed_nd::<3>(rects))
        }
        _ => panic!("Unsupported dimensions for NdRTree"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstar::primitives::GeomWithData;

    #[test]
    fn rtree_1d_neighbours() {
        // intervals: [0,1], [1,2], [3,4] (embedded as y∈[0,1])
        let rects = vec![
            GeomWithData::new(rectangle_from_extents_1d_as2d(&[0.0, 1.0]), 0),
            GeomWithData::new(rectangle_from_extents_1d_as2d(&[1.0, 2.0]), 1),
            GeomWithData::new(rectangle_from_extents_1d_as2d(&[3.0, 4.0]), 2),
        ];
        let tree = NdRTree::D1(bulk_load_indexed_nd::<2>(rects));

        // neighbours of [1,2] (index=1) — touching at x=1 counts as intersecting
        let n = tree.find_neighbours(&[1.0, 2.0], 1);
        assert!(n.contains(&0));
        assert!(!n.contains(&2));
        assert!(!n.contains(&1), "must not include itself");
    }

    #[test]
    fn rtree_2d_neighbours() {
        // squares: [0,0]-[1,1], [1,0]-[2,1], [3,3]-[4,4]
        let rects = vec![
            GeomWithData::new(rectangle_from_extents_nd::<2>(&[0.0, 0.0, 1.0, 1.0]), 0),
            GeomWithData::new(rectangle_from_extents_nd::<2>(&[1.0, 0.0, 2.0, 1.0]), 1),
            GeomWithData::new(rectangle_from_extents_nd::<2>(&[3.0, 3.0, 4.0, 4.0]), 2),
        ];
        let tree = NdRTree::D2(bulk_load_indexed_nd::<2>(rects));

        // Query neighbours for rect 1
        let n = tree.find_neighbours(&[1.0, 0.0, 2.0, 1.0], 1);
        assert!(n.contains(&0), "touching edge should count as intersecting");
        assert!(!n.contains(&2), "far square shouldn't intersect");
        assert!(!n.contains(&1), "must not include itself");

        // Query a non-overlapping box (empty result)
        let empty = tree.find_neighbours(&[10.0, 10.0, 11.0, 11.0], usize::MAX);
        assert!(empty.is_empty());
    }

    #[test]
    fn rtree_3d_neighbours() {
        // cubes: [0,0,0]-[1,1,1], [1,0,0]-[2,1,1], [3,3,3]-[4,4,4]
        let rects = vec![
            GeomWithData::new(
                rectangle_from_extents_nd::<3>(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                0,
            ),
            GeomWithData::new(
                rectangle_from_extents_nd::<3>(&[1.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
                1,
            ),
            GeomWithData::new(
                rectangle_from_extents_nd::<3>(&[3.0, 3.0, 3.0, 4.0, 4.0, 4.0]),
                2,
            ),
        ];
        let tree = NdRTree::D3(bulk_load_indexed_nd::<3>(rects));

        // Query neighbours for cube 1
        let n = tree.find_neighbours(&[1.0, 0.0, 0.0, 2.0, 1.0, 1.0], 1);
        assert!(n.contains(&0));
        assert!(!n.contains(&2));
        assert!(!n.contains(&1));
    }

    #[test]
    fn rtree_find_neighbours_excludes_self_in_all_dims() {
        // 2D example — but behaviour is the same across dims
        let rects = vec![GeomWithData::new(
            rectangle_from_extents_nd::<2>(&[0.0, 0.0, 1.0, 1.0]),
            42,
        )];
        let tree = NdRTree::D2(bulk_load_indexed_nd::<2>(rects));
        let n = tree.find_neighbours(&[0.0, 0.0, 1.0, 1.0], 42);
        assert!(n.is_empty(), "self should not be listed as neighbour");
    }
}
