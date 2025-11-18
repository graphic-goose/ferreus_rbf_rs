/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides a simple KD-tree implementation for nearest-neighbour queries in RBF workflows.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, Row, RowRef};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    InfinityNorm,
}

// Define a wrapper for a single row to implement the trait.
#[derive(Debug, Clone, PartialEq)]
pub struct PointRowWithId {
    pub coords: Row<f64>,
    pub id: i32,
}

impl PointRowWithId {
    pub fn new(coords: &RowRef<f64>, id: &i32) -> Self {
        let new_row = coords.to_owned();
        Self {
            coords: new_row,
            id: *id,
        }
    }

    /// Standard euclidean distance
    pub fn distance_euclidean(&self, other: &Self) -> f64 {
        let dist: f64 = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        dist.sqrt()
    }

    /// infinity-norm distance
    pub fn distance_inf(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }
}

/// A node in the KDTree
#[derive(Debug)]
struct Node {
    point: PointRowWithId,
    left: Option<usize>,
    right: Option<usize>,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
struct Neighbour {
    distance_sq: f64,
    id: i32,
}

impl Eq for Neighbour {}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for max-heap
        other.distance_sq.partial_cmp(&self.distance_sq)
    }
}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// The KDTree structure
#[derive(Debug)]
pub struct KDTree {
    nodes: Vec<Node>,
}

impl KDTree {
    /// Constructs a new KDTree from a Mat of points.
    pub fn new(nd_array: &Mat<f64>) -> Self {
        let shape = nd_array.shape();
        let nd_array_nrows = shape.0;
        let mut points: Vec<PointRowWithId> = Vec::new();

        (0..nd_array_nrows)
            .into_iter()
            .for_each(|i| points.push(PointRowWithId::new(&nd_array.row(i), &(i as i32))));

        let mut tree = KDTree { nodes: Vec::new() };
        tree.build_tree(&mut points, 0);
        tree
    }

    /// Recursively builds the KDTree and stores nodes in a flat vector.
    fn build_tree(&mut self, points: &mut [PointRowWithId], depth: usize) -> Option<usize> {
        if points.is_empty() {
            return None;
        }

        // Determine splitting axis
        let axis = depth % points[0].coords.ncols();

        // Sort points by the current axis
        points.sort_by(|a, b| {
            a.coords[axis]
                .partial_cmp(&b.coords[axis])
                .unwrap_or(Ordering::Equal)
        });

        // Choose the median as the pivot
        let mid = points.len() / 2;
        let median_point = &points[mid];

        // Create the current node
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            point: median_point.clone(),
            left: None,
            right: None,
        });

        // Recursively build left and right subtrees
        self.nodes[node_index].left = self.build_tree(&mut points[..mid], depth + 1);
        self.nodes[node_index].right = self.build_tree(&mut points[mid + 1..], depth + 1);

        Some(node_index)
    }

    fn point_dist(metric: DistanceMetric, p: &PointRowWithId, q: &PointRowWithId) -> f64 {
        match metric {
            DistanceMetric::Euclidean => p.distance_euclidean(q),
            DistanceMetric::InfinityNorm => p.distance_inf(q),
        }
    }

    fn axis_diff_ok(diff: f64, best_radius: f64) -> bool {
        // |diff| <= radius ⇒ hypersphere / hyper‑cube intersects the splitting plane
        diff.abs() <= best_radius
    }

    /// Performs a radius search around a target point.
    pub fn radius_search(
        &self,
        target: &PointRowWithId,
        radius: f64,
        metric: DistanceMetric,
    ) -> Vec<i32> {
        let mut result = Vec::new();
        self.radius_search_impl(0, target, radius, 0, metric, &mut result);
        result
    }

    fn radius_search_impl(
        &self,
        node_index: usize,
        target: &PointRowWithId,
        radius: f64,
        depth: usize,
        metric: DistanceMetric,
        result: &mut Vec<i32>,
    ) {
        if node_index >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_index];
        let dist = Self::point_dist(metric, target, &node.point);

        if dist <= radius {
            result.push(node.point.id);
        }

        let axis = depth % node.point.coords.ncols();
        let diff = target.coords[axis] - node.point.coords[axis];

        if Self::axis_diff_ok(diff, radius) {
            if let Some(left) = node.left {
                self.radius_search_impl(left, target, radius, depth + 1, metric, result);
            }
            if let Some(right) = node.right {
                self.radius_search_impl(right, target, radius, depth + 1, metric, result);
            }
        } else if diff < 0.0 {
            if let Some(left) = node.left {
                self.radius_search_impl(left, target, radius, depth + 1, metric, result);
            }
        } else {
            if let Some(right) = node.right {
                self.radius_search_impl(right, target, radius, depth + 1, metric, result);
            }
        }
    }

    #[allow(dead_code)]
    // Not currently used in the library, but might be useful later?
    pub fn k_nearest_neighbors(
        &self,
        target: &PointRowWithId,
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(i32, f64)> {
        let mut heap = BinaryHeap::with_capacity(k);
        self.k_nearest_impl(0, target, k, 0, metric, &mut heap);

        let mut result: Vec<_> = heap.into_sorted_vec();
        result.reverse(); // closest first
        result.into_iter().map(|n| (n.id, n.distance_sq)).collect()
    }

    #[allow(dead_code)]
    // Not currently used in the library, but might be useful later?
    pub fn k_nearest_neighbors_batch(
        &self,
        query_points: &Mat<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> (Mat<i32>, Mat<f64>) {
        let num_queries = query_points.shape().0;
        let mut all_ids: Mat<i32> = Mat::from_fn(num_queries, k, |_i, _j| 0);
        let mut all_dists = Mat::<f64>::zeros(num_queries, k);

        for i in 0..num_queries {
            let query = PointRowWithId::new(&query_points.row(i), &-1);
            let mut heap = BinaryHeap::with_capacity(k);
            self.k_nearest_impl(0, &query, k, 0, metric, &mut heap);

            let mut nbrs: Vec<_> = heap.into_sorted_vec();
            nbrs.reverse();

            all_ids
                .row_mut(i)
                .iter_mut()
                .zip(nbrs.iter())
                .for_each(|(slot, n)| *slot = n.id);

            all_dists
                .row_mut(i)
                .iter_mut()
                .zip(nbrs.iter())
                .for_each(|(slot, n)| *slot = n.distance_sq);
        }
        (all_ids, all_dists)
    }

    // Not currently used in the library, but might be useful later?
    fn k_nearest_impl(
        &self,
        node_index: usize,
        target: &PointRowWithId,
        k: usize,
        depth: usize,
        metric: DistanceMetric,
        heap: &mut BinaryHeap<Neighbour>,
    ) {
        if node_index >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_index];
        let dist = Self::point_dist(metric, target, &node.point);

        if heap.len() < k {
            heap.push(Neighbour {
                distance_sq: dist,
                id: node.point.id,
            });
        } else if dist < heap.peek().unwrap().distance_sq {
            heap.pop();
            heap.push(Neighbour {
                distance_sq: dist,
                id: node.point.id,
            });
        }

        let axis = depth % node.point.coords.ncols();
        let diff = target.coords[axis] - node.point.coords[axis];

        let (near_idx, far_idx) = if diff < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(near) = near_idx {
            self.k_nearest_impl(near, target, k, depth + 1, metric, heap);
        }

        if let Some(far) = far_idx {
            if heap.len() < k || Self::axis_diff_ok(diff, heap.peek().unwrap().distance_sq) {
                self.k_nearest_impl(far, target, k, depth + 1, metric, heap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat, RowRef};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn make_query_point(point_row: RowRef<f64>) -> PointRowWithId {
        PointRowWithId::new(&point_row, &-1)
    }

    fn brute_force_radius_ids(
        points: &Mat<f64>,
        target: &PointRowWithId,
        radius: f64,
        metric: DistanceMetric,
    ) -> Vec<i32> {
        let mut ids = Vec::new();
        for i in 0..points.nrows() {
            let p = PointRowWithId::new(&points.row(i), &(i as i32));
            let d = match metric {
                DistanceMetric::Euclidean => p.distance_euclidean(&target),
                DistanceMetric::InfinityNorm => p.distance_inf(&target),
            };
            if d <= radius {
                ids.push(i as i32);
            }
        }
        ids.sort_unstable();
        ids
    }

    fn random_points(n: usize, dim: usize, seed: u64) -> Mat<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Mat::from_fn(n, dim, |_, _| rng.random_range(0.0..1.0))
    }

    fn assert_same_ids(mut a: Vec<i32>, mut b: Vec<i32>) {
        a.sort_unstable();
        b.sort_unstable();
        assert!(a == b, "ID sets differ:\na={:?}\nb={:?}", a, b);
    }

    #[test]
    fn radius_search_matches_bruteforce_euclidean_1d_2d_3d() {
        for (n, d, seed, rmax) in [
            (200, 1, 42u64, 0.4),
            (300, 2, 123u64, 0.35),
            (400, 3, 999u64, 0.3),
        ] {
            let points = random_points(n, d, seed);
            let tree = KDTree::new(&points);
            let mut rng = StdRng::seed_from_u64(seed + 50);

            for _ in 0..25 {
                let q_idx = rng.random_range(0..points.nrows());
                let q = make_query_point(points.row(q_idx));
                let r = rng.random_range(0.0..rmax);

                let kd_ids = tree.radius_search(&q, r, DistanceMetric::Euclidean);
                let bf_ids = brute_force_radius_ids(&points, &q, r, DistanceMetric::Euclidean);
                assert_same_ids(kd_ids, bf_ids);
            }
        }
    }

    #[test]
    fn radius_search_matches_bruteforce_infnorm_1d_2d_3d() {
        for (n, d, seed) in [(150, 1, 1u64), (200, 2, 2u64), (250, 3, 3u64)] {
            let points = random_points(n, d, seed);
            let tree = KDTree::new(&points);
            let mut rng = StdRng::seed_from_u64(seed + 100);

            for _ in 0..25 {
                let q_idx = rng.random_range(0..points.nrows());
                let q = make_query_point(points.row(q_idx));
                let r = rng.random_range(0.0..0.25);

                let kd_ids = tree.radius_search(&q, r, DistanceMetric::InfinityNorm);
                let bf_ids = brute_force_radius_ids(&points, &q, r, DistanceMetric::InfinityNorm);
                assert_same_ids(kd_ids, bf_ids);
            }
        }
    }

    #[test]
    fn boundary_inclusion_euclidean_and_infnorm() {
        // Construct a tiny, hand-checkable set
        // points: origin and axis-aligned points at distance exactly r
        let mut points = Mat::<f64>::zeros(4, 2);
        // 0: (0,0), 1: (0.2,0), 2: (0,0.2), 3: (0.2,0.2)
        points[(0, 0)] = 0.0; points[(0, 1)] = 0.0;
        points[(1, 0)] = 0.2; points[(1, 1)] = 0.0;
        points[(2, 0)] = 0.0; points[(2, 1)] = 0.2;
        points[(3, 0)] = 0.2; points[(3, 1)] = 0.2;

        let tree = KDTree::new(&points);
        let q = make_query_point(points.row(0));

        // Euclidean radius 0.2: should include 0,1,2 (distance 0, 0.2, 0.2). Point 3 is ~0.282842.
        let kd_ids_e = tree.radius_search(&q, 0.2, DistanceMetric::Euclidean);
        let expect_e = vec![0, 1, 2];
        assert_same_ids(kd_ids_e, expect_e);

        // Infinity norm radius 0.2: should include all 4 (max(|dx|,|dy|) <= 0.2)
        let kd_ids_i = tree.radius_search(&q, 0.2, DistanceMetric::InfinityNorm);
        let expect_i = vec![0, 1, 2, 3];
        assert_same_ids(kd_ids_i, expect_i);
    }

    #[test]
    fn empty_tree_returns_empty() {
        let points = Mat::<f64>::zeros(0, 3);
        let tree = KDTree::new(&points);
        let q = make_query_point(Mat::<f64>::zeros(1, 3).row(0));
        let out = tree.radius_search(&q, 1.0, DistanceMetric::Euclidean);
        assert!(out.is_empty());
    }

    #[test]
    fn single_point_tree_behaves() {
        let mut points = Mat::<f64>::zeros(1, 3);
        points[(0, 0)] = 0.5;
        points[(0, 1)] = 0.5;
        points[(0, 2)] = 0.5;
        let tree = KDTree::new(&points);

        let q_same = make_query_point(points.row(0));
        let ids_r0 = tree.radius_search(&q_same, 0.0, DistanceMetric::Euclidean);
        assert_same_ids(ids_r0, vec![0]);

        let ids_small = tree.radius_search(&q_same, 1e-12, DistanceMetric::Euclidean);
        assert_same_ids(ids_small, vec![0]);

        let q_far = make_query_point(Mat::<f64>::from_fn(1, 3, |_, _| 0.0).row(0));
        let ids_far = tree.radius_search(&q_far, 0.1, DistanceMetric::Euclidean);
        assert!(ids_far.is_empty());
    }

    #[test]
    fn negative_radius_returns_empty() {
        let points = random_points(10, 2, 44);
        let tree = KDTree::new(&points);
        let q = make_query_point(points.row(0));
        let out = tree.radius_search(&q, -0.1, DistanceMetric::Euclidean);
        assert!(out.is_empty());
    }

    #[test]
    fn duplicates_are_all_returned_at_zero_radius() {
        // Two identical points at index 0 and 1
        let mut points = Mat::<f64>::zeros(2, 2);
        points[(0, 0)] = 0.3; points[(0, 1)] = 0.7;
        points[(1, 0)] = 0.3; points[(1, 1)] = 0.7;

        let tree = KDTree::new(&points);
        let q = make_query_point(points.row(0));
        let out = tree.radius_search(&q, 0.0, DistanceMetric::Euclidean);
        // both should be at distance 0
        assert_same_ids(out, vec![0, 1]);
    }

    #[test]
    fn small_batch_random_queries_match_bruteforce() {
        let points = random_points(300, 3, 2025);
        let tree = KDTree::new(&points);
        let mut rng = StdRng::seed_from_u64(55);

        // choose 10 random queries and radii
        for _ in 0..10 {
            let q_idx = rng.random_range(0..points.nrows());
            let q = make_query_point(points.row(q_idx));
            let r = rng.random_range(0.0..0.35);

            for &metric in &[DistanceMetric::Euclidean, DistanceMetric::InfinityNorm] {
                let kd_ids = tree.radius_search(&q, r, metric);
                let bf_ids = brute_force_radius_ids(&points, &q, r, metric);
                assert_same_ids(kd_ids, bf_ids);
            }
        }
    }

    #[test]
    fn ids_are_valid_indices() {
        // sanity: every returned id must be a valid row index into the original matrix
        let points = random_points(200, 2, 77);
        let tree = KDTree::new(&points);
        let q = make_query_point(points.row(0));
        let ids = tree.radius_search(&q, 0.5, DistanceMetric::Euclidean);
        let set: HashSet<i32> = ids.into_iter().collect();
        assert!(set.iter().all(|&i| (i as usize) < points.nrows()));
    }
}
