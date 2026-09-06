/////////////////////////////////////////////////////////////////////////////////////////////
//
// Builds a multi-level overlapping domain decomposition hierarchy for Schwarz preconditioning.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # domain_decomposition
//!
//! Module that builds a multi-level overlapping domain decomposition (DDM) for
//! use as a Schwarz preconditioner for RBF interpolation.
//!
//! The approach is to decompose the domain into non-overlapping subdomains and
//! then expand these domains to include 'overlapping' internal points from
//! neighbouring subdomains.
//!
//! # References
//! 1.  R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis
//!     function interpolation equations: domain decomposition methods. SIAM J. Sci.
//!     Comput., 22(5):1717–1740 (electronic), 2000.
//! 2.  Haase, G., Martin, D., Schiffmann, P., Offner, G. (2018). A Domain Decomposition
//!     Multilevel Preconditioner for Interpolation with Radial Basis Functions.
//!     In: Lirkov, I., Margenov, S. (eds) Large-Scale Scientific Computing. LSSC 2017.

use rayon::prelude::*;
use std::{collections::{HashMap, VecDeque}, sync::Arc};

use crate::{
    common, config::DDMParams, domain::Domain, global_trend::GlobalTrendTransform,
    interpolant_config::InterpolantSettings, rtree,
};

use faer::{Mat, MatRef, RowRef};
use ferreus_rbf_utils;

/// A single level in the DDM.
pub struct Level {
    /// Union of internal point indices for all domains
    /// in the level.
    pub point_indices: Vec<usize>,

    /// Overlapping leaf domains.
    pub leaf_domains: Vec<Domain>,
}

impl Level {
    /// Creates a level with the given active point ids.
    fn new(point_indices: &Vec<usize>) -> Self {
        Self {
            point_indices: point_indices.clone(),
            leaf_domains: Vec::new(),
        }
    }
}

/// A multi-level DDM hierarchy from finest (index 0) to coarsest (last).
pub struct DDMTree {
    // Levels ordered from finest -> coarsest.
    pub levels: Vec<Level>,
}

impl DDMTree {
    /// Builds a DDM hierarchy over `points`.
    pub fn new(
        points: MatRef<f64>,
        interpolant_settings: &Arc<InterpolantSettings>,
        ddm_params: DDMParams,
        global_trend: &Option<GlobalTrendTransform>,
    ) -> Self {
        let (num_points, points_ncols) = points.shape();

        let mut levels: Vec<Level> = Vec::new();

        let dimensions = points_ncols;

        let mut active_point_indices: Vec<usize> = (0..num_points).into_iter().collect();

        let num_level_reductions = get_num_level_reductions(
            active_point_indices.len(),
            ddm_params.coarse_threshold,
            ddm_params.coarse_reduction_factor,
        );

        while active_point_indices.len() > ddm_params.coarse_threshold {
            let mut root = Domain::new(active_point_indices.clone());
            root.internal_points_mask = vec![true; active_point_indices.len()];

            let overlapping_points =
                ferreus_rbf_utils::select_mat_rows(points, &root.overlapping_point_indices);

            root.extents = ferreus_rbf_utils::get_pointarray_extents(overlapping_points.as_ref());

            let mut active_domains: VecDeque<Domain> = VecDeque::new();

            active_domains.push_front(root);

            let mut fine_level = Level::new(&active_point_indices);
            let mut level_course_points: Vec<usize> = Vec::new();
            let mut owner_map: HashMap<usize, usize> = HashMap::new();

            while active_domains.len() > 0 {
                let current_domain = active_domains.pop_front().unwrap();
                let current_indices = &current_domain.overlapping_point_indices;
                let num_domain_points = current_indices.len();

                let current_points = ferreus_rbf_utils::select_mat_rows(points, &current_indices);
                let current_points_extents =
                    ferreus_rbf_utils::get_pointarray_extents(current_points.as_ref());

                let axis_lengths: Vec<f64> = (0..dimensions as usize)
                    .into_iter()
                    .map(|idx| {
                        current_points_extents[idx + dimensions] - current_points_extents[idx]
                    })
                    .collect();

                // Split by the longest axis relative to the points within the domain.
                let split_axis = ferreus_rbf_utils::argmax(&axis_lengths, &None);
                let axis_column = current_points.col(split_axis);
                let axis_column_vec: Vec<f64> = axis_column.iter().cloned().collect();

                // Median split indices along that axis.
                let sort_indices = ferreus_rbf_utils::argsort(&axis_column_vec.as_slice());
                let sorted_indices: Vec<usize> = sort_indices
                    .iter()
                    .map(|idx| current_indices[*idx].clone())
                    .collect();

                let mid_index = axis_column.nrows() / 2;

                // Left/right children inherit the parent extents, with one boundary clamped
                // at the split plane (the midpoint coordinate).
                let mut left_indices = sorted_indices[..mid_index].to_vec();
                left_indices.sort();

                let mut right_indices = sorted_indices[mid_index..].to_vec();
                right_indices.sort();

                let mid_row_idx = sorted_indices[mid_index];
                let mid_point = points.row(mid_row_idx);

                let mut left_domain = Domain::new(left_indices.clone());
                left_domain.extents = current_domain.extents.clone();
                left_domain.extents[split_axis + dimensions] = mid_point[split_axis].clone();

                let mut right_domain = Domain::new(right_indices.clone());
                right_domain.extents = current_domain.extents.clone();
                right_domain.extents[split_axis] = mid_point[split_axis].clone();

                let mut new_domains = vec![left_domain, right_domain];

                // If splitting would still produce leaves above the threshold, keep splitting;
                // otherwise, accept as leaves.
                if num_domain_points as f64
                    >= 2.0 * ddm_params.leaf_threshold as f64
                {
                    active_domains.extend(new_domains);
                } else {
                    new_domains.iter_mut().for_each(|domain| {
                        domain.internal_points_mask =
                            vec![true; domain.overlapping_point_indices.len()];
                    });

                    for (didx, domain) in new_domains.iter().enumerate() {
                        for index in &domain.overlapping_point_indices {
                            owner_map.insert(*index, didx + fine_level.leaf_domains.len());
                        }
                    }

                    fine_level.leaf_domains.extend(new_domains);
                }
            }

            // Calculate the exact number of coarse points required for the next level.
            let remaining_level_reductions = num_level_reductions
                .saturating_sub(levels.len())
                .max(1);

            let num_level_coarse_points = get_num_level_coarse_points(
                active_point_indices.len(),
                ddm_params.coarse_threshold,
                remaining_level_reductions,
            );

            // Distribute the global coarse-point count between the leaf domains.
            let domain_internal_sizes: Vec<usize> = fine_level
                .leaf_domains
                .iter()
                .map(|domain| domain.overlapping_point_indices.len())
                .collect();

            let domain_coarse_quotas =
                get_domain_coarse_quotas(&domain_internal_sizes, num_level_coarse_points);

            // Build an R-tree over all points in the level for nearest neighbour queries.
            let rtree = rtree::build_nd_point_rtree(
                dimensions,
                points,
                &active_point_indices
            );

            // Find symmetric cross-domain k-nearest-neighbour endpoints.
            let mut overlap_endpoints: Vec<(usize, usize)> = active_point_indices
                .par_iter()
                .flat_map_iter(|source_index| {
                    let mut local_endpoints = Vec::new();

                    let source_owner = *owner_map.get(source_index).unwrap();
                    let coordinates: Vec<f64> =
                        points.row(*source_index).iter().copied().collect();

                    for (target_index, _) in
                        rtree.nearest(&coordinates, ddm_params.overlap_knn + 1)
                    {
                        if target_index == *source_index {
                            continue;
                        }

                        let target_owner = *owner_map.get(&target_index).unwrap();

                        if target_owner == source_owner {
                            continue;
                        }

                        // Add both endpoints so every cross-domain relationship is symmetric.
                        local_endpoints.push((source_owner, target_index));
                        local_endpoints.push((target_owner, *source_index));
                    }

                    local_endpoints
                })
                .collect();

            // Sorting by domain and then global point index makes the overlap deterministic.
            overlap_endpoints.par_sort_unstable();
            overlap_endpoints.dedup();

            let mut domain_overlap_indices =
                vec![Vec::new(); fine_level.leaf_domains.len()];

            for (domain_index, point_index) in overlap_endpoints {
                domain_overlap_indices[domain_index].push(point_index);
            }

            // For each leaf select its coarse points and add overlap from neighbours.
            for i in 0..fine_level.leaf_domains.len() {
                let internal_indices: Vec<usize> = fine_level.leaf_domains[i]
                    .overlapping_point_indices
                    .iter()
                    .zip(fine_level.leaf_domains[i].internal_points_mask.iter())
                    .filter_map(|(&index, &mask)| if mask { Some(index) } else { None })
                    .collect();

                let num_domain_internal_points = internal_indices.len();

                let internal_points =
                    ferreus_rbf_utils::select_mat_rows(points, &internal_indices);

                let sample_size = num_domain_internal_points.min(domain_coarse_quotas[i]);

                let mut coarse_indices = if sample_size == 0 {
                    Vec::new()
                } else {
                    let selected_indices = {
                        // Approach here is to get the closest point to the center of the domain,
                        // then use the Farthest Point Sampling algorithm to get points furthest
                        // from the center.
                        // Perhaps not super robust, but it's at least deterministic and seems to
                        // greatly speed up convergence compared with selecting random points.
                        let center = get_centroid(&internal_points);

                        let distances: Vec<f64> = internal_points
                            .row_iter()
                            .map(|row| {
                                ferreus_rbf_utils::get_distance(
                                    RowRef::from_slice(center.as_slice()),
                                    row,
                                )
                            })
                            .collect();

                        let center_index = ferreus_rbf_utils::argmin(&distances, &None);

                        common::farthest_point_sampling(
                            &internal_points,
                            &sample_size,
                            &center_index,
                        )
                    };

                    let coarse_indices: Vec<usize> = selected_indices
                        .iter()
                        .map(|idx| internal_indices[*idx])
                        .collect();

                    coarse_indices
                };

                coarse_indices.sort();

                level_course_points.extend(coarse_indices);

                let new_indices = &domain_overlap_indices[i];
                let num_overlap_points = new_indices.len();

                // Extend this leaf's overlapping set with the neighbour points and mark them as
                // non-internal.
                fine_level.leaf_domains[i]
                    .overlapping_point_indices
                    .extend(new_indices.iter().copied());

                fine_level.leaf_domains[i].internal_points_mask.extend(vec![
                    false;
                    num_overlap_points
                        as usize
                ]);
            }

            // Factorise all leaves at this level.
            fine_level.leaf_domains.par_iter_mut().for_each(|domain| {
                domain.factorise(points, interpolant_settings.clone(), false, &global_trend);
            });

            levels.push(fine_level);

            level_course_points.sort();
            level_course_points.dedup();

            assert_eq!(
                level_course_points.len(),
                num_level_coarse_points
            );

            // The per-leaf coarse selections become the next level’s active set.
            active_point_indices = level_course_points;
        }

        // Once the number of active points is less than the coarse threshold then
        // build a single domain for the coarse level.
        let mut coarse_level = Level::new(&active_point_indices);

        let mut coarse_domain = Domain::new(coarse_level.point_indices.clone());
        coarse_domain.internal_points_mask =
            vec![true; coarse_domain.overlapping_point_indices.len()];

        coarse_domain.factorise(
            points,
            interpolant_settings.clone(),
            interpolant_settings.basis_size != 0,
            &global_trend,
        );

        coarse_level.leaf_domains.push(coarse_domain);

        levels.push(coarse_level);

        Self { levels }
    }
}

/// Computes the centroid of a point matrix.
fn get_centroid(points: &Mat<f64>) -> Vec<f64> {
    let (nrows, ncols) = points.shape();

    (0..ncols)
        .map(|col| {
            let column: Vec<f64> = points.col(col).iter().map(|val| *val).collect();
            column.into_iter().sum::<f64>() / nrows as f64
        })
        .collect()
}

/// Calculates the number of level reductions required to reach the
/// direct coarse problem.
fn get_num_level_reductions(
    num_points: usize,
    coarse_threshold: usize,
    coarse_reduction_factor: usize,
) -> usize {
    assert!(coarse_threshold > 0);

    if num_points <= coarse_threshold {
        return 0;
    }

    let mut covered_points = coarse_threshold;
    let mut num_level_reductions = 0;

    while covered_points < num_points {
        covered_points = covered_points
            .saturating_mul(coarse_reduction_factor);

        num_level_reductions += 1;
    }

    num_level_reductions
}

/// Calculates the number of points required at the next coarser level.
fn get_num_level_coarse_points(
    num_active_points: usize,
    coarse_threshold: usize,
    remaining_level_reductions: usize,
) -> usize {
    let coarse_ratio =
        (coarse_threshold as f64 / num_active_points as f64)
            .powf(1.0 / remaining_level_reductions as f64);

    ((num_active_points as f64 * coarse_ratio).ceil() as usize)
        .clamp(1, num_active_points - 1)
}

/// Distributes an exact coarse-point count between domains in proportion
/// to their internal point counts.
fn get_domain_coarse_quotas(
    domain_sizes: &[usize],
    num_coarse_points: usize,
) -> Vec<usize> {
    let total_points = domain_sizes.iter().sum::<usize>();
    let num_coarse_points = num_coarse_points.min(total_points);

    if num_coarse_points == 0 {
        return vec![0; domain_sizes.len()];
    }

    let num_nonempty_domains = domain_sizes
        .iter()
        .filter(|&&size| size > 0)
        .count();

    let reserve_one_per_domain =
        num_coarse_points >= num_nonempty_domains;

    // Give each nonempty domain one coverage point when the global
    // coarse-point budget permits it.
    let mut domain_quotas: Vec<usize> = domain_sizes
        .iter()
        .map(|&size| {
            usize::from(reserve_one_per_domain && size > 0)
        })
        .collect();

    let num_reserved_points = domain_quotas.iter().sum::<usize>();
    let num_remaining_points =
        num_coarse_points - num_reserved_points;

    if num_remaining_points == 0 {
        return domain_quotas;
    }

    let total_remaining_capacity =
        total_points - num_reserved_points;

    let mut remainders =
        Vec::with_capacity(domain_sizes.len());

    // Allocate the remaining budget in proportion to the capacity
    // remaining in each domain.
    for (domain_index, (&domain_size, quota)) in domain_sizes
        .iter()
        .zip(&mut domain_quotas)
        .enumerate()
    {
        let capacity = domain_size - *quota;
        let numerator =
            num_remaining_points.saturating_mul(capacity);

        *quota += numerator / total_remaining_capacity;

        remainders.push((
            numerator % total_remaining_capacity,
            domain_index,
        ));
    }

    // Assign the points lost through integer division to the domains
    // with the largest fractional remainders.
    remainders.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
    });

    let num_unassigned_points =
        num_coarse_points - domain_quotas.iter().sum::<usize>();

    for &(_, domain_index) in remainders
        .iter()
        .take(num_unassigned_points)
    {
        domain_quotas[domain_index] += 1;
    }

    assert_eq!(
        domain_quotas.iter().sum::<usize>(),
        num_coarse_points
    );

    domain_quotas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolant_config::{InterpolantSettings, RBFKernelType};
    use faer::Mat;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::collections::HashSet;
    use std::sync::Arc;

    fn generate_interpolant_settings() -> Arc<InterpolantSettings> {
        Arc::new(InterpolantSettings::builder(RBFKernelType::Spheroidal).build())
    }

    fn generate_points(n: usize, d: usize) -> Mat<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        Mat::from_fn(n, d, |_, _| rng.random_range(0.0..1.0))
    }

    #[test]
    fn hierarchy_reductions_use_configured_factor() {
        assert_eq!(get_num_level_reductions(4_096, 4_096, 128), 0);
        assert_eq!(get_num_level_reductions(524_288, 4_096, 128), 1);
        assert_eq!(get_num_level_reductions(524_289, 4_096, 128), 2);
        assert_eq!(get_num_level_reductions(32_768, 4_096, 8), 1);
        assert_eq!(get_num_level_reductions(32_769, 4_096, 8), 2);
    }

    fn run_union_test(dim: usize) {
        let params = DDMParams {
            leaf_threshold: 5,
            overlap_knn: 2,
            coarse_reduction_factor: 128,
            coarse_threshold: 10,
        };
        let interpolant_settings = generate_interpolant_settings();

        let n = 100;
        let points = generate_points(n, dim);
        let ddm = DDMTree::new(points.as_ref(), &interpolant_settings, params, &None);

        for (lvl_idx, level) in ddm.levels.iter().enumerate() {
            let mut union_internal: Vec<usize> = level
                .leaf_domains
                .iter()
                .flat_map(|dom| {
                    dom.overlapping_point_indices
                        .iter()
                        .zip(&dom.internal_points_mask)
                        .filter_map(|(&gi, &mask)| mask.then_some(gi))
                })
                .collect();
            union_internal.sort_unstable();

            let expected = level.point_indices.clone();

            assert_eq!(
                union_internal, expected,
                "dim={dim} level={lvl_idx}: union(internal) must equal level.point_indices"
            );
        }
    }

    #[test]
    fn union_match_1d() {
        run_union_test(1);
    }
    #[test]
    fn union_match_2d() {
        run_union_test(2);
    }
    #[test]
    fn union_match_3d() {
        run_union_test(3);
    }

    fn run_disjointness_test(dim: usize) {
        let params = DDMParams {
            leaf_threshold: 8,
            overlap_knn: 2,
            coarse_reduction_factor: 128,
            coarse_threshold: 12,
        };
        let interpolant_settings = generate_interpolant_settings();
        let points = generate_points(96, dim);
        let ddm = DDMTree::new(points.as_ref(), &interpolant_settings, params, &None);

        for (lvl_idx, level) in ddm.levels.iter().enumerate() {
            let mut seen = HashSet::<usize>::new();
            for (d_idx, dom) in level.leaf_domains.iter().enumerate() {
                for (li, &gi) in dom.overlapping_point_indices.iter().enumerate() {
                    if dom.internal_points_mask[li] {
                        assert!(
                            seen.insert(gi),
                            "dim={dim} level={lvl_idx} domain={d_idx}: internal index {gi} appears in multiple domains"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn disjoint_internal_1d() {
        run_disjointness_test(1);
    }
    #[test]
    fn disjoint_internal_2d() {
        run_disjointness_test(2);
    }
    #[test]
    fn disjoint_internal_3d() {
        run_disjointness_test(3);
    }

    fn run_overlap_bound_test(dim: usize) {
        let params = DDMParams {
            leaf_threshold: 9,
            overlap_knn: 2,
            coarse_reduction_factor: 128,
            coarse_threshold: 14,
        };
        let interpolant_settings = generate_interpolant_settings();
        let points = generate_points(80, dim);
        let ddm = DDMTree::new(points.as_ref(), &interpolant_settings, params.clone(), &None);

        if let Some(lvl0) = ddm.levels.first() {
            for dom in &lvl0.leaf_domains {
                let internal = dom.internal_points_mask.iter().filter(|&&b| b).count();
                let overlap = dom.internal_points_mask.len() - internal;
                let bound = internal * params.overlap_knn;

                assert!(
                    overlap <= bound,
                    "dim={dim}: overlap {overlap} exceeds bound {bound} for a leaf"
                );

                if let Some(first_false) = dom.internal_points_mask.iter().position(|&b| !b) {
                    assert!(
                        dom.internal_points_mask[first_false..].iter().all(|&b| !b),
                        "dim={dim}: non-internal entries should be appended at the tail"
                    );
                }
            }
        }
    }

    #[test]
    fn overlap_bound_1d() {
        run_overlap_bound_test(1);
    }
    #[test]
    fn overlap_bound_2d() {
        run_overlap_bound_test(2);
    }
    #[test]
    fn overlap_bound_3d() {
        run_overlap_bound_test(3);
    }

    fn run_monotone_levels_and_coarse_test(dim: usize) {
        let params = DDMParams {
            leaf_threshold: 8,
            overlap_knn: 2,
            coarse_reduction_factor: 128,
            coarse_threshold: 16,
        };
        let interpolant_settings = generate_interpolant_settings();
        let points = generate_points(96, dim);
        let ddm = DDMTree::new(points.as_ref(), &interpolant_settings, params, &None);

        // active set shrinks and is subset of previous
        for w in ddm.levels.windows(2) {
            let a = &w[0].point_indices;
            let b = &w[1].point_indices;
            assert!(b.len() <= a.len(), "dim={dim}: next level must not grow");
            let aset: HashSet<_> = a.iter().cloned().collect();
            assert!(
                b.iter().all(|gi| aset.contains(gi)),
                "dim={dim}: next level indices must be subset of previous"
            );
        }

        // coarse level invariants
        let coarse = ddm.levels.last().expect("must have at least one level");
        assert_eq!(
            coarse.leaf_domains.len(),
            1,
            "dim={dim}: coarse level should have exactly one domain"
        );
        let coarse_dom = &coarse.leaf_domains[0];
        assert!(
            coarse_dom.internal_points_mask.iter().all(|&b| b),
            "dim={dim}: coarse domain should mark all overlapping points as internal"
        );

        // point_indices sorted & unique
        for (lvl_idx, level) in ddm.levels.iter().enumerate() {
            let mut sorted = level.point_indices.clone();
            let mut uniq = level.point_indices.clone();
            sorted.sort_unstable();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(
                level.point_indices.len(),
                uniq.len(),
                "dim={dim} level={lvl_idx}: point_indices must be unique"
            );
            assert_eq!(
                level.point_indices, sorted,
                "dim={dim} level={lvl_idx}: point_indices should be sorted"
            );
        }
    }

    #[test]
    fn monotone_and_coarse_1d() {
        run_monotone_levels_and_coarse_test(1);
    }
    #[test]
    fn monotone_and_coarse_2d() {
        run_monotone_levels_and_coarse_test(2);
    }
    #[test]
    fn monotone_and_coarse_3d() {
        run_monotone_levels_and_coarse_test(3);
    }

    #[test]
    fn threshold_short_circuit_coarse_only() {
        let interpolant_settings = generate_interpolant_settings();
        let points = generate_points(25, 2);
        let params = DDMParams {
            leaf_threshold: 8,
            overlap_knn: 2,
            coarse_reduction_factor: 128,
            coarse_threshold: 25, // >= npoints -> no fine levels
        };
        let ddm = DDMTree::new(points.as_ref(), &interpolant_settings, params, &None);
        assert_eq!(ddm.levels.len(), 1, "only coarse level expected");
        assert_eq!(
            ddm.levels[0].leaf_domains.len(),
            1,
            "single coarse domain expected"
        );
    }
}
