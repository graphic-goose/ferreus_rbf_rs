/////////////////////////////////////////////////////////////////////////////////////////////
//
// Constructs linear Morton-encoded trees used as the spatial hierarchy for BBFMM.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    bbfmm::{Dimensions, FmmError, TreeLists},
    morton, morton_constants,
};
use crate::{EvaluationTargets, target_refinement::TargetRefinement};
use faer::MatRef;
use rayon::prelude::*;

pub fn build_tree(
    points: MatRef<f64>,
    center: &Vec<f64>,
    radius: f64,
    max_points_per_cell: usize,
    store_empty_leaves: bool,
    depth: &mut u64,
    dimensions: Dimensions,
    interpolation_order: usize,
    targets: Option<EvaluationTargets<'_>>,
) -> TreeLists {
    let displacement: Vec<f64> = center.iter().map(|&c| c - radius).collect();

    let refinement = if store_empty_leaves {
        targets.map(|targets| {
            TargetRefinement::new(
                points,
                center,
                radius,
                dimensions,
                interpolation_order,
                max_points_per_cell,
                targets,
            )
        })
    } else {
        None
    };
    let mut all_nodes = HashSet::from([0]);
    let mut leaf_nodes = HashSet::new();
    let mut children = HashMap::new();
    let mut level_cells_map = HashMap::from([(0, vec![0])]);
    let mut cells_point_indices: HashMap<u64, Vec<usize>> =
        HashMap::from([(0, (0..points.nrows()).collect())]);
    let mut leaf_source_indices = HashMap::new();
    let mut key_to_index_map = HashMap::from([(0, 0)]);
    let mut active_cells = VecDeque::from([0]);

    let mut current_level = 0;

    while !active_cells.is_empty() {
        let mut next_level_cells = HashSet::new();
        let child_level = current_level + 1;
        let side_length = morton::get_side_length(radius, child_level);

        while let Some(cell) = active_cells.pop_front() {
            let mut cell_children = HashSet::new();

            if let Some(cell_points) = cells_point_indices.get(&cell).cloned() {
                for &i in cell_points.iter() {
                    let point = points.row(i);
                    let anchor =
                        morton::point_to_anchor(point, child_level, &displacement, side_length);
                    let key = morton::encode_morton_point(anchor, &dimensions);
                    cell_children.insert(key);
                    cells_point_indices
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }

            let active_children: Vec<u64> = match store_empty_leaves {
                true => {
                    let all_children = morton::get_children(cell, &dimensions);
                    all_children
                }
                false => cell_children.iter().copied().collect(),
            };

            for &child in &active_children {
                all_nodes.insert(child);
                key_to_index_map.entry(child).or_insert(all_nodes.len() - 1);

                children.entry(child).or_insert_with(Vec::new);
                level_cells_map
                    .entry(child_level)
                    .or_insert_with(Vec::new)
                    .push(child);

                let source_count = cells_point_indices.get(&child).map_or(0, Vec::len);

                let split = source_count > max_points_per_cell
                    || refinement
                        .as_ref()
                        .is_some_and(|plan| plan.should_split(child));
                if split && child_level < morton_constants::MAXIMUM_LEVEL {
                    next_level_cells.insert(child);
                } else {
                    leaf_nodes.insert(child);

                    if source_count > 0 {
                        leaf_source_indices.insert(child, cells_point_indices[&child].clone());
                    }
                }
            }

            children.insert(cell, active_children.clone());
        }

        if !next_level_cells.is_empty() {
            active_cells.extend(next_level_cells);
            current_level += 1;
        }
    }

    let (u_lists, v_lists, source_key_to_index_map) = get_interaction_lists(
        &all_nodes,
        &leaf_nodes,
        &children,
        &leaf_source_indices,
        center,
        radius,
        &dimensions,
    );

    *depth = current_level + 1;

    TreeLists {
        tree: all_nodes,
        leaves: leaf_nodes,
        children,
        u_lists,
        v_lists,
        level_cells_map,
        key_to_index_map,
        source_key_to_index_map,
        leaf_source_indices,
        leaf_target_indices: HashMap::new(),
    }
}

pub fn get_interaction_lists(
    complete_tree: &HashSet<u64>,
    leaves_set: &HashSet<u64>,
    children: &HashMap<u64, Vec<u64>>,
    leaf_sources: &HashMap<u64, Vec<usize>>,
    tree_center: &[f64],
    tree_radius: f64,
    dimensions: &Dimensions,
) -> (
    HashMap<u64, HashSet<u64>>,
    HashMap<u64, HashSet<u64>>,
    HashMap<u64, usize>,
) {
    // Definitions
    // -----------

    // colleagues
    // ----------
    // - For any cell, B, its colleagues are defined as the adjacent cells that are in the same tree level.

    // u_list
    // ------
    // - Only defined for leaf cells.
    // - For a leaf cell, B, the u_list of B contains the source leaf cells whose source points must be evaluated
    //   directly at B's target points.
    // - The u_list for a cell contains:
    //   - B itself, if it contains sources.
    //   - Source leaf cells adjacent to B.
    //   - All source leaf descendants of B's same-level colleagues, including descendants that are not adjacent to B.
    //   - Coarser source leaf cells adjacent to any ancestor of B. These cells are inherited from the ancestor's direct list.
    // - A cell is defined as 'adjacent' to B if they share a vertex, edge or face.
    // - Compute the interaction of U's source points with B's target points, including those handled by the X and W lists
    //   in the traditional adaptive FMM scheme. Testing proved this to be significantly more efficient than P2L and M2P.

    // v_list
    // ------
    // - The v_list of a cell, B (leaf OR non leaf), consists of those children of the colleagues of B's
    // parent cell, P(B), which are not adjacent to B.
    // - Compute the interaction from V to B using M2L translation, since two boxes are well-separated.

    // ┌───────────────────────────────────────┐───────────────────┐───────────────────┐───────────────────┐───────────────────┐
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                                       |         V         |         V         |         V         |         V         |
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                   U                   |───────────────────|───────────────────|───────────────────|───────────────────|
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                                       |         U         |         U         |         V         |         V         |
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |                                       |                   |                   |                   |                   |
    // |───────────────────┐───────────────────│───────────────────│───────────────────│───────────────────────────────────────│
    // |                   |                   │                   │                   │                                       |
    // |                   |                   │                   │                   │                                       |
    // |        V          |          U        │         B         │         U         │                                       |
    // |                   |                   │                   │                   │                                       |
    // |                   |                   │                   │                   │                                       |
    // |                   |                   │                   │                   │                                       |
    // |───────────────────|───────────────────│─────────┐────┐────┐────┐────┐─────────┐                   U                   |
    // |                   |                   │         │ U  │ U  │ U  │ U  │         │                                       |
    // |                   |                   │    U    │────│────│────│────│    U    │                                       |
    // |                   |                   │         │ U  │ U  │ U  │ U  │         │                                       |
    // |        V          |         U         │─────────│────┘────┘────┘────│─────────│                                       |
    // |                   |                   │         │         │         │         │                                       |
    // |                   |                   │    U    │    U    │    U    │    U    │                                       |
    // |                   |                   │         │         │         │         │                                       |
    // │───────────────────|───────────────────│─────────└─────────│─────────└─────────│───────────────────────────────────────│
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |         V         |         V         |         V         |         V         |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |───────────────────|───────────────────|───────────────────|───────────────────|                  U                    |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |        V          |          V        |        V          |          V        |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // └───────────────────└───────────────────┘───────────────────└───────────────────┘───────────────────────────────────────┘

    let source_cells: HashSet<u64> = leaf_sources
        .iter()
        .filter(|(_, indices)| !indices.is_empty())
        .flat_map(|(key, _)| morton::get_ancestors(*key, dimensions))
        .collect();

    let lists: Vec<_> =
        complete_tree
            .par_iter()
            .map(|&key| {
                let mut direct = HashSet::new();
                let mut far = HashSet::new();

                if let Some(parent) = morton::get_parent(key, dimensions) {
                    for peer in morton::get_neighbours(parent, dimensions) {
                        if let Some(peer_children) = children.get(&peer) {
                            for &source in peer_children {
                                if source_cells.contains(&source)
                                    && !morton::are_adjacent(
                                        key,
                                        source,
                                        &tree_center,
                                        tree_radius,
                                        dimensions,
                                    )
                                {
                                    far.insert(source);
                                }
                            }
                        }
                    }
                }
                let mut colleagues = morton::get_neighbours(key, dimensions);
                colleagues.push(key);

                if leaves_set.contains(&key) {
                    let mut pending: Vec<u64> = colleagues
                        .into_iter()
                        .filter(|source| source_cells.contains(source))
                        .collect();

                    while let Some(source) = pending.pop() {
                        if leaves_set.contains(&source) {
                            direct.insert(source);
                        } else if let Some(source_children) = children.get(&source) {
                            pending.extend(
                                source_children
                                    .iter()
                                    .copied()
                                    .filter(|child| source_cells.contains(child)),
                            );
                        }
                    }
                } else {
                    direct.extend(colleagues.into_iter().filter(|source| {
                        leaves_set.contains(source) && source_cells.contains(source)
                    }));
                }
                (key, direct, far)
            })
            .collect();

    let mut leaf_directs = Vec::with_capacity(leaves_set.len());
    let mut inherited_directs = HashMap::new();
    let mut v_lists = HashMap::new();

    for (key, direct, far) in lists {
        if leaves_set.contains(&key) {
            leaf_directs.push((key, direct));
        } else if !direct.is_empty() {
            inherited_directs.insert(key, direct);
        }
        if !far.is_empty() {
            v_lists.insert(key, far);
        }
    }

    let u_lists = leaf_directs
        .into_par_iter()
        .filter_map(|(leaf, mut direct)| {
            let mut ancestor = morton::get_parent(leaf, dimensions);

            while let Some(key) = ancestor {
                if let Some(inhereted) = inherited_directs.get(&key) {
                    direct.extend(inhereted.iter().copied());
                }
                ancestor = morton::get_parent(key, dimensions);
            }
            if direct.is_empty() {
                None
            } else {
                Some((leaf, direct))
            }
        })
        .collect();

    let mut source_cells: Vec<u64> = source_cells.iter().cloned().collect();
    source_cells.sort_unstable();

    let source_key_to_index_map = source_cells
        .into_iter()
        .enumerate()
        .map(|(idx, key)| (key, idx))
        .collect();

    (u_lists, v_lists, source_key_to_index_map)
}

pub fn points_to_keys(
    points: MatRef<f64>,
    leaves_set: &HashSet<u64>,
    depth: u64,
    center: &[f64],
    radius: f64,
    dimensions: &Dimensions,
) -> Result<Vec<u64>, FmmError> {
    let side_length = morton::get_side_length(radius, depth);
    let displacement: Vec<f64> = center.iter().map(|&c| c - radius).collect();

    points
        .par_row_iter()
        .enumerate()
        .map(|(idx, point)| {
            let anchor = morton::point_to_anchor(point, depth, &displacement, side_length);
            let mut current_key = morton::encode_morton_point(anchor, &dimensions);

            while !leaves_set.contains(&current_key) {
                current_key = morton::get_parent(current_key, &dimensions)
                    .ok_or(FmmError::PointOutsideTree { point_index: idx })?;
            }

            Ok(current_key)
        })
        .collect()
}

pub fn get_points_to_leaves_map(point_keys: &Vec<u64>) -> HashMap<u64, Vec<usize>> {
    let mut indices_map: HashMap<u64, Vec<usize>> = HashMap::new();

    for (i, &value) in point_keys.iter().enumerate() {
        indices_map.entry(value).or_default().push(i);
    }

    indices_map.iter_mut().for_each(|(_k, v)| {
        v.sort();
    });

    indices_map
}
