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

use super::{bbfmm::{Dimensions, TreeLists, FmmError}, morton, morton_constants};
use faer::Mat;
use rayon::prelude::*;

pub fn build_tree(
    points: &Mat<f64>,
    center: &Vec<f64>,
    radius: &f64,
    max_points_per_cell: &usize,
    store_empty_leaves: &bool,
    depth: &mut u64,
    dimensions: &Dimensions,
    adaptive_tree: &bool,
) -> TreeLists {
    let displacement: Vec<f64> = center.iter().map(|&c| c - radius).collect();
    let n_points = points.nrows() as f64;
    let optimal_depth = (n_points.log2() / *dimensions as isize as f64).ceil() as u64;

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
        let side_length = morton::get_side_length(radius, &child_level);
        let mut any_child_exceeds = false;

        while let Some(cell) = active_cells.pop_front() {
            let mut cell_children = HashSet::new();

            if let Some(cell_points) = cells_point_indices.get(&cell).cloned() {
                for &i in cell_points.iter() {
                    let point = points.row(i);
                    let anchor =
                        morton::point_to_anchor(point, &child_level, &displacement, &side_length);
                    let key = morton::encode_morton_point(anchor, dimensions);
                    cell_children.insert(key);
                    cells_point_indices
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(i);
                }
            }

            let active_children: Vec<u64> = match *store_empty_leaves {
                true => {
                    let all_children = morton::get_children(&cell, dimensions);
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

                if let Some(child_points) = cells_point_indices.get(&child) {
                    if *adaptive_tree {
                        if child_points.len() > *max_points_per_cell
                            && child_level < morton_constants::MAXIMUM_LEVEL
                        {
                            next_level_cells.insert(child);
                        } else {
                            leaf_nodes.insert(child);
                            leaf_source_indices
                                .entry(child)
                                .or_insert_with(Vec::new)
                                .extend(child_points.clone());
                        }
                    } else if child_points.len() > *max_points_per_cell {
                        any_child_exceeds = true;
                    }
                } else if *adaptive_tree && *store_empty_leaves {
                    leaf_nodes.insert(child);
                }
            }

            children.insert(cell, active_children.clone());

            if !*adaptive_tree {
                next_level_cells.extend(active_children);
            }
        }

        let should_subdivide = *adaptive_tree
            || (any_child_exceeds
                && child_level < morton_constants::MAXIMUM_LEVEL
                && child_level < optimal_depth);

        if should_subdivide && !next_level_cells.is_empty() {
            active_cells.extend(next_level_cells);
            current_level += 1;
        } else if !*adaptive_tree {
            for &leaf in &next_level_cells {
                if let Some(indices) = cells_point_indices.get(&leaf) {
                    leaf_source_indices.entry(leaf).or_insert(indices.clone());
                }
            }
            leaf_nodes.extend(next_level_cells);
        }
    }

    let (u_lists, v_lists, x_lists, w_lists) = match *adaptive_tree {
        true => {
            let (u_lists, v_lists, x_lists, w_lists) = get_interaction_lists_adaptive(
                &all_nodes,
                &leaf_nodes,
                &center,
                &radius,
                &dimensions,
            );

            (u_lists, v_lists, Some(x_lists), Some(w_lists))
        }
        false => {
            let (u_lists, v_lists) = get_interaction_lists_regular(
                &all_nodes,
                &leaf_nodes,
                &cells_point_indices,
                &children,
                &center,
                &radius,
                &dimensions,
            );

            (u_lists, v_lists, None, None)
        }
    };

    *depth = current_level + 1;

    TreeLists {
        tree: all_nodes,
        leaves: leaf_nodes,
        children,
        u_lists,
        v_lists,
        x_lists,
        w_lists,
        level_cells_map,
        key_to_index_map,
        leaf_source_indices,
        leaf_target_indices: HashMap::new(),
    }
}

pub fn get_interaction_lists_adaptive(
    complete_tree: &HashSet<u64>,
    leaves_set: &HashSet<u64>,
    tree_center: &Vec<f64>,
    tree_radius: &f64,
    dim: &Dimensions,
) -> (
    HashMap<u64, HashSet<u64>>,
    HashMap<u64, HashSet<u64>>,
    HashMap<u64, HashSet<u64>>,
    HashMap<u64, HashSet<u64>>,
) {
    // Definitions
    // -----------

    // colleagues
    // ----------
    // - For any cell, B, its colleagues are defined as the adjacent cells that are in the same tree level.

    // u_list
    // ------
    // - Only defined for leaf cells.
    // - For a leaf cell, B, the u_list of B consists of all leaf cells adjacent to B, including B itself.
    // - A cell is defined as 'adjacent' to B if they share a vertex, edge or face.
    // - Direct computation of the interaction of U's source points with B's target points is necessary since U and B are adjacent.

    // v_list
    // ------
    // - The v_list of a cell, B (leaf OR non leaf), consists of those children of the colleagues of B's
    // parent cell, P(B), which are not adjacent to B.
    // - Compute the interaction from V to B using M2L translation, since two boxes are well-separated.

    // w_list
    // ------
    // - The w_list is only created for a leaf cell, B, and contains a cell, C, if and ONLY if:
    //     - C is a descendant of a colleague of B
    //     - C is not adjacent to Bl
    //     - The parent of C is adjacent to B.
    // - Evaluate directly at B's target points using the multipole coefficients of W, as B is in the far range of W.

    // x_list
    // ------
    // - The x_list of a cell B, consists of those cells, C, which have B on their w_list.
    // - Evaluate at B's Chebyshev nodes using X's sources points.

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
    // |───────────────────|───────────────────│─────────┐────┐────┐────┐────┐─────────┐                   X                   |
    // |                   |                   │         │ U  │ U  │ U  │ W  │         │                                       |
    // |                   |                   │    U    │────│────│────│────│    W    │                                       |
    // |                   |                   │         │ W  │ W  │ W  │ W  │         │                                       |
    // |        V          |         U         │─────────│────┘────┘────┘────│─────────│                                       |
    // |                   |                   │         │         │         │         │                                       |
    // |                   |                   │    W    │    W    │    W    │     W   │                                       |
    // |                   |                   │         │         │         │         │                                       |
    // │───────────────────|───────────────────│─────────└─────────│─────────└─────────│───────────────────────────────────────│
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |         V         |         V         |         V         |         V         |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |───────────────────|───────────────────|───────────────────|───────────────────|                  X                    |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |        V          |          V        |        V          |          V        |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // |                   |                   |                   |                   |                                       |
    // └───────────────────└───────────────────┘───────────────────└───────────────────┘───────────────────────────────────────┘

    // Process in parallel and collect results for each list
    let results: Vec<(u64, HashSet<u64>, HashSet<u64>, HashSet<u64>)> = complete_tree
        .par_iter()
        .map(|key| {
            let mut cell_u_list: HashSet<u64> = HashSet::new();
            let mut cell_v_list: HashSet<u64> = HashSet::new();
            let mut cell_w_list: HashSet<u64> = HashSet::new();

            if let Some(parent) = morton::get_parent(&key, &dim) {
                let parent_colleagues = morton::get_neighbours(&parent, dim);

                let parent_colleagues_children: Vec<u64> = parent_colleagues
                    .iter()
                    .flat_map(|col| morton::get_children(col, dim))
                    .collect();

                parent_colleagues_children
                    .iter()
                    .filter(|pcc| {
                        complete_tree.contains(pcc)
                            && !morton::are_adjacent(&key, &pcc, &tree_center, &tree_radius, &dim)
                    })
                    .for_each(|pcc| {
                        cell_v_list.insert(*pcc);
                    });

                if leaves_set.contains(&key) {
                    let colleagues = morton::get_neighbours(&key, &dim);
                    let colleagues_children: Vec<u64> = colleagues
                        .iter()
                        .flat_map(|col| morton::get_children(&col, &dim))
                        .collect();

                    let mut colleagues_ancestors: VecDeque<u64> =
                        colleagues.iter().cloned().collect();

                    let mut visited_cells: HashSet<u64> = HashSet::new();

                    while let Some(current_cell) = colleagues_ancestors.pop_front() {
                        // Skip cells that have already been processed
                        if !visited_cells.insert(current_cell) {
                            continue;
                        }

                        if morton::are_adjacent(
                            &key,
                            &current_cell,
                            &tree_center,
                            &tree_radius,
                            &dim,
                        ) {
                            if leaves_set.contains(&current_cell) {
                                cell_u_list.insert(current_cell);
                            } else {
                                if let Some(parent) = morton::get_parent(&current_cell, &dim) {
                                    colleagues_ancestors.push_back(parent);
                                }
                            }
                        }
                    }

                    let mut colleagues_descendants: VecDeque<u64> = colleagues_children
                        .iter()
                        .filter(|child| complete_tree.contains(child))
                        .cloned()
                        .collect();

                    while let Some(current_cell) = colleagues_descendants.pop_front() {
                        if morton::are_adjacent(
                            &key,
                            &current_cell,
                            &tree_center,
                            &tree_radius,
                            &dim,
                        ) {
                            // Adjacent cells to the key go to the u_list
                            if leaves_set.contains(&current_cell) {
                                cell_u_list.insert(current_cell);
                            } else {
                                // Expand non-leaf adjacent cells
                                let next_level_children: Vec<u64> =
                                    morton::get_children(&current_cell, &dim)
                                        .iter()
                                        .filter(|child| complete_tree.contains(child))
                                        .cloned()
                                        .collect();

                                colleagues_descendants.extend(next_level_children);
                            }
                        } else {
                            // Non-adjacent cells go to the w_list (and are not traversed further)
                            cell_w_list.insert(current_cell);
                        }
                    }

                    cell_u_list.insert(*key);
                }
            }
            (*key, cell_u_list, cell_v_list, cell_w_list)
        })
        .collect();

    let mut u_lists = HashMap::new();
    let mut v_lists = HashMap::new();
    let mut w_lists = HashMap::new();
    let mut x_lists = HashMap::new();

    for (key, u, v, w) in results.iter() {
        if u.len() > 0 {
            u_lists.insert(*key, u.clone());
        }
        if v.len() > 0 {
            v_lists.insert(*key, v.clone());
        }
        if w.len() > 0 {
            w_lists.insert(*key, w.clone());
        }
    }

    w_lists.iter().for_each(|(cell, w_list)| {
        for w in w_list {
            x_lists
                .entry(*w)
                .or_insert_with(HashSet::new)
                .insert(*cell);
        }
    });

    (u_lists, v_lists, x_lists, w_lists)
}

fn get_interaction_lists_regular(
    tree: &HashSet<u64>,
    leaves: &HashSet<u64>,
    cells_points_indices: &HashMap<u64, Vec<usize>>,
    children: &HashMap<u64, Vec<u64>>,
    center: &Vec<f64>,
    radius: &f64,
    dimensions: &Dimensions,
) -> (HashMap<u64, HashSet<u64>>, HashMap<u64, HashSet<u64>>) {
    let tree_vec: Vec<u64> = tree.iter().copied().collect();

    let interactions: Vec<(u64, HashSet<u64>, HashSet<u64>)> = tree_vec
        .par_iter()
        .map(|&cell| {
            let (u_list, v_list) = compute_u_v_list(
                &cell,
                &children,
                &cells_points_indices,
                &dimensions,
                &tree,
                &leaves,
                &center,
                &radius,
            );
            (cell, u_list, v_list)
        })
        .collect();

    let mut u_lists = HashMap::new();
    let mut v_lists = HashMap::new();

    for (cell, u, v) in interactions {
        if leaves.contains(&cell) {
            u_lists.insert(cell, u);
        }

        v_lists.insert(cell, v);
    }

    (u_lists, v_lists)
}

fn compute_u_v_list(
    cell: &u64,
    children: &HashMap<u64, Vec<u64>>,
    cells_points_indices: &HashMap<u64, Vec<usize>>,
    dimensions: &Dimensions,
    tree: &HashSet<u64>,
    leaves: &HashSet<u64>,
    center: &Vec<f64>,
    radius: &f64,
) -> (HashSet<u64>, HashSet<u64>) {
    let mut u_list = HashSet::new();
    let mut v_list = HashSet::new();

    if let Some(parent) = morton::get_parent(&cell, dimensions) {
        if leaves.contains(&cell) {
            if let Some(siblings) = children.get(&parent) {
                for sib in siblings {
                    if cells_points_indices.get(&*sib).is_some() {
                        u_list.insert(sib.clone());
                    }
                }
            }
        }
        let parent_colleagues: Vec<u64> = morton::get_neighbours(&parent, &dimensions)
            .into_iter()
            .filter(|key| tree.contains(key))
            .collect();

        for pc in parent_colleagues {
            if let Some(pcc) = children.get(&pc) {
                for colleague in pcc {
                    if cells_points_indices.get(&*colleague).is_some() {
                        if morton::are_adjacent(&cell, &colleague, &center, &radius, &dimensions) {
                            if leaves.contains(&cell) {
                                u_list.insert(*colleague);
                            }
                        } else {
                            v_list.insert(*colleague);
                        }
                    }
                }
            }
        }
    }

    (u_list, v_list)
}

pub fn points_to_keys(
    points: &Mat<f64>,
    leaves_set: &HashSet<u64>,
    depth: &u64,
    center: &[f64],
    radius: &f64,
    dimensions: &Dimensions,
) -> Result<Vec<u64>, FmmError> {
    let side_length = morton::get_side_length(&radius, &depth);
    let displacement: Vec<f64> = center.iter().map(|&c| c - radius).collect();

    let results: Vec<Result<u64, FmmError>> = points
        .par_row_iter()
        .enumerate()
        .map(|(idx, point)| {
            let anchor = morton::point_to_anchor(point, &depth, &displacement, &side_length);
            let mut current_key = morton::encode_morton_point(anchor, &dimensions);

            while !leaves_set.contains(&current_key) {
                current_key = morton::get_parent(&current_key, &dimensions).ok_or(
                    FmmError::PointOutsideTree { point_index: idx },
                )?;
            }

            Ok(current_key)
        })
        .collect();

    let mut keys = Vec::with_capacity(points.nrows());
    for r in results {
        keys.push(r?);
    }

    Ok(keys)
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
