/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements Morton (Z-order) encoding, decoding, and neighbourhood queries for FMM trees.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::bbfmm::Dimensions;
use faer::RowRef;
use std::collections::HashSet;

use crate::morton_constants::{
    BYTE_DISPLACEMENT, BYTE_MASK, DIRECTION_VECTORS_1D, DIRECTION_VECTORS_2D, DIRECTION_VECTORS_3D,
    EIGHT_BIT_MASK, LEVEL_DISPLACEMENT, LEVEL_MASK, MORTON_DECODE_1D_LOOKUP,
    MORTON_DECODE_2D_X_LOOKUP, MORTON_DECODE_2D_Y_LOOKUP, MORTON_DECODE_3D_X_LOOKUP,
    MORTON_DECODE_3D_Y_LOOKUP, MORTON_DECODE_3D_Z_LOOKUP, MORTON_ENCODE_1D_LOOKUP,
    MORTON_ENCODE_2D_X_LOOKUP, MORTON_ENCODE_2D_Y_LOOKUP, MORTON_ENCODE_3D_X_LOOKUP,
    MORTON_ENCODE_3D_Y_LOOKUP, MORTON_ENCODE_3D_Z_LOOKUP, NINE_BIT_MASK,
};

/// The morton code implementation for the linear heirarchical tree is based on that of:
/// - [`AdaptOctree`](https://github.com/Excalibur-SLE/AdaptOctree)
/// - [`Libmorton`](https://github.com/Forceflow/libmorton)

/// Gets the side length of a cell for the current level.
pub fn get_side_length(radius: &f64, level: &u64) -> f64 {
    let side_length = 2.0 * radius / ((1 << level) as f64);
    side_length
}

/// Finds the 'anchor' (origin) of the cell in which a point in world coordinates lies.
pub fn point_to_anchor(
    point: RowRef<f64>,
    level: &u64,
    displacement: &Vec<f64>,
    side_length: &f64,
) -> Vec<u64> {
    let n_dims = point.ncols();

    let mut anchor = Vec::with_capacity(n_dims + 1);

    for (i, col) in point.iter().enumerate() {
        anchor.push(((col - displacement[i]) / side_length).floor() as u64);
    }
    anchor.push(*level as u64);

    anchor
}

/// Morton encode a set of anchor coordinates and their octree level. Assume a
/// maximum of 16 bits for each anchor coordinate, and 15 bits for level.
/// The strategy is to examine each coordinate byte by byte, from most to
/// least significant bytes, and find interleaving using the lookup table.
/// Finally, level information is appended to the tail.
pub fn encode_morton_point(anchor: Vec<u64>, dimensions: &Dimensions) -> u64 {
    let level = anchor[*dimensions as usize];
    let mut morton_code = 0;

    match dimensions {
        Dimensions::One => {
            let x = anchor[0];

            morton_code |= MORTON_ENCODE_1D_LOOKUP[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

            morton_code <<= 8;

            morton_code |= MORTON_ENCODE_1D_LOOKUP[(x & BYTE_MASK) as usize];

            morton_code <<= LEVEL_DISPLACEMENT;

            morton_code |= level;
        }
        Dimensions::Two => {
            let x = anchor[0];
            let y = anchor[1];

            morton_code |=
                MORTON_ENCODE_2D_Y_LOOKUP[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];
            morton_code |=
                MORTON_ENCODE_2D_X_LOOKUP[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

            morton_code <<= 16;

            morton_code |= MORTON_ENCODE_2D_Y_LOOKUP[(y & BYTE_MASK) as usize];
            morton_code |= MORTON_ENCODE_2D_X_LOOKUP[(x & BYTE_MASK) as usize];

            morton_code <<= LEVEL_DISPLACEMENT;

            morton_code |= level;
        }
        Dimensions::Three => {
            let x = anchor[0];
            let y = anchor[1];
            let z = anchor[2];

            morton_code |=
                MORTON_ENCODE_3D_Z_LOOKUP[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];
            morton_code |=
                MORTON_ENCODE_3D_Y_LOOKUP[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];
            morton_code |=
                MORTON_ENCODE_3D_X_LOOKUP[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

            morton_code <<= 24;

            morton_code |= MORTON_ENCODE_3D_Z_LOOKUP[(z & BYTE_MASK) as usize];
            morton_code |= MORTON_ENCODE_3D_Y_LOOKUP[(y & BYTE_MASK) as usize];
            morton_code |= MORTON_ENCODE_3D_X_LOOKUP[(x & BYTE_MASK) as usize];

            morton_code <<= LEVEL_DISPLACEMENT;

            morton_code |= level;
        }
    }

    morton_code
}

/// Gets the last 15 bits of a key, corresponding to a level.
pub fn get_level(key: &u64) -> u64 {
    key & LEVEL_MASK
}

/// Decode a Morton encoded key into an anchor using the provided lookup tables.
pub fn decode_key(key: &u64, dimensions: &Dimensions) -> Vec<u64> {
    let level = get_level(&key);
    let key_no_level = key >> LEVEL_DISPLACEMENT;
    let num_loops = 7;
    let mut anchor = vec![0; *dimensions as usize + 1];
    anchor[*dimensions as usize] = level;

    let start_shift: u64 = 0;

    match dimensions {
        Dimensions::One => {
            anchor[0] |= MORTON_DECODE_1D_LOOKUP[((key_no_level >> 8) & BYTE_MASK) as usize] << 8;
            anchor[0] |= MORTON_DECODE_1D_LOOKUP[(key_no_level & BYTE_MASK) as usize];
        }
        Dimensions::Two => {
            for i in 0..num_loops {
                anchor[0] |= MORTON_DECODE_2D_X_LOOKUP
                    [((key_no_level >> ((i * 8) + start_shift)) & EIGHT_BIT_MASK) as usize]
                    << (4 * i);
                anchor[1] |= MORTON_DECODE_2D_Y_LOOKUP
                    [((key_no_level >> ((i * 8) + start_shift)) & EIGHT_BIT_MASK) as usize]
                    << (4 * i);
            }
        }
        Dimensions::Three => {
            for i in 0..num_loops {
                anchor[0] |= MORTON_DECODE_3D_X_LOOKUP
                    [((key_no_level >> ((i * 9) + start_shift)) & NINE_BIT_MASK) as usize]
                    << (3 * i);
                anchor[1] |= MORTON_DECODE_3D_Y_LOOKUP
                    [((key_no_level >> ((i * 9) + start_shift)) & NINE_BIT_MASK) as usize]
                    << (3 * i);
                anchor[2] |= MORTON_DECODE_3D_Z_LOOKUP
                    [((key_no_level >> ((i * 9) + start_shift)) & NINE_BIT_MASK) as usize]
                    << (3 * i);
            }
        }
    }

    anchor
}

/// Gets the Morton key of the parent of the current key.
pub fn get_parent(key: &u64, dimensions: &Dimensions) -> Option<u64> {
    let level = get_level(key);

    if level == 0 {
        return None;
    }

    let key_no_level = key >> LEVEL_DISPLACEMENT;
    let parent_level = level - 1;

    let mut parent = match dimensions {
        Dimensions::One => key_no_level >> 1,
        Dimensions::Two => key_no_level >> 2,
        Dimensions::Three => key_no_level >> 3,
    };

    parent <<= LEVEL_DISPLACEMENT;
    parent |= parent_level;

    Some(parent)
}

// Gets the Morton key of all ancestors, including the key itself.
pub fn get_ancestors(key: &u64, dimensions: &Dimensions) -> HashSet<u64> {
    let mut current_key = *key;
    let current_level = get_level(key);

    let mut ancestors = HashSet::new();
    ancestors.insert(current_key);

    for _ in (0..current_level).rev() {
        if let Some(ancestor) = get_parent(&current_key, dimensions) {
            ancestors.insert(ancestor);
            current_key = ancestor;
        } else {
            break;
        }
    }

    ancestors
}

/// Gets the Morton keys for all potential neighbours of a cell at the same level.
/// Can include cells that are outside the tree extents.
pub fn get_neighbours(key: &u64, dimensions: &Dimensions) -> Vec<u64> {
    let level = get_level(&key);
    let max_num_boxes = 1 << level;
    let anchor = decode_key(&key, dimensions);
    let mut neighbours = Vec::new();

    match dimensions {
        Dimensions::One => {
            for direction in DIRECTION_VECTORS_1D {
                let x_d = (anchor[0] as i64) + direction;

                if x_d >= 0 && x_d < max_num_boxes {
                    let new_anchor = vec![x_d as u64, level];
                    neighbours.push(encode_morton_point(new_anchor, dimensions));
                }
            }
        }
        Dimensions::Two => {
            for direction in DIRECTION_VECTORS_2D {
                let x_d = (anchor[0] as i64) + direction[0];
                let y_d = (anchor[1] as i64) + direction[1];

                if x_d >= 0 && y_d >= 0 && x_d < max_num_boxes && y_d < max_num_boxes {
                    let new_anchor = vec![x_d as u64, y_d as u64, level];
                    neighbours.push(encode_morton_point(new_anchor, dimensions));
                }
            }
        }
        Dimensions::Three => {
            for direction in DIRECTION_VECTORS_3D {
                let x_d = (anchor[0] as i64) + direction[0];
                let y_d = (anchor[1] as i64) + direction[1];
                let z_d = (anchor[2] as i64) + direction[2];

                if x_d >= 0
                    && y_d >= 0
                    && z_d >= 0
                    && x_d < max_num_boxes
                    && y_d < max_num_boxes
                    && z_d < max_num_boxes
                {
                    let new_anchor = vec![x_d as u64, y_d as u64, z_d as u64, level];
                    neighbours.push(encode_morton_point(new_anchor, dimensions));
                }
            }
        }
    };

    neighbours
}

// Gets the key of all siblings of the current key.
pub fn get_siblings(key: &u64, dimensions: &Dimensions) -> Vec<u64> {
    let num_siblings = 2u64.pow(*dimensions as u32);

    let level = get_level(&key);
    let key_no_level = key >> LEVEL_DISPLACEMENT;

    let root = match dimensions {
        Dimensions::One => (key_no_level >> 1) << 1,
        Dimensions::Two => (key_no_level >> 2) << 2,
        Dimensions::Three => (key_no_level >> 3) << 3,
    };

    (0..num_siblings)
        .into_iter()
        .map(|suffix| {
            let sibling = ((root | suffix) << LEVEL_DISPLACEMENT) | level;
            sibling
        })
        .collect()
}

// Gets the keys for all children of the current key.
pub fn get_children(key: &u64, dimensions: &Dimensions) -> Vec<u64> {
    let level = get_level(&key);
    let key_no_level = key >> LEVEL_DISPLACEMENT;

    let mut child = key_no_level << *dimensions as isize;
    child <<= LEVEL_DISPLACEMENT;
    child |= level + 1;

    get_siblings(&child, &dimensions)
}

// Gets the child index of a key.
pub fn get_child_index(child: &u64, dimensions: &Dimensions) -> usize {
    let key_no_level = child >> LEVEL_DISPLACEMENT;
    let sibling_mask = (1u64 << *dimensions as usize) - 1;
    let child_index = key_no_level & sibling_mask;
    child_index as usize
}

// Tests whether two cells in the tree are adjacent.
pub fn are_adjacent(
    cell_a: &u64,
    cell_b: &u64,
    tree_center: &Vec<f64>,
    tree_radius: &f64,
    dimensions: &Dimensions,
) -> bool {
    let tolerance = 1e-6;
    let (center_a, length_a) = get_center_length(&cell_a, &tree_center, &tree_radius, &dimensions);
    let (center_b, length_b) = get_center_length(&cell_b, &tree_center, &tree_radius, &dimensions);

    let length = 0.5 * (length_a + length_b);

    center_a
        .iter()
        .zip(center_b)
        .all(|(value_a, value_b)| (value_b - value_a).abs() <= tolerance + length)
}

// Gets the center and radius of the cell, given a Morton key and tree center and radius.
pub fn get_center_length(
    key: &u64,
    tree_center: &Vec<f64>,
    tree_radius: &f64,
    dimensions: &Dimensions,
) -> (Vec<f64>, f64) {
    let mut anchor = decode_key(&key, &dimensions);
    let level = anchor.pop().unwrap();
    let side_length = get_side_length(&tree_radius, &level);
    let displacement: Vec<f64> = tree_center.iter().map(|&c| c - *tree_radius).collect();

    let cell_center: Vec<f64> = anchor
        .iter()
        .enumerate()
        .map(|(idx, &value)| (value as f64 + 0.5) * side_length + displacement[idx])
        .collect();

    (cell_center, side_length)
}

// Calculates the center and radius of the tree required by the given extents.
pub fn calculate_tree_center_and_radius(extents: &Vec<f64>) -> (Vec<f64>, f64) {
    let eps = 1E-3;
    let dimensions = extents.len() / 2;
    let mut lower_bounds: Vec<f64> = extents[0..dimensions].to_vec();
    let mut upper_bounds: Vec<f64> = extents[dimensions..].to_vec();

    lower_bounds
        .iter_mut()
        .for_each(|elem| *elem = elem.floor());
    upper_bounds.iter_mut().for_each(|elem| *elem = elem.ceil());

    let center: Vec<f64> = lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&lower, &upper)| (lower + upper) / 2.0)
        .collect();

    let radius = lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&lower, &upper)| (upper - lower) / 2.0 + eps)
        .fold(f64::NEG_INFINITY, f64::max);

    (center, radius)
}
