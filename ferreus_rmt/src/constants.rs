/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides constants useful for the Regularised Marching Tetrahedra algorithm.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # constants
//! This module provides helper constants that define the order of sample point edges, as well
//! as the lookup tables provided in the original paper for use in topology tests.
//!
//! The sample lattice is built from a stack of XY sample planes. From the paper:
//! "The locations of each of the sample points on neighbouring planes alternate,
//! so that two planes taken together form a rectangular lattice of sample points
//! with half the spacing of an individual plane."
//!
//! We exploit this “two-plane fine grid” to simplify ijk indexing. Instead of treating
//! each parity plane separately, we index points on the fine grid directly, so that
//! neighbour offsets become small integer (i,j,k) steps.
//!
//! With this indexing, each sample point owns a set of edges that define a single
//! parallelepiped. This parallelepiped is the union of the six owned tetrahedra for
//! that sample point and is the fundamental cell we use for wavefront propagation,
//! topology checks, clustering, and triangulation.
//!
//! Parallelepiped corner indices (top-down / plan-view projection)
//! - Corners 0, 1, 2, 3 lie on the upper XY plane (k plane).
//! - Corners 4, 5, 6, 7 lie on the lower XY plane (k-1 plane).
//! - The lower face is the upper face translated by the skew vector w.
//!
//! Spanning vectors:
//! - u is the in-plane edge from corner 0 to corner 1.
//! - v is the in-plane edge from corner 0 to corner 3.
//! - w is the cross-plane edge from corner 0 to corner 7.
//!
//! ```text
//!            1----------4
//!           ⟋ ⟍       ⟋  ⟍
//!         ⟋     ⟍   ⟋      ⟍
//!       ⟋         ⟍          ⟍
//!     ⟋         ⟋   u          ⟍
//!   ⟋         ⟋       ⟍          ⟍
//!  2---------5          0 ----w----7
//!   ⟍         ⟍       ⟋          ⟋
//!     ⟍         ⟍   v          ⟋
//!       ⟍         ⟋          ⟋
//!         ⟍     ⟋   ⟍      ⟋
//!           ⟍ ⟋       ⟍  ⟋
//!            3----------6
//! ```
//!
//! Parallelepiped faces:
//! - 0 (-ve u) face is corners 0, 3, 6, 7
//! - 1 (+ve u) face is corners 1, 2, 5, 4
//! - 2 (-ve v) face is corners 0, 1, 4, 7
//! - 3 (+ve v) face is corners 3, 2, 5, 6
//! - 4 (-ve w) face is corners 0, 1, 2, 3
//! - 5 (+ve w) face is corners 4, 5, 6, 7
//!
//! ```text
//!            o----------o
//!           ⟋ ⟍       ⟋  ⟍
//!         ⟋     ⟍   ⟋      ⟍
//!       ⟋     1   ⟍      2   ⟍
//!     ⟋         ⟋   u          ⟍
//!   ⟋         ⟋       ⟍          ⟍
//!  o---------o 4        o-----w----o
//!   ⟍         ⟍       ⟋  5       ⟋
//!     ⟍         ⟍   v          ⟋
//!       ⟍     3   ⟋      0   ⟋
//!         ⟍     ⟋   ⟍      ⟋
//!           ⟍ ⟋       ⟍  ⟋
//!            o----------o
//! ```
//!
//! Parallelepiped edge indices (top-down / plan-view projection)
//! - Edges 0, 1, 2 lie entirely on the upper XY plane (within-plane edges).
//! - Edges 3, 4, 5, 6 connect the upper and lower planes (cross-plane edges).
//!
//! ```text
//!            o          o
//!             ⟍         |
//!               ⟍       |
//!                 0     3
//!                   ⟍   |
//!                     ⟍ |
//!  o— —1— — —o-----4--- o------6-----o
//!                     ⟋ |
//!                   ⟋   |
//!                 2     5
//!               ⟋       |
//!             ⟋         |
//!            o          o
//! ```

/// The IJK offsets from the sample point for each of the edges.
pub const EDGE_DELTAS: [[i8; 3]; 14] = [
    [-1, 1, 0],  // 0
    [-2, 0, 0],  // 1
    [-1, -1, 0], // 2
    [0, 1, -1],  // 3
    [-1, 0, -1], // 4
    [0, -1, -1], // 5
    [1, 0, -1],  // 6
    [1, -1, 0],  // 7
    [2, 0, 0],   // 8
    [1, 1, 0],   // 9
    [0, -1, 1],  // 10
    [1, 0, 1],   // 11
    [0, 1, 1],   // 12
    [-1, 0, 1],  // 13
];

/// The spanning vectors.
pub const U: [i8; 3] = EDGE_DELTAS[0];
pub const V: [i8; 3] = EDGE_DELTAS[2];
pub const W: [i8; 3] = EDGE_DELTAS[6];

/// The edge deltas that define each face of the parallelpiped.
pub const FACES: [[usize; 4]; 6] = [
    [0, 3, 6, 7], // u=0  (across -u)
    [1, 2, 5, 4], // u=1  (across +u)
    [0, 1, 4, 7], // v=0  (across -v)
    [3, 2, 5, 6], // v=1  (across +v)
    [0, 1, 2, 3], // w=0  (across -w)
    [4, 5, 6, 7], // w=1  (across +w)
];

/// The normal direction for each face.
pub const FACE_DIRS: [[i8; 3]; 6] = [
    [-U[0], -U[1], -U[2]], // -U
    U,                     // +U
    [-V[0], -V[1], -V[2]], // -V
    V,                     // +V
    [-W[0], -W[1], -W[2]], // -W
    W,                     // +W
];

/// Given an IJK delta offset, returns the edge number.
pub const DELTA_TO_EDGE: [([i8; 3], usize); EDGE_DELTAS.len()] = make_delta_to_edge(EDGE_DELTAS);

const fn make_delta_to_edge<const N: usize>(deltas: [[i8; 3]; N]) -> [([i8; 3], usize); N] {
    let mut out = [([0, 0, 0], 0); N];

    let mut i = 0;
    while i < N {
        out[i] = (deltas[i], i);
        i += 1;
    }

    out
}

pub fn delta_to_edge(delta: [i8; 3]) -> Option<usize> {
    DELTA_TO_EDGE
        .iter()
        .find_map(|&(d, edge)| if d == delta { Some(edge) } else { None })
}

// Reverse edge label for each edge label (0<->7, 1<->8, 2<->9, 3<->10, 4<->11, 5<->12, 6<->13)
pub const REVERSE_EDGE: [usize; 14] = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6];

// Table 3: Neighbor masks for topology checks
pub const NEIGHBOUR_MASKS: [u16; 14] = [
    0x321A, 0x2015, 0x24B2, 0x0251, 0x006F, 0x00D4, 0x03B8, 0x0D64, 0x0AC0, 0x1949, 0x2884, 0x3780,
    0x2A01, 0x1C07,
];

pub const NEIGHBOUR_EDGE_PLANE_PAIRS: [&[[u8; 2]]; 14] = [
    &[[9, 1], [12, 4], [3, 13]],  // 0
    &[[0, 2], [4, 13]],           // 1
    &[[1, 7], [13, 5], [4, 10]],  // 2
    &[[9, 4], [6, 0]],            // 3
    &[[0, 5], [3, 2], [1, 6]],    // 4
    &[[4, 7], [2, 6]],            // 5
    &[[5, 9], [7, 3], [8, 4]],    // 6
    &[[10, 6], [5, 11], [2, 8]],  // 7
    &[[7, 9], [11, 6]],           // 8
    &[[8, 0], [11, 3], [6, 12]],  // 9
    &[[2, 11], [13, 7]],          // 10
    &[[13, 8], [7, 12], [10, 9]], // 11
    &[[9, 13], [11, 0]],          // 12
    &[[0, 10], [12, 2], [11, 1]], // 13
];

pub const PHI_1: f64 = 0.955316618125;
pub const PHI_2: f64 = 1.230959417341;

pub const NEIGHBOUR_EDGE_PLANE_PHIS: [&[[f64; 2]]; 14] = [
    &[[PHI_2, PHI_1], [PHI_1, PHI_2], [PHI_1, PHI_2]], // 0
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 1
    &[[PHI_1, PHI_2], [PHI_2, PHI_1], [PHI_2, PHI_1]], // 2
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 3
    &[[PHI_2, PHI_1], [PHI_1, PHI_2], [PHI_1, PHI_2]], // 4
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 5
    &[[PHI_1, PHI_2], [PHI_2, PHI_1], [PHI_1, PHI_2]], // 6
    &[[PHI_1, PHI_2], [PHI_1, PHI_2], [PHI_2, PHI_1]], // 7
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 8
    &[[PHI_1, PHI_2], [PHI_2, PHI_1], [PHI_2, PHI_1]], // 9
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 10
    &[[PHI_2, PHI_1], [PHI_2, PHI_1], [PHI_1, PHI_2]], // 11
    &[[PHI_1, PHI_1], [PHI_1, PHI_1]],                 // 12
    &[[PHI_2, PHI_1], [PHI_1, PHI_2], [PHI_2, PHI_1]], // 13
];

// Table 4: Flat hole detection
pub const FLAT_HOLE_MASKS: [[u16; 2]; 36] = [
    [0x0003, 0x2010],
    [0x0009, 0x0210],
    [0x0011, 0x000A],
    [0x0201, 0x1008],
    [0x1001, 0x2200],
    [0x2001, 0x1002],
    [0x0006, 0x2010],
    [0x0012, 0x0005],
    [0x2002, 0x0005],
    [0x0014, 0x0022],
    [0x0024, 0x0090],
    [0x0084, 0x0420],
    [0x0404, 0x2080],
    [0x2004, 0x0402],
    [0x0018, 0x0041],
    [0x0048, 0x0210],
    [0x0208, 0x0041],
    [0x0030, 0x0044],
    [0x0050, 0x0028],
    [0x0060, 0x0090],
    [0x00A0, 0x0044],
    [0x00C0, 0x0120],
    [0x0140, 0x0280],
    [0x0240, 0x0108],
    [0x0180, 0x0840],
    [0x0480, 0x0804],
    [0x0880, 0x0500],
    [0x0300, 0x0840],
    [0x0900, 0x0280],
    [0x0A00, 0x1100],
    [0x1200, 0x0801],
    [0x0C00, 0x2080],
    [0x2400, 0x0804],
    [0x1800, 0x2200],
    [0x2800, 0x1400],
    [0x3000, 0x0801],
];

/// The bit mask for all 14 edges, used as a comparator.
pub const ALL14_MASK: u16 = (1 << 14) - 1;

/// 6 tetrahedra owned by each sample point (edge labels to 3 other vertices)
pub const OWNED_TET_EDGES: [[usize; 3]; 6] = [
    [0, 4, 1],
    [0, 3, 4],
    [3, 6, 4],
    [1, 4, 2],
    [2, 4, 5],
    [4, 6, 5],
];

/// Tetrahedron edges (vertex index pairs) - indexed 0-5
pub const TET_EDGE_PAIRS: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Marching Tetrahedra lookup table (right-handed, normals outward).
/// Each entry is a list of triangles; each triangle is 3 edge indices (0..5).
pub const MT_TABLE: [&[[u8; 3]]; 16] = [
    &[],                     // 0
    &[[0, 1, 2]],            // 1
    &[[0, 4, 3]],            // 2
    &[[3, 1, 2], [3, 2, 4]], // 3
    &[[1, 3, 5]],            // 4
    &[[5, 2, 0], [5, 0, 3]], // 5
    &[[5, 1, 0], [5, 0, 4]], // 6
    &[[2, 4, 5]],            // 7
    &[[2, 5, 4]],            // 8
    &[[4, 0, 1], [4, 1, 5]], // 9
    &[[3, 0, 2], [3, 2, 5]], // 10
    &[[1, 5, 3]],            // 11
    &[[4, 2, 1], [4, 1, 3]], // 12
    &[[0, 3, 4]],            // 13
    &[[0, 2, 1]],            // 14
    &[],                     // 15
];
