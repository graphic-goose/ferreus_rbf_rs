/////////////////////////////////////////////////////////////////////////////////////////////
//
// Topology tests to determine whether owned sample point intersections can be clustered.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # topology
//! This module provides the topology tests as defined in the reference paper to determine
//! whether the intersections for an owned sample point can be clustered without changing
//! the topology of the resultant surface.
//!
//! From the paper:
//! "There are a variety of situations where clustering the surface intersections near a sample
//! point would lead to a change of topology in the resulting surface. The possible cases are:
//!     (a) Closed-surface: The sample point has a value opposite in sign from all the surrounding
//!         points. Clustering the surface intersections to a single vertex would result in the
//!         elimination of this surface, so the original surface intersections become the new
//!         vertices.
//!     (b) Multi-hole: There is a single surface, but with a hole in it. Clustering the surface
//!         intersections would close up the hole, so once again the original surface intersections
//!         remain.
//!     (c) Flat-hole: There is a single surface with no hole, but the surface is concave such
//!         that clustering the surface intersections might result in ‘flattening’ the surface,
//!         leading to two edges or two triangles being back to back. Again, the original surface
//!         intersections remain.
//!     (d) Multi-surface: The surface intersections form more than one separate surface, so one
//!         new vertex is required for each of these surfaces.
//!     (e) Simple-surface: The simple (and most common) case where there is only one surface can
//!         be clustered to a single new vertex with no change in surface topology."
use super::{
    constants::{ALL14_MASK, EDGE_DELTAS, FLAT_HOLE_MASKS, NEIGHBOUR_MASKS},
    isosurface,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum TopologyCase {
    ClosedSurface,
    MultiHole,
    FlatHole,
    MultiSurface,
    SimpleSurface,
    DoNotCluster,
}

#[derive(Debug, Clone)]
pub struct Cluster {
    pub edges: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TopologyResult {
    pub case: TopologyCase,
    pub clusters: Vec<Cluster>,
}

impl TopologyResult {
    pub fn iter_clusters(&self) -> impl Iterator<Item = &Cluster> + '_ {
        self.clusters.iter()
    }
}
/// Iterator over set-bit indices (LSB-first), e.g. mask=0b10100 -> yields 2,4
pub struct BitIndexIter {
    mask: u16,
}

impl BitIndexIter {
    #[inline]
    pub fn new(mask: u16) -> Self {
        Self { mask }
    }
}

impl Iterator for BitIndexIter {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        }
        let tz = self.mask.trailing_zeros() as u8;
        self.mask ^= 1u16 << tz;
        Some(tz)
    }
}

#[inline]
fn lowbit(x: u16) -> u16 {
    // x & -x in unsigned form
    x & x.wrapping_neg()
}

/// Connected components in the 14-edge adjacency graph (Table 3 neighbour masks).
/// From the paper:
/// "...finding all the surface intersections which belong to one surface is performed by selecting
/// an arbitrary edge with an intersection, then including any of the neighbouring edges that also
/// contain surface intersections in a list of edges to check. This list then determines the next
/// edge to check. The process is iterated until there are no more edges to check."
///
/// Returns a list of bitmasks, each bitmask is a connected component.
pub fn connected_components_masks(edge_mask: u16) -> Vec<u16> {
    let mut remaining = edge_mask & ALL14_MASK;
    let mut comps: Vec<u16> = Vec::new();

    while remaining != 0 {
        let seed = lowbit(remaining);
        remaining ^= seed;

        let mut comp: u16 = 0;
        let mut frontier: u16 = seed;

        while frontier != 0 {
            let b = lowbit(frontier);
            frontier ^= b;

            let e = b.trailing_zeros() as usize;
            comp |= b;

            let nbrs = NEIGHBOUR_MASKS[e] & remaining;
            remaining ^= nbrs;
            frontier |= nbrs;
        }

        comps.push(comp);
    }

    comps
}

/// Gets the values for the given sample point.
#[inline]
fn edge_endpoint_value(idx: [i64; 3], edge: usize, values: &HashMap<[i64; 3], f64>) -> Option<f64> {
    let [di, dj, dk] = EDGE_DELTAS[edge];
    let endpoint = [idx[0] + di as i64, idx[1] + dj as i64, idx[2] + dk as i64];

    values.get(&endpoint).copied().filter(|v| v.is_finite())
}

/// For a given mask, returns the two edge indices.
#[inline]
fn two_edge_indices(mask: u16) -> Option<[usize; 2]> {
    let mut edges = BitIndexIter::new(mask & ALL14_MASK);
    let a = edges.next()? as usize;
    let b = edges.next()? as usize;
    if edges.next().is_some() {
        return None;
    }
    Some([a, b])
}

#[inline]
fn is_inside(v: f64) -> bool {
    const EPS: f64 = 1.0e-9;
    v < -EPS
}

#[inline]
fn crossing_alpha(a_val: f64, b_val: f64) -> Option<f64> {
    if is_inside(a_val) == is_inside(b_val) {
        return None;
    }

    Some(isosurface::lerp_alpha(a_val, b_val))
}

/// Flat-hole test from the paper's Figure 6 criterion.
/// From the paper:
/// "O is the current sample point, A...D are some of the neighbouring sample points. The existence
/// of edges OA and OB with no near surface intersection, and edges OC and OD with near surface
/// intersections, will lead to a collapsed surface if it is connected around O, and the surface
/// intersections on edges AD and AC are both near A, or BC and BD are both near B."
///
/// The statement in Table 4 conflicts with the above statement from earlier in the paper. So we
/// interpret a Table 4 row as OA/OB in `edge_mask` and OC/OD in `opposite_mask`.
fn is_flat_hole(surface_comp: u16, idx: [i64; 3], values: &HashMap<[i64; 3], f64>) -> bool {
    let sm = surface_comp & ALL14_MASK;

    for [edge_mask, opposite_mask] in FLAT_HOLE_MASKS.iter() {
        if (sm & *edge_mask) != 0 {
            continue;
        }
        if (sm & *opposite_mask) != *opposite_mask {
            continue;
        }

        let Some([a, b]) = two_edge_indices(*edge_mask) else {
            continue;
        };
        let Some([c, d]) = two_edge_indices(*opposite_mask) else {
            continue;
        };

        let Some(a_val) = edge_endpoint_value(idx, a, values) else {
            continue;
        };
        let Some(b_val) = edge_endpoint_value(idx, b, values) else {
            continue;
        };
        let Some(c_val) = edge_endpoint_value(idx, c, values) else {
            continue;
        };
        let Some(d_val) = edge_endpoint_value(idx, d, values) else {
            continue;
        };

        let near_a = crossing_alpha(a_val, d_val).is_some_and(|t| t < 0.5)
            && crossing_alpha(a_val, c_val).is_some_and(|t| t < 0.5);
        let near_b = crossing_alpha(b_val, d_val).is_some_and(|t| t < 0.5)
            && crossing_alpha(b_val, c_val).is_some_and(|t| t < 0.5);

        if near_a || near_b {
            return true;
        }
    }
    false
}

/// Vetos a cluster and returns each intersection as a separate cluster.
fn do_not_cluster(edge_mask: u16) -> Vec<Cluster> {
    BitIndexIter::new(edge_mask)
        .map(|e| Cluster { edges: vec![e] })
        .collect()
}

/// For a given sample point's intersections, determine its topology case and
/// the relevant clusters.
pub fn test_topology(
    near_mask: u16,
    cluster: bool,
    ijk: [i64; 3],
    values: &HashMap<[i64; 3], f64>,
) -> TopologyResult {
    let m = near_mask & ALL14_MASK;

    // No intersections, nothing to check.
    if m == 0 {
        return TopologyResult {
            case: TopologyCase::SimpleSurface,
            clusters: Vec::default(),
        };
    }

    // The user has specifically asked for no clustering.
    if !cluster {
        return TopologyResult {
            case: TopologyCase::DoNotCluster,
            clusters: do_not_cluster(m),
        };
    }

    // All edges have a near intersection, forming a closed surface. Don't cluster.
    if m == ALL14_MASK {
        return TopologyResult {
            case: TopologyCase::ClosedSurface,
            clusters: do_not_cluster(m),
        };
    }

    let mut parts: Vec<Cluster> = Vec::new();

    // Determine the number of separate surface components.
    let surface_components = connected_components_masks(m);

    // Test for multi-surface and multi-hole cases.
    // If multi-surface we can test each surface component to determine if it can be clustered.
    // For multi-hole don't cluster.
    match surface_components.len() > 1 {
        true => {
            for mut comp in surface_components {
                comp &= ALL14_MASK;

                parts.push(Cluster {
                    edges: BitIndexIter::new(comp).collect(),
                });
            }
            return TopologyResult {
                case: TopologyCase::MultiSurface,
                clusters: parts,
            };
        }
        false => {
            // holes = connected components of complement within 14 edges
            let holes = connected_components_masks(ALL14_MASK & !m);

            if holes.len() != 1 {
                return TopologyResult {
                    case: TopologyCase::MultiHole,
                    clusters: do_not_cluster(m),
                };
            };

            if is_flat_hole(m, ijk, values) {
                return TopologyResult {
                    case: TopologyCase::FlatHole,
                    clusters: do_not_cluster(m),
                };
            };

            parts.push(Cluster {
                edges: BitIndexIter::new(m).collect(),
            });

            return TopologyResult {
                case: TopologyCase::SimpleSurface,
                clusters: parts,
            };
        }
    };
}
