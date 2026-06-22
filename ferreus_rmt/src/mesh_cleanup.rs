/////////////////////////////////////////////////////////////////////////////////////////////
//
// Cleans extracted triangle meshes before conversion to the public mesh representation.
//
// Created on: 17 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Mesh cleanup for extracted RMT triangle surfaces.
//!
//! The extraction and boundary-closure stages may sometimes create duplicate vertices, repeated
//! facets, degenerate triangles, or tiny disconnected components. This module removes those
//! artefacts while preserving the triangle orientation and vertex order of the remaining surface.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::geometry::Point;

/// Minimum number of facets a connected component must contain to be retained.
const MIN_CONNECTED_COMPONENT_FACETS: usize = 2;

/// Cleans raw row-major mesh buffers produced by extraction.
///
/// The cleanup pass:
/// 1. Deduplicates vertices within `eps`.
/// 2. Drops triangles that collapse after deduplication.
/// 3. Drops near-zero-area triangles.
/// 4. Drops duplicate triangles regardless of winding.
/// 5. Removes disconnected components smaller than [`MIN_CONNECTED_COMPONENT_FACETS`].
pub(crate) fn clean_mesh(
    vertices: Vec<f64>,
    facets: Vec<usize>,
    eps: f64,
) -> (Vec<f64>, Vec<usize>) {
    let mut vertex_map: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut old_to_new = Vec::with_capacity(vertices.len() / 3);
    let mut compact_vertices = Vec::new();

    for p in vertices.chunks_exact(3) {
        let id = push_dedup_point(
            &mut vertex_map,
            &mut compact_vertices,
            [p[0], p[1], p[2]],
            eps,
        );
        old_to_new.push(id);
    }

    let mut compact_facets = Vec::new();
    let mut seen = HashSet::new();
    for tri in facets.chunks_exact(3) {
        let a = old_to_new[tri[0]];
        let b = old_to_new[tri[1]];
        let c = old_to_new[tri[2]];
        if a == b || b == c || a == c {
            continue;
        }

        let pa = [
            compact_vertices[3 * a],
            compact_vertices[3 * a + 1],
            compact_vertices[3 * a + 2],
        ];
        let pb = [
            compact_vertices[3 * b],
            compact_vertices[3 * b + 1],
            compact_vertices[3 * b + 2],
        ];
        let pc = [
            compact_vertices[3 * c],
            compact_vertices[3 * c + 1],
            compact_vertices[3 * c + 2],
        ];
        let ab = pb.sub(pa);
        let ac = pc.sub(pa);
        let area2 = ab.cross(ac).norm2();
        if area2 <= eps.powi(4) {
            continue;
        }

        let mut key = [a, b, c];
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        compact_facets.extend_from_slice(&[a, b, c]);
    }

    remove_small_disconnected_components(
        compact_vertices,
        compact_facets,
        MIN_CONNECTED_COMPONENT_FACETS,
    )
}

/// Removes connected triangle components with fewer than `min_component_facets` facets.
///
/// Connectivity is vertex-based: two triangles are in the same component if they share at least
/// one vertex, directly or through a chain of neighbouring triangles.
fn remove_small_disconnected_components(
    vertices: Vec<f64>,
    facets: Vec<usize>,
    min_component_facets: usize,
) -> (Vec<f64>, Vec<usize>) {
    let nfacets = facets.len() / 3;
    if nfacets == 0 || min_component_facets <= 1 {
        return (vertices, facets);
    }

    let mut vertex_facets: HashMap<usize, Vec<usize>> = HashMap::new();
    for tri_idx in 0..nfacets {
        let base = tri_idx * 3;
        for &vid in &facets[base..base + 3] {
            vertex_facets.entry(vid).or_default().push(tri_idx);
        }
    }

    let mut keep_facet = vec![false; nfacets];
    let mut seen_facet = vec![false; nfacets];
    let mut queue = VecDeque::new();
    let mut component = Vec::new();

    for seed in 0..nfacets {
        if seen_facet[seed] {
            continue;
        }

        queue.clear();
        component.clear();
        seen_facet[seed] = true;
        queue.push_back(seed);

        while let Some(tri_idx) = queue.pop_front() {
            component.push(tri_idx);
            let base = tri_idx * 3;
            for &vid in &facets[base..base + 3] {
                let Some(neighbours) = vertex_facets.get(&vid) else {
                    continue;
                };
                for &next_tri in neighbours {
                    if seen_facet[next_tri] {
                        continue;
                    }
                    seen_facet[next_tri] = true;
                    queue.push_back(next_tri);
                }
            }
        }

        if component.len() >= min_component_facets {
            for &tri_idx in &component {
                keep_facet[tri_idx] = true;
            }
        }
    }

    compact_kept_facets(vertices, facets, &keep_facet)
}

/// Rebuilds vertex and facet buffers containing only retained facets.
///
/// This removes vertices that are no longer referenced after component filtering and rewrites
/// facet indices to the compacted vertex buffer.
fn compact_kept_facets(
    vertices: Vec<f64>,
    facets: Vec<usize>,
    keep_facet: &[bool],
) -> (Vec<f64>, Vec<usize>) {
    let mut old_to_new = vec![usize::MAX; vertices.len() / 3];
    let mut kept_vertices = Vec::new();
    let mut kept_facets = Vec::new();

    for (tri_idx, tri) in facets.chunks_exact(3).enumerate() {
        if !keep_facet.get(tri_idx).copied().unwrap_or(false) {
            continue;
        }

        for &old_vid in tri {
            if old_to_new[old_vid] == usize::MAX {
                let new_vid = kept_vertices.len() / 3;
                old_to_new[old_vid] = new_vid;
                kept_vertices.extend_from_slice(&vertices[3 * old_vid..3 * old_vid + 3]);
            }
            kept_facets.push(old_to_new[old_vid]);
        }
    }

    (kept_vertices, kept_facets)
}

/// Quantises a point to the spatial hash grid used during vertex deduplication.
fn quantized_point_key(p: [f64; 3], eps: f64) -> (i64, i64, i64) {
    let q = |v: f64| (v / eps.max(1.0e-12)).round() as i64;
    (q(p[0]), q(p[1]), q(p[2]))
}

/// Inserts `p` into the compact vertex buffer, or returns an existing nearby vertex id.
///
/// The lookup checks the quantised cell containing `p` and the 26 neighbouring cells so that
/// points within `eps` are found even when they lie across a cell boundary.
fn push_dedup_point(
    map: &mut HashMap<(i64, i64, i64), Vec<usize>>,
    vertices: &mut Vec<f64>,
    p: [f64; 3],
    eps: f64,
) -> usize {
    let key = quantized_point_key(p, eps);
    let eps2 = eps * eps;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let neighbour = (key.0 + dx, key.1 + dy, key.2 + dz);
                let Some(ids) = map.get(&neighbour) else {
                    continue;
                };
                for &id in ids {
                    let q = [vertices[3 * id], vertices[3 * id + 1], vertices[3 * id + 2]];
                    if p.sub(q).norm2() <= eps2 {
                        return id;
                    }
                }
            }
        }
    }

    let id = vertices.len() / 3;
    vertices.extend_from_slice(p.as_slice());
    map.entry(key).or_default().push(id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_mesh_removes_single_triangle_components() {
        let vertices = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            10.0, 0.0, 0.0, //
            11.0, 0.0, 0.0, //
            10.0, 1.0, 0.0,
        ];
        let facets = vec![0, 1, 2, 1, 3, 2, 4, 5, 6];

        let (clean_vertices, clean_facets) = clean_mesh(vertices, facets, 1.0e-9);

        assert_eq!(clean_vertices.len() / 3, 4);
        assert_eq!(clean_facets.len() / 3, 2);
    }
}
