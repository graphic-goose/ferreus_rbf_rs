/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements surface-following regularised marching tetrahedra extraction.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Iso-surface extraction using regularised marching tetrahedra.
//!
//! This module owns the main extraction pipeline: seed projection, wavefront expansion,
//! topology-aware vertex clustering, triangle generation, rollback of invalid clustered
//! vertices, clipping, mesh cleanup, and optional boundary closure.
//!
//! The public entry point is [`build_isosurface`]. Most other helpers in this module operate
//! on internal row-major buffers and lattice-space `ijk` coordinates while the final result is
//! returned as a [`Mesh`].

use std::collections::{HashMap, HashSet};

use crate::{
    aabb_clipping::{bbox_eps, clip_mesh_to_aabb, facet_fully_inside_aabb},
    boundary_closure::{self, BoundaryClosure},
    constants::{
        self, EDGE_DELTAS, FACE_DIRS, FACES, MT_TABLE, OWNED_TET_EDGES, REVERSE_EDGE,
        TET_EDGE_PAIRS,
    },
    curvature_weighting,
    lattice::SampleLattice,
    mesh::Mesh,
    mesh_cleanup::clean_mesh,
    mesh_intersections,
    progress::{IsosurfaceStage, ProgressMsg, ProgressSink},
    seed_projection,
    topology::{self, TopologyCase},
};

use faer::{Mat, MatRef};

pub use crate::aabb_clipping::AABB;

/// Vertex clustering strategy used when converting edge intersections into mesh vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMethod {
    /// No vertex clustering is performed. Raw marching tetrahedra triangles are produced.
    None,

    /// Cluster topology-compatible intersections to their arithmetic mean.
    Average,

    /// Cluster topology-compatible intersections using the local curvature estimate.
    CurvatureWeighted,
}

/// Per-lattice-sample extraction state.
#[derive(Debug, Clone, Copy)]
struct SamplePoint {
    /// Bit mask of owned lattice edges containing an isosurface intersection.
    intersections: u16,
}

impl Default for SamplePoint {
    fn default() -> Self {
        Self {
            intersections: 0u16,
        }
    }
}

/// Candidate vertex generated from one topology-compatible cluster of edge intersections.
#[derive(Debug)]
struct VertexCandidate {
    /// Proposed world-space vertex location.
    point: [f64; 3],

    /// Lattice edges represented by this clustered vertex.
    edge_endpoints: Vec<([i64; 3], [i64; 3])>,

    /// Sample point that owns this candidate.
    owner: [i64; 3],
}

/// Returns the lattice points addressed by the first `E` edge offsets from `ijk`.
#[inline]
pub fn get_edge_points<const E: usize>(ijk: &[i64; 3]) -> [[i64; 3]; E] {
    let mut corners: [[i64; 3]; E] = [<[i64; 3]>::default(); E];

    corners[0][0] = ijk[0];
    corners[0][1] = ijk[1];
    corners[0][2] = ijk[2];

    for d in 0..(E - 1) {
        let eds = EDGE_DELTAS[d];
        let corner_idx = d + 1;
        corners[corner_idx][0] = ijk[0] + eds[0] as i64;
        corners[corner_idx][1] = ijk[1] + eds[1] as i64;
        corners[corner_idx][2] = ijk[2] + eds[2] as i64;
    }

    corners
}

/// Returns `(owner, other_endpoint, owned_label)` for an edge in the 7-associated ownership map.
pub(crate) fn get_edge_owner(u: [i64; 3], v: [i64; 3]) -> Option<([i64; 3], [i64; 3], usize)> {
    let delta = [
        (v[0] - u[0]) as i8,
        (v[1] - u[1]) as i8,
        (v[2] - u[2]) as i8,
    ];
    let eid = constants::delta_to_edge(delta)?;
    if eid < 7 {
        Some((u, v, eid))
    } else {
        Some((v, u, REVERSE_EDGE[eid]))
    }
}

/// Looks up the mesh vertex assigned to the lattice edge `(u, v)`.
#[inline]
fn edge_ref_get(
    edge_ref: &HashMap<([i64; 3], usize), usize>,
    u: [i64; 3],
    v: [i64; 3],
) -> Option<usize> {
    let (owner, _other, lab) = get_edge_owner(u, v)?;
    edge_ref.get(&(owner, lab)).copied()
}

/// Stores the mesh vertex assigned to the lattice edge `(u, v)`.
#[inline]
fn edge_ref_set(
    edge_ref: &mut HashMap<([i64; 3], usize), usize>,
    u: [i64; 3],
    v: [i64; 3],
    vid: usize,
) {
    if let Some((owner, _other, lab)) = get_edge_owner(u, v) {
        edge_ref.insert((owner, lab), vid);
    }
}

/// Appends a point to the row-major vertex buffer and returns its vertex id.
#[inline]
fn push_vertex(vertices: &mut Vec<f64>, p: [f64; 3]) -> usize {
    let vid = vertices.len() / 3;
    vertices.extend_from_slice(p.as_slice());
    vid
}

/// Computes the world-space isosurface intersection point on a lattice edge.
#[inline]
pub fn edge_intersection_point(
    u: [i64; 3],
    v: [i64; 3],
    evaluated: &HashMap<[i64; 3], f64>,
    lattice: &SampleLattice,
) -> Option<[f64; 3]> {
    let vu = *evaluated.get(&u)?;
    let vv = *evaluated.get(&v)?;
    if !vu.is_finite() || !vv.is_finite() {
        return None;
    }

    let pu = lattice.ijk_to_world(u);
    let pv = lattice.ijk_to_world(v);
    Some(interpolate_points(pu, pv, lerp_alpha(vu, vv)))
}

/// Calculates the alpha weight ('t') for linearly interpolating along the edge.
#[inline]
pub fn lerp_alpha(vu: f64, vv: f64) -> f64 {
    let denom = vu - vv;
    if denom.abs() < 1e-30 {
        0.5
    } else {
        (vu / denom).clamp(0.0, 1.0)
    }
}

/// Returns the arithmetic mean of a non-empty point slice.
fn average_point(pts: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for p in pts {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    let inv_n = 1.0 / (pts.len() as f64);
    [sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n]
}

/// Linearly interpolates between two points.
fn interpolate_points(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Returns an edge delta as signed lattice coordinates.
#[inline]
fn edge_delta(e: usize) -> [i64; 3] {
    [
        EDGE_DELTAS[e][0] as i64,
        EDGE_DELTAS[e][1] as i64,
        EDGE_DELTAS[e][2] as i64,
    ]
}

/// Adds two lattice-space coordinate triples.
#[inline]
fn add_ijk(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Marches the owned tetrahedra for `keys`, resolving edge vertices and emitting triangle ids.
///
/// `resolve` maps lattice edges to mesh vertex ids. `emit` receives each non-degenerate triangle
/// generated from the marching-tetrahedra table.
#[inline]
fn march_tets<Resolve, Emit>(
    keys: &[[i64; 3]],
    evaluated: &HashMap<[i64; 3], f64>,
    mut resolve: Resolve,
    mut emit: Emit,
) where
    Resolve: FnMut([i64; 3], [i64; 3]) -> Option<usize>,
    Emit: FnMut([usize; 3]),
{
    for &c0 in keys {
        for [ea, eb, ec] in OWNED_TET_EDGES {
            let corners = [
                c0,
                add_ijk(c0, edge_delta(ea)),
                add_ijk(c0, edge_delta(eb)),
                add_ijk(c0, edge_delta(ec)),
            ];

            let vals = match [
                evaluated.get(&corners[0]).copied(),
                evaluated.get(&corners[1]).copied(),
                evaluated.get(&corners[2]).copied(),
                evaluated.get(&corners[3]).copied(),
            ] {
                [Some(v0), Some(v1), Some(v2), Some(v3)]
                    if v0.is_finite() && v1.is_finite() && v2.is_finite() && v3.is_finite() =>
                {
                    [v0, v1, v2, v3]
                }
                _ => continue,
            };

            let mut case = 0usize;
            for (i, s) in vals.iter().enumerate() {
                if is_inside(*s) {
                    case |= 1 << i;
                }
            }

            for tri in MT_TABLE[case] {
                let mut vids = [0usize; 3];
                let mut ok = true;
                for i in 0..3 {
                    let [a, b] = TET_EDGE_PAIRS[tri[i] as usize];
                    if let Some(id) = resolve(corners[a], corners[b]) {
                        vids[i] = id;
                    } else {
                        ok = false;
                        break;
                    }
                }
                // Don't keep degenerate triangles created from clustering.
                if !ok || vids[0] == vids[1] || vids[1] == vids[2] || vids[0] == vids[2] {
                    continue;
                }
                emit(vids);
            }
        }
    }
}

/// Returns whether a scalar value lies inside the extracted isosurface.
fn is_inside(v: f64) -> bool {
    let eps = 1E-9;
    v < -eps
}

/// Returns an undirected vertex-edge key.
#[inline]
fn sorted_vertex_edge(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Returns the three vertex ids for a triangle in a row-major facet buffer.
#[inline]
fn facet_vertex_ids(facets: &[usize], tri_idx: usize) -> Option<[usize; 3]> {
    let base = tri_idx * 3;
    if base + 2 >= facets.len() {
        return None;
    }
    Some([facets[base], facets[base + 1], facets[base + 2]])
}

/// Adds the cluster owners referenced by one facet to `bad_owners`.
fn add_facet_cluster_owners(
    facets: &[usize],
    tri_idx: usize,
    cluster_vertex_owner: &HashMap<usize, [i64; 3]>,
    bad_owners: &mut HashSet<[i64; 3]>,
) {
    let Some(ids) = facet_vertex_ids(facets, tri_idx) else {
        return;
    };

    for vid in ids {
        if let Some(owner) = cluster_vertex_owner.get(&vid) {
            bad_owners.insert(*owner);
        }
    }
}

/// Finds clustered sample owners that participate in non-manifold mesh edges.
fn collect_invalid_topology_cluster_owners(
    facets: &[usize],
    cluster_vertex_owner: &HashMap<usize, [i64; 3]>,
) -> HashSet<[i64; 3]> {
    let nfacets = facets.len() / 3;
    let mut bad_owners = HashSet::new();
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for tri_idx in 0..nfacets {
        let Some(ids) = facet_vertex_ids(facets, tri_idx) else {
            continue;
        };

        for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            let key = sorted_vertex_edge(a, b);
            edge_faces.entry(key).or_default().push(tri_idx);
        }
    }

    for faces in edge_faces.values() {
        if faces.len() <= 2 {
            continue;
        }

        for &tri_idx in faces {
            add_facet_cluster_owners(facets, tri_idx, cluster_vertex_owner, &mut bad_owners);
        }
    }

    bad_owners
}

/// Replaces invalid clustered vertices with their original per-edge intersection vertices.
fn rollback_cluster_owners(
    bad_owners: HashSet<[i64; 3]>,
    owner_cluster_vertices: &mut HashMap<[i64; 3], Vec<usize>>,
    cluster_vertex_edges: &mut HashMap<usize, Vec<([i64; 3], [i64; 3])>>,
    cluster_vertex_owner: &mut HashMap<usize, [i64; 3]>,
    edge_ref: &mut HashMap<([i64; 3], usize), usize>,
    vertices: &mut Vec<f64>,
    evaluated: &HashMap<[i64; 3], f64>,
    lattice: &SampleLattice,
) -> usize {
    let mut bad_vertices = HashSet::new();
    let mut rolled_back_owner_count = 0usize;
    for owner in bad_owners {
        if let Some(owned_vids) = owner_cluster_vertices.remove(&owner) {
            if !owned_vids.is_empty() {
                rolled_back_owner_count += 1;
            }
            bad_vertices.extend(owned_vids);
        }
    }

    for vid in bad_vertices {
        let Some(edges) = cluster_vertex_edges.remove(&vid) else {
            continue;
        };
        cluster_vertex_owner.remove(&vid);

        for (u, v) in edges {
            if let Some(p) = edge_intersection_point(u, v, evaluated, lattice) {
                let split_vid = push_vertex(vertices, p);
                edge_ref_set(edge_ref, u, v, split_vid);
            }
        }
    }

    rolled_back_owner_count
}

/// Emits a structured progress update if a sink was provided.
#[inline]
fn emit_progress(
    progress_callback: Option<&dyn ProgressSink>,
    isovalue: f64,
    stage: IsosurfaceStage,
    progress: f64,
) {
    if let Some(sink) = progress_callback {
        sink.emit(ProgressMsg::IsosurfaceProgress {
            isovalue,
            stage,
            progress,
        });
    }
}

/// Emits a text progress message if a sink was provided.
#[inline]
fn emit_message(progress_callback: Option<&dyn ProgressSink>, message: impl Into<String>) {
    if let Some(sink) = progress_callback {
        sink.emit(ProgressMsg::Message {
            message: message.into(),
        });
    }
}

/// Convenience wrapper for [`build_isosurface`]` that can extract multiple meshes from a vec of
/// isovalues at once.
pub fn build_isosurfaces<F>(
    seed_points: MatRef<f64>,
    extents: &[f64],
    resolution: f64,
    isovalues: Vec<f64>,
    isosurface_fn: &mut F,
    gradient_fn: Option<&mut dyn FnMut(MatRef<f64>) -> (Mat<f64>, Mat<f64>)>,
    cluster_method: ClusterMethod,
    boundary_closure: BoundaryClosure,
    progress_callback: Option<&dyn ProgressSink>,
) -> Vec<Mesh>
where
    F: FnMut(MatRef<f64>) -> Mat<f64>,
{
    let mut meshes = Vec::with_capacity(isovalues.len());

    match gradient_fn {
        Some(gradient_fn) => {
            for &isovalue in &isovalues {
                meshes.push(build_isosurface(
                    seed_points,
                    extents,
                    resolution,
                    isovalue,
                    isosurface_fn,
                    Some(&mut *gradient_fn),
                    cluster_method,
                    boundary_closure,
                    progress_callback,
                ));
            }
        }
        None => {
            for &isovalue in &isovalues {
                meshes.push(build_isosurface(
                    seed_points,
                    extents,
                    resolution,
                    isovalue,
                    isosurface_fn,
                    None,
                    cluster_method,
                    boundary_closure,
                    progress_callback,
                ));
            }
        }
    }

    meshes
}

/// Extracts an isosurface from an implicit function using regularised marching tetrahedra.
///
/// `seed_points` is an `N x 3` matrix of points on, or near, the target isosurface. `extents`
/// must contain `[xmin, ymin, zmin, xmax, ymax, zmax]`. Seeds are projected onto
/// `f(x) = isovalue`, then used to expand a surface-following wavefront through the tetrahedral
/// lattice defined by `extents` and `resolution`.
///
/// `isosurface_fn` must return one scalar value per input point. If `gradient_fn` is provided it
/// is used for seed projection; otherwise gradients are estimated by central differences. The
/// selected [`ClusterMethod`] controls how topology-compatible edge intersections are combined
/// into mesh vertices. [`BoundaryClosure`] controls whether clipped AABB boundaries are closed.
pub fn build_isosurface<F>(
    seed_points: MatRef<f64>,
    extents: &[f64],
    resolution: f64,
    isovalue: f64,
    isosurface_fn: &mut F,
    gradient_fn: Option<&mut dyn FnMut(MatRef<f64>) -> (Mat<f64>, Mat<f64>)>,
    cluster_method: ClusterMethod,
    boundary_closure: BoundaryClosure,
    progress_callback: Option<&dyn ProgressSink>,
) -> Mesh
where
    F: FnMut(MatRef<f64>) -> Mat<f64>,
{
    assert_eq!(extents.len(), 6, "extents must have length 6");
    let extents = AABB {
        min_corner: [extents[0], extents[1], extents[2]],
        max_corner: [extents[3], extents[4], extents[5]],
    };
    let lattice = SampleLattice::new(resolution, extents);
    let bbox_eps = bbox_eps(extents);
    let mut vertices: Vec<f64> = Vec::new();
    let mut edge_ref: HashMap<([i64; 3], usize), usize> = HashMap::new();

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::ProjectingSeeds,
        0.0,
    );

    // Project the seed points to the level-set isosurface.
    let mut wavefront = if let Some(gradient_fn) = gradient_fn {
        seed_projection::get_unique_seed_point_ijks(seed_points, gradient_fn, &lattice, isovalue)
    } else {
        let mut gradient_fn = |targets: MatRef<f64>| {
            seed_projection::central_difference_values_and_gradients(
                targets,
                isosurface_fn,
                &lattice,
            )
        };

        seed_projection::get_unique_seed_point_ijks(
            seed_points,
            &mut gradient_fn,
            &lattice,
            isovalue,
        )
    };

    let mut sample_points: HashMap<[i64; 3], SamplePoint> = HashMap::new();
    let mut seen_cells = wavefront.clone();
    let mut evaluated: HashMap<[i64; 3], f64> = HashMap::new();

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::ExpandingWavefront,
        0.05,
    );

    // Wavefront expansion
    while !wavefront.is_empty() {
        let mut next_wavefront: HashSet<[i64; 3]> = HashSet::new();
        let mut unevaluated_world: Vec<f64> = Vec::new();
        let mut unevaluated_ijk: Vec<[i64; 3]> = Vec::new();

        // Since the main use-case for this crate is RBF evaluations, which can be expensive,
        // it's far more efficient to collect all the unnevaluated sample points in each wavefront
        // iteration and perform a batch evaluation.
        for cell in &wavefront {
            sample_points.entry(*cell).or_insert(SamplePoint::default());
            let corners = get_edge_points::<8>(cell);
            for corner in corners {
                if !evaluated.contains_key(&corner) {
                    unevaluated_ijk.push(corner);
                    unevaluated_world.extend_from_slice(lattice.ijk_to_world(corner).as_slice());
                }
            }
        }

        if !unevaluated_ijk.is_empty() {
            let nrows = unevaluated_ijk.len();
            let evaluated_values =
                isosurface_fn(MatRef::from_row_major_slice(&unevaluated_world, nrows, 3));
            for (ijk, val) in unevaluated_ijk.iter().zip(evaluated_values.row_iter()) {
                evaluated.insert(*ijk, val[0] - isovalue);
            }
        }

        for cell in &wavefront {
            let corners = get_edge_points::<8>(&cell);
            let corner_vals: Vec<f64> =
                corners.iter().map(|c| *evaluated.get(c).unwrap()).collect();
            let s0 = corner_vals[0];
            let inside0 = is_inside(s0);

            let mut cell_has_intersections = false;

            // Mark edge intersections on closest sample point
            for corner_idx in 1..8usize {
                let s1 = corner_vals[corner_idx];
                let inside1 = is_inside(s1);
                if inside0 == inside1 {
                    continue;
                }

                let eid = corner_idx - 1;
                let t = s0 / (s0 - s1);

                let nbr_key = corners[corner_idx];
                let rev = REVERSE_EDGE[eid];

                if t < 0.5 {
                    sample_points.get_mut(cell).unwrap().intersections |= 1u16 << eid;
                } else {
                    sample_points
                        .entry(nbr_key)
                        .or_insert(SamplePoint::default())
                        .intersections |= 1u16 << rev;
                }

                cell_has_intersections = true;
            }

            if !cell_has_intersections {
                continue;
            }

            // Expand the wavefront across faces with intersections
            for (face_ids, d) in FACES.iter().zip(FACE_DIRS.iter()) {
                let mut any_inside = false;
                let mut any_outside = false;

                for id in face_ids {
                    if is_inside(corner_vals[*id]) {
                        any_inside = true;
                    } else {
                        any_outside = true;
                    }
                }

                if !(any_inside && any_outside) {
                    continue;
                }

                let nbr = [
                    cell[0] + d[0] as i64,
                    cell[1] + d[1] as i64,
                    cell[2] + d[2] as i64,
                ];

                let mut any_inbounds = false;

                let nbr_corners = get_edge_points::<8>(&nbr);

                for c in nbr_corners {
                    if lattice.extraction_ijk_inbounds(c) {
                        any_inbounds = true;
                        break;
                    }
                }
                if !any_inbounds {
                    continue;
                }

                if seen_cells.contains(&nbr) {
                    continue;
                }

                seen_cells.insert(nbr);
                sample_points.entry(nbr).or_insert(SamplePoint::default());
                next_wavefront.insert(nbr);
            }
        }
        wavefront = next_wavefront;
    }

    // Evaluate any missing lattice points connected to lattice points with intersections for topology tests.
    let mut missing: HashMap<[i64; 3], [f64; 3]> = HashMap::new();

    for (ijk, point) in sample_points.iter() {
        if point.intersections == 0 {
            continue;
        }

        if !evaluated.contains_key(ijk) {
            missing.insert(*ijk, lattice.ijk_to_world(*ijk));
        }

        for [di, dj, dk] in EDGE_DELTAS {
            let edge = [ijk[0] + di as i64, ijk[1] + dj as i64, ijk[2] + dk as i64];
            if !evaluated.contains_key(&edge) {
                missing.insert(edge, lattice.ijk_to_world(edge));
            }
        }
    }

    if !missing.is_empty() {
        let missing_items: Vec<([i64; 3], [f64; 3])> =
            missing.iter().map(|(k, v)| (*k, *v)).collect();
        let nrows = missing_items.len();
        let missing_worlds: Vec<f64> = missing_items.iter().flat_map(|(_k, v)| *v).collect();
        let missing_values = isosurface_fn(MatRef::from_row_major_slice(&missing_worlds, nrows, 3));
        for ((ijk, _world), val) in missing_items.iter().zip(missing_values.row_iter()) {
            evaluated.insert(*ijk, val[0] - isovalue);
        }
    }

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::ClusteringVertices,
        0.7,
    );

    let keys: Vec<[i64; 3]> = sample_points.keys().copied().collect();
    let mut candidates: Vec<VertexCandidate> = Vec::new();
    let mut candidate_ref: HashMap<([i64; 3], usize), usize> = HashMap::new();
    let mut num_closed_surface = 0usize;
    let mut num_multi_hole = 0usize;
    let mut num_flat_hole = 0usize;
    let mut num_multi_surface = 0usize;
    let mut num_simple_surface = 0usize;

    // Cluster the near intersections for each sample point.
    for ijk in &keys {
        let sample = sample_points.get(ijk).unwrap();
        let intersections = sample.intersections;

        if intersections == 0 {
            continue;
        }

        let should_cluster = !matches!(cluster_method, ClusterMethod::None);

        let topology_result =
            topology::test_topology(intersections, should_cluster, *ijk, &evaluated);

        match topology_result.case {
            TopologyCase::ClosedSurface => num_closed_surface += 1,
            TopologyCase::MultiHole => num_multi_hole += 1,
            TopologyCase::FlatHole => num_flat_hole += 1,
            TopologyCase::MultiSurface => num_multi_surface += 1,
            TopologyCase::SimpleSurface => num_simple_surface += 1,
            _ => {}
        }

        for cluster in topology_result.iter_clusters() {
            let mut edge_endpoints: Vec<([i64; 3], [i64; 3])> =
                Vec::with_capacity(cluster.edges.len());
            let mut pts: Vec<[f64; 3]> = Vec::new();

            for &edge in &cluster.edges {
                let [di, dj, dk] = EDGE_DELTAS[edge as usize];

                let nbr = [ijk[0] + di as i64, ijk[1] + dj as i64, ijk[2] + dk as i64];

                if let Some(interpolated_point) =
                    edge_intersection_point(*ijk, nbr, &evaluated, &lattice)
                {
                    edge_endpoints.push((*ijk, nbr));
                    pts.push(interpolated_point);
                }
            }

            if pts.len() == 0 {
                continue;
            }

            let candidate = match cluster_method {
                ClusterMethod::CurvatureWeighted => {
                    curvature_weighting::curvature_weighted_cluster_point(
                        edge_endpoints.as_slice(),
                        &evaluated,
                        &lattice,
                    )
                    .unwrap_or_else(|| {
                        if pts.len() == 1 {
                            pts[0]
                        } else {
                            average_point(&pts)
                        }
                    })
                }
                ClusterMethod::Average | ClusterMethod::None => {
                    if pts.len() == 1 {
                        pts[0]
                    } else {
                        average_point(&pts)
                    }
                }
            };
            let candidate_id = candidates.len();
            for &(u, v) in &edge_endpoints {
                if let Some((owner, _other, lab)) = get_edge_owner(u, v) {
                    let key = (owner, lab);
                    candidate_ref.insert(key, candidate_id);
                }
            }

            candidates.push(VertexCandidate {
                point: candidate,
                edge_endpoints,
                owner: *ijk,
            });
        }
    }
    let mut predicted_edge_counts: HashMap<(usize, usize), usize> = HashMap::new();

    emit_message(
        progress_callback,
        format!(
            "Closed surfaces: {num_closed_surface}\n\
            Multi-holes: {num_multi_hole}\n\
            Flat holes: {num_flat_hole}\n\
            Multi-surfaces: {num_multi_surface}\n\
            Simple surfaces: {num_simple_surface}"
        ),
    );

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::BuildingFacets,
        0.82,
    );

    // The paper suggests the topology tests should guarantee that clustering of near-intersections
    // that pass the topology tests won't produce non-manifold edges. However, from testing with
    // various real-world RBF datasets this doesn't always prove to be true, unfortunately.
    // So we need to process candidates and find any non-manifold edges generated from clustering.
    // Any clusters that generate non-manifold edges are simply unclustered back to their raw
    // unclustered vertices.
    march_tets(
        &keys,
        &evaluated,
        |u, v| {
            let (owner, _other, lab) = get_edge_owner(u, v)?;
            let key = (owner, lab);
            candidate_ref.get(&key).copied()
        },
        |[v0, v1, v2]| {
            for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
                let key = if a <= b { (a, b) } else { (b, a) };
                *predicted_edge_counts.entry(key).or_insert(0) += 1;
            }
        },
    );

    let mut split_candidates = HashSet::new();
    for ((a, b), count) in predicted_edge_counts {
        if count <= 2 {
            continue;
        }
        if candidates[a].edge_endpoints.len() > 1 {
            split_candidates.insert(a);
        }
        if candidates[b].edge_endpoints.len() > 1 {
            split_candidates.insert(b);
        }
    }
    let mut cluster_vertex_edges: HashMap<usize, Vec<([i64; 3], [i64; 3])>> = HashMap::new();
    let mut cluster_vertex_owner: HashMap<usize, [i64; 3]> = HashMap::new();
    let mut owner_cluster_vertices: HashMap<[i64; 3], Vec<usize>> = HashMap::new();

    for (candidate_id, candidate) in candidates.iter().enumerate() {
        if split_candidates.contains(&candidate_id) {
            for (u, v) in &candidate.edge_endpoints {
                if let Some(p) = edge_intersection_point(*u, *v, &evaluated, &lattice) {
                    let vid = push_vertex(&mut vertices, p);
                    edge_ref_set(&mut edge_ref, *u, *v, vid);
                }
            }
        } else {
            let vid = push_vertex(&mut vertices, candidate.point);
            if candidate.edge_endpoints.len() > 1 {
                cluster_vertex_edges.insert(vid, candidate.edge_endpoints.clone());
                cluster_vertex_owner.insert(vid, candidate.owner);
                owner_cluster_vertices
                    .entry(candidate.owner)
                    .or_default()
                    .push(vid);
            }
            for (u, v) in &candidate.edge_endpoints {
                edge_ref_set(&mut edge_ref, *u, *v, vid);
            }
        }
    }
    let mut facets = Vec::new();

    march_tets(
        &keys,
        &evaluated,
        |u, v| edge_ref_get(&edge_ref, u, v),
        |[a, b, c]| facets.extend_from_slice(&[a, b, c]),
    );

    if !cluster_vertex_edges.is_empty() && !facets.is_empty() {
        let mut non_manifold_rollback_count = 0usize;

        for _ in 0..4 {
            let bad_owners =
                collect_invalid_topology_cluster_owners(&facets, &cluster_vertex_owner);
            if bad_owners.is_empty() {
                break;
            }

            let rolled_back_count = rollback_cluster_owners(
                bad_owners,
                &mut owner_cluster_vertices,
                &mut cluster_vertex_edges,
                &mut cluster_vertex_owner,
                &mut edge_ref,
                &mut vertices,
                &evaluated,
                &lattice,
            );

            if rolled_back_count == 0 {
                break;
            }

            non_manifold_rollback_count += rolled_back_count;

            facets.clear();
            march_tets(
                &keys,
                &evaluated,
                |u, v| edge_ref_get(&edge_ref, u, v),
                |[a, b, c]| facets.extend_from_slice(&[a, b, c]),
            );
        }

        emit_message(
            progress_callback,
            format!(
                "Rolled back {non_manifold_rollback_count} sample points from non-manifold edges."
            ),
        );
    }

    // Similar to the case with non-manifold edges, in some tested scenarios clustering can cause
    // self-intersections in highly complex geometry. So we also need to test for self-intersections.
    // As with the non-manifold edges, any clusters that generate self-intersections are simply
    // unclustered back to their raw unclustered vertices.
    let mut self_intersection_rollback_count = 0usize;

    if !cluster_vertex_edges.is_empty() && !facets.is_empty() {
        let nverts = vertices.len() / 3;
        let mut inside_facets = Vec::with_capacity(facets.len());
        let mut inside_to_original = Vec::with_capacity(facets.len() / 3);
        for tri_idx in 0..(facets.len() / 3) {
            if !facet_fully_inside_aabb(&vertices, &facets, tri_idx, extents, bbox_eps) {
                continue;
            }

            let base = tri_idx * 3;
            inside_facets.extend_from_slice(&facets[base..base + 3]);
            inside_to_original.push(tri_idx);
        }

        let nfacets = inside_facets.len() / 3;

        let intersecting_tris = mesh_intersections::get_intersecting_triangles(
            MatRef::from_row_major_slice(vertices.as_slice(), nverts, 3),
            MatRef::from_row_major_slice(inside_facets.as_slice(), nfacets, 3),
        );

        if !intersecting_tris.is_empty() {
            let mut bad_owners = HashSet::new();
            for filtered_tri_idx in intersecting_tris {
                let tri_idx = inside_to_original[filtered_tri_idx];
                let base = tri_idx * 3;
                if base + 2 >= facets.len() {
                    continue;
                }

                let tri_vids = [facets[base], facets[base + 1], facets[base + 2]];
                for vid in tri_vids {
                    if let Some(owner) = cluster_vertex_owner.get(&vid) {
                        bad_owners.insert(*owner);
                    }
                }
            }

            if !bad_owners.is_empty() {
                let rolled_back_count = rollback_cluster_owners(
                    bad_owners,
                    &mut owner_cluster_vertices,
                    &mut cluster_vertex_edges,
                    &mut cluster_vertex_owner,
                    &mut edge_ref,
                    &mut vertices,
                    &evaluated,
                    &lattice,
                );
                if rolled_back_count > 0 {
                    self_intersection_rollback_count += rolled_back_count;

                    facets.clear();
                    march_tets(
                        &keys,
                        &evaluated,
                        |u, v| edge_ref_get(&edge_ref, u, v),
                        |[a, b, c]| facets.extend_from_slice(&[a, b, c]),
                    );
                }
            }

            emit_message(
                progress_callback,
                format!(
                    "Rolled back {self_intersection_rollback_count} sample points from self-intersections."
                ),
            );
        }
    }

    // Clip the facets to the axis-aligned bounding box extents, to ensure a clean cut.
    let (vertices, facets) = clip_mesh_to_aabb(vertices, facets, extents, bbox_eps);

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::CleaningMesh,
        0.94,
    );

    // Clean the mesh to remove any vertices that are now unused after unclustering of problematic
    // sample points, as well as removing any single floating triangles.
    let (vertices, facets) = clean_mesh(vertices, facets, bbox_eps);

    emit_progress(
        progress_callback,
        isovalue,
        IsosurfaceStage::BoundaryClosure,
        0.97,
    );

    // Perform boundary closure using the requested method.
    let (vertices, facets) = boundary_closure::cap_mesh_to_aabb(
        vertices,
        facets,
        extents,
        resolution,
        boundary_closure,
        bbox_eps,
    );
    let nverts = vertices.len() / 3;
    let nfacets = facets.len() / 3;
    let vertices = MatRef::from_row_major_slice(vertices.as_slice(), nverts, 3).to_owned();
    let facets = MatRef::from_row_major_slice(facets.as_slice(), nfacets, 3).cloned();

    emit_progress(progress_callback, isovalue, IsosurfaceStage::Finished, 1.0);

    Mesh { vertices, facets }
}
