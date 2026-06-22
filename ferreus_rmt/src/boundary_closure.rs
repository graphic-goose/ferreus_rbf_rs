/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines AABB boundary closure for clipped triangle meshes.
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Boundary closure for meshes clipped to an axis-aligned bounding box.
//!
//! Surface-following extraction does not require a full 3D sampling grid, so a
//! bounding box is used to limit the extracted region. When the surface crosses
//! that box, clipping leaves open boundary loops on the box faces.
//!
//! Boundary closure closes those loops to create a watertight mesh. The closure can
//! be selected by treating the unknown field outside the box as either positive
//! or negative relative to the extracted isovalue. [`BoundaryClosure::None`]
//! leaves the clipped surface open, [`BoundaryClosure::ClosePositive`] treats the
//! outside as above the isovalue, and [`BoundaryClosure::CloseNegative`] treats the
//! outside as below the isovalue.

use std::collections::{HashMap, HashSet, VecDeque};

use spade::{
    ConstrainedDelaunayTriangulation, HasPosition, Point2, Triangulation,
    handles::FixedVertexHandle,
};

use super::{
    geometry::{Triangle, Triangle3},
    isosurface::AABB,
};

/// Boundary closure mode for open surfaces clipped to an AABB. Isosurfaces can be left open or
/// capped to create a watertight suface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryClosure {
    /// Do not add cap triangles; leave the clipped surface open.
    None,

    /// Close the surface as if values outside the AABB are above the isovalue.
    ClosePositive,

    /// Close the surface as if values outside the AABB are below the isovalue.
    CloseNegative,
}

/// One of the six faces of the clipping AABB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BoxFace {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

/// All AABB faces.
const BOX_FACES: [BoxFace; 6] = [
    BoxFace::XMin,
    BoxFace::XMax,
    BoxFace::YMin,
    BoxFace::YMax,
    BoxFace::ZMin,
    BoxFace::ZMax,
];

/// Open boundary edge from the clipped mesh, including the AABB faces it lies on.
#[derive(Clone, Copy, Debug)]
struct BoundaryEdge {
    /// Faces that contain this boundary edge.
    faces: FaceMask,

    /// First mesh vertex id, preserving the original oriented boundary edge.
    a: usize,

    /// Second mesh vertex id, preserving the original oriented boundary edge.
    b: usize,
}

impl BoundaryEdge {
    /// Returns `true` if the boundary edge belongs to `face`.
    fn on_face(self, face: BoxFace) -> bool {
        self.faces.contains(face)
    }
}

/// Bit mask over the six AABB faces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FaceMask(u8);

impl FaceMask {
    /// Inserts `face` into the mask.
    fn insert(&mut self, face: BoxFace) {
        self.0 |= 1 << face.index();
    }

    /// Returns `true` if `face` is present in the mask.
    fn contains(self, face: BoxFace) -> bool {
        self.0 & (1 << face.index()) != 0
    }

    /// Returns `true` if no faces are present.
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of faces in the mask.
    fn len(self) -> usize {
        self.0.count_ones() as usize
    }
}

/// Index into the candidate cap vertex pool.
type CandidateVertexId = usize;

/// Undirected edge between two candidate cap vertices.
type CandidateEdge = (CandidateVertexId, CandidateVertexId);

/// Quantised 3D point key used to deduplicate candidate cap vertices.
type PointKey = (i64, i64, i64);

/// Vertex in the temporary cap surface.
#[derive(Clone, Debug)]
struct CandidateCapVertex {
    /// Three-dimensional point on the AABB boundary.
    point: [f64; 3],

    /// Original mesh vertex id to reuse, if this candidate came from the input mesh.
    original_mesh_vertex: Option<usize>,

    /// AABB faces that contain this vertex.
    faces: FaceMask,
}

/// Triangle in the temporary candidate cap surface.
#[derive(Clone, Copy, Debug)]
struct CandidateCapTri {
    /// AABB face that owns this face-local CDT triangle.
    face: BoxFace,

    /// Candidate vertex ids.
    vertices: Triangle<CandidateVertexId>,
}

/// Temporary cap surface built over all six AABB faces.
struct CandidateCapSurface {
    /// AABB that bounds the clipped mesh.
    extents: AABB<f64>,

    /// Target spacing for generated boundary and face-grid candidates.
    resolution: f64,

    /// Geometric tolerance used for point matching and boundary tests.
    eps: f64,

    /// Global candidate vertex pool shared by all face-local CDTs.
    vertices: Vec<CandidateCapVertex>,

    /// Quantised map used to keep AABB edge and corner vertices shared.
    point_to_candidate: HashMap<PointKey, CandidateVertexId>,

    /// Integer AABB-grid coordinates mapped to their shared candidate vertices.
    grid_to_candidate: HashMap<[usize; 3], CandidateVertexId>,

    /// Candidate vertex ids present on each AABB face.
    face_vertices: [Vec<CandidateVertexId>; 6],

    /// Candidate triangles generated by the six face-local CDTs.
    tris: Vec<CandidateCapTri>,

    /// Undirected original mesh boundary edges that block flood-fill traversal.
    boundary_edges: HashSet<CandidateEdge>,

    /// Oriented boundary edges used to seed the ClosePositive cap region.
    oriented_boundary_edges: Vec<(BoxFace, CandidateVertexId, CandidateVertexId)>,
}

impl CandidateCapSurface {
    /// Creates an empty temporary cap surface on the AABB boundary.
    ///
    /// This is not the output mesh: generated vertices are only copied to the
    /// output if a selected cap triangle uses them during the final append pass.
    fn new(extents: AABB<f64>, resolution: f64, eps: f64) -> Self {
        Self {
            extents,
            resolution,
            eps,
            vertices: Vec::new(),
            point_to_candidate: HashMap::new(),
            grid_to_candidate: HashMap::new(),
            face_vertices: std::array::from_fn(|_| Vec::new()),
            tris: Vec::new(),
            boundary_edges: HashSet::new(),
            oriented_boundary_edges: Vec::new(),
        }
    }

    /// Adds generated grid support vertices on all AABB faces.
    ///
    /// The regular AABB grid gives each face enough triangulation support. The
    /// global integer grid map keeps edge and corner points identical across
    /// adjacent faces.
    fn add_aabb_grid_vertices(&mut self, mesh_vertices: &[f64]) {
        self.seed_existing_aabb_edge_vertices(mesh_vertices);
        let axes: [Vec<f64>; 3] = std::array::from_fn(|axis| {
            axis_grid_coordinates(
                self.extents.min_corner[axis],
                self.extents.max_corner[axis],
                self.resolution,
                self.eps,
            )
        });
        for face in BOX_FACES {
            self.add_face_grid_points(face, &axes);
        }
    }

    /// Detects input mesh boundary edges and inserts them as candidate barriers.
    ///
    /// Open input boundary edges become barriers in the candidate cap mesh.
    /// Their original orientation is retained because it determines which
    /// adjacent candidate triangle seeds ClosePositive.
    fn inject_mesh_boundary_edges(&mut self, mesh_vertices: &[f64], facets: &[usize]) {
        let boundary_edges = detect_boundary_edges(mesh_vertices, facets, self.extents, self.eps);
        for boundary_edge in boundary_edges {
            let pa = vertex_point(mesh_vertices, boundary_edge.a);
            let pb = vertex_point(mesh_vertices, boundary_edge.b);
            let a = self.insert_candidate(pa, Some(boundary_edge.a), boundary_edge.faces);
            let b = self.insert_candidate(pb, Some(boundary_edge.b), boundary_edge.faces);
            if a == b {
                continue;
            }

            self.boundary_edges.insert(sorted_pair(a, b));
            for face in BOX_FACES {
                if boundary_edge.on_face(face) {
                    self.oriented_boundary_edges.push((face, a, b));
                }
            }
        }
    }

    /// Builds constrained Delaunay triangulations for all AABB faces.
    fn build_face_cdts(&mut self) {
        for face in BOX_FACES {
            self.build_face_cdt(face);
        }
    }

    /// Selects the candidate triangles belonging to the ClosePositive cap region.
    ///
    /// ClosePositive represents the cap chosen when the field outside the AABB is
    /// treated as above the isovalue. CloseNegative is represented by the
    /// complement of this selected set.
    fn select_closed_plus(&self) -> Vec<bool> {
        let mut selected = vec![false; self.tris.len()];
        let mut graph = vec![Vec::<usize>::new(); self.tris.len()];
        let mut edge_tris: HashMap<CandidateEdge, Vec<usize>> = HashMap::new();

        // Build an edge-to-triangle map for the temporary cap surface. This
        // lets us find all candidate triangles adjacent across a shared edge.
        for (tri_idx, tri) in self.tris.iter().enumerate() {
            for (a, b) in tri.vertices.edges() {
                edge_tris
                    .entry(sorted_pair(a, b))
                    .or_default()
                    .push(tri_idx);
            }
        }

        // Build the traversal graph between neighbouring cap triangles. Input
        // mesh boundary edges are barriers, so the flood-fill must not cross
        // them.
        for (edge, adjacent) in &edge_tris {
            if self.boundary_edges.contains(edge) {
                continue;
            }
            for i in 0..adjacent.len() {
                for j in i + 1..adjacent.len() {
                    let a = adjacent[i];
                    let b = adjacent[j];
                    graph[a].push(b);
                    graph[b].push(a);
                }
            }
        }

        // Seed ClosePositive from the side of each oriented boundary edge that
        // corresponds to outside-box values being above the isovalue.
        let mut queue = VecDeque::new();
        for &(face, a, b) in &self.oriented_boundary_edges {
            let Some(adjacent) = edge_tris.get(&sorted_pair(a, b)) else {
                continue;
            };
            let desired_ccw = face.ccw_is_outward();
            for &tri_idx in adjacent {
                if self.tris[tri_idx].face != face {
                    continue;
                }
                let oriented = self.orient_face_triangle(self.tris[tri_idx], desired_ccw);
                if oriented.contains_oriented_edge(b, a) && !selected[tri_idx] {
                    selected[tri_idx] = true;
                    queue.push_back(tri_idx);
                }
            }
        }

        // Flood-fill from those seeds across the candidate cap surface. The
        // unvisited candidate triangles form the CloseNegative complement.
        while let Some(tri_idx) = queue.pop_front() {
            for &next_tri in &graph[tri_idx] {
                if !selected[next_tri] {
                    selected[next_tri] = true;
                    queue.push_back(next_tri);
                }
            }
        }

        selected
    }

    /// Orients a candidate triangle geometrically in its face-local UV coordinates.
    fn orient_face_triangle(
        &self,
        tri: CandidateCapTri,
        desired_ccw: bool,
    ) -> Triangle<CandidateVertexId> {
        let [a, b, c] = tri
            .vertices
            .map(|id| tri.face.project(self.vertices[id].point))
            .vertices();
        let signed_area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if (signed_area2 > 0.0) == desired_ccw {
            tri.vertices
        } else {
            tri.vertices.reversed()
        }
    }

    /// Inserts a generated point on one AABB face into the candidate pool.
    fn insert_generated_on_face(&mut self, point: [f64; 3], face: BoxFace) -> CandidateVertexId {
        let mut faces = FaceMask::default();
        faces.insert(face);
        self.insert_candidate(point, None, faces)
    }

    /// Inserts or updates a global candidate cap vertex.
    ///
    /// Candidate ids are global across all six faces. A point on an AABB edge or
    /// corner can therefore belong to multiple face-local CDTs while still being
    /// one vertex in the temporary cap surface. If an existing mesh vertex
    /// matches a previously generated point key, the candidate is upgraded so
    /// the final output reuses the original mesh vertex.
    fn insert_candidate(
        &mut self,
        point: [f64; 3],
        original_mesh_vertex: Option<usize>,
        faces: FaceMask,
    ) -> CandidateVertexId {
        let key = quantized_point_key(point, self.eps);
        if let Some(&id) = self.point_to_candidate.get(&key) {
            let candidate = &mut self.vertices[id];
            for face in BOX_FACES {
                if faces.contains(face) {
                    candidate.faces.insert(face);
                    push_unique(&mut self.face_vertices[face.index()], id);
                }
            }
            if let Some(mesh_id) = original_mesh_vertex {
                candidate.original_mesh_vertex = Some(mesh_id);
                candidate.point = point;
            }
            return id;
        }

        let id = self.vertices.len();
        self.vertices.push(CandidateCapVertex {
            point,
            original_mesh_vertex,
            faces,
        });
        self.point_to_candidate.insert(key, id);
        for face in BOX_FACES {
            if faces.contains(face) {
                push_unique(&mut self.face_vertices[face.index()], id);
            }
        }
        id
    }

    /// Seeds existing mesh vertices that lie on AABB edges or corners.
    fn seed_existing_aabb_edge_vertices(&mut self, mesh_vertices: &[f64]) {
        for id in 0..(mesh_vertices.len() / 3) {
            let p = vertex_point(mesh_vertices, id);
            let faces = aabb_face_mask(p, p, self.extents, self.eps);
            if faces.len() >= 2 {
                self.insert_candidate(p, Some(id), faces);
            }
        }
    }

    /// Adds support points from the shared integer-indexed AABB grid to one face.
    fn add_face_grid_points(&mut self, face: BoxFace, axes: &[Vec<f64>; 3]) {
        let boundary_segments: Vec<([f64; 2], [f64; 2])> = self
            .oriented_boundary_edges
            .iter()
            .filter(|(boundary_face, _, _)| *boundary_face == face)
            .map(|&(_, a, b)| {
                (
                    face.project(self.vertices[a].point),
                    face.project(self.vertices[b].point),
                )
            })
            .collect();

        let [u_axis, v_axis] = face.uv_axes();
        let u_last = axes[u_axis].len() - 1;
        let v_last = axes[v_axis].len() - 1;
        let grid_max = [axes[0].len() - 1, axes[1].len() - 1, axes[2].len() - 1];

        for u_idx in 0..=u_last {
            for v_idx in 0..=v_last {
                let uv = [axes[u_axis][u_idx], axes[v_axis][v_idx]];
                let is_face_interior =
                    u_idx != 0 && u_idx != u_last && v_idx != 0 && v_idx != v_last;
                if is_face_interior
                    && boundary_segments
                        .iter()
                        .any(|&(a, b)| point_on_segment2(uv, a, b, self.resolution * 1.0e-9))
                {
                    continue;
                }

                let ijk = face.grid_ijk(u_idx, v_idx, grid_max);
                let point = [axes[0][ijk[0]], axes[1][ijk[1]], axes[2][ijk[2]]];
                self.insert_grid_candidate(ijk, point, face);
            }
        }
    }

    /// Inserts a generated AABB-grid point, sharing its candidate across adjacent faces.
    fn insert_grid_candidate(&mut self, ijk: [usize; 3], point: [f64; 3], face: BoxFace) {
        if let Some(&id) = self.grid_to_candidate.get(&ijk) {
            let candidate = &mut self.vertices[id];
            candidate.faces.insert(face);
            push_unique(&mut self.face_vertices[face.index()], id);
            return;
        }

        let id = self.insert_generated_on_face(point, face);
        self.grid_to_candidate.insert(ijk, id);
    }

    /// Builds the constrained Delaunay triangulation for one AABB face.
    ///
    /// Each cube face is triangulated independently in its own 2D plane. The
    /// resulting CDT triangles are immediately converted back to global
    /// candidate ids so the split step can traverse across shared AABB edges.
    fn build_face_cdt(&mut self, face: BoxFace) {
        let candidate_ids = self.face_vertices[face.index()].clone();
        if candidate_ids.len() < 3 {
            return;
        }

        let mut local_vertices = Vec::with_capacity(candidate_ids.len());
        let mut candidate_to_local = HashMap::with_capacity(candidate_ids.len());
        for &candidate_id in &candidate_ids {
            let uv = face.project(self.vertices[candidate_id].point);
            let local_id = local_vertices.len();
            local_vertices.push(LocalVertex {
                position: Point2::new(uv[0], uv[1]),
                candidate: candidate_id,
            });
            candidate_to_local.insert(candidate_id, local_id);
        }

        let mut constraints = HashSet::new();
        self.add_face_box_boundary_constraints(face, &candidate_to_local, &mut constraints);
        let mut boundary_constraints = Vec::new();
        for &(boundary_face, a, b) in &self.oriented_boundary_edges {
            if boundary_face != face {
                continue;
            }
            let (Some(&local_a), Some(&local_b)) =
                (candidate_to_local.get(&a), candidate_to_local.get(&b))
            else {
                continue;
            };
            if local_a != local_b {
                constraints.insert(sorted_pair(local_a, local_b));
                boundary_constraints.push((local_a, local_b));
            }
        }

        let constraint_edges: Vec<[usize; 2]> = constraints.iter().map(|&(a, b)| [a, b]).collect();
        let mut conflicts = Vec::new();
        let cdt = ConstrainedDelaunayTriangulation::<LocalVertex>::try_bulk_load_cdt(
            local_vertices,
            constraint_edges,
            |edge| conflicts.push(edge),
        )
        .unwrap_or_else(|err| panic!("failed to build AABB face CDT for {face:?}: {err:?}"));
        if !conflicts.is_empty() {
            panic!("conflicting AABB cap constraints on {face:?}: {conflicts:?}");
        }

        for (a, b) in &boundary_constraints {
            let ab = cdt.exists_constraint(
                FixedVertexHandle::from_index(*a),
                FixedVertexHandle::from_index(*b),
            );
            let ba = cdt.exists_constraint(
                FixedVertexHandle::from_index(*b),
                FixedVertexHandle::from_index(*a),
            );
            if !ab && !ba {
                panic!("boundary edge {a}-{b} was not preserved as a CDT constraint on {face:?}");
            }
        }

        self.tris.extend(cdt.inner_faces().map(|tri_face| {
            let verts = tri_face.vertices();
            CandidateCapTri {
                face,
                vertices: Triangle::new(
                    verts[0].data().candidate,
                    verts[1].data().candidate,
                    verts[2].data().candidate,
                ),
            }
        }));
    }

    /// Adds constrained boundary chains around the perimeter of one face.
    ///
    /// Each perimeter chain is built from every candidate vertex lying on that
    /// side, not just from regular grid samples. This keeps existing mesh
    /// vertices on AABB edges/corners inside the constrained boundary chain.
    fn add_face_box_boundary_constraints(
        &self,
        face: BoxFace,
        candidate_to_local: &HashMap<CandidateVertexId, usize>,
        constraints: &mut HashSet<(usize, usize)>,
    ) {
        for side in face_boundary_sides(face, self.extents) {
            let mut side_vertices = Vec::new();
            for &candidate_id in &self.face_vertices[face.index()] {
                let Some(t) = self.candidate_on_face_side(face, candidate_id, side) else {
                    continue;
                };
                let Some(&local_id) = candidate_to_local.get(&candidate_id) else {
                    continue;
                };
                side_vertices.push((t, local_id));
            }

            side_vertices.sort_by(|a, b| a.0.total_cmp(&b.0));
            side_vertices.dedup_by_key(|(_, local_id)| *local_id);

            for pair in side_vertices.windows(2) {
                let a = pair[0].1;
                let b = pair[1].1;
                if a != b {
                    constraints.insert(sorted_pair(a, b));
                }
            }
        }
    }

    /// Returns the face-side interpolation coordinate for a candidate on `side`.
    fn candidate_on_face_side(
        &self,
        face: BoxFace,
        candidate_id: CandidateVertexId,
        side: ([f64; 2], [f64; 2]),
    ) -> Option<f64> {
        let uv = face.project(self.vertices[candidate_id].point);
        let ([u0, v0], [u1, v1]) = side;
        let du = u1 - u0;
        let dv = v1 - v0;
        let side_length = du.abs().max(dv.abs());
        let eps_t = self.eps / side_length.max(self.eps);

        let t = if du.abs() >= dv.abs() {
            if (uv[1] - v0).abs() > self.eps {
                return None;
            }
            (uv[0] - u0) / du
        } else {
            if (uv[0] - u0).abs() > self.eps {
                return None;
            }
            (uv[1] - v0) / dv
        };

        if t >= -eps_t && t <= 1.0 + eps_t {
            Some(t.clamp(0.0, 1.0))
        } else {
            None
        }
    }

    /// Counts generated vertices in tests.
    #[cfg(test)]
    fn generated_vertex_count(&self) -> usize {
        self.vertices
            .iter()
            .filter(|vertex| vertex.original_mesh_vertex.is_none())
            .count()
    }
}

/// Vertex payload used by Spade's face-local constrained triangulation.
#[derive(Clone, Copy, Debug)]
struct LocalVertex {
    /// Two-dimensional position in a face-local UV plane.
    position: Point2<f64>,

    /// Global candidate vertex represented by this local CDT vertex.
    candidate: CandidateVertexId,
}

impl HasPosition for LocalVertex {
    type Scalar = f64;

    fn position(&self) -> Point2<f64> {
        self.position
    }
}

/// Caps open mesh boundaries that lie on an AABB boundary.
///
/// `vertices` and `facets` are flat row-major buffers. If `mode` is
/// [`BoundaryClosure::None`], or if there are no open AABB boundary edges, the
/// original buffers are returned unchanged. [`BoundaryClosure::ClosePositive`]
/// closes the mesh as if values outside the box are above the isovalue, while
/// [`BoundaryClosure::CloseNegative`] closes it as if they are below.
pub(crate) fn cap_mesh_to_aabb(
    mut vertices: Vec<f64>,
    mut facets: Vec<usize>,
    extents: AABB<f64>,
    resolution: f64,
    mode: BoundaryClosure,
    eps: f64,
) -> (Vec<f64>, Vec<usize>) {
    if mode == BoundaryClosure::None || facets.is_empty() {
        return (vertices, facets);
    }

    let mut candidates = CandidateCapSurface::new(extents, resolution, eps);
    candidates.inject_mesh_boundary_edges(&vertices, &facets);

    if candidates.boundary_edges.is_empty() {
        return (vertices, facets);
    }

    candidates.add_aabb_grid_vertices(&vertices);
    candidates.build_face_cdts();
    let closed_plus = candidates.select_closed_plus();
    if mode == BoundaryClosure::CloseNegative {
        for tri in facets.chunks_exact_mut(3) {
            tri.swap(1, 2);
        }
    }
    append_selected_candidate_caps(&mut vertices, &mut facets, &candidates, &closed_plus, mode);
    (vertices, facets)
}

/// Detects open boundary edges that lie on the AABB boundary.
///
/// Each detected boundary edge preserves the original oriented edge from its incident
/// triangle so the closure flood can seed the correct side.
fn detect_boundary_edges(
    vertices: &[f64],
    facets: &[usize],
    extents: AABB<f64>,
    eps: f64,
) -> Vec<BoundaryEdge> {
    let mut edges: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for tri in facets.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.entry(sorted_pair(a, b)).or_default().push((a, b));
        }
    }

    let mut boundary_edges = Vec::new();
    for ((a, b), incident) in edges {
        if incident.len() != 1 {
            continue;
        }

        let pa = vertex_point(vertices, a);
        let pb = vertex_point(vertices, b);
        let faces = aabb_face_mask(pa, pb, extents, eps);
        if faces.is_empty() {
            panic!(
                "open boundary edge {a}-{b} is not on the AABB clipping boundary; cannot cap mesh"
            );
        }

        let (oa, ob) = incident[0];
        boundary_edges.push(BoundaryEdge {
            faces,
            a: oa,
            b: ob,
        });
    }
    boundary_edges
}

/// Appends selected candidate cap triangles to the output mesh.
///
/// This is the only place generated candidates become real output vertices.
/// Skipped CDT triangles therefore leave no unused vertices behind. Winding is
/// oriented outward from the AABB. For CloseNegative the input surface is reversed
/// before this pass because it bounds a cavity in the complementary volume.
fn append_selected_candidate_caps(
    vertices: &mut Vec<f64>,
    facets: &mut Vec<usize>,
    candidates: &CandidateCapSurface,
    closed_plus_selected: &[bool],
    mode: BoundaryClosure,
) {
    let mut materialized = vec![None; candidates.vertices.len()];
    let mut seen = HashSet::new();

    for (tri_idx, tri) in candidates.tris.iter().enumerate() {
        let keep = match mode {
            BoundaryClosure::None => false,
            BoundaryClosure::ClosePositive => closed_plus_selected[tri_idx],
            BoundaryClosure::CloseNegative => !closed_plus_selected[tri_idx],
        };

        if !keep {
            continue;
        }

        let desired_ccw = tri.face.ccw_is_outward();
        let ids = candidates
            .orient_face_triangle(*tri, desired_ccw)
            .map(|candidate_id| {
                materialize_candidate(candidate_id, candidates, vertices, &mut materialized)
            });
        // Triangle3::area2() returns the cross-product magnitude.
        if mesh_triangle(vertices, ids).area2() <= 1.0e-12 {
            continue;
        }
        let key = ids.sorted_vertices();
        if !seen.insert(key) {
            continue;
        }
        facets.extend_from_slice(&ids.vertices());
    }
}

/// Returns the output mesh vertex id for a candidate, appending it if needed.
fn materialize_candidate(
    candidate_id: CandidateVertexId,
    candidates: &CandidateCapSurface,
    vertices: &mut Vec<f64>,
    materialized: &mut [Option<usize>],
) -> usize {
    if let Some(mesh_id) = materialized[candidate_id] {
        return mesh_id;
    }

    let mesh_id = match candidates.vertices[candidate_id].original_mesh_vertex {
        Some(mesh_id) => mesh_id,
        None => push_vertex(vertices, candidates.vertices[candidate_id].point),
    };
    materialized[candidate_id] = Some(mesh_id);
    mesh_id
}

/// Returns `true` if a 2D point lies on a segment within tolerance.
fn point_on_segment2(p: [f64; 2], a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let bp = [p[0] - b[0], p[1] - b[1]];
    let cross = ab[0] * ap[1] - ab[1] * ap[0];
    if cross.abs() > eps * ab[0].hypot(ab[1]).max(1.0) {
        return false;
    }
    ap[0] * bp[0] + ap[1] * bp[1] <= eps * eps
}

/// Pushes `value` only if it is not already present.
fn push_unique<T: Eq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

/// Returns a stable undirected ordering for a pair.
fn sorted_pair<T: Ord>(a: T, b: T) -> (T, T) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Reads a 3D point from a flat row-major vertex buffer.
fn vertex_point(vertices: &[f64], id: usize) -> [f64; 3] {
    [vertices[3 * id], vertices[3 * id + 1], vertices[3 * id + 2]]
}

/// Appends `p` to a flat row-major vertex buffer and returns its id.
fn push_vertex(vertices: &mut Vec<f64>, p: [f64; 3]) -> usize {
    let id = vertices.len() / 3;
    vertices.extend_from_slice(&p);
    id
}

/// Returns evenly spaced coordinates whose interval count is derived from a target resolution.
fn axis_grid_coordinates(min: f64, max: f64, resolution: f64, eps: f64) -> Vec<f64> {
    let length = max - min;
    let segments = (length / resolution.max(eps)).ceil().max(1.0) as usize;
    (0..=segments)
        .map(|i| {
            if i == segments {
                max
            } else {
                min + length * (i as f64 / segments as f64)
            }
        })
        .collect()
}

/// Quantises a point into an integer key for tolerance-based deduplication.
fn quantized_point_key(p: [f64; 3], eps: f64) -> (i64, i64, i64) {
    let scale = eps.max(1.0e-12);
    (
        (p[0] / scale).round() as i64,
        (p[1] / scale).round() as i64,
        (p[2] / scale).round() as i64,
    )
}

/// Returns the AABB faces containing both endpoints of an edge.
fn aabb_face_mask(a: [f64; 3], b: [f64; 3], extents: AABB<f64>, eps: f64) -> FaceMask {
    let mut mask = FaceMask::default();
    for face in BOX_FACES {
        if face.point_on(a, extents, eps) && face.point_on(b, extents, eps) {
            mask.insert(face);
        }
    }
    mask
}

/// Builds a [`Triangle3`] by looking up vertex ids in a flat vertex buffer.
fn mesh_triangle(vertices: &[f64], ids: Triangle<usize>) -> Triangle3 {
    ids.map(|id| vertex_point(vertices, id))
}

/// Returns the four oriented perimeter sides of a face in UV coordinates.
fn face_boundary_sides(face: BoxFace, extents: AABB<f64>) -> [([f64; 2], [f64; 2]); 4] {
    let [umin, umax, vmin, vmax] = face.uv_bounds(extents);
    [
        ([umin, vmin], [umax, vmin]),
        ([umax, vmin], [umax, vmax]),
        ([umax, vmax], [umin, vmax]),
        ([umin, vmax], [umin, vmin]),
    ]
}

impl BoxFace {
    /// Returns the stable array index for this face.
    fn index(self) -> usize {
        match self {
            BoxFace::XMin => 0,
            BoxFace::XMax => 1,
            BoxFace::YMin => 2,
            BoxFace::YMax => 3,
            BoxFace::ZMin => 4,
            BoxFace::ZMax => 5,
        }
    }

    /// Returns `true` if `p` lies on this AABB face.
    fn point_on(self, p: [f64; 3], extents: AABB<f64>, eps: f64) -> bool {
        match self {
            BoxFace::XMin => (p[0] - extents.min_corner[0]).abs() <= eps,
            BoxFace::XMax => (p[0] - extents.max_corner[0]).abs() <= eps,
            BoxFace::YMin => (p[1] - extents.min_corner[1]).abs() <= eps,
            BoxFace::YMax => (p[1] - extents.max_corner[1]).abs() <= eps,
            BoxFace::ZMin => (p[2] - extents.min_corner[2]).abs() <= eps,
            BoxFace::ZMax => (p[2] - extents.max_corner[2]).abs() <= eps,
        }
    }

    /// Projects a 3D point on this face into face-local UV coordinates.
    fn project(self, p: [f64; 3]) -> [f64; 2] {
        match self {
            BoxFace::XMin | BoxFace::XMax => [p[1], p[2]],
            BoxFace::YMin | BoxFace::YMax => [p[0], p[2]],
            BoxFace::ZMin | BoxFace::ZMax => [p[0], p[1]],
        }
    }

    /// Returns the world axes represented by this face's local U and V coordinates.
    fn uv_axes(self) -> [usize; 2] {
        match self {
            BoxFace::XMin | BoxFace::XMax => [1, 2],
            BoxFace::YMin | BoxFace::YMax => [0, 2],
            BoxFace::ZMin | BoxFace::ZMax => [0, 1],
        }
    }

    /// Converts face-local grid indices to a global AABB-grid coordinate.
    fn grid_ijk(self, u: usize, v: usize, grid_max: [usize; 3]) -> [usize; 3] {
        match self {
            BoxFace::XMin => [0, u, v],
            BoxFace::XMax => [grid_max[0], u, v],
            BoxFace::YMin => [u, 0, v],
            BoxFace::YMax => [u, grid_max[1], v],
            BoxFace::ZMin => [u, v, 0],
            BoxFace::ZMax => [u, v, grid_max[2]],
        }
    }

    /// Returns `[umin, umax, vmin, vmax]` for this face's UV domain.
    fn uv_bounds(self, extents: AABB<f64>) -> [f64; 4] {
        match self {
            BoxFace::XMin | BoxFace::XMax => [
                extents.min_corner[1],
                extents.max_corner[1],
                extents.min_corner[2],
                extents.max_corner[2],
            ],
            BoxFace::YMin | BoxFace::YMax => [
                extents.min_corner[0],
                extents.max_corner[0],
                extents.min_corner[2],
                extents.max_corner[2],
            ],
            BoxFace::ZMin | BoxFace::ZMax => [
                extents.min_corner[0],
                extents.max_corner[0],
                extents.min_corner[1],
                extents.max_corner[1],
            ],
        }
    }

    /// Returns whether counter-clockwise UV winding points outward for this face.
    fn ccw_is_outward(self) -> bool {
        match self {
            BoxFace::XMin => false,
            BoxFace::XMax => true,
            BoxFace::YMin => true,
            BoxFace::YMax => false,
            BoxFace::ZMin => false,
            BoxFace::ZMax => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary_edge_count(facets: &[usize]) -> usize {
        edge_incidence_counts(facets)
            .values()
            .filter(|&&count| count == 1)
            .count()
    }

    fn edge_incidence_counts(facets: &[usize]) -> HashMap<(usize, usize), usize> {
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for tri in facets.chunks_exact(3) {
            for (a, b) in Triangle::new(tri[0], tri[1], tri[2]).edges() {
                *edges.entry(sorted_pair(a, b)).or_default() += 1;
            }
        }
        edges
    }

    fn assert_consistently_oriented(facets: &[usize]) {
        let mut edge_directions: HashMap<(usize, usize), i32> = HashMap::new();
        for tri in facets.chunks_exact(3) {
            for (a, b) in Triangle::new(tri[0], tri[1], tri[2]).edges() {
                let (edge, direction) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                *edge_directions.entry(edge).or_default() += direction;
            }
        }
        for (edge, direction) in edge_directions {
            assert_eq!(direction, 0, "edge {edge:?} has inconsistent winding");
        }
    }

    fn all_vertices_on_z(vertices: &[f64], tri: &[usize], z: f64) -> bool {
        tri.iter()
            .all(|&vid| (vertices[3 * vid + 2] - z).abs() <= 1.0e-9)
    }

    fn unreferenced_vertex_count(vertices: &[f64], facets: &[usize]) -> usize {
        let nverts = vertices.len() / 3;
        let mut used = vec![false; nverts];
        for &vid in facets {
            used[vid] = true;
        }
        used.into_iter().filter(|used| !*used).count()
    }

    fn horizontal_sheet_case() -> (AABB<f64>, Vec<f64>, Vec<usize>, Vec<(usize, usize)>) {
        let extents = AABB {
            min_corner: [0.0, 0.0, 0.0],
            max_corner: [1.0, 1.0, 1.0],
        };
        let vertices = vec![0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0, 0.5];
        let facets = vec![0, 1, 2, 0, 2, 3];
        let boundary_edges = vec![
            sorted_pair(0, 1),
            sorted_pair(1, 2),
            sorted_pair(2, 3),
            sorted_pair(0, 3),
        ];
        (extents, vertices, facets, boundary_edges)
    }

    fn assert_original_boundary_edges_have_one_cap_triangle(
        facets: &[usize],
        original_boundary_edges: &[(usize, usize)],
    ) {
        let counts = edge_incidence_counts(facets);
        for edge in original_boundary_edges {
            assert_eq!(counts.get(edge).copied(), Some(2), "boundary edge {edge:?}");
        }
    }

    fn assert_aabb_edge_segments_are_shared(
        vertices: &[f64],
        facets: &[usize],
        extents: AABB<f64>,
    ) {
        let counts = edge_incidence_counts(facets);
        for (&(a, b), &count) in &counts {
            let pa = vertex_point(vertices, a);
            let pb = vertex_point(vertices, b);
            if aabb_face_mask(pa, pb, extents, 1.0e-9).len() >= 2 {
                assert_eq!(count, 2, "AABB-edge segment {a}-{b} is not shared");
            }
        }
    }

    fn build_horizontal_sheet_caps(
        mode: BoundaryClosure,
    ) -> (Vec<f64>, Vec<usize>, CandidateCapSurface, Vec<bool>) {
        let (extents, mut vertices, facets, _boundary_edges) = horizontal_sheet_case();
        let mut candidates = CandidateCapSurface::new(extents, 0.25, 1.0e-9);
        candidates.inject_mesh_boundary_edges(&vertices, &facets);
        candidates.add_aabb_grid_vertices(&vertices);
        candidates.build_face_cdts();
        let plus_selected = candidates.select_closed_plus();
        let mut out_facets = facets;
        append_selected_candidate_caps(
            &mut vertices,
            &mut out_facets,
            &candidates,
            &plus_selected,
            mode,
        );
        (vertices, out_facets, candidates, plus_selected)
    }

    #[test]
    fn closed_plus_reuses_single_face_boundary_edges() {
        let extents = AABB {
            min_corner: [0.0, 0.0, 0.0],
            max_corner: [1.0, 1.0, 1.0],
        };
        let vertices = vec![0.25, 0.25, 0.0, 0.75, 0.25, 0.0, 0.25, 0.75, 0.0];
        let facets = vec![0, 1, 2];

        let (_vertices, facets) = cap_mesh_to_aabb(
            vertices,
            facets,
            extents,
            0.25,
            BoundaryClosure::ClosePositive,
            1e-9,
        );

        assert_eq!(boundary_edge_count(&facets), 0);
        assert_eq!(unreferenced_vertex_count(&_vertices, &facets), 0);
        assert!(facets.len() >= 6);
    }

    #[test]
    fn closed_minus_without_aabb_boundary_edges_leaves_mesh_unchanged() {
        let extents = AABB {
            min_corner: [0.0, 0.0, 0.0],
            max_corner: [1.0, 1.0, 1.0],
        };
        let vertices = vec![
            0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75,
        ];
        let facets = vec![0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];

        let (out_vertices, out_facets) = cap_mesh_to_aabb(
            vertices.clone(),
            facets.clone(),
            extents,
            0.25,
            BoundaryClosure::CloseNegative,
            1e-9,
        );

        assert_eq!(out_vertices, vertices);
        assert_eq!(out_facets, facets);
    }

    #[test]
    fn rejected_cdt_generated_points_are_not_materialized() {
        let (vertices, _facets, candidates, _selected) =
            build_horizontal_sheet_caps(BoundaryClosure::ClosePositive);
        let materialized = vertices.len() / 3 - horizontal_sheet_case().1.len() / 3;

        assert!(materialized < candidates.generated_vertex_count());
        assert_eq!(
            materialized,
            vertices.len() / 3 - horizontal_sheet_case().1.len() / 3
        );
    }

    #[test]
    fn closed_plus_global_region_includes_bottom_face_without_direct_boundary_edge() {
        let (extents, vertices, facets, boundary_edges) = horizontal_sheet_case();

        let (out_vertices, out_facets) = cap_mesh_to_aabb(
            vertices,
            facets,
            extents,
            0.25,
            BoundaryClosure::ClosePositive,
            1e-9,
        );

        assert_eq!(boundary_edge_count(&out_facets), 0);
        assert_eq!(unreferenced_vertex_count(&out_vertices, &out_facets), 0);
        assert_eq!(&out_facets[..6], &[0, 1, 2, 0, 2, 3]);
        assert_original_boundary_edges_have_one_cap_triangle(&out_facets, &boundary_edges);
        assert_aabb_edge_segments_are_shared(&out_vertices, &out_facets, extents);
        assert_consistently_oriented(&out_facets);
        assert!(
            out_facets.chunks_exact(3).any(|tri| all_vertices_on_z(
                &out_vertices,
                tri,
                extents.min_corner[2]
            )),
            "ClosePositive should flood onto the bottom face even though no boundary edge lies on it"
        );
    }

    #[test]
    fn closed_minus_global_complement_excludes_bottom_face_without_direct_boundary_edge() {
        let (extents, vertices, facets, boundary_edges) = horizontal_sheet_case();

        let (out_vertices, out_facets) = cap_mesh_to_aabb(
            vertices,
            facets,
            extents,
            0.25,
            BoundaryClosure::CloseNegative,
            1e-9,
        );

        assert_eq!(boundary_edge_count(&out_facets), 0);
        assert_eq!(unreferenced_vertex_count(&out_vertices, &out_facets), 0);
        assert_eq!(&out_facets[..6], &[0, 2, 1, 0, 3, 2]);
        assert_original_boundary_edges_have_one_cap_triangle(&out_facets, &boundary_edges);
        assert_aabb_edge_segments_are_shared(&out_vertices, &out_facets, extents);
        assert_consistently_oriented(&out_facets);
        assert!(
            !out_facets.chunks_exact(3).any(|tri| all_vertices_on_z(
                &out_vertices,
                tri,
                extents.min_corner[2]
            )),
            "CloseNegative should use the global complement and exclude the bottom face"
        );
    }

    #[test]
    fn off_grid_existing_aabb_edge_vertices_are_boundary_constraints() {
        let extents = AABB {
            min_corner: [0.0, 0.0, 0.0],
            max_corner: [1.0, 1.0, 1.0],
        };
        let vertices = vec![
            0.0, 0.0, 0.37, 1.0, 0.0, 0.37, 1.0, 1.0, 0.37, 0.0, 1.0, 0.37,
        ];
        let facets = vec![0, 1, 2, 0, 2, 3];
        let boundary_edges = vec![
            sorted_pair(0, 1),
            sorted_pair(1, 2),
            sorted_pair(2, 3),
            sorted_pair(0, 3),
        ];

        let (out_vertices, out_facets) = cap_mesh_to_aabb(
            vertices,
            facets,
            extents,
            0.5,
            BoundaryClosure::ClosePositive,
            1.0e-9,
        );

        assert_eq!(boundary_edge_count(&out_facets), 0);
        assert_eq!(unreferenced_vertex_count(&out_vertices, &out_facets), 0);
        assert_original_boundary_edges_have_one_cap_triangle(&out_facets, &boundary_edges);
        assert_aabb_edge_segments_are_shared(&out_vertices, &out_facets, extents);
    }

    #[test]
    fn closed_minus_non_even_grid_is_watertight() {
        let extents = AABB {
            min_corner: [-1.1, -1.1, -1.1],
            max_corner: [0.0, 1.1, 1.1],
        };
        let vertices = vec![
            0.0, -0.5, -0.5, //
            0.0, 0.5, -0.5, //
            0.0, 0.5, 0.5, //
            0.0, -0.5, 0.5,
        ];
        // Reversed XMax winding makes CloseNegative select the global box-face complement.
        let facets = vec![0, 2, 1, 0, 3, 2];

        let (out_vertices, out_facets) = cap_mesh_to_aabb(
            vertices,
            facets,
            extents,
            0.2,
            BoundaryClosure::CloseNegative,
            1.0e-9,
        );

        assert_eq!(boundary_edge_count(&out_facets), 0);
        assert!(
            edge_incidence_counts(&out_facets)
                .values()
                .all(|&count| count == 2)
        );
        assert_aabb_edge_segments_are_shared(&out_vertices, &out_facets, extents);
        assert_consistently_oriented(&out_facets);
    }
}
