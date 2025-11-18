/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements a surface-following Surface Nets algorithm for extracting isosurfaces from RBF fields.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    progress::ProgressMsg,
    rbf::RBFInterpolator,
};
use faer::{Mat, Row, RowRef, row, stats};
use std::collections::{HashMap, HashSet};

const CUBE_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

const CUBE_EDGES: [(usize, usize, usize); 12] = [
    (0, 1, 0),
    (2, 3, 0),
    (4, 5, 0),
    (6, 7, 0),
    (0, 2, 1),
    (1, 3, 1),
    (4, 6, 1),
    (5, 7, 1),
    (0, 4, 2),
    (1, 5, 2),
    (2, 6, 2),
    (3, 7, 2),
];

fn get_cell_corners(ijk: &(i32, i32, i32)) -> [(i32, i32, i32); 8] {
    let (i, j, k) = *ijk;
    CUBE_CORNERS.map(|[dx, dy, dz]| (i + dx, j + dy, k + dz))
}

fn ijk_from_world(world: &RowRef<f64>, min_corner: &[f64], resolution: f64) -> (i32, i32, i32) {
    (
        ((world[0] - min_corner[0]) / resolution).floor() as i32,
        ((world[1] - min_corner[1]) / resolution).floor() as i32,
        ((world[2] - min_corner[2]) / resolution).floor() as i32,
    )
}

fn world_from_ijk(ijk: &(i32, i32, i32), min_corner: &[f64], resolution: f64) -> Row<f64> {
    row![
        min_corner[0] + ijk.0 as f64 * resolution,
        min_corner[1] + ijk.1 as f64 * resolution,
        min_corner[2] + ijk.2 as f64 * resolution
    ]
}

fn _get_neighbours(ijk: &(i32, i32, i32)) -> [(i32, i32, i32); 6] {
    let (i, j, k) = *ijk;
    [
        (i + 1, j, k),
        (i - 1, j, k),
        (i, j + 1, k),
        (i, j - 1, k),
        (i, j, k + 1),
        (i, j, k - 1),
    ]
}

fn is_inside_extent(point: &RowRef<f64>, min: &[f64], max: &[f64]) -> bool {
    point
        .iter()
        .zip(min.iter().zip(max.iter()))
        .all(|(&v, (&lo, &hi))| v >= lo && v <= hi)
}

fn seed_points_to_unique_ijk_from_values(
    seed_points: &Mat<f64>,
    seed_values: &Mat<f64>,
    isovalue: f64,
    tol: f64,
    min_corner: &[f64],
    resolution: f64,
) -> HashSet<(i32, i32, i32)> {
    seed_points
        .row_iter()
        .zip(seed_values.col(0).iter().copied())
        .filter_map(|(row, f)| {
            if (f - isovalue).abs() <= tol {
                Some(ijk_from_world(&row, min_corner, resolution))
            } else {
                None
            }
        })
        .collect()
}

#[inline]
fn sgn_from_bit(bit: i32) -> i32 {
    if bit == 0 { -1 } else { 1 }
}

/// Return up to 3 neighbour cells (relative to `ijk`) that share the edge
/// identified by corner index `i1` (the edge’s “min-bit” corner) and `axis`.
fn neighbours_sharing_edge(
    ijk: (i32, i32, i32),
    i1: usize,
    axis: usize, // 0=x, 1=y, 2=z
) -> [(i32, i32, i32); 3] {
    let (i, j, k) = ijk;
    let a = CUBE_CORNERS[i1];
    let xb = a[0] as i32;
    let yb = a[1] as i32;
    let zb = a[2] as i32;

    match axis {
        // Edge runs along X -> step across Y and Z faces
        0 => {
            let sy = sgn_from_bit(yb);
            let sz = sgn_from_bit(zb);
            [(i, j + sy, k), (i, j, k + sz), (i, j + sy, k + sz)]
        }
        // Edge runs along Y-> step across X and Z faces
        1 => {
            let sx = sgn_from_bit(xb);
            let sz = sgn_from_bit(zb);
            [(i + sx, j, k), (i, j, k + sz), (i + sx, j, k + sz)]
        }
        // Edge runs along Z -> step across X and Y faces
        _ => {
            let sx = sgn_from_bit(xb);
            let sy = sgn_from_bit(yb);
            [(i + sx, j, k), (i, j + sy, k), (i + sx, j + sy, k)]
        }
    }
}

fn get_cell_intersections(
    rbfi: &mut RBFInterpolator,
    resolution: &f64,
    min_corner: &[f64],
    max_corner: &[f64],
    isovalue: &f64,
) -> (
    HashMap<(i32, i32, i32), Vec<(((i32, i32, i32), (i32, i32, i32)), usize)>>,
    HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>>, // edge key -> world pos
    HashMap<(i32, i32, i32), f64>,                         // corner -> sdf value
) {
    let mut evaluated_corners: HashMap<(i32, i32, i32), f64> = HashMap::new();
    let mut active_cells = HashSet::new();
    let mut cell_intersections: HashMap<
        (i32, i32, i32),
        Vec<(((i32, i32, i32), (i32, i32, i32)), usize)>, // (edge_key, axis)
    > = HashMap::new();
    let mut edge_intersections: HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>> =
        HashMap::new();

    let mut frontier = seed_points_to_unique_ijk_from_values(
        &rbfi.points,
        &rbfi.point_values,
        *isovalue,
        *resolution,
        min_corner,
        *resolution,
    );

    while !frontier.is_empty() {
        let mut next_frontier = HashSet::new();
        let mut unevaluated_corners = HashSet::new();
        let mut cell_corner_map = HashMap::new();

        for ijk in &frontier {
            if active_cells.contains(ijk) {
                continue;
            }
            let corners = get_cell_corners(ijk);
            cell_corner_map.insert(*ijk, corners);
            for &corner in &corners {
                if !evaluated_corners.contains_key(&corner) {
                    unevaluated_corners.insert(corner);
                }
            }
        }

        if !unevaluated_corners.is_empty() {
            let mut corner_array = Mat::zeros(unevaluated_corners.len(), 3);
            for (i, corner) in unevaluated_corners.iter().enumerate() {
                corner_array
                    .row_mut(i)
                    .copy_from(world_from_ijk(corner, min_corner, *resolution));
            }
            let values = rbfi.evaluate_targets(&corner_array);
            for (corner, &val) in unevaluated_corners.iter().zip(values.col(0).iter()) {
                evaluated_corners.insert(*corner, val);
            }
        }

        for ijk in &frontier {
            if active_cells.contains(ijk) {
                continue;
            }
            if let Some(corners) = cell_corner_map.get(ijk) {
                let corner_vals: Vec<f64> = corners.iter().map(|c| evaluated_corners[c]).collect();
                let mut has_valid_intersection = false;

                for &(i1, i2, axis) in &CUBE_EDGES {
                    let corner_a = corners[i1];
                    let corner_b = corners[i2];
                    let v1 = corner_vals[i1];
                    let v2 = corner_vals[i2];

                    if (v1 > *isovalue) != (v2 > *isovalue) {
                        let key = (corner_a, corner_b);
                        let entry = edge_intersections.entry(key).or_insert_with(|| {
                            let mut pt = world_from_ijk(&corner_a, min_corner, *resolution);
                            let t = (isovalue - v1) / (v2 - v1);
                            pt[axis] += t * resolution;
                            pt
                        });
                        if is_inside_extent(&entry.as_ref(), min_corner, max_corner) {
                            has_valid_intersection = true;
                            cell_intersections
                                .entry(*ijk)
                                .or_default()
                                .push((key, axis));
                        }
                    }
                }

                if has_valid_intersection {
                    active_cells.insert(*ijk);
                    if let Some(edges) = cell_intersections.get(ijk) {
                        for &(edge_key, axis) in edges {
                            // Recover i1 from corner_a = edge_key.0 relative to this cell’s min corner (ijk)
                            let (ci, cj, ck) = edge_key.0; // corner_a (absolute grid-corner coords)
                            let (i0, j0, k0) = *ijk;
                            let xb = (ci - i0) as usize;
                            let yb = (cj - j0) as usize;
                            let zb = (ck - k0) as usize;

                            // Corner index in the same scheme as CUBE_CORNERS
                            let i1 = (xb | (yb << 1) | (zb << 2)) as usize;

                            let neighbours = neighbours_sharing_edge(*ijk, i1, axis);
                            for nbr in neighbours {
                                if !active_cells.contains(&nbr) {
                                    next_frontier.insert(nbr);
                                }
                            }
                        }
                    }
                }
            }
        }

        frontier = next_frontier;
    }

    (cell_intersections, edge_intersections, evaluated_corners)
}

#[inline]
fn trilinear_grad(
    u: f64,
    v: f64,
    w: f64,
    hx: f64,
    hy: f64,
    hz: f64,
    f000: f64,
    f100: f64,
    f010: f64,
    f110: f64,
    f001: f64,
    f101: f64,
    f011: f64,
    f111: f64,
) -> (f64, f64, f64) {
    // ∂f/∂u on [0,1]
    let c00 = f100 - f000;
    let c10 = f110 - f010;
    let c01 = f101 - f001;
    let c11 = f111 - f011;
    let df_du = (1.0 - v) * ((1.0 - w) * c00 + w * c01) + v * ((1.0 - w) * c10 + w * c11);

    // ∂f/∂v on [0,1]
    let d00 = f010 - f000;
    let d10 = f110 - f100;
    let d01 = f011 - f001;
    let d11 = f111 - f101;
    let df_dv = (1.0 - u) * ((1.0 - w) * d00 + w * d01) + u * ((1.0 - w) * d10 + w * d11);

    // ∂f/∂w on [0,1]
    let e00 = f001 - f000;
    let e10 = f101 - f100;
    let e01 = f011 - f010;
    let e11 = f111 - f110;
    let df_dw = (1.0 - u) * ((1.0 - v) * e00 + v * e01) + u * ((1.0 - v) * e10 + v * e11);

    // Map local -> world: ∂/∂x = (1/hx) ∂/∂u, etc.
    (df_du / hx, df_dv / hy, df_dw / hz)
}

fn normals_from_intersections_in_cell(
    ijk: &(i32, i32, i32),
    intersections: &Vec<((i32, i32, i32), (i32, i32, i32))>, // edge keys for this cell
    edge_intersections: &HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>>,
    evaluated_corners: &HashMap<(i32, i32, i32), f64>,
    min_corner: &[f64],
    resolution: f64,
) -> Vec<Row<f64>> {
    if intersections.is_empty() {
        return Vec::new();
    }

    // 8 corners in the same order as CUBE_CORNERS
    let corners = get_cell_corners(ijk);
    let vals: Vec<f64> = corners.iter().map(|c| evaluated_corners[c]).collect();
    let (f000, f100, f010, f110, f001, f101, f011, f111) = (
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7],
    );

    let origin = world_from_ijk(ijk, min_corner, resolution); // cell origin
    let (hx, hy, hz) = (resolution, resolution, resolution);

    let mut out = Vec::with_capacity(intersections.len());
    for key in intersections {
        // intersection world pos
        let p = edge_intersections
            .get(key)
            .expect("missing intersection point");
        // local coords in [0,1]
        let u = (p[0] - origin[0]) / hx;
        let v = (p[1] - origin[1]) / hy;
        let w = (p[2] - origin[2]) / hz;

        let (gx, gy, gz) = trilinear_grad(
            u, v, w, hx, hy, hz, f000, f100, f010, f110, f001, f101, f011, f111,
        );

        // normalize
        let mut n = row![gx, gy, gz];
        let len = n.squared_norm_l2().sqrt();
        if len > f64::EPSILON {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            n.fill(0.0);
        }
        out.push(n);
    }
    out
}

#[inline]
fn face_normal(v0: &Row<f64>, v1: &Row<f64>, v2: &Row<f64>) -> Row<f64> {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    row![
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]
}

fn get_quads(
    cell_vertices: HashMap<(i32, i32, i32), Row<f64>>,
    edge_intersections: HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>>,
    edge_normals: &HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>>, // NEW
) -> (Mat<f64>, Mat<usize>) {
    let mut vertex_indices: HashMap<(i32, i32, i32), usize> = HashMap::new();
    let mut vertices: Vec<Row<f64>> = Vec::with_capacity(cell_vertices.len());

    for (idx, ijk) in cell_vertices.keys().enumerate() {
        vertex_indices.insert(*ijk, idx);
        vertices.push(cell_vertices[ijk].clone());
    }

    let mut quads: Vec<[usize; 4]> = Vec::new();

    for (&(c1, c2), _) in edge_intersections.iter() {
        // Face axis detection
        let delta = (
            c1.0 as isize - c2.0 as isize,
            c1.1 as isize - c2.1 as isize,
            c1.2 as isize - c2.2 as isize,
        );
        let abs_delta = (delta.0.abs(), delta.1.abs(), delta.2.abs());
        let axis = if abs_delta.0 > abs_delta.1 && abs_delta.0 > abs_delta.2 {
            0
        } else if abs_delta.1 > abs_delta.2 {
            1
        } else {
            2
        };

        let (i, j, k) = c1;
        let (ijk0, ijk1, ijk2, ijk3) = match axis {
            0 => ((i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)), // YZ face
            1 => ((i - 1, j, k - 1), (i, j, k - 1), (i, j, k), (i - 1, j, k)), // XZ face
            2 => ((i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)), // XY face
            _ => continue,
        };

        if [ijk0, ijk1, ijk2, ijk3]
            .iter()
            .all(|&ijk| vertex_indices.contains_key(&ijk))
        {
            // default winding (ijk0, ijk1, ijk2, ijk3)
            let mut quad = [
                vertex_indices[&ijk0],
                vertex_indices[&ijk1],
                vertex_indices[&ijk2],
                vertex_indices[&ijk3],
            ];

            // Compute face normal from current winding
            let v0 = &vertices[quad[0]];
            let v1 = &vertices[quad[1]];
            let v2 = &vertices[quad[2]];
            let fnrm = face_normal(v0, v1, v2);

            // Reference normal from the intersection edge we're processing
            if let Some(n_ref) = edge_normals
                .get(&(c1, c2))
                .or_else(|| edge_normals.get(&(c2, c1)))
            {
                let dot = fnrm[0] * n_ref[0] + fnrm[1] * n_ref[1] + fnrm[2] * n_ref[2];
                if dot < 0.0 {
                    // flip winding: (0,1,2,3) -> (0,3,2,1)
                    quad = [quad[0], quad[3], quad[2], quad[1]];
                }
            }
            quads.push(quad);
        }
    }

    let vertex_mat = Mat::from_fn(vertices.len(), 3, |i, j| vertices[i][j]);
    let quad_mat = Mat::from_fn(quads.len(), 4, |i, j| quads[i][j]);
    (vertex_mat, quad_mat)
}

fn quads_to_faces(quads: Mat<usize>) -> Mat<usize> {
    let num_quads = quads.nrows();

    let mut faces = Mat::full(2 * num_quads, 3, 0);

    for i in 0..num_quads {
        let q0 = quads[(i, 0)];
        let q1 = quads[(i, 1)];
        let q2 = quads[(i, 2)];
        let q3 = quads[(i, 3)];

        // First triangle: (0, 1, 2)
        faces[(2 * i, 0)] = q0;
        faces[(2 * i, 1)] = q1;
        faces[(2 * i, 2)] = q2;

        // Second triangle: (2, 3, 0)
        faces[(2 * i + 1, 0)] = q2;
        faces[(2 * i + 1, 1)] = q3;
        faces[(2 * i + 1, 2)] = q0;
    }

    faces
}

pub fn surface_nets(
    extents: &Vec<f64>,
    resolution: f64,
    isovalues: &Vec<f64>,
    rbfi: &mut RBFInterpolator,
) -> (Vec<Mat<f64>>, Vec<Mat<usize>>) {
    let dimensions = rbfi.points.ncols();

    let mut evaluator_extents = extents.clone();

    evaluator_extents[0..dimensions]
        .iter_mut()
        .for_each(|val| *val -= resolution * 2.0);

    evaluator_extents[dimensions..]
        .iter_mut()
        .for_each(|val| *val += resolution * 2.0);

    rbfi.build_evaluator(Some(evaluator_extents));

    let (min_corner, max_corner) = extents.split_at(dimensions);

    let mut all_isosurface_points = Vec::new();
    let mut all_isosurface_faces = Vec::new();

    for isovalue in isovalues {
        if let Some(sink) = &rbfi.progress_callback {
            sink.emit(ProgressMsg::SurfacingProgress {
                isovalue: *isovalue,
                stage: String::from("Calculating surface intersections"),
                progress: 0.0,
            });
        }

        let (cell_intersections, edge_intersections, evaluated_corners) = get_cell_intersections(
            rbfi,
            &resolution,
            &min_corner,
            &max_corner,
            &isovalue,
        );

        // Store normals per edge intersection
        let mut edge_normals: HashMap<((i32, i32, i32), (i32, i32, i32)), Row<f64>> =
            HashMap::new();

        // Build cell vertices with trilinear normals
        let cell_vertices: HashMap<(i32, i32, i32), Row<f64>> = cell_intersections
            .iter()
            .map(|(key, intersections)| {
                let num_rows = intersections.len();
                let mut points = Mat::<f64>::zeros(num_rows, 3);

                let normals_vec = {
                    let edge_keys: Vec<_> = intersections.iter().map(|(ek, _axis)| *ek).collect();
                    normals_from_intersections_in_cell(
                        key,
                        &edge_keys,
                        &edge_intersections,
                        &evaluated_corners,
                        &min_corner,
                        resolution,
                    )
                };                

                for (idx, (edge_key, _axis)) in intersections.iter().enumerate() {
                    points
                        .row_mut(idx)
                        .copy_from(edge_intersections.get(edge_key).unwrap().cloned());

                    // Stash/average normals per edge
                    let n = &normals_vec[idx];
                    match edge_normals.get_mut(edge_key) {
                        Some(old) => {
                            let nx = 0.5 * (old[0] + n[0]);
                            let ny = 0.5 * (old[1] + n[1]);
                            let nz = 0.5 * (old[2] + n[2]);
                            let mut avg = row![nx, ny, nz];
                            let len = avg.squared_norm_l2().sqrt();
                            if len > f64::EPSILON {
                                avg[0] /= len;
                                avg[1] /= len;
                                avg[2] /= len;
                            } else {
                                avg.fill(0.0);
                            }
                            *old = avg;
                        }
                        None => {
                            edge_normals.insert(*edge_key, n.clone());
                        }
                    }                    
                }

                // pack normals
                let mut normals = Mat::<f64>::zeros(num_rows, 3);
                for (i, n) in normals_vec.iter().enumerate() {
                    normals.row_mut(i).copy_from(n.clone());
                }

                let mut best_point = Row::<f64>::zeros(3);
                stats::row_mean(best_point.as_mut(), points.as_ref(), stats::NanHandling::Ignore);
                (key.clone(), Row::from_iter(best_point.iter().cloned()))
            })
            .collect();

        if let Some(sink) = &rbfi.progress_callback {
            sink.emit(ProgressMsg::SurfacingProgress {
                isovalue: *isovalue,
                stage: String::from("Building quads"),
                progress: 0.8,
            });
        }
        let (vertices, quads) = get_quads(cell_vertices, edge_intersections, &edge_normals);

        if let Some(sink) = &rbfi.progress_callback {
            sink.emit(ProgressMsg::SurfacingProgress {
                isovalue: *isovalue,
                stage: String::from("Building faces"),
                progress: 0.9,
            });
        }

        let faces = quads_to_faces(quads);
        all_isosurface_faces.push(faces);
        all_isosurface_points.push(vertices);

        if let Some(sink) = &rbfi.progress_callback {
            sink.emit(ProgressMsg::SurfacingProgress {
                isovalue: *isovalue,
                stage: String::from("Finished"),
                progress: 1.0,
            });
        }
    }

    (all_isosurface_points, all_isosurface_faces)
}
