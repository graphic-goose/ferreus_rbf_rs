/////////////////////////////////////////////////////////////////////////////////////////////
//
// Exposes the public API and high-level documentation for regularised marching tetrahedra.
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # Fast surface-following isosurface extraction
//!
//! This crate implements a surface-following variant of regularised marching
//! tetrahedra (RMT) for extracting isosurfaces from implicit scalar fields.
//!
//! Traditional isosurface extraction methods, such as marching cubes and marching
//! tetrahedra, usually sample the full volume of interest. That can be expensive
//! for costly implicit functions, and can produce meshes with large numbers of
//! poorly shaped triangles.
//!
//! Regularised marching tetrahedra combines marching tetrahedra with vertex
//! clustering. The result is an isosurface mesh that remains consistent with the
//! sampled field while typically using fewer faces and producing better-shaped
//! triangles than standard marching tetrahedra or marching cubes.
//!
//! This crate extends RMT with surface-following extraction. Instead of evaluating
//! the implicit function across the entire volume, extraction starts from one or
//! more user-provided seed points and expands across the surface. This is
//! particularly useful for signed-distance fields, radial basis function (RBF)
//! interpolants, and other implicit functions where evaluations are expensive and
//! an approximate surface location is already known.
//!
//! Seed points do not need to lie exactly on the target isosurface. Before
//! wavefront expansion begins, each seed is projected onto the surface using a
//! gradient-based search. Users may provide an analytic gradient function, or allow
//! gradients to be estimated from the scalar field using central differences.
//!
//! See the repository's examples directory for complete usage examples.
//!
//! # Features
//!
//! - Surface-following isosurface extraction from implicit scalar fields
//! - Regularised marching tetrahedra with vertex clustering
//! - Reduced evaluation counts compared with full-volume sampling
//! - Improved triangle quality compared with standard marching tetrahedra
//! - Manifold, self-intersection-free mesh generation
//! - Optional watertight extraction against an axis-aligned bounding box (AABB)
//!
//! # Examples
//!
//! ```
//! use faer::{mat, row, Mat, MatRef};
//! use ferreus_rmt::{self, BoundaryClosure, ClusterMethod};
//!
//! // Define the implicit function for a sphere.
//! pub fn sphere(pts: MatRef<'_, f64>) -> Mat<f64> {
//!     Mat::<f64>::from_fn(pts.nrows(), 1, |r, _| {
//!         let row = pts.row(r);
//!         row.norm_l2() - 1.0
//!     })
//! }
//!
//! // Define the axis-aligned bounding box extents to extract the isosurface within:
//! // [xmin, ymin, zmin, xmax, ymax, zmax].
//! let extents = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5];
//!
//! // Define the resolution of the sample lattice.
//! let resolution = 0.2;
//!
//! // Define the isovalue of the implicit function to surface.
//! let isovalue = 0.0;
//!
//! // Define a closure for the isosurface fucntion.
//! let mut surface_fn = move |targets: MatRef<f64>| sphere(targets);
//!
//! // Define some seed points on, or near, the isosurface to seed the wavefront.
//! let seed_points = mat![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],];
//!
//! // Define the vertex clustering method. Curvature-weighted is recommended.
//! let cluster_method = ClusterMethod::CurvatureWeighted;
//!
//! // Define the boundary closure method. In this case, since the extents are beyond
//! // the sphere and we know the sphere will be closed None is fine.
//! let boundary_closure = BoundaryClosure::None;
//!
//! // Extract the isosurface.
//! let mesh = ferreus_rmt::build_isosurface(
//!     seed_points.as_ref(),
//!     &extents,
//!     resolution,
//!     isovalue,
//!     &mut surface_fn,
//!     None,
//!     cluster_method,
//!     boundary_closure,
//!     None,
//! );
//!
//! // Save the isosurface out to an obj file
//! let name = "sphere";
//! let outpath = "sphere.obj";
//! mesh.save_obj(outpath, &name);
//!
//! assert_eq!(mesh.vertices.nrows(), 540);
//! assert_eq!(mesh.facets.nrows(), 1076);
//! ```
//!
//! # References
//! 1.  G.M. Treece, R.W. Prager, and A.H. Gee. Regularised marching tetrahedra: improved
//!     iso-surface extraction. Computers & Graphics, 23(4):583–598, 1999.

mod aabb_clipping;
mod boundary_closure;
mod constants;
mod curvature_weighting;
mod geometry;
mod isosurface;
mod lattice;
mod mesh;
mod mesh_cleanup;
mod mesh_intersections;
mod moller;
pub mod progress;
mod seed_projection;
mod topology;

pub use {
    boundary_closure::BoundaryClosure,
    isosurface::{ClusterMethod, build_isosurface, build_isosurfaces},
    mesh::Mesh,
};
