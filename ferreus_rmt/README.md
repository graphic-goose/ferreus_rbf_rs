# Fast surface-following isosurface extraction

This crate implements a surface-following variant of regularised marching
tetrahedra (RMT) for extracting isosurfaces from implicit scalar fields.

Traditional isosurface extraction methods, such as marching cubes and marching
tetrahedra, usually sample the full volume of interest. That can be expensive
for costly implicit functions, and can produce meshes with large numbers of
poorly shaped triangles.

Regularised marching tetrahedra combines marching tetrahedra with vertex
clustering. The result is an isosurface mesh that remains consistent with the
sampled field while typically using fewer faces and producing better-shaped
triangles than standard marching tetrahedra or marching cubes.

This crate extends RMT with surface-following extraction. Instead of evaluating
the implicit function across the entire volume, extraction starts from one or
more user-provided seed points and expands across the surface. This is
particularly useful for signed-distance fields, radial basis function (RBF)
interpolants, and other implicit functions where evaluations are expensive and
an approximate surface location is already known.

Seed points do not need to lie exactly on the target isosurface. Before
wavefront expansion begins, each seed is projected onto the surface using a
gradient-based search. Users may provide an analytic gradient function, or allow
gradients to be estimated from the scalar field using central differences.

See the examples section for complete usage examples.

---
# Features
- Regularised marching tetrahedra with vertex clustering
- Reduced evaluation counts compared with full-volume sampling
- Improved triangle quality compared with standard marching tetrahedra
- Manifold, self-intersection-free mesh generation
- Optional watertight extraction against an axis-aligned bounding box (AABB)

---
## Getting started

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
ferreus_rmt = "0.1"
```

---
## Quick start

```Rust
use faer::{mat, row, Mat, MatRef};
use ferreus_rmt::{self, BoundaryClosure, ClusterMethod};

// Define the implicit function for a sphere.
pub fn sphere(pts: MatRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(pts.nrows(), 1, |r, _| {
        let row = pts.row(r);
        row.norm_l2() - 1.0
    })
}

// Define the axis-aligned bounding box extents to extract the isosurface within:
// [xmin, ymin, zmin, xmax, ymax, zmax].
let extents = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5];

// Define the resolution of the sample lattice.
let resolution = 0.2;

// Define the isovalue of the implicit function to surface.
let isovalue = 0.0;

// Define a closure for the isosurface fucntion.
let mut surface_fn = move |targets: MatRef<f64>| sphere(targets);

// Define some seed points on, or near, the isosurface to seed the wavefront.
let seed_points = mat![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],];

// Define the vertex clustering method. Curvature-weighted is recommended.
let cluster_method = ClusterMethod::CurvatureWeighted;

// Define the boundary closure method. In this case, since the extents are beyond
// the sphere and we know the sphere will be closed None is fine.
let boundary_closure = BoundaryClosure::None;

// Extract the isosurface.
let mesh = ferreus_rmt::build_isosurface(
    seed_points.as_ref(),
    &extents,
    resolution,
    isovalue,
    &mut surface_fn,
    None,
    cluster_method,
    boundary_closure,
    None,
);

// Save the isosurface out to an obj file
let name = "sphere";
let outpath = "sphere.obj";
mesh.save_obj(outpath, &name);

```
![Implicit sphere isosurface](docs/assets/images/isosurface_sphere.png)

---
## References
1.  G.M. Treece, R.W. Prager, and A.H. Gee. Regularised marching tetrahedra: improved
    iso-surface extraction. Computers & Graphics, 23(4):583–598, 1999.

---
## Attribution and licensing

This package was developed while the author was working at
[Maptek](https://www.maptek.com) and has been approved for open-source
distribution under the terms of the MIT license.

Unless otherwise stated, the following copyright applies:

> Copyright (c) 2026 Maptek Pty Ltd.  
> All rights reserved.

This copyright applies to all files in this repository, whether or not an
individual file contains an explicit notice.

The code is released under the MIT License - see the top-level `LICENSE` file
for details.
