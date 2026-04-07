/////////////////////////////////////////////////////////////////////////////////////////////
//
// Example 3D isosurface generation of the unit sphere.
//
// Created on: 7 April 2026     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef, mat};
use ferreus_rbf::isosurfacing::{save_obj, surface_nets};
use std::env;

/// Nice float formatter for filenames: trims trailing zeros and dots.
fn fmt_num(x: f64) -> String {
    let s = format!("{:.6}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Define a function to evaluate an isosurface from
/// In this example we'll create a unit sphere
pub fn sphere(pts: MatRef<'_, f64>) -> Mat<f64> {
    assert_eq!(pts.ncols(), 3, "sphere() expects pts to be N x 3");

    Mat::<f64>::from_fn(pts.nrows(), 1, |r, _| {
        let row = pts.row(r);
        row.norm_l2() - 1.0
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir().unwrap();

    // Define the extents for the isosurfacer
    let extents = vec![-1.1, -1.1, -1.1, 1.1, 1.1, 1.1];

    // Define the resolution to evaluate at
    let resolution = 0.1;

    // Define some seed points on the isosurface
    let seed_points = mat![[1.0, 0.0, 0.0,], [-1.0, 0.0, 0.0],];

    // Define the function values of the seed points
    let seed_values = mat![[0.0,], [0.0],];

    // Define the isovalue at which to surface
    let isovalue = 0.0;

    // Extract the isosurface
    let (verts, faces) = surface_nets(
        &extents,
        resolution,
        isovalue,
        &mut sphere,
        seed_points.as_ref(),
        seed_values.as_ref(),
        &None,
    );

    //Save the isosurface out to an obj file
    let name = format!("isosurface_sphere_{}m", fmt_num(resolution));
    let outpath = cwd.join(format!("{}.obj", &name));
    save_obj(outpath, &name, verts.as_ref(), faces.as_ref())?;

    Ok(())
}
