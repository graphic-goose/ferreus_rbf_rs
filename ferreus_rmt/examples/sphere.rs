/////////////////////////////////////////////////////////////////////////////////////////////
//
// Example sphere isosurface generation
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef, mat, row};
use ferreus_rmt::{self, BoundaryClosure, ClusterMethod};

pub fn sphere(pts: MatRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(pts.nrows(), 1, |r, _| {
        let row = pts.row(r);
        row.norm_l2() - 1.0
    })
}

pub fn sphere_gradient(pts: MatRef<'_, f64>) -> (Mat<f64>, Mat<f64>) {
    let nrows = pts.nrows();
    let mut values = Mat::<f64>::zeros(nrows, 1);
    let mut gradients = Mat::<f64>::zeros(nrows, 3);
    const EPS: f64 = 1.0e-12;

    for i in 0..nrows {
        let row = pts.row(i);
        let r = row.norm_l2();
        values[(i, 0)] = r - 1.0;

        let grad = match r > EPS {
            true => {
                let inv_r = 1.0 / r;
                &row * inv_r
            }
            false => row![0.0, 0.0, 0.0],
        };

        gradients.row_mut(i).copy_from(grad);
    }

    (values, gradients)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extents = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5];
    let resolution = 0.2;
    let isovalue = 0.0;

    let mut surface_fn = move |targets: MatRef<f64>| sphere(targets);

    let mut gradient_fn = move |targets: MatRef<f64>| sphere_gradient(targets);

    let seed_points = mat![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],];

    let mesh = ferreus_rmt::build_isosurface(
        seed_points.as_ref(),
        &extents,
        resolution,
        isovalue,
        &mut surface_fn,
        Some(&mut gradient_fn),
        ClusterMethod::CurvatureWeighted,
        BoundaryClosure::None,
        None,
    );

    //Save the isosurface out to an obj file
    let name = "sphere";
    let outpath = "sphere.obj";
    mesh.save_obj(outpath, &name)?;

    Ok(())
}
