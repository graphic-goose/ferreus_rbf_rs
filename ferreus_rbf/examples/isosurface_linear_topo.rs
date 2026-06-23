/////////////////////////////////////////////////////////////////////////////////////////////
//
// Example 3D isosurface generation clipped by topography using the linear RBF kernel.
//
// Created on: 23 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef};
use ferreus_rbf::{
    RBFInterpolator, csv_to_point_arrays,
    interpolant_config::{
        FittingAccuracy, FittingAccuracyType, InterpolantSettings, RBFKernelType,
    },
    isosurfacing::{BoundaryClosure, ClusterMethod, build_isosurface},
    progress::{ProgressMsg, ProgressSink, ProgressSinkExt, closure_sink},
};
use ferreus_rbf_utils;
use std::{cell::RefCell, env, path::Path, rc::Rc, sync::Arc};

/// Nice float formatter for filenames: trims trailing zeros and dots.
fn fmt_num(x: f64) -> String {
    let s = format!("{:.6}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Generates a callback closure_sink
fn get_callback_sink() -> Arc<dyn ProgressSink> {
    let (sink, _listener) = closure_sink(256, |msg| match msg {
        ProgressMsg::SolverIteration {
            iter,
            residual,
            progress,
        } => {
            println!(
                "Iteration: {:>3}    {:>.5E}    {:>.1}%",
                iter,
                residual,
                progress * 100.0
            );
        }
        ProgressMsg::SurfacingProgress {
            isovalue,
            stage,
            progress,
        } => {
            println!(
                "Isovalue: {:?}    Stage: {}    {:>.1}%",
                isovalue,
                stage,
                progress * 100.0
            );
        }
        ProgressMsg::DuplicatesRemoved { num_duplicates } => {
            println!("Removed {:>3} duplicate points", num_duplicates);
        }

        ProgressMsg::Message { message } => {
            println!("{message}");
        }
    });

    sink
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env::current_dir().unwrap();

    // Define the filepaths for the albatite signed distance and topography datasets.
    let datasets_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("datasets");
    let pointset_path = datasets_path.join("albatite_SD_points.csv");
    let topo_path = datasets_path.join("Topo points.csv");

    // Extract the source point coordinates and signed distance values.
    let (source_points, source_values): (Mat<f64>, Mat<f64>) =
        csv_to_point_arrays(pointset_path.to_str().expect("valid UTF-8 path"), true)?;

    // Extract the topography points as a 2D interpolant: XY locations with Z elevation values.
    let (topo_points, topo_values): (Mat<f64>, Mat<f64>) =
        csv_to_point_arrays(topo_path.to_str().expect("valid UTF-8 path"), true)?;
    let topo_source_extents = ferreus_rbf_utils::get_pointarray_extents(topo_points.as_ref());

    // Define the RBF kernel to use.
    let kernel_type = RBFKernelType::Linear;

    // Define the desired fitting accuracy.
    let fitting_accuracy = FittingAccuracy {
        tolerance: 0.01,
        tolerance_type: FittingAccuracyType::Absolute,
    };

    // Initialise an InterpolantSettings instance.
    let interpolant_settings = InterpolantSettings::builder(kernel_type)
        .fitting_accuracy(fitting_accuracy)
        .build();

    // Create a callback to receive progress updates from the RBFInterpolator.
    let callback = get_callback_sink();

    // Setup and solve the signed distance and topography RBF systems.
    let mut rbfi = RBFInterpolator::builder(
        source_points.clone(),
        source_values.clone(),
        interpolant_settings.clone(),
    )
    .progress_callback(callback.clone())
    .build();

    let mut topo_rbfi = RBFInterpolator::builder(topo_points, topo_values, interpolant_settings)
        .progress_callback(callback.clone())
        .build();

    // Define the sampling grid resolution for the surfacer.
    let resolution = 5.0;

    // Define the signed distance isovalue to surface before topography clipping.
    let isovalue = 20.0;

    // Define the isosurfacing extents: [minx, miny, minz, maxx, maxy, maxz].
    let extents = vec![
        329105.0, 7744370.0, -320.0, 329845.0, 7745275.0, 435.0,
    ];

    // When setting up the RBF evaluators for isosurfacing we need to add a buffer to
    // the evaluator extents so we don't evaluate points outside the evaluator domains.
    let evaluator_padding = 10.0 * resolution;

    let source_point_extents = ferreus_rbf_utils::get_pointarray_extents(source_points.as_ref());
    let rbf_evaluator_extents = vec![
        source_point_extents[0].min(extents[0]) - evaluator_padding,
        source_point_extents[1].min(extents[1]) - evaluator_padding,
        source_point_extents[2].min(extents[2]) - evaluator_padding,
        source_point_extents[3].max(extents[3]) + evaluator_padding,
        source_point_extents[4].max(extents[4]) + evaluator_padding,
        source_point_extents[5].max(extents[5]) + evaluator_padding,
    ];
    rbfi.build_evaluator(Some(rbf_evaluator_extents));

    let topo_evaluator_extents = vec![
        topo_source_extents[0].min(extents[0]) - evaluator_padding,
        topo_source_extents[1].min(extents[1]) - evaluator_padding,
        topo_source_extents[2].max(extents[3]) + evaluator_padding,
        topo_source_extents[3].max(extents[4]) + evaluator_padding,
    ];
    topo_rbfi.build_evaluator(Some(topo_evaluator_extents));

    let rbfi = Rc::new(RefCell::new(rbfi));
    let topo_rbfi = Rc::new(RefCell::new(topo_rbfi));

    let rbfi_surface = Rc::clone(&rbfi);
    let topo_surface = Rc::clone(&topo_rbfi);

    let mut surface_fn = move |targets: MatRef<f64>| {
        // The surfacer calls this closure with batches of 3D sample locations.
        // It expects one scalar value per sample. The zero contour of those returned
        // values is the surface that will be extracted.
        let rbf_values = rbfi_surface
            .borrow_mut()
            .evaluate_targets(targets.as_ref());

        // The topography interpolant is 2D: it maps XY locations to a topographic Z
        // elevation. For every 3D sample point, evaluate the topography directly
        // below or above it using only the sample's X and Y coordinates.
        let topo_targets =
            Mat::<f64>::from_fn(targets.nrows(), 2, |row, col| targets[(row, col)]);
        let topo_values = topo_surface
            .borrow_mut()
            .evaluate_targets(topo_targets.as_ref());

        Mat::<f64>::from_fn(targets.nrows(), 1, |row, _| {
            // Shift the signed-distance RBF so the requested isovalue becomes zero.
            // For example, if isovalue is 20, points where the RBF evaluates to 20 now
            // return 0, values below 20 are negative, and values above 20 are positive.
            let rbf_isovalue_value = rbf_values[(row, 0)] - isovalue;

            // Build a second implicit field for the topography clipping surface.
            // This is negative below the topography, zero on it, and positive above it.
            let topo_clip_value = targets[(row, 2)] - topo_values[(row, 0)];

            // Taking the maximum combines the two implicit fields as an intersection:
            // the result is negative only where both fields are negative. Surfacing the
            // combined field at zero keeps the RBF isosurface only where it is below the
            // topography, and the topography itself closes the clipped portion.
            rbf_isovalue_value.max(topo_clip_value)
        })
    };

    let rmt_callback = callback.clone().into_rmt_progress();

    // surface_fn shifts the RBF by isovalue, so the combined clipped field is surfaced at 0.
    let mesh = build_isosurface(
        source_points.as_ref(),
        &extents,
        resolution,
        0.0,
        &mut surface_fn,
        None,
        ClusterMethod::CurvatureWeighted,
        BoundaryClosure::ClosePositive,
        Some(rmt_callback.as_ref()),
    );

    // Save the isosurface out to an obj file.
    let name = format!(
        "isosurface_linear_topo_{}_{}m",
        fmt_num(isovalue),
        fmt_num(resolution)
    );
    let outpath = cwd.join("examples").join(format!("{}.obj", &name));
    mesh.save_obj(outpath, &name)?;

    Ok(())
}
