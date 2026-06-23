/////////////////////////////////////////////////////////////////////////////////////////////
//
// Example 3D isosurface generation with 35,801 input signed distance points using the linear
// RBF kernel and the default constant drift.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
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

    // Define the filepath for the albatite test dataset
    let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("datasets")
        .join("albatite_SD_points.csv");

    // Extract the source point coordinates and signed distance values
    let (source_points, source_values): (Mat<f64>, Mat<f64>) =
        csv_to_point_arrays(file_path.to_str().expect("valid UTF-8 path"), true)?;

    // Get the axis aligned bounding box extents of the source points
    // to use for the isosurface extraction
    let source_point_extents = ferreus_rbf_utils::get_pointarray_extents(source_points.as_ref());

    // Define the RBF kernel to use
    let kernel_type = RBFKernelType::Linear;

    // Define the desired fitting accuracy
    let fitting_accuracy = FittingAccuracy {
        tolerance: 0.01,
        tolerance_type: FittingAccuracyType::Absolute,
    };

    // Initialise an InterpolantSettings instance
    let interpolant_settings = InterpolantSettings::builder(kernel_type)
        .fitting_accuracy(fitting_accuracy)
        .build();

    // Create a callback to receive progress updates from the RBFInterpolator
    let callback = get_callback_sink();

    // Setup and solve the RBF system
    let mut rbfi = RBFInterpolator::builder(
        source_points.clone(),
        source_values.clone(),
        interpolant_settings,
    )
    .progress_callback(callback.clone())
    .build();

    // Define the sampling grid resolution for the surfacer
    let resolution = 10.0;

    let evaluator_extents: Vec<f64> = source_point_extents
        .iter()
        .enumerate()
        .map(|(idx, val)| match idx < 3 {
            true => val - resolution * 10.0,
            false => val + resolution * 10.0,
        })
        .collect();

    rbfi.build_evaluator(Some(evaluator_extents));

    let rbfi = Rc::new(RefCell::new(rbfi));
    let rbfi_surface = Rc::clone(&rbfi);
    let rbfi_grad = Rc::clone(&rbfi);

    let mut surface_fn =
        move |targets: MatRef<f64>| rbfi_surface.borrow_mut().evaluate_targets(targets.as_ref());

    let mut gradient_fn = move |targets: MatRef<f64>| {
        rbfi_grad
            .borrow_mut()
            .evaluate_targets_with_gradients(targets.as_ref())
    };

    let rmt_callback = callback.clone().into_rmt_progress();

    //  Define the isovalue at which to surface
    let isovalue = 0.0;

    // Generate an isosurface directly from the RMT extractor.
    let mesh = build_isosurface(
        source_points.as_ref(),
        &source_point_extents,
        resolution,
        isovalue,
        &mut surface_fn,
        Some(&mut gradient_fn),
        ClusterMethod::CurvatureWeighted,
        BoundaryClosure::ClosePositive,
        Some(rmt_callback.as_ref()),
    );

    //Save the isosurface out to an obj file
    let name = format!("isosurface_linear_{}m", fmt_num(resolution));
    let outpath = cwd.join("examples").join(format!("{}.obj", &name));
    mesh.save_obj(outpath, &name)?;

    Ok(())
}
