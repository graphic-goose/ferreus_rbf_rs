/////////////////////////////////////////////////////////////////////////////////////////////
//
// Example 3D isosurface generation with 35,801 input signed distance points using the linear
// RBF kernel, default constant drift and a global trend.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::Mat;
use ferreus_rbf::{
    GlobalTrend, RBFInterpolator, csv_to_point_arrays,
    interpolant_config::{
        FittingAccuracy, FittingAccuracyType, InterpolantSettings, RBFKernelType,
    },
    progress::{ProgressMsg, ProgressSink, closure_sink},
    save_obj,
};
use ferreus_rbf_utils;
use std::{env, sync::Arc};

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
    let file_path = cwd
        .join("examples")
        .join("datasets")
        .join("albatite_SD_points.csv");

    // Extract the source point coordinates and signed distance values
    let (source_points, source_values): (Mat<f64>, Mat<f64>) =
        csv_to_point_arrays(file_path.to_str().expect("valid UTF-8 path"), true)?;

    // Get the axis aligned bounding box extents of the source points
    // to use for the isosurface extraction
    let source_point_extents = ferreus_rbf_utils::get_pointarray_extents(&source_points);

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

    // Define a global trend
    let dip = 87.0;
    let dip_direction = 255.0;
    let pitch = 25.0;
    let major_ratio = 4.0;
    let semi_major_ratio = 2.0;
    let minor_ratio = 1.0;

    let global_trend = GlobalTrend::Three {
        dip,
        dip_direction,
        pitch,
        major_ratio,
        semi_major_ratio,
        minor_ratio,
    };

    // Create a callback to receive progress updates from the RBFInterpolator
    let callback = get_callback_sink();

    // Setup and solve the RBF system
    let mut rbfi = RBFInterpolator::builder(source_points, source_values, interpolant_settings)
        .progress_callback(callback.clone())
        .global_trend(global_trend)
        .build();

    // Define the sampling grid resolution for the surfacer
    let resolution = 5.0;

    //  Define the isovalues at which to surface
    let isovalues = vec![0.0];

    // Generate an isosurface
    let (all_isosurface_points, all_isosurface_faces) =
        rbfi.build_isosurfaces(&source_point_extents, &resolution, &isovalues);

    //Save the isosurface out to an obj file
    let name = format!("albatite_isosurface_global_trend_{}m", fmt_num(resolution));
    let outpath = cwd.join("examples").join(format!("{}.obj", &name));
    save_obj(
        outpath,
        &name,
        all_isosurface_points[0].as_ref(),
        all_isosurface_faces[0].as_ref(),
    )?;

    Ok(())
}