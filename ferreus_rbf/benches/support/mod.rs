/////////////////////////////////////////////////////////////////////////////////////////////
//
// Shared fixtures for the ferreus_rbf benchmarks.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

#![allow(dead_code)] // Each bench target uses a different subset of these helpers.

use std::path::PathBuf;

use faer::Mat;
use ferreus_rbf::{
    RBFInterpolator, RBFTestFunctions, csv_to_point_arrays, generate_random_points,
    interpolant_config::{FittingAccuracy, FittingAccuracyType, InterpolantSettings, RBFKernelType},
};

/// Fixed seed for the synthetic source points, so every run fits the same system.
pub const SOURCE_SEED: u64 = 42;

/// Fixed seed for synthetic target points.
pub const TARGET_SEED: u64 = 9;

/// Default fitting tolerance. Absolute, and matched to the unit-cube test
/// functions below; it sets how many solver iterations a fit actually takes,
/// so keep it fixed when comparing baselines.
pub const TOLERANCE: f64 = 0.01;

/// Interpolant settings for `kernel` at the default benchmark tolerance.
pub fn settings(kernel: RBFKernelType) -> InterpolantSettings {
    InterpolantSettings::builder(kernel)
        .fitting_accuracy(FittingAccuracy {
            tolerance: TOLERANCE,
            tolerance_type: FittingAccuracyType::Absolute,
        })
        .build()
}

/// Synthetic source points and values in the unit cube/square.
///
/// Uses Franke's function in 2D and test function 1 in 3D, both of which vary on
/// the scale of the domain so the fit is a realistic amount of work.
pub fn synthetic(n: usize, dimensions: usize) -> (Mat<f64>, Mat<f64>) {
    let points = generate_random_points(n, dimensions, Some(SOURCE_SEED));
    let values = match dimensions {
        2 => RBFTestFunctions::franke_2d(&points),
        3 => RBFTestFunctions::f1_3d(&points),
        d => panic!("no benchmark test function for {d}D"),
    };
    (points, values)
}

/// Synthetic target points in the unit cube, inset slightly so they stay within
/// evaluator extents derived from the source points.
pub fn targets(n: usize, dimensions: usize) -> Mat<f64> {
    let raw = generate_random_points(n, dimensions, Some(TARGET_SEED));
    Mat::from_fn(raw.nrows(), raw.ncols(), |i, j| 0.05 + raw[(i, j)] * 0.9)
}

/// Path to a file in the workspace `datasets/` directory.
pub fn dataset_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("datasets")
        .join(name)
}

/// The albatite signed-distance dataset: 35,801 real survey points in 3D.
///
/// Returns `None` if the dataset is not checked out, so the benchmarks degrade
/// to the synthetic cases rather than failing outright.
pub fn albatite() -> Option<(Mat<f64>, Mat<f64>)> {
    let path = dataset_path("albatite_SD_points.csv");
    if !path.exists() {
        eprintln!("skipping albatite benchmarks: {} not found", path.display());
        return None;
    }
    Some(csv_to_point_arrays(path.to_str().expect("valid UTF-8 path"), true).expect("read dataset"))
}

/// Fits an interpolant to be reused as a fixture by evaluation benchmarks.
pub fn fit(points: Mat<f64>, values: Mat<f64>, kernel: RBFKernelType) -> RBFInterpolator {
    RBFInterpolator::builder(points, values, settings(kernel)).build()
}
