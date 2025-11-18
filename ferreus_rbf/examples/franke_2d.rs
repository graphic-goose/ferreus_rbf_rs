use ferreus_rbf::generate_random_points;
use ferreus_rbf::{
    RBFInterpolator, RBFTestFunctions, create_evaluation_grid,
    interpolant_config::{InterpolantSettings, RBFKernelType},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define input source points in a 2D grid within [0, 1]^2
    let dim = 2usize;
    let num_points = 100usize;
    let points = generate_random_points(num_points, dim, Some(42));

    // Define some values at the source points using Franke's function
    let point_values = RBFTestFunctions::franke_2d(&points);

    // Select the Linear RBF kernel
    let interpolant_settings = InterpolantSettings::builder(RBFKernelType::ThinPlateSpline).build();

    // Setup and solve the RBF
    let rbfi = RBFInterpolator::builder(points, point_values, interpolant_settings)
        .build();

    // Build a 2D grid of target points in [0, 1]^2 to evaluate the RBF at
    let n = 50;
    let target_points = create_evaluation_grid(&[(0.0, 1.0), (0.0, 1.0)], &[n, n]);

    // Evaluate the RBF at the target points
    let _interpolated_values = rbfi.evaluate(&target_points);

    Ok(())
}