// Complexity Benchmark for ferreus_rbf_rs
// Tests scaling of the FastRBF implementation
//
// Expected: O(N log N) time and O(N) memory

use ferreus_rbf::{
    RBFInterpolator,
    interpolant_config::{
        InterpolantSettings,
        RBFKernelType,
        FittingAccuracy,
        FittingAccuracyType
    },
    config::{Params, FmmParams, DDMParams},
};
use faer::Mat;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn generate_random_points_3d(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Mat::from_fn(n, 3, |_, _| rng.random_range(0.0..1.0))
}

fn generate_values(points: &Mat<f64>) -> Mat<f64> {
    // Simple polynomial function for testing
    Mat::from_fn(points.nrows(), 1, |i, _| {
        let x = points[(i, 0)];
        let y = points[(i, 1)];
        let z = points[(i, 2)];
        x + 2.0 * y + 3.0 * z
    })
}

fn run_benchmark(n: usize, seed: u64) -> (f64, usize, f64) {
    println!("  Generating {} random 3D points...", n);
    let points = generate_random_points_3d(n, seed);
    let values = generate_values(&points);

    let fitting_accuracy = FittingAccuracy {
        tolerance: 0.1,
        tolerance_type: FittingAccuracyType::Relative,
    };

    let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear)
        .fitting_accuracy(fitting_accuracy)
        .build();

    let params = Params {
        fmm: FmmParams {
            interpolation_order: 6,
            max_points_per_cell: 256,
            epsilon: 1e-6,
            use_adaptive_tree: true,
        },
        ddm: DDMParams {
            leaf_threshold: 128,
            overlap_quota: 0.3,
            coarse_ratio: 0.3,
            coarse_threshold: 256,
        },
        max_inner_iterations: 30,
        max_outer_iterations: 10,
    };

    println!("  Building RBF interpolator...");
    let start = Instant::now();

    let mut rbfi = RBFInterpolator::builder(points, values, interpolant_settings)
        .params(params)
        .build();

    let elapsed = start.elapsed().as_secs_f64();
    let iterations = rbfi.iterations;

    // Estimate memory (rough approximation)
    let memory_gb = (n * 8 * 10) as f64 / 1e9; // Very rough estimate

    println!("  Completed in {:.2}s ({} iterations)", elapsed, iterations);

    (elapsed, iterations, memory_gb)
}

fn main() {
    println!("=== ferreus_rbf_rs Complexity Benchmark ===\n");
    println!("Testing O(N log N) scaling hypothesis\n");

    let test_sizes = vec![
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
    ];

    println!("N\t\tTime(s)\t\tIters\tTime/(N*log(N))\tMemory(GB)");
    println!("-".repeat(70));

    for &n in &test_sizes {
        println!("\nTesting N = {}:", n);

        let (time, iters, mem) = run_benchmark(n, 42);

        let n_f64 = n as f64;
        let n_log_n = n_f64 * n_f64.log2();
        let time_per_nlogn = time / n_log_n * 1e6; // microseconds per N log N

        println!("{}\t\t{:.3}\t\t{}\t{:.3}\t\t{:.3}",
                 n, time, iters, time_per_nlogn, mem);
    }

    println!("\n=== Analysis ===");
    println!("If Time/(N*log(N)) remains roughly constant, the algorithm is O(N log N).");
    println!("If Time/(N*log(N)) increases linearly with N, it's O(N²).");
    println!("If Time/(N*log(N)) increases quadratically with N, it's O(N³).");
}
