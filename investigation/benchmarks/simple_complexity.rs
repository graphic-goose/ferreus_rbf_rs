// Simple complexity test - can be run as a standalone binary
// Place in ferreus_rbf/examples/ and run with: cargo run --example simple_complexity --release

use ferreus_rbf::{
    RBFInterpolator, generate_random_points,
    interpolant_config::{InterpolantSettings, RBFKernelType, FittingAccuracy, FittingAccuracyType},
};
use faer::Mat;
use std::time::Instant;

fn main() {
    println!("=== Simple Complexity Test ===\n");

    let test_sizes = vec![500, 1000, 2000, 4000, 8000];

    println!("{:<10} {:<12} {:<15}", "N", "Time(s)", "Time/(N*log2(N))");
    println!("{}", "-".repeat(40));

    for n in test_sizes {
        let points = generate_random_points(n, 3, Some(42));
        let values = Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];
            x + y + z
        });

        let settings = InterpolantSettings::builder(RBFKernelType::Linear)
            .fitting_accuracy(FittingAccuracy {
                tolerance: 0.1,
                tolerance_type: FittingAccuracyType::Relative,
            })
            .build();

        let start = Instant::now();
        let _rbfi = RBFInterpolator::builder(points, values, settings).build();
        let elapsed = start.elapsed().as_secs_f64();

        let n_f64 = n as f64;
        let normalized = elapsed / (n_f64 * n_f64.log2()) * 1e6;

        println!("{:<10} {:<12.3} {:<15.3}", n, elapsed, normalized);
    }

    println!("\nIf the rightmost column stays roughly constant, complexity is O(N log N)");
}
