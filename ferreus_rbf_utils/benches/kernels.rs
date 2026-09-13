/////////////////////////////////////////////////////////////////////////////////////////////
//
// Benchmarks for the shared kernel evaluation and dense kernel matrix assembly routines.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use std::{hint::black_box, time::Duration};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime, SamplingMode,
};
use faer::Mat;
use ferreus_rbf_utils::{
    KernelParams, KernelType, get_a_matrix, get_a_matrix_symmetric_solver, get_pointarray_extents,
    kernel_phi,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Criterion spends `warm_up_time + measurement_time` on every benchmark regardless
/// of how fast the code under test is, so with this many benchmarks that fixed
/// budget - not the work itself - sets the suite runtime. These three policies keep
/// it proportionate.
///
/// * `micro` - iterations well under a millisecond; a short window still collects
///   thousands of samples.
/// * `medium` - iterations up to a few hundred milliseconds.
/// * `heavy` - iterations of a second or more. These use flat sampling: Criterion's
///   default linear sampling runs sample *i* for *i* iterations, so "10 samples"
///   means 1+2+...+10 = 55 runs of the benchmark. That is fine when Criterion can
///   fit the schedule inside `measurement_time`, but for second-scale iterations it
///   cannot, and the group takes 5.5x longer than the sample count suggests. Flat
///   sampling runs the same number of iterations per sample, and `measurement_time`
///   is deliberately short so that number stays at one for anything slower than
///   about half a second. Criterion then prints "unable to complete 10 samples in
///   5.0s" - that warning is expected here, and means the floor is the work itself.
#[allow(dead_code)]
fn micro(group: &mut BenchmarkGroup<'_, WallTime>) {
    group
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
}

#[allow(dead_code)]
fn medium(group: &mut BenchmarkGroup<'_, WallTime>) {
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
}

#[allow(dead_code)]
fn heavy(group: &mut BenchmarkGroup<'_, WallTime>) {
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
}

/// Every kernel in the registry, with a short label for the benchmark id.
const KERNELS: &[(&str, KernelType)] = &[
    ("linear", KernelType::LinearRbf),
    ("thin_plate_spline", KernelType::ThinPlateSplineRbf),
    ("cubic", KernelType::CubicRbf),
    ("spheroidal3", KernelType::Spheroidal3Rbf),
    ("spheroidal5", KernelType::Spheroidal5Rbf),
    ("spheroidal7", KernelType::Spheroidal7Rbf),
    ("spheroidal9", KernelType::Spheroidal9Rbf),
    ("wendlands_c2", KernelType::WendlandsC2Rbf),
    ("spherical", KernelType::SphericalRbf),
    ("exponential", KernelType::ExponentialRbf),
    ("gaussian", KernelType::GaussianRbf),
    ("cubic2", KernelType::Cubic2Rbf),
    ("inverse_multiquadratic", KernelType::InverseMultiquadraticRbf),
    ("laplacian", KernelType::Laplacian),
    ("one_over_r2", KernelType::OneOverR2),
    ("one_over_r4", KernelType::OneOverR4),
];

fn params(kernel: KernelType) -> KernelParams {
    // `base_range` must be >= `total_sill`; these values keep every kernel in its
    // interesting regime for radii in (0, 2].
    KernelParams::builder(kernel)
        .base_range(1.0)
        .total_sill(1.0)
        .build()
}

fn points(n: usize, dim: usize, seed: u64) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Mat::from_fn(n, dim, |_, _| rng.random_range(0.0..1.0))
}

/// Scalar kernel evaluation cost, per kernel. Radii are precomputed so this
/// measures `phi` alone rather than the distance calculation.
fn bench_phi(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernels/phi");
    micro(&mut group);

    const N: usize = 4096;
    let radii: Vec<f64> = (0..N).map(|i| (i as f64 + 0.5) * 2.0 / N as f64).collect();

    group.throughput(Throughput::Elements(N as u64));
    for (name, kernel) in KERNELS {
        let p = params(*kernel);
        group.bench_with_input(BenchmarkId::from_parameter(name), &p, |b, p| {
            b.iter(|| {
                let mut acc = 0.0;
                for &r in &radii {
                    acc += kernel_phi(black_box(r), p);
                }
                black_box(acc)
            })
        });
    }

    group.finish();
}

/// Dense kernel matrix assembly. `n = 256` matches the default domain decomposition
/// leaf threshold, and is the size actually assembled inside the preconditioner.
fn bench_a_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernels/a_matrix");
    medium(&mut group);

    for n in [256usize, 1024, 4096] {
        let targets = points(n, 3, 1);
        let sources = points(n, 3, 2);

        // Entries produced, i.e. kernel evaluations.
        group.throughput(Throughput::Elements((n * n) as u64));

        // Two kernels only: assembly is dominated by the traversal and allocation,
        // not by `phi`, so linear and cubic measure the same to within noise. Linear
        // and spheroidal3 bracket the cheap and expensive ends; `kernels/phi` above
        // already covers per-kernel scalar cost.
        for (name, kernel) in [
            ("linear", KernelType::LinearRbf),
            ("spheroidal3", KernelType::Spheroidal3Rbf),
        ] {
            let p = params(kernel);
            group.bench_with_input(BenchmarkId::new(format!("general_{name}"), n), &p, |b, p| {
                b.iter(|| black_box(get_a_matrix(black_box(&targets), black_box(&sources), *p)))
            });
            group.bench_with_input(
                BenchmarkId::new(format!("symmetric_{name}"), n),
                &p,
                |b, p| {
                    b.iter(|| {
                        black_box(get_a_matrix_symmetric_solver(
                            black_box(&targets),
                            black_box(&targets),
                            p,
                            &0.0,
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

/// Small helpers that run over whole point arrays on every solve.
fn bench_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernels/utils");
    micro(&mut group);

    for n in [10_000usize, 100_000] {
        let pts = points(n, 3, 3);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("pointarray_extents", n), &n, |b, _| {
            b.iter(|| black_box(get_pointarray_extents(black_box(pts.as_ref()))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_phi, bench_a_matrix, bench_utils);
criterion_main!(benches);
