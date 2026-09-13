/////////////////////////////////////////////////////////////////////////////////////////////
//
// Benchmarks for fitting the global RBF system: the direct dense path, the
// preconditioned FGMRES path, and the parameters that drive their cost.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

mod support;

use std::{hint::black_box, time::Duration};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime, SamplingMode,
};
use ferreus_rbf::{
    RBFInterpolator,
    config::{DDMParams, Params},
    interpolant_config::{
        FittingAccuracy, FittingAccuracyType, InterpolantSettings, RBFKernelType, SpheroidalOrder,
    },
};
use support::{TOLERANCE, albatite, settings, synthetic};

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

/// Fits below `Params::naive_solve_threshold` (4096 by default) take the dense
/// LU path, which is O(N^3) and has nothing to do with the iterative solver.
fn bench_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/fit/direct");
    medium(&mut group);

    for n in [500usize, 2_000, 4_000] {
        let (points, values) = synthetic(n, 3);
        let s = settings(RBFKernelType::Linear);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("3d_linear", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    RBFInterpolator::builder(points.clone(), values.clone(), s).build(),
                )
            })
        });
    }

    group.finish();
}

/// Fits above the direct threshold, which exercise domain decomposition
/// preconditioning plus FMM-accelerated FGMRES. This is the headline path.
fn bench_iterative(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/fit/iterative");
    heavy(&mut group);

    for n in [10_000usize, 50_000] {
        let (points, values) = synthetic(n, 3);
        let s = settings(RBFKernelType::Linear);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("3d_linear", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    RBFInterpolator::builder(points.clone(), values.clone(), s).build(),
                )
            })
        });
    }

    // Dimensionality changes both the tree branching factor and the conditioning
    // of the system, so 2D and 3D fits of the same size are not comparable work.
    for dimensions in [2usize, 3] {
        let n = 10_000;
        let (points, values) = synthetic(n, dimensions);
        let s = settings(RBFKernelType::Linear);
        group.bench_with_input(
            BenchmarkId::new("n10k_linear_dim", dimensions),
            &dimensions,
            |b, _| {
                b.iter(|| {
                    black_box(
                        RBFInterpolator::builder(points.clone(), values.clone(), s).build(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Kernel choice drives the default FMM interpolation order (7 for linear, 9 for
/// thin plate spline, 11 for cubic), which dominates fit cost. Expect an order of
/// magnitude between the cheapest and most expensive entries here.
fn bench_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/fit/kernel");
    heavy(&mut group);

    // 5,000 rather than 10,000: the cubic kernel defaults to FMM interpolation
    // order 11, and at 10,000 points a single fit takes seconds.
    let n = 5_000;
    let (points, values) = synthetic(n, 3);

    for (name, s) in [
        ("linear", settings(RBFKernelType::Linear)),
        ("cubic", settings(RBFKernelType::Cubic)),
        ("thin_plate_spline", settings(RBFKernelType::ThinPlateSpline)),
        (
            "spheroidal3",
            InterpolantSettings::builder(RBFKernelType::Spheroidal)
                .spheroidal_order(SpheroidalOrder::Three)
                .fitting_accuracy(FittingAccuracy {
                    tolerance: TOLERANCE,
                    tolerance_type: FittingAccuracyType::Absolute,
                })
                .build(),
        ),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(name), &s, |b, s| {
            b.iter(|| {
                black_box(
                    RBFInterpolator::builder(points.clone(), values.clone(), *s).build(),
                )
            })
        });
    }

    group.finish();
}

/// Fitting tolerance sets the FGMRES stopping criterion, and therefore the
/// iteration count. Useful for separating per-iteration cost from convergence rate.
fn bench_tolerance(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/fit/tolerance");
    heavy(&mut group);

    let n = 10_000;
    let (points, values) = synthetic(n, 3);

    for tolerance in [0.1f64, 0.01, 0.001] {
        let s = InterpolantSettings::builder(RBFKernelType::Linear)
            .fitting_accuracy(FittingAccuracy {
                tolerance,
                tolerance_type: FittingAccuracyType::Absolute,
            })
            .build();
        group.bench_with_input(BenchmarkId::from_parameter(tolerance), &s, |b, s| {
            b.iter(|| {
                black_box(
                    RBFInterpolator::builder(points.clone(), values.clone(), *s).build(),
                )
            })
        });
    }

    group.finish();
}

/// Domain decomposition tuning: leaf size sets how much direct work each
/// subdomain solve does, and `test_unique` is a whole-dataset preprocessing pass.
fn bench_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/fit/params");
    heavy(&mut group);

    let n = 10_000;
    let (points, values) = synthetic(n, 3);
    let s = settings(RBFKernelType::Linear);

    for leaf_threshold in [128usize, 256, 512] {
        let params = Params::builder(RBFKernelType::Linear)
            .ddm_params(DDMParams {
                leaf_threshold,
                ..DDMParams::default()
            })
            .build();
        group.bench_with_input(
            BenchmarkId::new("leaf_threshold", leaf_threshold),
            &params,
            |b, params| {
                b.iter(|| {
                    black_box(
                        RBFInterpolator::builder(points.clone(), values.clone(), s)
                            .params(params.clone())
                            .build(),
                    )
                })
            },
        );
    }

    for test_unique in [true, false] {
        let params = Params::builder(RBFKernelType::Linear)
            .test_unique(test_unique)
            .build();
        group.bench_with_input(
            BenchmarkId::new("test_unique", test_unique),
            &params,
            |b, params| {
                b.iter(|| {
                    black_box(
                        RBFInterpolator::builder(points.clone(), values.clone(), s)
                            .params(params.clone())
                            .build(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// The albatite dataset: 35,801 real signed-distance points, clustered rather
/// than uniform. Tree balance and preconditioner quality both behave differently
/// here than on uniform random points, so keep this as the realism check.
fn bench_dataset(c: &mut Criterion) {
    let Some((points, values)) = albatite() else {
        return;
    };

    let mut group = c.benchmark_group("rbf/fit/dataset");
    heavy(&mut group);
    group.throughput(Throughput::Elements(points.nrows() as u64));

    let s = settings(RBFKernelType::Linear);
    group.bench_function("albatite_linear", |b| {
        b.iter(|| black_box(RBFInterpolator::builder(points.clone(), values.clone(), s).build()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct,
    bench_iterative,
    bench_kernel,
    bench_tolerance,
    bench_params,
    bench_dataset
);
criterion_main!(benches);
