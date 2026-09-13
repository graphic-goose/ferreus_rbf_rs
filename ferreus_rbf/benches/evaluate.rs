/////////////////////////////////////////////////////////////////////////////////////////////
//
// Benchmarks for evaluating a fitted interpolant, covering both the one-shot
// evaluator and the cached evaluator used for repeated batches.
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
use ferreus_rbf::interpolant_config::RBFKernelType;
use support::{fit, synthetic, targets};

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

/// Number of source points in the fitted fixture shared by these benchmarks.
const FIT_SIZE: usize = 20_000;

/// `evaluate` builds and discards an FMM tree per call; `evaluate_targets` reuses
/// one built by `build_evaluator`. The gap between them is the setup cost callers
/// pay for by not caching the evaluator.
fn bench_evaluate(c: &mut Criterion) {
    let (points, values) = synthetic(FIT_SIZE, 3);
    let mut rbfi = fit(points, values, RBFKernelType::Linear);

    let mut group = c.benchmark_group("rbf/evaluate");
    medium(&mut group);

    for n in [10_000usize, 100_000] {
        let t = targets(n, 3);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("one_shot", n), &t, |b, t| {
            b.iter(|| black_box(rbfi.evaluate(black_box(t.as_ref()))))
        });
        group.bench_with_input(BenchmarkId::new("one_shot_gradients", n), &t, |b, t| {
            b.iter(|| black_box(rbfi.evaluate_with_gradients(black_box(t.as_ref()))))
        });
    }

    // Building the cached evaluator: a full tree build plus upward pass.
    group.bench_function("build_evaluator", |b| {
        b.iter(|| rbfi.build_evaluator(black_box(None)))
    });

    rbfi.build_evaluator(None);

    for n in [10_000usize, 100_000] {
        let t = targets(n, 3);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("cached", n), &t, |b, t| {
            b.iter(|| black_box(rbfi.evaluate_targets(black_box(t.as_ref()))))
        });
        group.bench_with_input(BenchmarkId::new("cached_gradients", n), &t, |b, t| {
            b.iter(|| black_box(rbfi.evaluate_targets_with_gradients(black_box(t.as_ref()))))
        });
    }

    group.finish();
}

/// Evaluation back at the source locations, used to check the achieved fit.
fn bench_at_source(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf/evaluate_at_source");
    medium(&mut group);

    for n in [10_000usize, 50_000] {
        let (points, values) = synthetic(n, 3);
        let rbfi = fit(points, values, RBFKernelType::Linear);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| black_box(rbfi.evaluate_at_source(black_box(true))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_evaluate, bench_at_source);
criterion_main!(benches);
