/////////////////////////////////////////////////////////////////////////////////////////////
//
// Benchmarks for the black box fast multipole method: tree construction, the upward pass,
// and the downward/leaf evaluation passes.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use std::{hint::black_box, sync::Arc, time::Duration};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime, SamplingMode,
};
use faer::{Mat, RowRef, mat::AsMatRef};
use ferreus_bbfmm::{FmmParams, FmmTree, KernelFunction, M2LCompressionType};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// The linear RBF kernel, `-r`. Duplicated here rather than pulled in from
/// `ferreus_rbf_utils` so that this crate's benchmarks stay dependency-free.
struct LinearRbfKernel;

impl KernelFunction for LinearRbfKernel {
    #[inline(always)]
    fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
        let mut dist = 0.0;
        for (t, s) in target.iter().zip(source.iter()) {
            let diff = t - s;
            dist += diff * diff;
        }
        -dist.sqrt()
    }

    /// `d/dx (-r) = -(x - x_s) / r`. Required for the gradient benchmarks; without
    /// it the tree rejects gradient evaluation.
    #[inline(always)]
    fn evaluate_value_gradient(
        &self,
        target: RowRef<f64>,
        source: RowRef<f64>,
        gradient_out: &mut [f64],
    ) -> Option<f64> {
        let mut r2 = 0.0;
        for (i, (t, s)) in target.iter().zip(source.iter()).enumerate() {
            let diff = t - s;
            gradient_out[i] = diff;
            r2 += diff * diff;
        }

        if r2 <= f64::EPSILON {
            gradient_out.fill(0.0);
            return Some(-r2.sqrt());
        }

        let r = r2.sqrt();
        let scale = -1.0 / r;
        for g in gradient_out.iter_mut() {
            *g *= scale;
        }
        Some(-r)
    }
}

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

/// Uniformly distributed points in the unit cube, from a fixed seed.
fn points(n: usize, dim: usize, seed: u64) -> Arc<Mat<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    Arc::new(Mat::from_fn(n, dim, |_, _| rng.random_range(-1.0..1.0)))
}

/// Source weights for `nrhs` right-hand sides, from a fixed seed.
fn weights(n: usize, nrhs: usize, seed: u64) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Mat::from_fn(n, nrhs, |_, _| rng.random_range(-1.0..1.0))
}

/// Default benchmark configuration: order 7, adaptive, sparse, ACA-compressed M2L.
fn tree(points: Arc<Mat<f64>>, order: usize, params: Option<FmmParams>) -> FmmTree<LinearRbfKernel> {
    FmmTree::new(points, order, LinearRbfKernel, true, true, None, params)
}

/// Tree construction, which precomputes and compresses the M2L operators.
fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmm/build");
    medium(&mut group);

    for n in [10_000usize, 100_000] {
        let pts = points(n, 3, 42);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("3d_order7", n), &n, |b, _| {
            b.iter(|| black_box(tree(pts.clone(), 7, None)))
        });
    }

    // Interpolation order drives the M2L precompute cost as roughly O(order^(2d)),
    // so it dominates build time and trades directly against far-field accuracy.
    let pts = points(50_000, 3, 42);
    for order in [4usize, 5, 6, 7, 8] {
        group.bench_with_input(BenchmarkId::new("3d_n50k_order", order), &order, |b, &o| {
            b.iter(|| black_box(tree(pts.clone(), o, None)))
        });
    }

    // M2L compression strategy: the cost paid at build time to make evaluation cheaper.
    for (name, compression) in [
        ("none", M2LCompressionType::None),
        ("svd", M2LCompressionType::SVD),
        ("aca", M2LCompressionType::ACA),
    ] {
        let mut params = FmmParams::new_defaults(7);
        params.compression_type = compression;
        group.bench_with_input(
            BenchmarkId::new("3d_n50k_compression", name),
            &params,
            |b, p| b.iter(|| black_box(tree(pts.clone(), 7, Some(*p)))),
        );
    }

    group.finish();
}

/// The upward pass: P2M at the leaves followed by M2M up the tree.
fn bench_set_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmm/set_weights");
    micro(&mut group);

    for n in [10_000usize, 100_000] {
        let pts = points(n, 3, 42);
        let mut t = tree(pts.clone(), 7, None);
        for nrhs in [1usize, 4] {
            let w = weights(n, nrhs, 7);
            group.throughput(Throughput::Elements((n * nrhs) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("3d_nrhs{nrhs}"), n),
                &w,
                |b, w| b.iter(|| t.set_weights(black_box(w.as_ref()))),
            );
        }
    }

    group.finish();
}

/// The downward pass (M2L, L2L) plus the leaf pass (L2P, P2P). This is the inner
/// loop of every FGMRES iteration in `ferreus_rbf`, so it is the hottest path here.
fn bench_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("fmm/evaluate");
    medium(&mut group);

    for n in [10_000usize, 100_000] {
        let pts = points(n, 3, 42);
        let w = weights(n, 1, 7);
        let mut t = tree(pts.clone(), 7, None);
        t.set_weights(w.as_ref());

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("3d_values", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    t.evaluate(black_box(w.as_ref()), pts.as_mat_ref())
                        .unwrap(),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("3d_values_grads", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    t.evaluate_with_gradients(black_box(w.as_ref()), pts.as_mat_ref())
                        .unwrap(),
                )
            })
        });
    }

    // Dimensionality changes the branching factor of the tree (2/4/8 children per cell)
    // and the size of the tensor-product node set, so 1D/2D/3D behave very differently.
    for dim in [1usize, 2, 3] {
        let n = 50_000;
        let pts = points(n, dim, 42);
        let w = weights(n, 1, 7);
        let mut t = tree(pts.clone(), 7, None);
        t.set_weights(w.as_ref());

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n50k_dim", dim), &dim, |b, _| {
            b.iter(|| {
                black_box(
                    t.evaluate(black_box(w.as_ref()), pts.as_mat_ref())
                        .unwrap(),
                )
            })
        });
    }

    // Multiple right-hand sides share a single tree traversal, so throughput per
    // RHS should improve with `nrhs`.
    let n = 50_000;
    let pts = points(n, 3, 42);
    let mut t = tree(pts.clone(), 7, None);
    for nrhs in [1usize, 2, 4, 8] {
        let w = weights(n, nrhs, 7);
        t.set_weights(w.as_ref());
        group.throughput(Throughput::Elements((n * nrhs) as u64));
        group.bench_with_input(BenchmarkId::new("3d_n50k_nrhs", nrhs), &w, |b, w| {
            b.iter(|| {
                black_box(
                    t.evaluate(black_box(w.as_ref()), pts.as_mat_ref())
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_build, bench_set_weights, bench_evaluate);
criterion_main!(benches);
