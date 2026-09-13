/////////////////////////////////////////////////////////////////////////////////////////////
//
// Benchmarks for surface-following regularised marching tetrahedra extraction.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use std::{
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime, SamplingMode,
};
use faer::{Mat, MatRef, mat, row};
use ferreus_rmt::{BoundaryClosure, ClusterMethod, build_isosurface};

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

/// Unit sphere as a signed distance field.
fn sphere(pts: MatRef<'_, f64>) -> Mat<f64> {
    Mat::<f64>::from_fn(pts.nrows(), 1, |r, _| pts.row(r).norm_l2() - 1.0)
}

/// Analytic value and gradient for the unit sphere.
fn sphere_with_gradient(pts: MatRef<'_, f64>) -> (Mat<f64>, Mat<f64>) {
    let nrows = pts.nrows();
    let mut values = Mat::<f64>::zeros(nrows, 1);
    let mut gradients = Mat::<f64>::zeros(nrows, 3);
    const EPS: f64 = 1.0e-12;

    for i in 0..nrows {
        let p = pts.row(i);
        let r = p.norm_l2();
        values[(i, 0)] = r - 1.0;
        let grad = match r > EPS {
            true => &p * (1.0 / r),
            false => row![0.0, 0.0, 0.0],
        };
        gradients.row_mut(i).copy_from(grad);
    }

    (values, gradients)
}

/// A torus with major radius 1.0 and minor radius 0.35. Genus 1, so the wavefront
/// has to close a non-trivial topology rather than a single convex shell.
fn torus(pts: MatRef<'_, f64>) -> Mat<f64> {
    const R: f64 = 1.0;
    const T: f64 = 0.35;
    Mat::<f64>::from_fn(pts.nrows(), 1, |i, _| {
        let (x, y, z) = (pts[(i, 0)], pts[(i, 1)], pts[(i, 2)]);
        let q = (x * x + y * y).sqrt() - R;
        (q * q + z * z).sqrt() - T
    })
}

const SPHERE_EXTENTS: [f64; 6] = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5];
const TORUS_EXTENTS: [f64; 6] = [-1.5, -1.5, -0.5, 1.5, 1.5, 0.5];

fn sphere_seeds() -> Mat<f64> {
    mat![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
}

fn torus_seeds() -> Mat<f64> {
    mat![[1.35, 0.0, 0.0], [-1.35, 0.0, 0.0]]
}

/// Counts how many field samples one extraction takes, so results can be reported
/// as implicit-function evaluations per second. Reducing this count is the whole
/// point of surface following, so it is the metric worth tracking.
fn count_evaluations(
    seeds: MatRef<f64>,
    extents: &[f64],
    resolution: f64,
    field: fn(MatRef<'_, f64>) -> Mat<f64>,
    cluster_method: ClusterMethod,
    boundary_closure: BoundaryClosure,
) -> u64 {
    let evaluations = AtomicU64::new(0);
    let mut counting_fn = |targets: MatRef<f64>| {
        evaluations.fetch_add(targets.nrows() as u64, Ordering::Relaxed);
        field(targets)
    };
    build_isosurface(
        seeds,
        extents,
        resolution,
        None,
        0.0,
        &mut counting_fn,
        None,
        cluster_method,
        boundary_closure,
        None,
    );
    evaluations.load(Ordering::Relaxed)
}

/// Extraction cost as the lattice is refined. Halving the resolution roughly
/// quadruples the surface sample count.
fn bench_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmt/resolution");
    medium(&mut group);

    let seeds = sphere_seeds();

    for resolution in [0.2f64, 0.1, 0.05, 0.025] {
        let evaluations = count_evaluations(
            seeds.as_ref(),
            &SPHERE_EXTENTS,
            resolution,
            sphere,
            ClusterMethod::CurvatureWeighted,
            BoundaryClosure::None,
        );
        group.throughput(Throughput::Elements(evaluations));

        let mut field = sphere;
        group.bench_with_input(
            BenchmarkId::new("sphere", resolution),
            &resolution,
            |b, &resolution| {
                b.iter(|| {
                    black_box(build_isosurface(
                        seeds.as_ref(),
                        &SPHERE_EXTENTS,
                        resolution,
                        None,
                        0.0,
                        &mut field,
                        None,
                        ClusterMethod::CurvatureWeighted,
                        BoundaryClosure::None,
                        None,
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Vertex clustering strategy, which dominates post-processing cost.
fn bench_cluster_method(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmt/cluster_method");
    medium(&mut group);

    let seeds = sphere_seeds();
    let resolution = 0.05;

    for (name, cluster_method) in [
        ("none", ClusterMethod::None),
        ("average", ClusterMethod::Average),
        ("curvature_weighted", ClusterMethod::CurvatureWeighted),
    ] {
        let evaluations = count_evaluations(
            seeds.as_ref(),
            &SPHERE_EXTENTS,
            resolution,
            sphere,
            cluster_method,
            BoundaryClosure::None,
        );
        group.throughput(Throughput::Elements(evaluations));

        let mut field = sphere;
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &cluster_method,
            |b, &cluster_method| {
                b.iter(|| {
                    black_box(build_isosurface(
                        seeds.as_ref(),
                        &SPHERE_EXTENTS,
                        resolution,
                        None,
                        0.0,
                        &mut field,
                        None,
                        cluster_method,
                        BoundaryClosure::None,
                        None,
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Analytic gradients versus central differences. Without a gradient function the
/// seed projection and curvature estimates fall back to extra field evaluations.
fn bench_gradient_source(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmt/gradient_source");
    medium(&mut group);

    let seeds = sphere_seeds();
    let resolution = 0.05;

    let mut field = sphere;
    group.bench_function("central_differences", |b| {
        b.iter(|| {
            black_box(build_isosurface(
                seeds.as_ref(),
                &SPHERE_EXTENTS,
                resolution,
                None,
                0.0,
                &mut field,
                None,
                ClusterMethod::CurvatureWeighted,
                BoundaryClosure::None,
                None,
            ))
        })
    });

    let mut field = sphere;
    let mut gradient_fn = sphere_with_gradient;
    group.bench_function("analytic", |b| {
        b.iter(|| {
            black_box(build_isosurface(
                seeds.as_ref(),
                &SPHERE_EXTENTS,
                resolution,
                None,
                0.0,
                &mut field,
                Some(&mut gradient_fn),
                ClusterMethod::CurvatureWeighted,
                BoundaryClosure::None,
                None,
            ))
        })
    });

    group.finish();
}

/// AABB clipping and cap generation, exercised by shrinking the extents so the
/// sphere is cut by every face of the box.
fn bench_boundary_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmt/boundary_closure");
    medium(&mut group);

    // Tighter than the unit sphere, so the surface intersects all six faces.
    let clipped_extents = [-0.7f64, -0.7, -0.7, 0.7, 0.7, 0.7];
    let seeds = mat![[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]];
    let resolution = 0.05;

    for (name, boundary_closure) in [
        ("none", BoundaryClosure::None),
        ("close_positive", BoundaryClosure::ClosePositive),
        ("close_negative", BoundaryClosure::CloseNegative),
    ] {
        let mut field = sphere;
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &boundary_closure,
            |b, &boundary_closure| {
                b.iter(|| {
                    black_box(build_isosurface(
                        seeds.as_ref(),
                        &clipped_extents,
                        resolution,
                        None,
                        0.0,
                        &mut field,
                        None,
                        ClusterMethod::CurvatureWeighted,
                        boundary_closure,
                        None,
                    ))
                })
            },
        );
    }

    group.finish();
}

/// A genus-1 field, to cover wavefront expansion around non-convex topology.
fn bench_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmt/topology");
    medium(&mut group);

    let seeds = torus_seeds();
    let resolution = 0.05;

    let evaluations = count_evaluations(
        seeds.as_ref(),
        &TORUS_EXTENTS,
        resolution,
        torus,
        ClusterMethod::CurvatureWeighted,
        BoundaryClosure::None,
    );
    group.throughput(Throughput::Elements(evaluations));

    let mut field = torus;
    group.bench_function("torus", |b| {
        b.iter(|| {
            black_box(build_isosurface(
                seeds.as_ref(),
                &TORUS_EXTENTS,
                resolution,
                None,
                0.0,
                &mut field,
                None,
                ClusterMethod::CurvatureWeighted,
                BoundaryClosure::None,
                None,
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_resolution,
    bench_cluster_method,
    bench_gradient_source,
    bench_boundary_closure,
    bench_topology
);
criterion_main!(benches);
