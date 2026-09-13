/////////////////////////////////////////////////////////////////////////////////////////////
//
// End-to-end benchmarks for isosurface extraction from a fitted RBF interpolant.
// This is the full production workload: RMT wavefront expansion driving FMM
// evaluation of the interpolant.
//
// Copyright (c) 2026, Leo Timmins. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

mod support;

use std::{hint::black_box, time::Duration};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, SamplingMode, measurement::WallTime,
};
use ferreus_rbf::{interpolant_config::RBFKernelType, isosurfacing::BoundaryClosure};
use ferreus_rbf_utils::get_pointarray_extents;
use support::{albatite, fit};

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

/// Pads extents outwards by ten cells, matching what the examples do so the
/// surface is not clipped by the sampling domain.
fn padded_extents(extents: &[f64], resolution: f64) -> Vec<f64> {
    let dimensions = extents.len() / 2;
    extents
        .iter()
        .enumerate()
        .map(|(i, v)| match i < dimensions {
            true => v - resolution * 10.0,
            false => v + resolution * 10.0,
        })
        .collect()
}

/// Isosurfacing the albatite signed-distance interpolant at several lattice
/// resolutions. The fit is done once as a fixture; only extraction is measured.
fn bench_albatite(c: &mut Criterion) {
    let Some((points, values)) = albatite() else {
        return;
    };
    let source_extents = get_pointarray_extents(points.as_ref());
    let mut rbfi = fit(points, values, RBFKernelType::Linear);

    let mut group = c.benchmark_group("rbf/isosurface");
    heavy(&mut group);

    for resolution in [25.0f64, 10.0, 5.0] {
        let extents = padded_extents(&source_extents, resolution);
        group.bench_with_input(
            BenchmarkId::new("albatite_linear", resolution),
            &resolution,
            |b, &resolution| {
                b.iter(|| {
                    black_box(rbfi.build_isosurface(
                        &extents,
                        resolution,
                        0.0,
                        BoundaryClosure::ClosePositive,
                    ))
                })
            },
        );
    }

    // Cap generation against the sampling AABB, which adds clipping and
    // retriangulation work on every boundary cell.
    let resolution = 10.0;
    let extents = padded_extents(&source_extents, resolution);
    for (name, boundary_closure) in [
        ("none", BoundaryClosure::None),
        ("close_positive", BoundaryClosure::ClosePositive),
    ] {
        group.bench_with_input(
            BenchmarkId::new("albatite_closure", name),
            &boundary_closure,
            |b, &boundary_closure| {
                b.iter(|| {
                    black_box(rbfi.build_isosurface(&extents, resolution, 0.0, boundary_closure))
                })
            },
        );
    }

    // Several isovalues share a single evaluator and lattice setup.
    let isovalues = vec![-50.0, 0.0, 50.0];
    group.bench_function("albatite_multi_isovalue", |b| {
        b.iter(|| {
            black_box(rbfi.build_isosurfaces(
                &extents,
                resolution,
                &isovalues,
                BoundaryClosure::ClosePositive,
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_albatite);
criterion_main!(benches);
