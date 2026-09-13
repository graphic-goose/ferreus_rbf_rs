# Benchmarking

This workspace uses [Criterion](https://github.com/criterion-rs/criterion.rs) for
statistical benchmarking. Benchmarks live in each crate's `benches/` directory and
are wired up with `harness = false`.

## What is covered

| Crate | Bench target | Groups | What it measures |
| --- | --- | --- | --- |
| `ferreus_bbfmm` | `fmm` | `fmm/build` | Tree construction and M2L operator precompute, swept over point count, interpolation order and compression type |
| | | `fmm/set_weights` | The upward pass (P2M, M2M), swept over point count and number of right-hand sides |
| | | `fmm/evaluate` | The downward and leaf passes, with and without gradients, swept over point count, dimensionality and right-hand sides |
| `ferreus_rbf_utils` | `kernels` | `kernels/phi` | Scalar evaluation cost of every kernel in the registry |
| | | `kernels/a_matrix` | Dense kernel matrix assembly, general and symmetric-with-nugget |
| | | `kernels/utils` | Whole-array helpers used on every solve |
| `ferreus_rmt` | `isosurface` | `rmt/resolution` | Extraction cost as the sample lattice is refined |
| | | `rmt/cluster_method` | `None` vs `Average` vs `CurvatureWeighted` vertex clustering |
| | | `rmt/gradient_source` | Analytic gradients vs central differences |
| | | `rmt/boundary_closure` | AABB clipping and cap generation |
| | | `rmt/topology` | A genus-1 field (torus), for wavefront expansion around non-convex topology |
| `ferreus_rbf` | `fit` | `rbf/fit/direct` | The dense LU path taken below `naive_solve_threshold` |
| | | `rbf/fit/iterative` | Domain decomposition + FMM-accelerated FGMRES, the headline path |
| | | `rbf/fit/kernel` | Kernel choice, which sets the default FMM interpolation order |
| | | `rbf/fit/tolerance` | Fitting tolerance, which sets the solver iteration count |
| | | `rbf/fit/params` | Domain decomposition leaf size and the input uniqueness check |
| | | `rbf/fit/dataset` | The real 35,801-point albatite signed-distance dataset |
| `ferreus_rbf` | `evaluate` | `rbf/evaluate` | One-shot `evaluate` vs cached `build_evaluator` + `evaluate_targets`, with and without gradients |
| | | `rbf/evaluate_at_source` | Evaluation back at the source locations |
| `ferreus_rbf` | `isosurface` | `rbf/isosurface` | End-to-end extraction from a fitted interpolant: RMT wavefront expansion driving FMM evaluation |

Benchmarks that need the albatite dataset skip themselves with a message on stderr
if `datasets/albatite_SD_points.csv` is not checked out.

## Running

```bash
# Everything. Takes a while - see "Runtime" below.
cargo bench --workspace

# One crate, or one bench target.
cargo bench -p ferreus_bbfmm
cargo bench -p ferreus_rbf --bench evaluate

# One group or one benchmark. The filter is a regex matched against the full
# benchmark id, so group prefixes work as filters.
cargo bench -p ferreus_rbf --bench fit -- 'rbf/fit/iterative'
cargo bench -p ferreus_rmt -- 'rmt/cluster_method/average'

# Check that every benchmark still runs, without measuring anything.
# This is the fast smoke test; use it in CI and after refactors.
cargo bench --workspace -- --test
```

Criterion writes an HTML report to `target/criterion/report/index.html`.

## Comparing before and after

This is the workflow the harness exists for. Save a baseline from the current
code, make a change, then compare against it:

```bash
# On the unmodified code:
cargo bench -p ferreus_bbfmm -- --save-baseline before

# ... make your change ...

cargo bench -p ferreus_bbfmm -- --baseline before
```

The second run prints a `change:` line per benchmark with a confidence interval and
a significance verdict. Criterion labels anything inside its noise threshold as
`Change within noise threshold`, so treat only `Performance has improved` and
`Performance has regressed` as real.

Baselines are stored under `target/criterion/`, so they do not survive
`cargo clean`. Save a baseline under a name you will recognise (`before`,
`main`, `pre-simd`) rather than relying on the implicit `base` baseline, which every
unnamed run overwrites.

## Keeping measurements honest

These benchmarks are multithreaded via Rayon and will use every core, so results
depend on machine load far more than a single-threaded microbenchmark would.

**Never run two benchmark targets at once.** Two concurrent suites on a 16-core
machine inflated measurements here by 3x to 11x - a 633 ms fit reported as 2.49 s -
with no warning from Criterion and no obvious sign in the output. `cargo bench`
serialises targets within one invocation; the hazard is starting a second invocation,
or kicking one off in a background shell, while another is still going. If a number
looks implausible, check nothing else was running before believing it.

Beyond that:

- Close other work before a comparison run, and compare only baselines taken on the
  same machine.
- `RAYON_NUM_THREADS=1 cargo bench ...` removes scheduling noise and makes changes in
  total work visible. Use it to check algorithmic improvements, and the default
  thread count to check scaling.
- On a laptop, pin the CPU governor to `performance` for the duration of a
  comparison; thermal throttling across a long run shows up as a fake regression in
  whatever runs last.
- `-- --quick` cuts sample counts substantially. Fine for a rough look, not for a
  decision.
- To sanity-check a suspicious result, time the operation standalone in a small
  `examples/` binary. The two should agree to within a few percent; if they do not,
  something about the measurement environment is wrong rather than the code.

Inputs are all generated from fixed seeds, so a given benchmark fits the same system
and extracts the same surface on every run. If you change a seed, a tolerance, or a
test function, previously saved baselines are no longer comparable - resave them.

## Runtime

Criterion spends `warm_up_time + measurement_time` on *every* benchmark, whether the
code under test takes nanoseconds or seconds. With ~100 benchmarks that fixed budget,
not the work itself, sets the suite runtime. Each group therefore picks one of three
policies, defined at the top of each bench file:

| Policy | Warm-up | Measurement | Samples | Sampling | For |
| --- | --- | --- | --- | --- | --- |
| `micro` | 1s | 3s | 100 | linear | Iterations well under a millisecond |
| `medium` | 1s | 5s | 20 | linear | Iterations up to a few hundred milliseconds |
| `heavy` | 1s | 5s | 10 | **flat** | Iterations of a second or more |

The sampling mode matters more than it looks. Under Criterion's default *linear*
sampling, sample *i* runs *i* iterations, so `sample_size(10)` means
`1+2+...+10 = 55` runs of the benchmark. Criterion normally compresses that schedule
to fit `measurement_time`, but it cannot go below the linear minimum, so for
second-scale iterations a group silently takes 5.5x longer than its sample count
suggests. *Flat* sampling runs the same number of iterations per sample, and with a
short `measurement_time` that number stays at one.

So `heavy` benchmarks print `Warning: Unable to complete 10 samples in 5.0s`. That is
expected, and it means the floor is the work itself rather than a Criterion setting.

If you are reading the logs to check a group's cost, the number to look at is the
iteration count, not the sample count:

```
Collecting 10 samples in estimated 39.163 s (55 iterations)   <- linear, 55 runs
Collecting 10 samples in estimated 17.249 s (10 iterations)   <- flat, 10 runs
```

Measured wall-clock on an idle 16-core machine, one target at a time:

| Target | Measured |
| --- | --- |
| `ferreus_rbf --bench isosurface` | 1m 29s |
| `ferreus_rbf --bench evaluate` | 1m 33s |
| `ferreus_rmt --bench isosurface` | 1m 45s |
| `ferreus_rbf_utils --bench kernels` | 2m 56s |
| `ferreus_bbfmm --bench fmm` | 3m 17s |
| `ferreus_rbf --bench fit` | 3m 19s |
| **Full suite** | **~14m 20s** |

`fit` cannot be tuned down much further: a 50,000-point fit genuinely takes about a
second, and Criterion's minimum `sample_size` is 10, so that one benchmark has a
ten-second floor by construction. Filter to the group you are working on rather than
running the whole suite on every iteration.

Two flags help when iterating:

```bash
# Skip plot and HTML generation (~0.6 s per benchmark).
cargo bench -p ferreus_rbf_utils -- --noplot

# Much lower sample counts. Fine for a rough look, not for a decision.
cargo bench -p ferreus_rbf_utils -- --quick
```

Criterion uses gnuplot for plots when it is installed and a slower pure-Rust backend
otherwise; installing gnuplot shaves a little off every run.

## Profiling

`[profile.bench]` in the workspace root manifest sets `debug = true`, so benchmark
binaries carry symbols. To profile one benchmark, run its binary directly with
`--profile-time`, which skips Criterion's statistics and just runs the benchmark for
a fixed number of seconds.

Ask Cargo where the binary is rather than guessing - the path is a hashed build
directory, not `target/release/deps`:

```bash
BENCH=$(cargo bench -p ferreus_rbf --bench fit --no-run --message-format=json \
        | jq -r 'select(.executable != null) | .executable' | tail -1)

perf record -g "$BENCH" --bench 'rbf/fit/iterative' --profile-time 20
perf report

# Or with samply, which gives a shareable flamegraph in the browser:
samply record "$BENCH" --bench 'rbf/fit/iterative' --profile-time 20
```

There is also a `profiling` profile for the examples and Python extension modules -
release optimisation with full debug info retained:

```bash
cargo build --profile profiling --example isosurface_linear_rmt
```

## Adding a benchmark

1. Put it in the `benches/` directory of the crate that owns the code. Benchmarks
   may only use a crate's public API plus its `[dev-dependencies]`.
2. Register it in that crate's `Cargo.toml`:

   ```toml
   [[bench]]
   name = "my_bench"
   harness = false
   ```

3. Generate inputs from a fixed seed. The `ferreus_rbf` benches share fixtures via
   `benches/support/mod.rs`; add new shared fixtures there rather than duplicating
   them across targets.
4. Wrap inputs and results in `std::hint::black_box` so nothing gets optimised away.
5. Do expensive setup (fitting an interpolant, building an FMM tree) *outside*
   `b.iter`, unless the setup is what you are measuring.
6. Set `Throughput::Elements` where there is a meaningful unit of work - points,
   matrix entries, field evaluations - so results stay comparable across sizes.
7. Apply one of the `micro` / `medium` / `heavy` policies to the group rather than
   setting `sample_size` and `measurement_time` by hand, so suite runtime stays
   predictable as benchmarks are added.
8. Prefer adding a sweep point that distinguishes something. Every entry costs the
   full time budget, so a variant that measures the same as its neighbour is pure
   runtime with no information.
