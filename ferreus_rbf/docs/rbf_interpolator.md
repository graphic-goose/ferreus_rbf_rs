Radial basis function (RBF) interpolator.

An [`RBFInterpolator`] represents a fitted RBF model built from input
data points, their associated values, and a chosen kernel. Once
constructed, it can be used to evaluate interpolated values at new
locations, or serialized for later reuse.

The interpolator stores:
- The original input points and values.
- The solved RBF and polynomial coefficients.
- Kernel settings and solver parameters used during fitting.
- Optional global trend transforms (e.g. anisotropy/scaling/rotation).
- An optional Fast Multipole Method (FMM) tree evaluator for efficient queries.

# Construction:

An interpolator is constructed using the [`RBFInterpolatorBuilder`], via the
[`RBFInterpolator::builder`] method, which applies sensible defaults and allows
incremental configuration:

```rust
use ferreus_rbf::{
    RBFInterpolator,
    interpolant_config::{
        InterpolantSettings, 
        RBFKernelType,
    },
    generate_random_points,
    RBFTestFunctions,
};

// Generate some random data in the unit hypercube
let dimensions = 2;
let num_points = 100;
let source_points = generate_random_points(num_points, dimensions, Some(42));

// Assign some values to the source points using Franke's function
let source_values = RBFTestFunctions::franke_2d(&source_points);

// Create a InterpolantSettings instance
let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear).build();

// Setup and solve the RBF
let mut rbfi = RBFInterpolator::builder(source_points, source_values, interpolant_settings).build();
```

# Saving and loading an RBFInterpolator to disc:

```rust ignore
// Save to JSON
let model_save_path = std::path::PathBuf::from("rbf_model.json");

rbfi.save_model(&model_save_path)?;

let mut loaded_rbfi = RBFInterpolator::load_model(model_save_path)?;
```

# Getting progress updates from the inteprolator

For large numbers of input points an iterative solver is used and can take some time.
The [`RBFInterpolator`] outputs progress updates via the `progress_callback` option,
which returns different [`ProgressMsg`] variants.

```rust
use ferreus_rbf::{
    RBFInterpolator,
    interpolant_config::{
        InterpolantSettings, 
        RBFKernelType,
    },
    generate_random_points,
    RBFTestFunctions,
    progress::{
        ProgressMsg, 
        closure_sink,
    },
};

// Generate some random data in the unit hypercube
let dimensions = 2;
let num_points = 10000;
let source_points = generate_random_points(num_points, dimensions, Some(42));

// Assign some values to the source points using Franke's function
let source_values = RBFTestFunctions::franke_2d(&source_points);

// Use the closure_sink function to setup a way to process the progress messages.
// In this example we just print each message as it arrives.
let (sink, _listener) = closure_sink(256, |msg| match msg {
    ProgressMsg::SolverIteration {
        iter,
        residual,
        progress,
    } => {
        println!(
            "Iteration: {:>3}    {:>.5E} {:>.1}%",
            iter,
            residual,
            progress * 100.0
        );
    }
    ProgressMsg::SurfacingProgress {
        isovalue,
        stage,
        progress,
    } => {
        println!(
            "Isovalue: {:?}    Stage: {} {:>.1}%",
            isovalue,
            stage,
            progress * 100.0
        );
    }
    ProgressMsg::DuplicatesRemoved { num_duplicates } => {
        println!("Removed {:>3} duplicate points", num_duplicates);
    }

    ProgressMsg::Message { message } => {
        println!("{message}");
    }
});

let kernel = RBFKernelType::Linear;

// Create a InterpolantSettings instance
let interpolant_settings = InterpolantSettings::builder(kernel).build();

// Setup and solve the RBF
let mut rbfi = RBFInterpolator::builder(
        source_points, 
        source_values, 
        interpolant_settings,
    )
    .progress_callback(sink.clone())
    .build();
```

See [`RBFInterpolatorBuilder`] and [`Params`] for configuration options.