Holds the configuration parameters for an RBF kernel.

An [`InterpolantSettings`] instance is created using the
[`InterpolantSettings::builder`] method, which provides defaults and convenience
methods for setting optional fields.

In most cases, end users should prefer [`InterpolantSettings::builder()`] over
constructing this struct directly.

The only required input is the [`RBFKernelType`].  

If no additional options are provided, defaults are applied:
 - [`Drift`] is set to the minimum valid choice for the kernel.
 - For spheroidal kernels, `base_range` and `total_sill` default to `1.0`.
 - The nugget defaults to `0.0` but may be specified for any kernel.
 - `fitting_accuracy`: [`FittingAccuracy::default()`]

# Linear kernel with default drift:
```rust
use ferreus_rbf::{interpolant_config::{RBFKernelType, InterpolantSettings, Drift}};

let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Linear).build();

assert_eq!(
    (interpolant_settings.kernel_type, interpolant_settings.drift),
    (RBFKernelType::Linear, Drift::Constant) 
)
```

# Spheroidal kernel with custom range and sill:
```rust
use ferreus_rbf::{interpolant_config::{RBFKernelType, InterpolantSettings, Drift}};

let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Spheroidal)
    .base_range(100.0)
    .total_sill(10.0)
    .build();

assert_eq!(
    (
        interpolant_settings.kernel_type, 
        interpolant_settings.drift,
        interpolant_settings.base_range,
        interpolant_settings.total_sill,
    ),
    (
        RBFKernelType::Spheroidal, 
        Drift::None,
        100.0,
        10.0
    ) 
)
```

# Spheroidal kernel with constant drift and nugget:
```rust
use ferreus_rbf::{interpolant_config::{RBFKernelType, InterpolantSettings, Drift, SpheroidalOrder}};

let interpolant_settings = InterpolantSettings::builder(RBFKernelType::Spheroidal)
    .spheroidal_order(SpheroidalOrder::Five)
    .drift(Drift::Constant)
    .nugget(10.0)
    .base_range(100.0)
    .total_sill(10.0)
    .build();

assert_eq!(
    (
        interpolant_settings.kernel_type, 
        interpolant_settings.spheroidal_order,
        interpolant_settings.drift,
        interpolant_settings.nugget,
        interpolant_settings.base_range,
        interpolant_settings.total_sill,
    ),
    (
        RBFKernelType::Spheroidal, 
        SpheroidalOrder::Five,
        Drift::Constant,
        10.0,
        100.0,
        10.0
    ) 
)
```
