Configuration parameters controlling how an RBF system is solved.

A [`Params`] instance specifies solver options, accuracy targets,
domain decomposition behaviour, fast multipole settings, and other
controls for model fitting and evaluation.

This struct is created with the [`Params::builder`] method,
which provides sensible defaults and convenience methods for customizing
individual fields.

Defaults:
- `solver_type`: [`Solvers::FGMRES`]
- `ddm_params`: [`DDMParams::default()`]
- `fmm_params`: [`FmmParams::new_defaults(kernel_type)`]
- `naive_solve_threshold`: `4096`
- `test_unique`: `true`

# Examples
```rust
use ferreus_rbf::{
    config::Params, 
    interpolant_config::RBFKernelType
};

let params = Params::builder(RBFKernelType::Linear).build();
assert_eq!(params.solver_type, ferreus_rbf::config::Solvers::FGMRES);
```

```rust
use ferreus_rbf::{
    config::{Params, ParamsBuilder, Solvers}, 
    interpolant_config::RBFKernelType,
};

let params = Params::builder(RBFKernelType::Linear)
    .solver_type(Solvers::DDM)
    .naive_solve_threshold(2048)
    .test_unique(false)
    .build();

assert_eq!(
    (params.solver_type, params.naive_solve_threshold, params.test_unique),
    (Solvers::DDM, 2048, false)
);
```