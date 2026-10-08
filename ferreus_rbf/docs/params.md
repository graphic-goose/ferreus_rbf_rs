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
- `direct_eval_threshold`: `4096` (direct evaluation for fewer sources; `0` forces FMM)
- `direct_eval_batch_size`: `128` (positive target batch size, independent of FMM)
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
Direct evaluation computes exact kernel sums over parallel target chunks with bounded
working memory, including gradients and all value columns. Its threshold is independent of the fitting
threshold. It applies to one-shot queries, stored evaluators, source-point
queries, and isosurface sampling. Direct evaluators do not require target extents.

Direct evaluation uses `direct_eval_batch_size` as its positive target batch size.
FMM evaluation uses `fmm_params.eval_chunk_size` independently.
Kernel selection occurs once per query; batching and parallelism are handled by
the typed kernel helper in `ferreus_rbf_utils`.
