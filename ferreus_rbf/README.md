# ferreus_rbf

Fast, memory‑efficient global radial basis function (RBF) interpolation in Rust.

## Overview

Direct RBF solvers based on dense linear algebra require **O(N²)** memory and
**O(N³)** work, which quickly becomes impractical beyond tens of thousands of
points. `ferreus_rbf` provides a scalable alternative suitable for millions of
points in up to three dimensions.

The crate combines:

- **Domain decomposition** (following established RBF preconditioning schemes),
  used as a preconditioner within a Flexible GMRES (FGMRES) iterative solver, and
- A **fast multipole method (FMM)** evaluator via the `ferreus_bbfmm` crate,
  which accelerates matrix–vector products during each iteration.

Together these techniques reduce the asymptotic cost to roughly **O(N log N)**,
enabling fast global interpolation and 3D surface extraction on large scattered
datasets.

## Features
- Supports 1D, 2D and 3D input domains
- Scales to datasets with over 1,000,000 source points (subject to hardware)
- Optional global trend transforms to capture large‑scale structure in the data
- Provides fast 3D isosurface extraction using a surface-following
  regularised marching tetrahedra method via the ['ferreus_rmt'](https://docs.rs/ferreus_rmt/latest/ferreus_rmt/) crate
- Optional simultaneous evaluation of RBF values and gradients
- Built on [`faer`](https://docs.rs/faer/latest/faer/) for linear algebra, avoiding
  complex build dependencies

## Getting started

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
ferreus_rbf = "0.2.2"
```

Then construct an `RBFInterpolator` with your source points, values and
configuration. See the crate documentation and the `examples/` directory for
complete examples.

## Attribution and licensing

This crate was developed while the author was working at
[Maptek](https://www.maptek.com) and has been approved for open‑source
distribution under the terms of the MIT license.

Unless otherwise stated, the following copyright applies:

> Copyright (c) 2025 Maptek Pty Ltd.  
> All rights reserved.

This copyright applies to all files in this repository, whether or not an
individual file contains an explicit notice.

The code is released under the MIT License – see the top‑level `LICENSE` file
for details.
