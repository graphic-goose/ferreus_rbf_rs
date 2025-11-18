# ferreus_rbf_rs

Fast global radial basis function (RBF) interpolation in Rust, with Python bindings.

## Overview

Radial basis function (RBF) interpolation is a flexible, mesh‑free approach for
approximating scattered data, but direct solvers require **O(N²)** memory and
**O(N³)** work, which becomes impractical beyond modest problem sizes.

This workspace provides a scalable alternative by combining:

- **Domain decomposition preconditioning** for the global RBF system, and
- A **black box fast multipole method (BBFMM)** evaluator for fast matrix–vector products,

reducing the overall complexity to roughly **O(N log N)** and enabling global
interpolation on millions of points in up to three dimensions.

## Crates and packages

- `ferreus_rbf` – Fast, memory‑efficient global RBF interpolation in 1D, 2D and 3D,
  using domain decomposition and FGMRES with an FMM‑based evaluator.
- `ferreus_bbfmm` – Parallel black box fast multipole method (BBFMM) implementation
  for smooth kernels in 1D, 2D and 3D, supporting adaptive trees and multiple RHS.
- `ferreus_rbf_utils` – Shared kernels, tree utilities and helper functions used by
  `ferreus_rbf`, `ferreus_bbfmm` and the Python bindings.
- `py_ferreus_rbf` – Python bindings for `ferreus_rbf`, providing a high‑level API for
  fast global RBF interpolation from Python.
- `py_ferreus_bbfmm` – Python bindings for `ferreus_bbfmm`, exposing fast kernel
  matrix–vector products and related FMM functionality to Python.

For more detailed API documentation and examples, see the individual crate and
package READMEs, the Rustdoc pages, and the `docs/` and `examples/` directories
in each sub‑project.

## Documentation

- Rust:
  - `ferreus_rbf` - [https://docs.rs/ferreus_rbf/latest/ferreus_rbf/](https://docs.rs/ferreus_rbf/latest/ferreus_rbf/)
  - `ferreus_bbfmm` - [https://docs.rs/ferreus_bbfmm/latest/ferreus_bbfmm/](https://docs.rs/ferreus_bbfmm/latest/ferreus_bbfmm/)
- Python:
  - `ferreus_rbf` - [https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_rbf/](https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_rbf/)
  - `ferreus_bbfmm` - [https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_bbfmm/](https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_bbfmm/)

## Installation

### Rust

Add the desired crate to your `Cargo.toml`, for example:

```toml
[dependencies]
ferreus_rbf = "0.1"
```

or

```toml
[dependencies]
ferreus_bbfmm = "0.1"
```

Refer to crates.io for the latest published versions.

### Python

For the Python bindings, install from PyPI:

```bash
pip install ferreus_rbf
pip install ferreus_bbfmm
```

Then, in Python:

```python
import ferreus_rbf
import ferreus_bbfmm
```

See the `docs/` and `examples/` folders in `py_ferreus_rbf` and
`py_ferreus_bbfmm` for more detailed usage.

## Attribution and licensing

This project was developed while the author was working at
[Maptek](https://www.maptek.com) and has been approved for open‑source
distribution under the terms of the MIT license.

Unless otherwise stated, the following copyright applies:

> Copyright (c) 2025 Maptek Pty Ltd.  
> All rights reserved.

This copyright applies to all files in this repository, whether or not an
individual file contains an explicit notice.

The code is released under the MIT License – see `LICENSE` for details.
