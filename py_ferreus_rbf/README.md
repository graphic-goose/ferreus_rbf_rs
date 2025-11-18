# ferreus_rbf (Python bindings)

Python bindings for the `ferreus_rbf` Rust crate, providing fast global radial
basis function (RBF) interpolation from Python.

## Overview

Direct RBF solvers based on dense linear algebra require **O(N²)** memory and
**O(N³)** work, which becomes impractical on large datasets. The underlying
Rust implementation combines:

- **Domain decomposition** preconditioning in a Flexible GMRES (FGMRES) solver, and
- A **fast multipole method (FMM)** evaluator via `ferreus_bbfmm`,

to reduce the cost to roughly **O(N log N)**. This makes it possible to build
global interpolants and extract isosurfaces on datasets with millions of points
from Python.

## Limitations

The current isosurface extraction method does **not** guarantee manifold or
watertight meshes. Surfaces may contain trifurcations or self‑intersections
and may therefore be unsuitable for downstream boolean or other topology‑
sensitive operations without additional post‑processing.

## Install

```bash
pip install ferreus_rbf
```

Then in Python:

```python
import ferreus_rbf
```

See the [docs](https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_rbf/) and `examples/` directories in this package for more detailed
usage and API documentation.

## Attribution and licensing

This package was developed while the author was working at
[Maptek](https://www.maptek.com) and has been approved for open‑source
distribution under the terms of the MIT license.

Unless otherwise stated, the following copyright applies:

> Copyright (c) 2025 Maptek Pty Ltd.  
> All rights reserved.

This copyright applies to all files in this repository, whether or not an
individual file contains an explicit notice.

The code is released under the MIT License – see the top‑level `LICENSE` file
for details.
