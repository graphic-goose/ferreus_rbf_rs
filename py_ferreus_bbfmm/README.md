# ferreus_bbfmm (Python bindings)

Python bindings for the `ferreus_bbfmm` Rust crate.

## Overview

`ferreus_bbfmm` implements the Black Box Fast Multipole Method (BBFMM), a
kernel‑independent hierarchical algorithm for rapidly evaluating all pairwise
interactions in a collection of particles. These Python bindings expose the
same fast, parallel evaluator to Python users.

The library is well‑suited to problems where:

- The interaction kernel is smooth (non‑oscillatory), and
- Many kernel matrix–vector products are required, for example in iterative
  solvers or large‑scale interpolation.

## Install

```bash
pip install ferreus_bbfmm
```

Then in Python:

```python
import ferreus_bbfmm
```

See the `docs/` and `examples/` directories in this package for more detailed
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
