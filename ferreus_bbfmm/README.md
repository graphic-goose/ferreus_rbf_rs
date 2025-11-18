# ferreus_bbfmm

Parallel black box fast multipole method (BBFMM) in Rust.

## Overview

`ferreus_bbfmm` is a parallel implementation of the Black Box Fast Multipole
Method. BBFMM is a kernel‑independent, hierarchical algorithm for rapidly
evaluating all pairwise interactions in a collection of particles.

While originally developed as the fast evaluator for radial basis function (RBF)
interpolation, this crate has been generalised to support a broad range of FMM
use‑cases where the kernel is smooth (i.e. non‑oscillatory).

## Features

- 1D (binary tree), 2D (quadtree) and 3D (octree) trees
- Optimised low‑rank M2L interactions that leverage symmetries and compression
- Adaptive and non‑adaptive tree structures
- Support for multiple right‑hand sides
- Designed to work with user‑defined kernels via traits

`ferreus_bbfmm` is used directly by the `ferreus_rbf` crate as the fast
evaluator for large‑scale RBF interpolation problems, and is also exposed to
Python via the `py_ferreus_bbfmm` bindings.

## Getting started

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
ferreus_bbfmm = "0.1"
```

Then define a kernel that implements the appropriate trait and construct an
`FmmTree` as shown in the crate documentation and examples.

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
