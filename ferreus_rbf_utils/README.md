# ferreus_rbf_utils

Shared utilities for the `ferreus_rbf` and `ferreus_bbfmm` crates and their
Python bindings.

## Overview

`ferreus_rbf_utils` provides reusable building blocks that underpin the RBF and
FMM implementations in this workspace, including:

- Implementations of radial basis function and related kernels
- Kernel parameter types and helper routines
- Common utilities for working with trees, distances and point sets

Although primarily intended as an internal support crate, it can also be used
directly by downstream crates that need access to the same kernel definitions
and helpers.

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
