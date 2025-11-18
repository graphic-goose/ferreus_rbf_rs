<!-- 
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the Home page of the Python documentation for the ferreus_bbfmm library.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
//////////////////////////////////////////////////////////////////////////////////////////// 
-->

# ferreus_bbfmm

# Black Box Fast Multipole Method (BBFMM)

This crate is a parallel implementation of the `Black Box Fast Multipole Method` in Rust.

BBFMM is a kernel-independent, hierarchical algorithm for rapidly evaluating
all pairwise interactions in a collection of particles.

While originally developed for radial basis function (RBF) interpolation problems,
`ferreus_bbfmm` has been generalised to support a broad range of FMM use cases where
the kernel is smooth (i.e. non-oscillatory).

---
# Features
- 1D (binary tree), 2D (quadtree) and 3D (octree)
- Optimised low-rank M2L interactions that leverage symmetries and compression
- Both adaptive and non-adaptive tree structures
- Multiple right-hand sides
---

## Install

```bash
pip install ferreus_bbfmm
```
--- 
## References
1.  Fong, W., & Darve, E. (2009).
    *[The black-box fast multipole method.](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf)*
    *Journal of Computational Physics*, **228**(23), 8712–8725.  
2.  Messner, M., Bramas, B., Coulaud, O., & Darve, E. (2012).
    *[Optimized M2L kernels for the Chebyshev interpolation-based fast multipole method.](https://arxiv.org/pdf/1210.7292)*  
3.  Pouransari, H., & Darve, E. (2015).
    *[Optimizing the adaptive fast multipole method for fractal sets.](https://doi.org/10.1137/140962681)*
    *SIAM Journal on Scientific Computing*, **37**, A1040–A1066.
