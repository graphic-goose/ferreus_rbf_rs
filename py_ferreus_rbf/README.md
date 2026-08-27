# Fast global Radial Basis Function (RBF) interpolation.

Radial Basis Function (RBF) interpolation is a powerful but computationally
expensive technique. Direct solvers (e.g. LU factorisation) require **O(N²)**
memory and **O(N³)** operations, which quickly becomes impractical beyond
~10,000 points on a typical machine.

This library provides a scalable alternative by combining
two key techniques:

- **Domain Decomposition** - following 1, used as a preconditioner within a
  Flexible Generalised Minimal Residual (FGMRES) iterative solver.
- **The Fast Multipole Method (FMM)** - via the [`ferreus_bbfmm`](https://docs.rs/ferreus_bbfmm/latest/ferreus_bbfmm/) crate, used as a
  fast evaluator to reduce per-iteration cost.

Together, these methods reduce the overall complexity to **O(N log N)**,
enabling efficient interpolation on datasets with millions of points in
up to three dimensions.

---
# Features
- Written in Rust
- Supports 1D, 2D, and 3D input domains
- Scales efficiently to datasets with over 1,000,000 input source points
- Optional global trend transforms to capture large-scale patterns in the data
- Provides fast 3D isosurface extraction using a surface-following
  regularised marching tetrahedra method via the ['ferreus_rmt'](https://docs.rs/ferreus_rmt/latest/ferreus_rmt/) crate
- Optional simultaneous evaluation of RBF values and gradients
- Built on [`faer`](https://docs.rs/faer/latest/faer/) for linear algebra, avoiding
  complex build dependencies

---
## Install

```bash
pip install ferreus_rbf
```
Then in Python:

```python
import ferreus_rbf
```

See the [docs](https://graphic-goose.github.io/ferreus_rbf_rs/ferreus_rbf/) and [examples](https://github.com/graphic-goose/ferreus_rbf_rs/tree/main/py_ferreus_rbf/examples) directory for more detailed usage and API documentation.

--- 
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

--- 
## References
1.  R. K. Beatson, W. A. Light, and S. Billings. Fast solution of the radial basis
    function interpolation equations: domain decomposition methods. SIAM J. Sci.
    Comput., 22(5):1717–1740 (electronic), 2000.
2.  Haase, G., Martin, D., Schiffmann, P., Offner, G. (2018). A Domain Decomposition
    Multilevel Preconditioner for Interpolation with Radial Basis Functions.
    In: Lirkov, I., Margenov, S. (eds) Large-Scale Scientific Computing. LSSC 2017.
3.  Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab. World Scientific Publishing Co.
4.  J. B. Cherrie. Fast Evaluation of Radial Basis Functions: Theory and Application.
    PhD thesis, University of Canterbury, 2000.
5.  G.M. Treece, R.W. Prager, and A.H. Gee. Regularised marching tetrahedra: improved
    iso-surface extraction. Computers & Graphics, 23(4):583–598, 1999.