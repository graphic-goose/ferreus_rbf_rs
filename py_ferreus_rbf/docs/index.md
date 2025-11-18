# ferreus_rbf

# Fast global Radial Basis Function (RBF) interpolation.

Radial Basis Function (RBF) interpolation is a powerful but computationally
expensive technique. Direct solvers (e.g. LU factorisation) require **O(N²)**
memory and **O(N³)** operations, which quickly becomes impractical beyond
~10,000 points on a typical machine.

This library provides a scalable alternative by combining
two key techniques:

- **Domain Decomposition** - following [1], used as a preconditioner within a
  Flexible Generalised Minimal Residual (FGMRES) iterative solver.
- **The Fast Multipole Method (FMM)** - via the [`ferreus_bbfmm`] crate, used as a
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
- Provides fast 3D isosurface extraction using a surface-following,
  non-adaptive Surface Nets method
- Built on [`faer`](https://docs.rs/faer/latest/faer/) for linear algebra, avoiding
  complex build dependencies

---

## Install

```bash
pip install ferreus_rbf
```

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
