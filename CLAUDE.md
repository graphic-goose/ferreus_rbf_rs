# CLAUDE.md - ferreus_rbf_rs Project Context

## Project Overview

**ferreus_rbf_rs** is a high-performance Rust library implementing fast global Radial Basis Function (RBF) interpolation with Python bindings. It provides a scalable, memory-efficient solution for RBF interpolation on datasets with millions of points in up to 3D.

### Key Innovation

This library implements **FastRBF** techniques to overcome the computational barriers of traditional RBF interpolation:
- Traditional RBF: O(N²) memory, O(N³) operations → impractical beyond ~10,000 points
- This implementation: O(N log N) complexity → scales to millions of points

### Potential Impact

This library could be a **bona fide open-source alternative to Leapfrog's geomodelling capabilities**, bringing advanced 3D geological implicit modelling to the open-source community.

## Architecture

### Workspace Structure

The project is organized as a Cargo workspace with multiple interconnected crates:

```
ferreus_rbf_rs/
├── ferreus_rbf/          # Core RBF interpolation library (Rust)
├── ferreus_bbfmm/        # Black Box Fast Multipole Method (Rust)
├── ferreus_rbf_utils/    # Shared utilities, kernels, traits
├── py_ferreus_rbf/       # Python bindings for RBF
└── py_ferreus_bbfmm/     # Python bindings for BBFMM
```

### Component Relationships

```
┌─────────────────────────────────────────────────┐
│              User Applications                   │
│     (Rust code or Python via bindings)          │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼────────┐  ┌────▼─────────────┐
│  ferreus_rbf   │  │ ferreus_bbfmm    │
│  (RBF Solver)  │  │ (FMM Evaluator)  │
└───────┬────────┘  └────┬─────────────┘
        │                │
        └────────┬────────┘
                 │
         ┌───────▼──────────┐
         │ ferreus_rbf_utils│
         │ (Shared code)    │
         └──────────────────┘
```

## Core Algorithms

### 1. Domain Decomposition Preconditioning

**Location:** `ferreus_rbf/src/preconditioning/`

**Purpose:** Breaks the global problem into smaller, manageable local problems

**Key Papers:**
- Beatson, Light & Billings (2000): "Fast solution of the radial basis function interpolation equations: domain decomposition methods"
- Haase et al. (2017): "A Domain Decomposition Multilevel Preconditioner for Interpolation with Radial Basis Functions"

**Implementation Details:**
- Uses overlapping domain decomposition
- Each subdomain solves a local RBF problem
- Serves as preconditioner for iterative solver
- Files: `domain_decomposition.rs`, `schwarz.rs`

### 2. Black Box Fast Multipole Method (BBFMM)

**Location:** `ferreus_bbfmm/`

**Purpose:** Accelerates matrix-vector products from O(N²) to O(N log N)

**Key Features:**
- Kernel-independent (works with any smooth kernel)
- Hierarchical tree structure (binary/quadtree/octree)
- Adaptive Chebyshev interpolation
- Optimized M2L (multipole-to-local) interactions
- Parallel execution with Rayon

**Key Files:**
- `bbfmm.rs` - Main FMM tree structure and algorithms
- `chebyshev.rs` - Chebyshev interpolation for kernel approximation
- `aca.rs` - Adaptive Cross Approximation for low-rank compression
- `linear_tree.rs` - Tree data structures
- `morton.rs` - Morton encoding for spatial indexing

### 3. Iterative Solver

**Location:** `ferreus_rbf/src/iterative_solvers.rs`

**Implementation:** Flexible Generalized Minimal Residual (FGMRES)

**Why FGMRES?**
- Allows non-constant preconditioner (domain decomposition)
- Handles non-symmetric systems
- Proven effective for RBF interpolation problems

### 4. RBF Kernels

**Location:** `ferreus_rbf_utils/src/rbf_kernels.rs`

**Supported Kernels:**
- **Polyharmonic splines:** Linear, Cubic, Quintic, ThinPlate
- **Spheroidal (compactly supported):** Orders 0-7
- **Others:** Gaussian, Multiquadric, InverseMultiquadric, InverseQuadratic

**Key for Geomodelling:**
- Linear and cubic are standard for geological implicit modelling
- Spheroidal kernels provide compact support (sparse matrices)

## Geomodelling Capabilities

### Implicit Surface Modelling

The library supports **implicit surface modelling**, the core technique used in modern 3D geological modelling software like Leapfrog:

1. **Input:** Scattered 3D points with scalar values (e.g., geological contacts, orientations)
2. **Process:** Fit RBF to create smooth implicit function
3. **Output:** Extract isosurfaces at specified values

### Isosurface Extraction

**Location:** `ferreus_rbf/src/surfacing/`

**Method:** Surface Nets (non-adaptive, surface-following variant)

**Capabilities:**
- Fast 3D isosurface extraction
- Follows surface during marching
- Generates triangulated meshes

**Limitations (Important):**
- **Not guaranteed to produce manifold/watertight meshes**
- May contain trifurcations or self-intersections
- Requires post-processing for topology-sensitive operations

### Drift Terms (Global Trends)

**Location:** `ferreus_rbf/src/global_trend.rs`, `ferreus_rbf/src/polynomials.rs`

**Purpose:** Capture large-scale trends in geological data

**Supported Drift Types:**
- None (pure RBF)
- Linear (planar trend)
- Quadratic (curved trend)

This is crucial for geological modelling where data often has regional trends.

## Key Dependencies

### Rust Dependencies
- **faer** (v0.23.2): Modern Rust linear algebra library
  - Used for all matrix operations
  - No BLAS/LAPACK required (simplified build)
- **rayon** (v1.11.0): Data parallelism
- **rstar** (v0.12.2): R-tree spatial indexing
- **serde**: Serialization support

### Python Bindings
- **maturin**: Build system for Rust-Python integration
- **pyo3**: Rust-Python FFI

## Development Practices

### Version Status
- Current version: 0.1.x (early release)
- Published on crates.io and PyPI
- Under active development

### Testing
Look for tests in:
- Unit tests: `#[cfg(test)]` modules in source files
- Integration tests: `tests/` directories
- Examples: `examples/` directories (serve as integration tests)

### Documentation
- **Rust docs:** docs.rs with KaTeX support for math equations
- **Python docs:** GitHub Pages
- Extensive inline documentation with mathematical notation

## Important Contexts for AI Assistance

### 1. Performance Considerations
- **Memory:** Key constraint for large datasets
- **Tree depth:** Affects FMM accuracy and performance
- **Overlap ratio:** Domain decomposition parameter affecting convergence
- **Fitting accuracy:** Trade-off between accuracy and iterations

### 2. Numerical Stability
- RBF interpolation is inherently ill-conditioned
- Domain decomposition helps conditioning
- Spheroidal kernels (compact support) better conditioned than global kernels

### 3. Geomodelling Context
If working on geomodelling applications:
- Geological data is sparse and irregularly distributed
- Often need to incorporate structural geology (orientations, faults)
- Trend removal is critical for realistic models
- Multiple scalar fields often needed (different rock types)

### 4. Comparison to Leapfrog
**Similarities:**
- Both use RBF interpolation for implicit modelling
- Support for constraints and orientations
- 3D isosurface extraction

**Key Questions to Investigate:**
- Does Leapfrog use similar FastRBF techniques?
- How do the algorithms compare in practice?
- What additional geological constraints does Leapfrog support?
- Are the kernels and drift terms equivalent?

## File Organization Patterns

### Source Files
- `lib.rs` - Public API and documentation
- `{feature}.rs` - Individual feature implementations
- `{feature}/mod.rs` - Module with submodules

### Configuration
- Builder pattern for `RBFInterpolator`
- Separate config structs: `InterpolantSettings`, `Params`, `FmmParams`, `DDMParams`

### Examples
- Rust examples: `{crate}/examples/*.rs`
- Python examples: `py_{crate}/examples/*.py`
- Often include test datasets in `examples/datasets/`

## Research References

### Core FastRBF Papers
1. **Beatson, Light & Billings (2000)** - Domain decomposition for RBF
2. **Haase et al. (2017)** - Multilevel preconditioner
3. **Fasshauer (2007)** - "Meshfree Approximation Methods with Matlab" (textbook)
4. **Cherrie (2000)** - PhD thesis on fast RBF evaluation

### FMM Background
- Greengard & Rokhlin (1987) - Original FMM paper
- Fong & Darve (2009) - Black Box FMM paper

### Geological Modelling
- Cowan et al. (2002) - Leapfrog's foundational work
- Various papers on implicit modelling in geosciences

## Build and Test Commands

### Rust
```bash
# Build all crates
cargo build --release

# Run tests
cargo test

# Run specific example
cargo run --example franke_2d --release

# Build documentation
cargo doc --no-deps --open

# Check code
cargo clippy
```

### Python
```bash
# Install in development mode
cd py_ferreus_rbf
pip install -e .

# Run examples
python examples/franke_2d.py

# Build for distribution
maturin build --release
```

## Common Tasks for AI Assistance

### Code Navigation
- "Find where [algorithm] is implemented"
- "Show me the domain decomposition code"
- "Where are the RBF kernels defined?"

### Understanding
- "Explain how the FMM tree construction works"
- "What's the role of Chebyshev interpolation in BBFMM?"
- "How does the preconditioner work?"

### Debugging
- "Why might convergence be slow?"
- "What could cause accuracy issues?"
- "How to diagnose memory problems?"

### Enhancement
- "Add a new RBF kernel"
- "Improve the isosurface extraction"
- "Add support for fault constraints"

### Benchmarking
- "Compare performance with direct solver"
- "Analyze scaling with dataset size"
- "Profile memory usage"

## Potential Areas for Investigation/Improvement

### Short-term Maturity Checks
1. Test coverage and quality
2. Error handling robustness
3. API stability and consistency
4. Documentation completeness
5. Edge case handling

### Medium-term Enhancements
1. Manifold isosurface extraction
2. Additional geological constraints (faults, unconformities)
3. Multiple RBF fields with coupling
4. Anisotropic kernels
5. Better conditioning for extreme geometries

### Long-term Research
1. Comparison with Leapfrog algorithms
2. GPU acceleration
3. Adaptive refinement strategies
4. Uncertainty quantification
5. Time-varying fields (4D)

## License and Attribution

- **License:** MIT
- **Copyright:** Maptek Pty Ltd (2025)
- **Author:** Daniel Owen
- **Context:** Developed at Maptek, approved for open-source release

This is significant because Maptek is a major player in mining software, suggesting professional-grade implementation quality.

## Quick Reference

### Key Types
- `RBFInterpolator` - Main interpolation interface
- `FmmTree` - Fast multipole method evaluator
- `InterpolantSettings` - Configuration for RBF fitting
- `RBFKernelType` - Enum of available kernels

### Key Modules
- `ferreus_rbf::rbf` - Core RBF fitting
- `ferreus_rbf::preconditioning` - Domain decomposition
- `ferreus_rbf::surfacing` - Isosurface extraction
- `ferreus_bbfmm::bbfmm` - FMM implementation

### Important Traits
- `Kernel` - Interface for kernel functions
- `PointCloud` - Interface for point data structures

---

**Last Updated:** 2025-11-23
**For:** Claude Code AI Assistant
**Project Version:** 0.1.x
