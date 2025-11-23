# FastRBF Implementation & Maturity Investigation Plan

**Project:** ferreus_rbf_rs
**Purpose:** Assess library maturity and verify bona fide FastRBF implementation
**Goal:** Determine viability as open-source alternative to Leapfrog geomodelling
**Date:** 2025-11-23

---

## Executive Summary

This investigation plan provides a systematic approach to evaluating whether `ferreus_rbf_rs` is:
1. A **true implementation** of FastRBF algorithms as described in academic literature
2. **Mature enough** for production use in geological modelling
3. A **viable open-source alternative** to commercial software like Leapfrog

The investigation is organized into 5 phases, from algorithmic verification to real-world validation.

---

## Phase 1: Algorithmic Correctness & FastRBF Verification

### Objective
Verify that the implementation faithfully follows FastRBF literature and achieves claimed complexity.

### 1.1 Core FastRBF Components Verification

#### 1.1.1 Domain Decomposition Implementation
- [ ] **Read and analyze:** `ferreus_rbf/src/preconditioning/domain_decomposition.rs`
- [ ] **Verify against Beatson et al. (2000) paper**
  - [ ] Check if overlapping domain decomposition matches paper description
  - [ ] Verify overlap ratio parameter implementation
  - [ ] Confirm local problem formulation
  - [ ] Check boundary condition handling
- [ ] **Compare with Haase et al. (2017) multilevel approach**
  - [ ] Identify if multilevel hierarchy is implemented
  - [ ] Check coarse-to-fine refinement strategy
- [ ] **Test Questions:**
  - How are domains constructed? (spatial partitioning method)
  - How is overlap handled? (overlap ratio, point assignment)
  - Are local solutions combined correctly? (Schwarz methods)

#### 1.1.2 Fast Multipole Method (FMM) Implementation
- [ ] **Read and analyze:** `ferreus_bbfmm/src/bbfmm.rs`
- [ ] **Verify BBFMM characteristics**
  - [ ] Check Chebyshev interpolation scheme (`chebyshev.rs`)
  - [ ] Verify adaptive cross approximation (`aca.rs`)
  - [ ] Confirm tree construction (adaptive vs fixed depth)
  - [ ] Check M2L (multipole-to-local) operator compression
- [ ] **Validate against classical FMM requirements**
  - [ ] P2M (particle-to-multipole) operations
  - [ ] M2M (multipole-to-multipole) translations
  - [ ] M2L (multipole-to-local) translations
  - [ ] L2L (local-to-local) translations
  - [ ] L2P (local-to-particle) operations
  - [ ] P2P (particle-to-particle) near-field direct evaluation
- [ ] **Test Questions:**
  - Is the tree construction optimal for RBF kernels?
  - How is the multipole approximation accuracy controlled?
  - Does it handle non-uniform point distributions well?

#### 1.1.3 Iterative Solver Verification
- [ ] **Read and analyze:** `ferreus_rbf/src/iterative_solvers.rs`
- [ ] **Verify FGMRES implementation**
  - [ ] Check Arnoldi process correctness
  - [ ] Verify flexible preconditioning interface
  - [ ] Confirm residual calculation and convergence criteria
  - [ ] Check restart mechanism (if present)
- [ ] **Test Questions:**
  - Why FGMRES over other Krylov methods? (documented rationale)
  - Is the preconditioner applied correctly at each iteration?
  - Are there convergence safeguards for ill-conditioned problems?

### 1.2 Complexity Analysis

#### 1.2.1 Theoretical Complexity Verification
- [ ] **Analyze computational complexity:**
  - [ ] FMM tree construction: should be O(N log N)
  - [ ] FMM matvec: should be O(N) or O(N log N)
  - [ ] Domain decomposition setup: should be O(N log N)
  - [ ] Iterative solver: O(k * N log N) for k iterations
  - [ ] Overall: should be O(N log N) if k is bounded
- [ ] **Identify complexity bottlenecks:**
  - [ ] Which operations dominate for small N? (< 1000)
  - [ ] Which operations dominate for large N? (> 100,000)
  - [ ] Is there a crossover point vs direct solve?

#### 1.2.2 Empirical Complexity Verification
- [ ] **Design scaling experiments:**
  - [ ] Create test datasets: N = 1K, 10K, 100K, 1M points
  - [ ] Use uniform random distributions (best case)
  - [ ] Use clustered distributions (worst case)
  - [ ] Measure time and memory for each N
- [ ] **Expected results if O(N log N):**
  - [ ] Time should roughly scale as N log N
  - [ ] Memory should scale as N (not N²)
  - [ ] Plot log-log graph: slope should be ~1.0-1.2
- [ ] **Implementation script:**
  ```rust
  // Benchmark various N values
  for N in [1000, 10000, 100000, 1000000] {
      let points = generate_random_points(N, 3, seed);
      let values = test_function(&points);

      let start = Instant::now();
      let rbf = RBFInterpolator::builder(points, values, settings).build();
      let elapsed = start.elapsed();

      // Log: N, time, memory_used
  }
  ```
- [ ] **Verification tasks:**
  - [ ] Run benchmark on representative hardware
  - [ ] Compare measured vs theoretical complexity
  - [ ] Check if N=1M is actually feasible (memory permitting)

### 1.3 Numerical Correctness

#### 1.3.1 Interpolation Accuracy Tests
- [ ] **Test interpolation exactness:**
  - [ ] Verify interpolation passes through source points
  - [ ] Test with different kernels (linear, cubic, gaussian)
  - [ ] Check with different fitting accuracy tolerances
- [ ] **Test with known analytical functions:**
  - [ ] Franke's function (2D benchmark)
  - [ ] Polynomial functions (should be exact with drift terms)
  - [ ] Radial functions (well-suited for RBF)
- [ ] **Analyze existing examples:**
  - [ ] `examples/franke_2d.rs` - Does it achieve expected error?
  - [ ] Check all examples in `ferreus_rbf/examples/`
  - [ ] Run examples and verify output quality

#### 1.3.2 Conditioning and Stability
- [ ] **Test ill-conditioned scenarios:**
  - [ ] Very small point separations
  - [ ] Very large domain extents
  - [ ] Highly non-uniform distributions
  - [ ] Collinear or coplanar point arrangements
- [ ] **Evaluate stability mechanisms:**
  - [ ] How does domain decomposition improve conditioning?
  - [ ] Are there regularization parameters?
  - [ ] What happens when convergence fails?

---

## Phase 2: Code Quality & Maturity Assessment

### Objective
Evaluate whether the codebase is production-ready and maintainable.

### 2.1 Code Structure & Organization

- [ ] **Architecture review:**
  - [ ] Is the separation of concerns clear? (RBF / FMM / utils)
  - [ ] Are abstractions appropriate? (trait design)
  - [ ] Is the public API well-designed and ergonomic?
  - [ ] Review `lib.rs` for each crate
- [ ] **Dependency analysis:**
  - [ ] Are dependencies minimal and well-maintained?
  - [ ] Check `Cargo.toml` for each crate
  - [ ] Any concerning dependencies? (unmaintained, security issues)
  - [ ] Why `faer` instead of `ndarray` or `nalgebra`? (justified?)

### 2.2 Test Coverage & Quality

#### 2.2.1 Test Discovery
- [ ] **Find all tests:**
  ```bash
  # Unit tests
  find . -name "*.rs" -exec grep -l "#\[cfg(test)\]" {} \;

  # Integration tests
  find . -type d -name "tests"

  # Doc tests
  cargo test --doc

  # Count test assertions
  rg "assert" --stats
  ```
- [ ] **Categorize tests:**
  - [ ] Unit tests (individual functions)
  - [ ] Integration tests (component interactions)
  - [ ] Examples (serve as tests)
  - [ ] Regression tests (known bug fixes)

#### 2.2.2 Test Quality Assessment
- [ ] **Test coverage analysis:**
  - [ ] Install and run `cargo-tarpaulin` or `cargo-llvm-cov`
  - [ ] Identify untested code paths
  - [ ] Prioritize critical paths (solver, FMM, preconditioning)
- [ ] **Test scenarios:**
  - [ ] Are edge cases tested? (empty inputs, single point, etc.)
  - [ ] Are error conditions tested? (invalid inputs, non-convergence)
  - [ ] Are numerical corner cases tested? (ill-conditioning, etc.)
- [ ] **Test data quality:**
  - [ ] Are test datasets realistic?
  - [ ] Are expected results documented and justified?
  - [ ] Are tolerances appropriate?

#### 2.2.3 Run All Tests
- [ ] **Execute test suites:**
  ```bash
  # All workspace tests
  cargo test --workspace --release

  # With output
  cargo test --workspace --release -- --show-output

  # Specific crate tests
  cargo test -p ferreus_rbf --release
  cargo test -p ferreus_bbfmm --release
  ```
- [ ] **Document results:**
  - [ ] Total test count
  - [ ] Pass/fail status
  - [ ] Any ignored tests? Why?
  - [ ] Performance of test execution

### 2.3 Documentation Quality

- [ ] **Rust documentation:**
  - [ ] Build docs: `cargo doc --no-deps --open --workspace`
  - [ ] Check completeness of public API docs
  - [ ] Verify math equations render correctly (KaTeX)
  - [ ] Check example code in doc comments runs
- [ ] **Python documentation:**
  - [ ] Review docs in `py_ferreus_rbf/docs/`
  - [ ] Check API reference completeness
  - [ ] Verify examples are runnable
- [ ] **High-level documentation:**
  - [ ] README quality and accuracy
  - [ ] Is the algorithm explained clearly?
  - [ ] Are limitations documented honestly?
  - [ ] Installation instructions accurate?

### 2.4 Error Handling & Robustness

- [ ] **Error handling patterns:**
  - [ ] Are `Result` types used appropriately?
  - [ ] Are panics avoided in library code?
  - [ ] Are error messages helpful?
  - [ ] Is there a consistent error type design?
- [ ] **Input validation:**
  - [ ] Are inputs validated before processing?
  - [ ] What happens with invalid dimensions?
  - [ ] What happens with NaN or Inf values?
  - [ ] What happens with empty or single-point datasets?
- [ ] **Graceful degradation:**
  - [ ] What happens if FMM accuracy is insufficient?
  - [ ] What happens if solver doesn't converge?
  - [ ] Are there reasonable defaults?

### 2.5 Performance & Resource Management

- [ ] **Memory management:**
  - [ ] Are large allocations avoided where possible?
  - [ ] Is memory reused effectively?
  - [ ] Are there memory leaks? (use `valgrind` or similar)
  - [ ] Can users control memory vs accuracy trade-offs?
- [ ] **Parallelism:**
  - [ ] Where is `rayon` used for parallelization?
  - [ ] Is thread count controllable?
  - [ ] Are parallel operations efficient? (low overhead)
  - [ ] Any thread safety issues?
- [ ] **Profiling:**
  - [ ] Profile a large problem: `cargo flamegraph --example isosurface_linear`
  - [ ] Identify hotspots
  - [ ] Are hotspots in expected places? (FMM, linear algebra)

---

## Phase 3: FastRBF Literature Comparison

### Objective
Compare implementation against published FastRBF algorithms and variants.

### 3.1 Primary Literature Review

#### 3.1.1 Beatson et al. (2000) - Domain Decomposition
- [ ] **Obtain and read paper:**
  - [ ] "Fast solution of the radial basis function interpolation equations: domain decomposition methods"
  - [ ] SIAM J. Sci. Comput., 22(5):1717–1740
- [ ] **Compare against implementation:**
  - [ ] Algorithm 1: Does the implementation match?
  - [ ] Overlap selection: How is overlap ratio chosen?
  - [ ] Local solver: What method is used for local problems?
  - [ ] Results section: Can we reproduce their benchmarks?

#### 3.1.2 Haase et al. (2017) - Multilevel Preconditioner
- [ ] **Obtain and read paper:**
  - [ ] "A Domain Decomposition Multilevel Preconditioner for Interpolation with Radial Basis Functions"
  - [ ] LSSC 2017 proceedings
- [ ] **Compare against implementation:**
  - [ ] Is multilevel hierarchy implemented?
  - [ ] If not, is single-level sufficient? Why?
  - [ ] What are the trade-offs?

#### 3.1.3 Cherrie (2000) - Fast RBF Evaluation
- [ ] **Obtain and read thesis:**
  - [ ] "Fast Evaluation of Radial Basis Functions: Theory and Application"
  - [ ] University of Canterbury PhD thesis
- [ ] **Compare FMM approach:**
  - [ ] Cherrie used classical FMM - this uses BBFMM
  - [ ] What are the differences?
  - [ ] Is BBFMM more general/flexible?

### 3.2 Alternative FastRBF Methods

- [ ] **Survey other FastRBF approaches:**
  - [ ] H-matrices (hierarchical matrices)
  - [ ] Compactly supported kernels (sparse matrices)
  - [ ] Partition of unity methods
  - [ ] Multilevel methods
- [ ] **Compare pros/cons:**
  - [ ] Why was BBFMM + domain decomposition chosen?
  - [ ] What are the trade-offs?
  - [ ] Are there scenarios where other methods excel?

### 3.3 Contemporary Research

- [ ] **Search for recent papers (2015-2025):**
  - [ ] Google Scholar: "fast RBF interpolation"
  - [ ] Google Scholar: "domain decomposition RBF"
  - [ ] Google Scholar: "FMM radial basis function"
- [ ] **Identify improvements:**
  - [ ] Have there been algorithmic advances since Beatson (2000)?
  - [ ] Could any be incorporated?
  - [ ] Is this implementation state-of-the-art?

---

## Phase 4: Geomodelling Capability Assessment

### Objective
Evaluate whether the library can serve as a Leapfrog alternative for implicit geological modelling.

### 4.1 Core Geomodelling Requirements

#### 4.1.1 Implicit Modelling Features
- [ ] **Supported features checklist:**
  - [ ] ✓ Scattered 3D point constraints
  - [ ] ✓ Scalar value interpolation (grade, property)
  - [ ] ✓ Isosurface extraction (geological boundaries)
  - [ ] ✓ Global trend removal (regional gradients)
  - [ ] ? Orientation constraints (structural dip/strike)
  - [ ] ? Fault modeling (discontinuities)
  - [ ] ? Stratigraphic constraints (relative age)
  - [ ] ? Multiple domain modeling
  - [ ] ? Anisotropic interpolation (directional)

#### 4.1.2 Geological Constraint Support
- [ ] **Investigate constraint handling:**
  - [ ] How would you add strike/dip constraints?
  - [ ] Can the RBF formulation handle inequality constraints?
  - [ ] Can multiple datasets be used simultaneously?
- [ ] **Code review:**
  - [ ] Check for constraint-related code
  - [ ] Review `global_trend.rs` for drift functionality
  - [ ] Look for extensibility points in API

#### 4.1.3 Workflow Integration
- [ ] **Data import/export:**
  - [ ] What file formats are supported?
  - [ ] Check `surfacing_io.rs` for mesh I/O
  - [ ] Can it read common geology file formats? (CSV, XYZ, etc.)
- [ ] **Interoperability:**
  - [ ] Python bindings quality for integration
  - [ ] Can output meshes be used in GIS/CAD software?
  - [ ] OBJ export capability (check examples)

### 4.2 Leapfrog Comparison

#### 4.2.1 Leapfrog Geo Capabilities (Public Knowledge)
- [ ] **Document Leapfrog's RBF approach:**
  - [ ] Review Leapfrog whitepapers (if available)
  - [ ] Academic papers by Cowan et al. (ARANZ founders)
  - [ ] User documentation on interpolation methods
- [ ] **Key Leapfrog features:**
  - [ ] Radial basis function used (likely cubic or thin-plate)
  - [ ] Structural geology incorporation
  - [ ] Fault handling and discontinuities
  - [ ] Geological trend modeling
  - [ ] Vein and dyke modeling
  - [ ] Grade estimation and uncertainty

#### 4.2.2 Feature Gap Analysis
- [ ] **Create comparison matrix:**
  ```
  Feature                    | ferreus_rbf | Leapfrog | Gap
  ---------------------------|-------------|----------|-----
  3D RBF interpolation       | ✓           | ✓        | None
  Isosurface extraction      | ✓           | ✓        | Quality?
  Orientation constraints    | ?           | ✓        | Investigate
  Fault modeling             | ✗           | ✓        | Major gap
  Multiple lithologies       | ?           | ✓        | Investigate
  Anisotropy                 | ✗           | ✓        | Gap
  Grade estimation           | ✓           | ✓        | None
  Uncertainty quantification | ✗           | ✓        | Gap
  ```
- [ ] **Prioritize gaps:**
  - [ ] Which gaps are show-stoppers?
  - [ ] Which are nice-to-have?
  - [ ] Which are feasible to implement?

### 4.3 Real-world Geological Test Cases

#### 4.3.1 Synthetic Geological Scenarios
- [ ] **Design test cases:**
  - [ ] Simple folded surface (synform/antiform)
  - [ ] Faulted horizon (discontinuous surface)
  - [ ] Multiple parallel horizons (stratigraphy)
  - [ ] Ore body with grade distribution
  - [ ] Complex 3D geometry (fold + fault)
- [ ] **Implement and test:**
  - [ ] Create synthetic data for each scenario
  - [ ] Run interpolation
  - [ ] Extract isosurfaces
  - [ ] Visually inspect results
  - [ ] Compare to expected geology

#### 4.3.2 Public Geological Datasets
- [ ] **Find open-source geological data:**
  - [ ] USGS datasets
  - [ ] Geological surveys (state/national)
  - [ ] Academic datasets from papers
  - [ ] Mining company open data (rare but exists)
- [ ] **Apply library to real data:**
  - [ ] Import data into library format
  - [ ] Run full workflow
  - [ ] Extract meaningful geological surfaces
  - [ ] Document successes and failures

#### 4.3.3 User Perspective Evaluation
- [ ] **Imagine being a geologist:**
  - [ ] Is the workflow intuitive?
  - [ ] Are parameter choices reasonable?
  - [ ] Are results geologically plausible?
  - [ ] What would make it more usable?
- [ ] **Consider production use:**
  - [ ] Can it handle typical mine-scale data? (100K-1M points)
  - [ ] Is performance acceptable for interactive work?
  - [ ] Are results reproducible?
  - [ ] Is there enough control over outputs?

---

## Phase 5: Ecosystem & Sustainability

### Objective
Assess the long-term viability and sustainability of the project.

### 5.1 Development Activity

- [ ] **GitHub repository analysis:**
  - [ ] Commit frequency and recency
  - [ ] Number of contributors
  - [ ] Issue tracker activity (open/closed ratio)
  - [ ] Pull request patterns
  - [ ] Release cadence
- [ ] **Community engagement:**
  - [ ] GitHub stars, forks, watchers
  - [ ] External references (Google Scholar citations)
  - [ ] Stack Overflow questions/mentions
  - [ ] Reddit, HN, or other forum discussions

### 5.2 Organizational Backing

- [ ] **Maptek relationship:**
  - [ ] Level of corporate support (ongoing or one-time release?)
  - [ ] Are Maptek employees active maintainers?
  - [ ] Is there a roadmap?
  - [ ] Will it receive continued funding/development?
- [ ] **Succession planning:**
  - [ ] Is the project dependent on one person?
  - [ ] Is there a contributor guide?
  - [ ] How welcoming is the project to contributions?

### 5.3 Competitive Landscape

- [ ] **Alternative open-source libraries:**
  - [ ] Survey other RBF libraries (Python, C++, Julia)
  - [ ] Survey other implicit modelling tools
  - [ ] Compare features, performance, maturity
- [ ] **Commercial alternatives:**
  - [ ] Leapfrog Geo (industry standard)
  - [ ] Geomodeller
  - [ ] SKUA-GOCAD
  - [ ] Others in geological modeling space
- [ ] **Positioning:**
  - [ ] What unique value does ferreus_rbf provide?
  - [ ] What niche could it fill?
  - [ ] Is "open-source Leapfrog" realistic or overpromising?

### 5.4 Adoption Barriers

- [ ] **Technical barriers:**
  - [ ] Rust knowledge required (limits audience)
  - [ ] Python bindings adequate for geologists?
  - [ ] Installation complexity
  - [ ] Platform support (Windows, Linux, Mac)
- [ ] **Domain barriers:**
  - [ ] Geological expertise needed for setup
  - [ ] Lack of GUI (vs Leapfrog's interface)
  - [ ] Documentation aimed at geologists?
  - [ ] Training materials available?
- [ ] **Practical barriers:**
  - [ ] Integration with existing workflows
  - [ ] File format compatibility
  - [ ] Licensing concerns (MIT is permissive, good)
  - [ ] Support and consulting availability

---

## Deliverables

### Investigation Report Structure

```markdown
# ferreus_rbf_rs: FastRBF Implementation Assessment

## Executive Summary
- One-page summary of findings
- Key strengths and weaknesses
- Final recommendation

## Part 1: FastRBF Verification
- Algorithm correctness analysis
- Complexity verification (theoretical & empirical)
- Comparison to literature
- Verdict: Is it true FastRBF?

## Part 2: Maturity Assessment
- Code quality evaluation
- Test coverage analysis
- Documentation review
- Performance profiling results
- Maturity score (with rubric)

## Part 3: Geomodelling Viability
- Feature comparison with Leapfrog
- Gap analysis
- Real-world test results
- Verdict: Can it replace Leapfrog?

## Part 4: Recommendations
- Short-term improvements (polish for 1.0)
- Medium-term features (geological capabilities)
- Long-term research directions
- Adoption strategy

## Appendices
- Detailed benchmark results
- Test execution logs
- Literature references
- Code examples
```

### Artifacts to Produce

1. **Benchmark suite:**
   - Scripts to reproduce complexity analysis
   - Performance comparison data
   - Plots showing O(N log N) scaling

2. **Test results:**
   - Test coverage report
   - List of failing/missing tests
   - Proposed new tests

3. **Comparison matrix:**
   - ferreus_rbf vs Leapfrog feature comparison
   - ferreus_rbf vs other open-source RBF libraries

4. **Example geological models:**
   - Synthetic test cases with results
   - Real-world data applications (if available)

5. **Code quality checklist:**
   - Static analysis results (clippy)
   - Dependency audit
   - Security considerations

---

## Success Criteria

### FastRBF Verification (Must Pass)
✓ **Algorithmic:** Domain decomposition + FMM correctly implemented
✓ **Complexity:** Empirically demonstrates O(N log N) scaling
✓ **Accuracy:** Produces correct interpolations within tolerance
✓ **Literature:** Matches published FastRBF algorithms

### Maturity Assessment (Scoring)
- **Code Quality:** 7/10 minimum (clean, maintainable code)
- **Test Coverage:** 60% minimum (critical paths well-tested)
- **Documentation:** 8/10 minimum (complete, accurate docs)
- **Robustness:** Handles edge cases and errors gracefully

### Geomodelling Viability (Qualitative)
✓ **Core capability:** Can model simple geological surfaces
✓ **Performance:** Handles mine-scale datasets (100K-1M points)
⚠ **Advanced features:** May lack some Leapfrog features (acceptable for v0.1)
✓ **Extensibility:** Architecture allows for future enhancements

### Overall Verdict
To be considered a **bona fide FastRBF implementation** and **viable Leapfrog alternative:**
- MUST pass all FastRBF verification criteria
- MUST score ≥ 7/10 on maturity assessment
- MUST demonstrate core geomodelling capability
- SHOULD have clear path to close feature gaps

---

## Timeline Estimate

- **Phase 1** (Algorithm verification): 2-3 days
- **Phase 2** (Code quality): 2-3 days
- **Phase 3** (Literature): 1-2 days (depends on paper availability)
- **Phase 4** (Geomodelling): 3-5 days
- **Phase 5** (Ecosystem): 1 day
- **Report writing**: 2 days

**Total:** 11-16 days of focused investigation

---

## Getting Started

### Immediate Next Steps

1. **Setup investigation environment:**
   ```bash
   # Clone repo (if not already done)
   git clone https://github.com/graphic-goose/ferreus_rbf_rs
   cd ferreus_rbf_rs

   # Build all crates
   cargo build --workspace --release

   # Run existing tests
   cargo test --workspace --release

   # Try examples
   cargo run --example franke_2d --release
   cargo run --example isosurface_linear --release
   ```

2. **Install analysis tools:**
   ```bash
   # Coverage
   cargo install cargo-tarpaulin
   # or
   cargo install cargo-llvm-cov

   # Profiling
   cargo install flamegraph

   # Audit
   cargo install cargo-audit
   ```

3. **Create investigation workspace:**
   ```bash
   mkdir investigation/
   cd investigation/

   # Subdirectories
   mkdir benchmarks/
   mkdir test_cases/
   mkdir results/
   mkdir papers/
   ```

4. **Begin Phase 1:**
   - Start with code reading: `ferreus_rbf/src/rbf.rs`
   - Follow the call chain through solver, preconditioner, FMM
   - Document understanding in notes

---

**Investigation Lead:** [Your Name]
**Date Started:** 2025-11-23
**Expected Completion:** TBD
**Status:** READY TO BEGIN
