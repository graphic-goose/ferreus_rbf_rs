/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements the main RBF interpolator, coefficient management, and solver orchestration logic.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    common,
    config::{Params, Solvers},
    domain::Domain,
    global_trend::{GlobalTrend, GlobalTrendTransform},
    interpolant_config::InterpolantSettings,
    iterative_solvers,
    kdtree::{DistanceMetric, KDTree, PointRowWithId},
    polynomials,
    preconditioning::{domain_decomposition::DDMTree, schwarz},
    progress::{ProgressMsg, ProgressSink},
    surfacing::surface_nets::surface_nets,
};

use faer::{Mat, MatRef, Row, concat, mat::AsMatRef};
use ferreus_bbfmm::FmmError;
use ferreus_rbf_utils::{self, FmmTree, KernelParams};
use roots;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    error::Error,
    f64,
    fmt::{self, Debug},
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

/// Coefficients of a solved RBF system.
///
/// After fitting, an RBF interpolator produces a set of coefficients
/// that define the contribution of each basis function. These
/// coefficients are stored in this struct and used during evaluation.
#[derive(Debug, Serialize, Deserialize)]
pub struct Coefficients {
    /// Coefficients associated with the RBF centers (data points).
    pub point_coefficients: Mat<f64>,

    /// Coefficients associated with the polynomial drift term, if present.
    ///
    /// This is `None` when no polynomial component was included in the
    /// system, or `Some(matrix)` otherwise.
    pub poly_coefficients: Option<Mat<f64>>,
}

impl Coefficients {
    /// Creates a new [`Coefficients`] instance from the given RBF and
    /// optional polynomial coefficients.
    pub(crate) fn new(point_coefficients: Mat<f64>, poly_coefficients: Option<Mat<f64>>) -> Self {
        Self {
            point_coefficients,
            poly_coefficients,
        }
    }
}

/// Internal helper that bundles together the data structures and routines
/// required for iterative solution of the RBF system.
///
/// This struct is not part of the public API. It exists to provide:
/// - Access to the FMM tree for fast matrix-vector products (`matvec`).
/// - Access to the DDM tree for Schwarz preconditioning (`precon`).
/// - Optional storage for monomial and orthonormal polynomial matrices
///   used when a polynomial drift term is included.
/// - A reference to the kernel settings that control evaluation.
///
/// Both `fmm_tree` and `ddm_tree` are wrapped in a `Mutex` since they may
/// be mutated during iterative solves, while `interpolant_settings` is shared
/// via `Arc` for cheap cloning and consistency.
struct IterativeSolver {
    /// Fast multipole method tree used for efficient matrix-vector products.
    pub fmm_tree: Mutex<FmmTree>,

    /// Domain decomposition tree used for Schwarz preconditioning.
    pub ddm_tree: Mutex<DDMTree>,

    /// Optional monomial basis matrix, present if a polynomial drift is included.
    pub monomial_matrix: Option<Mat<f64>>,

    /// Optional orthonormalized polynomial basis, used in preconditioning.
    pub orthonormal_poly: Option<Mat<f64>>,

    /// Kernel configuration shared across the solver.
    pub interpolant_settings: Arc<InterpolantSettings>,
}

impl IterativeSolver {
    /// Performs a full matrix-vector product with the RBF system matrix
    /// using the fast multipole method.
    pub fn matvec(&self, weights: &MatRef<f64>) -> Mat<f64> {
        let mut fmm = self.fmm_tree.lock().unwrap();
        fast_matrix_vector_product(
            &mut *fmm,
            weights,
            &self.interpolant_settings.basis_size,
            None,
            &self.monomial_matrix,
            &self.interpolant_settings.nugget,
        )
    }

    /// Performs a (possibly restricted) matrix-vector product, applying
    /// only to a subset of target indices when provided.
    pub fn matvec_partial(
        &self,
        weights: &MatRef<f64>,
        target_indices: Option<&Vec<usize>>,
    ) -> Mat<f64> {
        let mut fmm = self.fmm_tree.lock().unwrap();
        fast_matrix_vector_product(
            &mut *fmm,
            weights,
            &self.interpolant_settings.basis_size,
            target_indices,
            &self.monomial_matrix,
            &self.interpolant_settings.nugget,
        )
    }

    /// Applies the Schwarz preconditioner to the given residual vector.
    ///
    /// This uses the DDM tree and optional polynomial basis to compute
    /// a preconditioned residual that accelerates convergence of the
    /// iterative solver.
    pub fn precon(&self, residuals: &MatRef<f64>) -> Mat<f64> {
        let matvec = |weights: &MatRef<f64>, target_indices: Option<&Vec<usize>>| {
            self.matvec_partial(weights, target_indices)
        };

        schwarz::schwarz_preconditioner(
            residuals,
            &mut *self.ddm_tree.lock().unwrap(),
            &matvec,
            &self.interpolant_settings,
            &self.orthonormal_poly,
        )
    }
}

/// Convenience builder for constructing an [`RBFInterpolator`].
///
/// This builder provides an ergonomic way to configure and create an
/// interpolator instance from input data, kernel settings, and optional
/// parameters. Supplies sensible defaults and allows incremental configuration.
///
/// The builder should be called via the [`RBFInterpolator::builder`] method.
///
/// See [`RBFInterpolator`] for details on each field.
pub struct RBFInterpolatorBuilder {
    points: Mat<f64>,
    point_values: Mat<f64>,
    interpolant_settings: InterpolantSettings,
    params: Params,
    global_trend: Option<GlobalTrend>,
    progress_callback: Option<Arc<dyn ProgressSink>>,
}

impl RBFInterpolatorBuilder {
    /// Creates a new builder with the required inputs:
    /// - `points`: coordinates of the data points.
    /// - `point_values`: corresponding scalar values at each point.
    /// - `interpolant_settings`: configuration of the RBF kernel.
    ///
    /// Default [`Params`] will be created automatically for the given kernel type.
    fn new(
        points: Mat<f64>,
        point_values: Mat<f64>,
        interpolant_settings: InterpolantSettings,
    ) -> Self {
        Self {
            points: points,
            point_values: point_values,
            interpolant_settings: interpolant_settings,
            params: Params::builder(interpolant_settings.kernel_type).build(),
            global_trend: None,
            progress_callback: None,
        }
    }

    /// Sets custom solver and algorithm parameters.
    pub fn params(mut self, params: Params) -> Self {
        self.params = params;
        self
    }

    /// Attaches a [`GlobalTrend`] transform to the interpolator.
    ///
    /// A global trend applies a geometric transform (e.g. anisotropy/scaling/rotation)
    /// to the input space before solving and evaluating.
    pub fn global_trend(mut self, global_trend: GlobalTrend) -> Self {
        self.global_trend = Some(global_trend);
        self
    }

    /// Optional callback for reporting solver progress.
    ///
    /// Skipped during serialization.
    pub fn progress_callback(mut self, progress_callback: Arc<dyn ProgressSink>) -> Self {
        self.progress_callback = Some(progress_callback);
        self
    }

    /// Builds and returns the configured [`RBFInterpolator`].
    pub fn build(self) -> RBFInterpolator {
        RBFInterpolator::new(
            self.points,
            self.point_values,
            self.interpolant_settings,
            self.global_trend,
            self.params,
            self.progress_callback,
        )
    }
}

#[doc = include_str!("../docs/rbf_interpolator.md")]
#[derive(Serialize, Deserialize, Debug)]
pub struct RBFInterpolator {
    /// Coordinates of the input data points.
    pub points: Mat<f64>,

    /// Scalar values at each input point.
    pub point_values: Mat<f64>,

    /// Solved coefficients for the RBF and polynomial terms.
    pub coefficients: Coefficients,

    /// Kernel settings used to configure the interpolator.
    interpolant_settings: Arc<InterpolantSettings>,

    /// Per-dimension translation factor (used for scaling/normalization for the monomial matrix).
    translation_factor: Vec<f64>,

    /// Per-dimension scaling factor (used for scaling/normalization for the monomial matrix).
    scale_factor: Vec<f64>,

    /// Solver and algorithm parameters.
    pub params: Params,

    /// Optional fast multipole evaluator for efficient queries.
    ///
    /// Skipped during serialization.
    #[serde(skip, default)]
    evaluator: Option<FmmTree>,

    /// Optional global trend transform (anisotropy / rotation).
    global_trend: Option<GlobalTrendTransform>,

    /// Optional callback for reporting solver progress.
    /// Skipped during serialization.
    #[serde(skip, default)]
    pub(crate) progress_callback: Option<Arc<dyn ProgressSink>>,
}

impl RBFInterpolator {
    /// Creates a new [`RBFInterpolatorBuilder`] for the given points,
    /// values, and kernel settings.
    ///
    /// This is the way to construct an interpolator.
    pub fn builder(
        points: Mat<f64>,
        point_values: Mat<f64>,
        interpolant_settings: InterpolantSettings,
    ) -> RBFInterpolatorBuilder {
        RBFInterpolatorBuilder::new(points, point_values, interpolant_settings)
    }

    fn new(
        points: Mat<f64>,
        point_values: Mat<f64>,
        interpolant_settings: InterpolantSettings,
        global_trend: Option<GlobalTrend>,
        params: Params,
        progress_callback: Option<Arc<dyn ProgressSink>>,
    ) -> Self {
        let solver_start = Instant::now();

        let dimensions = points.ncols();

        assert!(
            (1..=3).contains(&dimensions),
            "Unsupported number of dimensions: {}",
            dimensions
        );

        let interpolant_settings = Arc::new({
            let mut ks = interpolant_settings;
            ks.set_basis_size(&dimensions);
            ks
        });

        let (mut unique_points, unique_point_values) = if params.test_unique {
            let idx = remove_duplicates(&points, &interpolant_settings);

            if idx.len() == points.nrows() {
                (points, point_values)
            } else {
                if let Some(sink) = &progress_callback {
                    sink.emit(ProgressMsg::DuplicatesRemoved {
                        num_duplicates: points.nrows() - idx.len(),
                    });
                }
                (
                    ferreus_rbf_utils::select_mat_rows(&points, &idx),
                    ferreus_rbf_utils::select_mat_rows(&point_values, &idx),
                )
            }
        } else {
            (points, point_values)
        };

        let global_trend_transform = match global_trend.is_some() {
            true => {
                let center = get_center(&unique_points);
                let global_trend_transform =
                    GlobalTrendTransform::new(center, global_trend.unwrap());
                unique_points = global_trend_transform.transform_points(&unique_points);
                Some(global_trend_transform)
            }
            false => None,
        };

        let mut interpolator = Self {
            points: unique_points,
            point_values: unique_point_values,
            coefficients: Coefficients {
                point_coefficients: Mat::<f64>::new(),
                poly_coefficients: None,
            },
            interpolant_settings,
            translation_factor: Vec::default(),
            scale_factor: Vec::default(),
            params: params,
            evaluator: None,
            global_trend: global_trend_transform,
            progress_callback: progress_callback,
        };

        interpolator.setup_and_solve();

        let solver_duration = solver_start.elapsed();

        if let Some(sink) = &interpolator.progress_callback {
            let msg = format!(
                "Took {:?} to solve RBF for {} points using the following settings:\n\
                Kernel: {:?}, Polynomial degree: {}\n\
                Fitting accuracy: {:?}, Tolerance type: {:?}",
                solver_duration,
                interpolator.points.nrows(),
                interpolator.interpolant_settings.kernel_type,
                interpolator.interpolant_settings.polynomial_degree,
                interpolator.interpolant_settings.fitting_accuracy.tolerance,
                interpolator
                    .interpolant_settings
                    .fitting_accuracy
                    .tolerance_type,
            );

            sink.emit(ProgressMsg::Message { message: msg });
        }

        interpolator
    }

    fn setup_and_solve(&mut self) {
        let num_points = self.points.nrows();
        let num_val_cols = self.point_values.ncols();

        if self.interpolant_settings.basis_size != 0 {
            (self.translation_factor, self.scale_factor) =
                common::get_cheb_cube_scaling_factors(&self.points);
        }

        if num_points < self.params.naive_solve_threshold {
            let naive_point_indices: Vec<usize> = (0..num_points as usize).into_iter().collect();
            let mut naive_domain = Domain::new(naive_point_indices);

            naive_domain.internal_points_mask =
                vec![true; naive_domain.overlapping_point_indices.len()];

            naive_domain.factorise(
                &self.points,
                self.interpolant_settings.clone(),
                true,
                &self.global_trend,
            );

            let domain_coefficients = naive_domain.solve(&self.point_values.as_ref());

            let mut global_point_coefficients = Mat::<f64>::zeros(num_points, num_val_cols);

            naive_domain
                .overlapping_point_indices
                .iter()
                .enumerate()
                .for_each(|(idx, global)| {
                    global_point_coefficients
                        .row_mut(*global)
                        .copy_from(domain_coefficients.point_coefficients.row(idx));
                });

            self.coefficients = Coefficients::new(
                global_point_coefficients,
                domain_coefficients.poly_coefficients,
            );
        } else {
            let adaptive_tree = true;
            let sparse_tree = true;

            let fmm_tree = FmmTree::new(
                self.points.clone(),
                self.params.fmm_params.interpolation_order.clone(),
                (*self.interpolant_settings).into(),
                adaptive_tree,
                sparse_tree,
                None,
                Some(self.params.fmm_params.into()),
            );

            let fmm_tree = Mutex::new(fmm_tree);

            let mut monomial_matrix: Option<Mat<f64>> = None;
            let mut orthonormal_poly: Option<Mat<f64>> = None;

            let mut rhs = self.point_values.clone();

            if self.interpolant_settings.basis_size != 0 {
                let monomial_points = match self.global_trend.is_some() {
                    true => {
                        let gt = self.global_trend.as_ref().unwrap();
                        gt.inverse_transform_points(&self.points)
                    }
                    false => self.points.clone(),
                };

                monomial_matrix = Some(polynomials::evaluate_monomials(
                    &monomial_points,
                    &self.interpolant_settings.polynomial_degree,
                    &self.interpolant_settings.basis_size,
                    &self.translation_factor,
                    &self.scale_factor,
                ));

                let qr = monomial_matrix.as_ref().unwrap().qr();

                orthonormal_poly = Some(qr.compute_thin_Q());

                rhs = concat![
                    [rhs],
                    [Mat::<f64>::zeros(
                        self.interpolant_settings.basis_size,
                        num_val_cols
                    )]
                ];
            }

            let ddm_tree = DDMTree::new(
                &self.points,
                &self.interpolant_settings,
                self.params.ddm_params,
                &self.global_trend,
            );

            let ddm_tree = Mutex::new(ddm_tree);

            let iterative_solver = IterativeSolver {
                fmm_tree: fmm_tree,
                ddm_tree: ddm_tree,
                monomial_matrix: monomial_matrix,
                orthonormal_poly: orthonormal_poly,
                interpolant_settings: self.interpolant_settings.clone(),
            };

            let matvec = |x: &MatRef<f64>| iterative_solver.matvec(x);
            let precon = |r: &MatRef<f64>| iterative_solver.precon(r);

            // n = number of RBF points, m = basis_size (may be 0)
            let n = num_points;
            let m = self.interpolant_settings.basis_size;

            let mut point_coefficients = Mat::<f64>::zeros(n, num_val_cols);
            let mut poly_coefficients = match m != 0 {
                true => Some(Mat::<f64>::zeros(m, num_val_cols)),
                false => None,
            };

            for col in 0..num_val_cols {
                let all_coefficients = match self.params.solver_type {
                    Solvers::FGMRES => iterative_solvers::fgmres(
                        &matvec,
                        rhs.submatrix(0, col, rhs.nrows(), 1),
                        Some(&precon),
                        None,
                        20,
                        5,
                        &self.interpolant_settings.fitting_accuracy,
                        self.progress_callback.clone(),
                    ),
                    Solvers::DDM => iterative_solvers::schwarz_ddm_solver(
                        &matvec,
                        rhs.submatrix(0, col, rhs.nrows(), 1),
                        Some(&precon),
                        100,
                        &self.interpolant_settings.fitting_accuracy,
                        self.progress_callback.clone(),
                    ),
                };

                if m != 0 {
                    // all_coefficients is (n + m) x 1 -> split into (n x 1, m x 1)
                    let (rbf_part, poly_part) = all_coefficients.split_at_row(n);

                    // Fill the destination columns (sizes match: n x 1 and m x 1)
                    point_coefficients.col_mut(col).copy_from(rbf_part.col(0));

                    if let Some(poly) = poly_coefficients.as_mut() {
                        poly.col_mut(col).copy_from(poly_part.col(0));
                    }
                } else {
                    // No polynomial basis: all_coefficients should be (n x 1)
                    point_coefficients
                        .col_mut(col)
                        .copy_from(all_coefficients.col(0));
                }
            }

            self.coefficients = Coefficients::new(point_coefficients, poly_coefficients);
        }

        if let Some(gt) = &self.global_trend {
            self.points = gt.inverse_transform_points(&self.points);
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    /// Internal: build and configure an FMM evaluator for this interpolator.
    ///
    /// - Applies `global_trend` to `points` and (if provided) to the `extents` via
    ///   corner transformation before building the tree.
    /// - If `extents` is `None`, derives them from the (possibly transformed) points.
    /// - `adaptive`: enable adaptive evaluation passes in the backend.
    /// - `sparse`: enable sparse/leaf-only evaluation strategies (used when evaluating
    ///   at the source points).
    fn _setup_fmmtree(&self, adaptive: bool, sparse: bool, extents: Option<Vec<f64>>) -> FmmTree {
        let mut points = self.points.clone();

        let mut evaluator_extents = extents.clone();

        if let Some(gt) = &self.global_trend {
            points = gt.transform_points(&points);

            if evaluator_extents.is_some() {
                let dimensions = self.points.ncols();
                let evaluator_extents_ref = evaluator_extents.unwrap();
                let mins = &evaluator_extents_ref[..dimensions];
                let maxs = &evaluator_extents_ref[dimensions..];

                let corner_mat = bounding_box_corners(mins, maxs);
                let transformed = gt.transform_points(&corner_mat);
                evaluator_extents = Some(ferreus_rbf_utils::get_pointarray_extents(&transformed));
            }
        }

        if evaluator_extents.is_none() {
            evaluator_extents = Some(ferreus_rbf_utils::get_pointarray_extents(&points));
        }

        let tree = FmmTree::new(
            points,
            self.params.fmm_params.interpolation_order.clone(),
            (*self.interpolant_settings).into(),
            adaptive,
            sparse,
            evaluator_extents,
            Some(self.params.fmm_params.into()),
        );

        tree
    }

    #[doc(hidden)]
    // #[inline(always)]
    /// Internal: run one evaluation against `tree` and return interpolated values.
    ///
    /// - Adds nugget on the diagonal when `add_nugget = true`.
    /// - Adds polynomial (monomial) contribution if a polynomial basis is enabled.
    fn _evaluate(
        &self,
        tree: &mut FmmTree,
        target_points: &Mat<f64>,
        add_nugget: bool,
    ) -> Mat<f64> {
        if let Err(err) = tree.evaluate(
            &self.coefficients.point_coefficients.as_mat_ref(),
            &target_points,
        ) {
            match err {
                FmmError::PointOutsideTree { point_index } => {
                    panic!(
                        "FMM evaluation failed during RBF evaluation: \
                         target point at row {} lies outside the FMM tree extents.",
                        point_index
                    );
                }
            }
        }

        let mut interpolated_values = tree.target_values().clone();

        if add_nugget {
            interpolated_values
                .row_iter_mut()
                .zip(self.coefficients.point_coefficients.row_iter())
                .for_each(|(mut a, b)| a += &b * self.interpolant_settings.nugget);
        }

        if self.interpolant_settings.basis_size != 0 {
            let monomials_mat = polynomials::evaluate_monomials(
                target_points,
                &self.interpolant_settings.polynomial_degree,
                &self.interpolant_settings.basis_size,
                &self.translation_factor,
                &self.scale_factor,
            );

            interpolated_values +=
                monomials_mat * self.coefficients.poly_coefficients.as_ref().unwrap();
        }

        interpolated_values
    }

    /// Evaluate the interpolant at `target_points` using a **one-shot** FMM evaluator.
    ///
    /// This is the most convenient way to evaluate a single batch: it builds a
    /// temporary FMM tree, evaluates, and discards the evaluator. If a
    /// `global_trend` is present, the target points are transformed for evaluation.
    ///
    /// Extents are computed as the **union** of the source and target point
    /// bounding boxes to ensure all targets can be assigned to tree boxes.
    ///
    /// ### Returns
    /// A `(n_targets × n_value_channels)` matrix of interpolated values.
    ///
    /// ### Accuracy & performance
    /// For repeated evaluations (e.g. meshing, isosurfacing), prefer
    /// [`RBFInterpolator::build_evaluator`] + [`RBFInterpolator::evaluate_targets`] to amortize setup cost.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::RBFInterpolator;
    /// # use faer::Mat;
    /// # let (rbfi, targets): (RBFInterpolator, Mat<f64>) = unimplemented!();
    /// let values = rbfi.evaluate(&targets);
    /// ```
    pub fn evaluate(&self, target_points: &Mat<f64>) -> Mat<f64> {
        let adaptive = true;
        let sparse = false;

        let source_extents = ferreus_rbf_utils::get_pointarray_extents(&self.points);
        let target_extents = ferreus_rbf_utils::get_pointarray_extents(&target_points);

        let combined_extents = union_extents(&source_extents, &target_extents);

        let mut tree = self._setup_fmmtree(adaptive, sparse, Some(combined_extents));

        tree.set_weights(&self.coefficients.point_coefficients.as_mat_ref());

        let eval_points = match self.global_trend.is_some() {
            true => {
                let gt = self.global_trend.as_ref().unwrap();
                &gt.transform_points(target_points)
            }
            false => target_points,
        };

        let interpolated_values = self._evaluate(&mut tree, eval_points, false);

        interpolated_values
    }

    /// Evaluate the interpolant **at the original source points**.
    ///
    /// Useful for **convergence checks** and diagnostics.
    ///
    /// - When `add_nugget = true`, the diagonal “nugget” term is added back so the
    ///   evaluated values should match the input samples to within the solver’s
    ///   tolerance (undoing any smoothing from the nugget).  
    /// - When `add_nugget = false`, you observe the smoothed/regularised fit.
    ///
    /// ### Returns
    /// A `(n_sources × n_value_channels)` matrix of values at the training sites.
    ///
    /// ### Notes
    /// This path uses a sparse/leaf-only evaluation strategy optimized for
    /// source-point queries.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::RBFInterpolator;
    /// # let rbfi: RBFInterpolator = unimplemented!();
    /// // Check training residual with nugget restored
    /// let fitted = rbfi.evaluate_at_source(true);
    /// ```
    pub fn evaluate_at_source(&self, add_nugget: bool) -> Mat<f64> {
        let adaptive = true;
        let sparse = true;

        let mut tree = self._setup_fmmtree(adaptive, sparse, None);

        tree.set_weights(&self.coefficients.point_coefficients.as_mat_ref());

        let tree_points = match self.global_trend.is_some() {
            true => {
                let gt = self.global_trend.as_ref().unwrap();
                &gt.transform_points(&self.points)
            }
            false => &self.points,
        };

        let interpolated_values = self._evaluate(&mut tree, tree_points, add_nugget);

        interpolated_values
    }

    /// Build and store an FMM evaluator for **repeated evaluations**.
    ///
    /// Use this when you’ll call [`RBFInterpolator::evaluate_targets`] many times.
    /// The evaluator is constructed once and saved inside the interpolator.
    ///
    /// ### Extents
    /// - If `extents` is `Some`, they define the evaluator domain
    ///   `[min_0.., max_0..]` and **must cover all future target points**.
    /// - If `extents` is `None`, extents are derived from the (transformed, if
    ///   applicable) source points.
    ///
    /// ### Panics
    /// - Evaluating targets **outside** the stored extents will make the backend
    ///   unable to assign them to tree boxes and will **panic**. Use a sufficiently
    ///   generous domain when building the evaluator.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::{RBFInterpolator};
    /// # let mut rbfi: RBFInterpolator = unimplemented!();
    /// // Build from source-point extents
    /// rbfi.build_evaluator(None);
    /// ```
    pub fn build_evaluator(&mut self, extents: Option<Vec<f64>>) {
        let adaptive = true;
        let sparse = false;

        let mut tree = self._setup_fmmtree(adaptive, sparse, extents);

        tree.set_weights(&self.coefficients.point_coefficients.as_mat_ref());

        tree.set_local_coefficients(&self.coefficients.point_coefficients.as_mat_ref());

        self.evaluator = Some(tree);
    }

    /// Evaluate using the **stored** evaluator built by [`RBFInterpolator::build_evaluator`].
    ///
    /// This is the fast path for repeated calls. If a `global_trend` is present,
    /// target points are transformed consistently with the stored evaluator.
    ///
    /// ### Panics
    /// - If called before [`RBFInterpolator::build_evaluator`].
    /// - If any `target_points` lie **outside** the extents used to build the
    ///   evaluator.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::RBFInterpolator;
    /// # use faer::Mat;
    /// # let (mut rbfi, targets): (RBFInterpolator, Mat<f64>) = unimplemented!();
    /// rbfi.build_evaluator(None);
    /// let values = rbfi.evaluate_targets(&targets);
    /// ```
    pub fn evaluate_targets(&mut self, target_points: &Mat<f64>) -> Mat<f64> {
        let tree = self.evaluator.as_mut().unwrap();

        let mut eval_points = target_points.clone();

        if let Some(gt) = &self.global_trend {
            eval_points = gt.transform_points(&target_points);
        }

        if let Err(err) = tree.evaluate_leaves(
            &self.coefficients.point_coefficients.as_mat_ref(),
            &eval_points,
        ) {
            match err {
                FmmError::PointOutsideTree { point_index } => {
                    panic!(
                        "FMM evaluation failed in evaluate_targets: \
                         target point at row {} lies outside the evaluator extents. \
                         Ensure the extents passed to build_evaluator(..) cover all \
                         target points.",
                        point_index
                    );
                }
            }
        }

        let mut interpolated_values = tree.target_values().cloned();

        if self.interpolant_settings.basis_size != 0 {
            let monomials_mat = polynomials::evaluate_monomials(
                &target_points,
                &self.interpolant_settings.polynomial_degree,
                &self.interpolant_settings.basis_size,
                &self.translation_factor,
                &self.scale_factor,
            );

            interpolated_values +=
                monomials_mat * self.coefficients.poly_coefficients.as_ref().unwrap();
        }

        interpolated_values
    }

    /// Build isosurfaces using a **surface-following, non-adaptive Dual Contouring** pipeline.
    ///
    /// The sampling `resolution` controls grid density; choose it relative to the
    /// data scale and desired detail. Multiple `isovalues` may be provided; each
    /// produces a separate surface.
    ///
    /// Seed cells are selected from samples within `resolution` of an isovalue.
    /// If no seeds are found for a given isovalue, the corresponding entry is
    /// empty.
    ///
    /// ### Parameters
    /// - `extents`: evaluation domain `[minx, miny, minz, maxx, maxy, maxz]`.
    /// - `resolution`: grid step in world units.
    /// - `isovalues`: list of scalar levels to extract.
    ///
    /// ### Returns
    /// `(points_per_iso, faces_per_iso)` where:
    /// - `points_per_iso[i]` is a `(V_i × 3)` matrix of vertex positions for the
    ///   `i`-th isosurface.
    /// - `faces_per_iso[i]` is an `(F_i × 3)` integer matrix of triangle vertex indices.
    ///
    /// ### Notes
    /// - Only implemented in 3D.
    /// - The current isosurface extraction method does **not** guarantee 
    ///   manifold or valid meshes; surfaces may contain trifurcations or
    ///   self-intersections and may not be suitable for downstream boolean
    ///   operations.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::{RBFInterpolator};
    /// # let mut rbfi: RBFInterpolator = unimplemented!();
    /// let extents = vec![0.0, 0.0, 0.0, 100.0, 100.0, 100.0]; // [mins..., maxs...]
    /// let resolution = 2.0;
    /// let isos = vec![0.0, 1.0];
    /// let (all_pts, all_faces) = rbfi.build_isosurfaces(&extents, &resolution, &isos);
    /// ```
    pub fn build_isosurfaces(
        &mut self,
        extents: &Vec<f64>,
        resolution: &f64,
        isovalues: &Vec<f64>,
    ) -> (Vec<Mat<f64>>, Vec<Mat<usize>>) {
        let (all_isosurface_points, all_isosurface_faces) =
            surface_nets::surface_nets(extents, *resolution, isovalues, self);

        (all_isosurface_points, all_isosurface_faces)
    }

    /// Save this interpolator to a **JSON envelope** `{ format, version, model }`.
    ///
    /// The on-disk format is versioned via `JSON_FORMAT_NAME` and `JSON_VERSION`.
    /// Files produced here are intended to be read back with [`RBFInterpolator::load_model`].
    ///
    /// ### Errors
    /// - Returns `ModelIOError::{Create, Serialize, Flush}` on I/O or serialization
    ///   failures.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::RBFInterpolator;
    /// # let rbfi: RBFInterpolator = unimplemented!();
    /// rbfi.save_model("rbf_model.json")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> ModelIOResult<()> {
        let path_ref = path.as_ref();
        let file = File::create(path_ref).map_err(|e| ModelIOError::Create {
            path: path_ref.to_path_buf(),
            source: e,
        })?;
        let mut w = BufWriter::new(file);

        let env = JsonEnvelopeRef {
            format: JSON_FORMAT_NAME,
            version: JSON_VERSION,
            model: self,
        };

        serde_json::to_writer_pretty(&mut w, &env).map_err(|e| ModelIOError::Serialize {
            path: path_ref.to_path_buf(),
            source: e,
        })?;
        w.flush().map_err(|e| ModelIOError::Flush {
            path: path_ref.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Load an interpolator from a versioned **JSON envelope**, validating format & version.
    ///
    /// If `progress` is `Some`, installs the sink into `self.params.progress_callback`
    /// on the returned model so subsequent long-running operations (solve, surface
    /// extraction, etc.) can report progress.
    ///
    /// ### Validation
    /// - Fails if `format != JSON_FORMAT_NAME` or `version != JSON_VERSION`.
    ///
    /// ### Errors
    /// - Returns `ModelIOError::{Open, Parse, FormatMismatch, VersionMismatch}` as appropriate.
    ///
    /// ### Example
    /// ```no_run
    /// # use ferreus_rbf::{RBFInterpolator, progress::{closure_sink, ProgressMsg}};
    /// let (sink, _listener) = closure_sink(256, |msg: ProgressMsg| { /* handle */ });
    /// let rbfi = RBFInterpolator::load_model("rbf_model.json", Some(sink))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_model<P: AsRef<Path>>(
        path: P,
        progress: Option<Arc<dyn ProgressSink>>,
    ) -> ModelIOResult<Self> {
        let path_ref = path.as_ref();

        let file = File::open(path_ref).map_err(|e| ModelIOError::Open {
            path: path_ref.to_path_buf(),
            source: e,
        })?;
        let reader = BufReader::new(file);

        let env: JsonEnvelopeOwned<Self> =
            serde_json::from_reader(reader).map_err(|e| ModelIOError::Parse {
                path: path_ref.to_path_buf(),
                source: e,
            })?;

        // Validate envelope
        if env.format != JSON_FORMAT_NAME {
            return Err(ModelIOError::FormatMismatch {
                path: path_ref.to_path_buf(),
                found: env.format,
                expected: JSON_FORMAT_NAME,
            });
        }

        if env.version != JSON_VERSION {
            return Err(ModelIOError::VersionMismatch {
                path: path_ref.to_path_buf(),
                found: env.version,
                expected: JSON_VERSION,
            });
        }

        let mut model = env.model;
        if let Some(sink) = progress {
            model.progress_callback = Some(sink);
        }
        Ok(model)
    }
}

fn get_center(points: &Mat<f64>) -> Row<f64> {
    points
        .col_iter()
        .map(|col| col.sum() / points.nrows() as f64)
        .collect()
}

fn bounding_box_corners(mins: &[f64], maxs: &[f64]) -> Mat<f64> {
    let dims = mins.len();
    let n = 1 << dims;
    Mat::from_fn(
        n,
        dims,
        |i, j| {
            if (i >> j) & 1 == 0 { mins[j] } else { maxs[j] }
        },
    )
}

/// Merge two extent vectors `[min0, …, min{D-1}, max0, …, max{D-1}]` into their union.
#[inline]
fn union_extents(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "extent vectors must have same length");
    assert!(
        a.len() % 2 == 0,
        "extent vector length must be even (mins then maxs)"
    );

    let d = a.len() / 2;
    let (a_min, a_max) = a.split_at(d);
    let (b_min, b_max) = b.split_at(d);

    let mins: Vec<f64> = a_min.iter().zip(b_min).map(|(x, y)| x.min(*y)).collect();
    let maxs: Vec<f64> = a_max.iter().zip(b_max).map(|(x, y)| x.max(*y)).collect();

    // same layout as input: mins first, then maxs
    mins.into_iter().chain(maxs.into_iter()).collect()
}

pub(crate) fn fast_matrix_vector_product(
    fmm_tree: &mut FmmTree,
    weights: &MatRef<f64>,
    basis_size: &usize,
    target_indices: Option<&Vec<usize>>,
    polynomial_matrix: &Option<Mat<f64>>,
    nugget: &f64,
) -> Mat<f64> {
    let mut result: Mat<f64> = Mat::zeros(weights.nrows(), 1);

    let weights_len = weights.nrows() - *basis_size;

    let evaluation_indices: Vec<usize>;
    if target_indices.is_none() {
        evaluation_indices = (0..weights_len).collect();
    } else {
        evaluation_indices = target_indices.unwrap().clone();
    }

    fmm_tree.set_weights(&weights);

    let target_points =
        ferreus_rbf_utils::select_mat_rows(&fmm_tree.source_points(), &evaluation_indices);

    if let Err(err) = fmm_tree.evaluate(&weights, &target_points) {
        match err {
            FmmError::PointOutsideTree { point_index } => {
                panic!(
                    "Internal FMM evaluation failed in fast_matrix_vector_product: \
                     source/target point at row {} lies outside the FMM tree extents.",
                    point_index
                );
            }
        }
    }

    evaluation_indices
        .iter()
        .enumerate()
        .for_each(|(fmm_idx, result_idx)| {
            result[(*result_idx, 0)] = *fmm_tree.target_values().get(fmm_idx, 0);
            result[(*result_idx, 0)] += weights.get(*result_idx, 0) * nugget;
            if polynomial_matrix.is_some() {
                result[(*result_idx, 0)] += &polynomial_matrix.as_ref().unwrap().row(*result_idx)
                    * &weights.subrows(weights_len, *basis_size).col(0);
            }
        });

    result
}

/// Estimate a duplicate cutoff distance for this kernel to keep the
/// augmented QTAQ RBF system strictly positive definite.
///
/// Some kernels have near-zero behaviour that can introduce numerical
/// noise and break SPD if the cutoff tolerance is too small. This function
/// probes the kernel response near r = 0 and scales the cutoff so that
/// |φ(r) - φ(0)| rises above machine epsilon relative to φ(h_ref).
///
/// Returns: cutoff distance in [0, h_ref], suitable as a minimum spacing
/// when removing duplicate/near-duplicate points.
fn duplicate_cutoff_distance(h_ref: f64, interpolant_settings: &InterpolantSettings) -> f64 {
    let kparams: KernelParams = interpolant_settings.clone().into();

    let phi = |r: f64| ferreus_rbf_utils::kernel_phi(r, &kparams);

    let f = 1.0;
    let eps = f64::EPSILON;
    let mut rtol = 1E-12;

    let phi0 = phi(0.0);
    let phih = phi(h_ref);
    let target = f * eps * (phih - phi0).abs();

    let resid = |r| (phi(r) - phi0).abs() - target;

    // If the reference point already meets the target, just return h_ref
    if resid(h_ref) <= 0.0 {
        return h_ref;
    };

    match roots::find_root_inverse_quadratic(0.0, h_ref, resid, &mut rtol) {
        Ok(r) => r,
        _ => h_ref,
    }
}

/// Remove duplicate or near-duplicate points to ensure the QTAQ RBF systems
/// remain strictly positive definite.
///
/// This function computes a kernel-dependent cutoff tolerance (via
/// [`duplicate_cutoff_distance`]) based on the spatial extent of the data.
/// Points closer than this tolerance are considered indistinguishable by
/// the kernel and can cause rank-deficiency, stalled convergence, or solver
/// breakdown.  
///
/// A KD-tree with infinity-norm distance is used to group points within the
/// cutoff radius; only the first point in each group is kept.
///
/// Returns: indices of unique points to keep.
fn remove_duplicates(points: &Mat<f64>, interpolant_settings: &InterpolantSettings) -> Vec<usize> {
    let dims = points.ncols();
    let extents = ferreus_rbf_utils::get_pointarray_extents(&points);
    let mins = &extents[..dims];
    let maxs = &extents[dims..];
    let max_length = maxs
        .iter()
        .zip(mins.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(f64::NEG_INFINITY, f64::max);

    let scaled_tolerance = duplicate_cutoff_distance(max_length, &interpolant_settings);

    let kdtree = KDTree::new(&points);

    let mut visited = HashSet::new();
    let mut unique_points = Vec::new();

    for (i, point) in points.row_iter().enumerate() {
        if visited.contains(&(i as i32)) {
            continue;
        }

        let target = PointRowWithId::new(&point, &-1);
        let neighbours =
            kdtree.radius_search(&target, scaled_tolerance, DistanceMetric::InfinityNorm);

        if !visited.contains(&(i as i32)) {
            unique_points.push(i);
            visited.extend(neighbours);
        }
    }

    unique_points
}

const JSON_FORMAT_NAME: &str = "ferreus_rbf.json";
const JSON_VERSION: u32 = 1;

/// Borrowing envelope for SAVE (no clone of the model).
#[derive(Serialize)]
struct JsonEnvelopeRef<'a, T: ?Sized> {
    format: &'static str,
    version: u32,
    #[serde(flatten)]
    model: &'a T,
}

/// Owning envelope for LOAD (generic over the concrete model).
#[derive(Serialize, Deserialize)]
struct JsonEnvelopeOwned<T> {
    format: String,
    version: u32,
    #[serde(flatten)]
    model: T,
}

type ModelIOResult<T> = std::result::Result<T, ModelIOError>;

/// Errors that can occur when saving or loading an [`RBFInterpolator`] model.
///
/// This is the error type returned by [`RBFInterpolator::save_model`] and
/// [`RBFInterpolator::load_model`], wrapping lower-level I/O and JSON
/// serialization issues as well as format/version validation failures.
#[derive(Debug)]
pub enum ModelIOError {
    /// Failed to create the target file before writing a model.
    Create {
        path: PathBuf,
        source: io::Error,
    },
    /// Failed to open an existing model file for reading.
    Open {
        path: PathBuf,
        source: io::Error,
    },
    /// Low-level write error while streaming the model to disk.
    Write {
        path: PathBuf,
        source: io::Error,
    },
    /// Failed to flush buffered output when finishing a write.
    Flush {
        path: PathBuf,
        source: io::Error,
    },
    /// Error serializing the in-memory model to JSON.
    Serialize {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// Error parsing JSON when reading a model from disk.
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    /// The JSON `format` field does not match the expected model format.
    FormatMismatch {
        path: PathBuf,
        found: String,
        expected: &'static str,
    },
    /// The JSON `version` field does not match the supported version.
    VersionMismatch {
        path: PathBuf,
        found: u32,
        expected: u32,
    },
}

impl fmt::Display for ModelIOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelIOError::Create { path, source } => {
                write!(f, "creating {}: {}", path.display(), source)
            }
            ModelIOError::Open { path, source } => {
                write!(f, "opening {}: {}", path.display(), source)
            }
            ModelIOError::Write { path, source } => {
                write!(f, "writing {}: {}", path.display(), source)
            }
            ModelIOError::Flush { path, source } => {
                write!(f, "flushing {}: {}", path.display(), source)
            }
            ModelIOError::Serialize { path, source } => {
                write!(f, "serializing JSON to {}: {}", path.display(), source)
            }
            ModelIOError::Parse { path, source } => {
                write!(f, "parsing JSON in {}: {}", path.display(), source)
            }
            ModelIOError::FormatMismatch {
                path,
                found,
                expected,
            } => write!(
                f,
                "unsupported format {:?} (expected {:?}) in {}",
                found,
                expected,
                path.display()
            ),
            ModelIOError::VersionMismatch {
                path,
                found,
                expected,
            } => write!(
                f,
                "unsupported version {} (expected {}) in {}",
                found,
                expected,
                path.display()
            ),
        }
    }
}

impl Error for ModelIOError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ModelIOError::Create { source, .. }
            | ModelIOError::Open { source, .. }
            | ModelIOError::Write { source, .. }
            | ModelIOError::Flush { source, .. } => Some(source),
            ModelIOError::Serialize { source, .. } | ModelIOError::Parse { source, .. } => {
                Some(source)
            }
            ModelIOError::FormatMismatch { .. } | ModelIOError::VersionMismatch { .. } => None,
        }
    }
}
