/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements PyO3 bindings, configuration wrappers, and NumPy conversion utilities for ferreus_rbf.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef};
use faer_ext::IntoFaer;
use ferreus_rbf::{self, config, interpolant_config};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};
use std::sync::Arc;

/// Convert a NumPy array into a 'faer::MatRef<T>'.
fn numpy_to_matref<'py, T>(
    py: Python<'py>,
    obj: &'py Py<PyAny>,
) -> Result<MatRef<'py, T>, &'static str> 
where
    T: numpy::Element + Copy
{
    if let Ok(array2) = obj.extract::<PyReadonlyArray2<T>>(py) {
        return Ok(array2.into_faer());
    }

    if let Ok(array1) = obj.extract::<PyReadonlyArray1<T>>(py) {
        return Ok(array1.into_faer().as_mat());
    }

    Err("Expected a 1D or 2D NumPy array of the requested dtype")
}

/// Convert a `faer::Mat<T>` to a NumPy array.
pub fn mat_to_numpy<'py, T>(mat: &Mat<T>, py: Python<'py>) -> Bound<'py, PyArray2<T>>
where
    T: numpy::Element + Copy,
{
    let (nrows, ncols) = mat.shape();

    let array = PyArray2::<T>::zeros(py, [nrows, ncols], false);
    let mut slice_mut = unsafe { array.as_array_mut() };

    for i in 0..nrows {
        for j in 0..ncols {
            slice_mut[[i, j]] = *mat.get(i, j);
        }
    }

    array
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Solvers {
    DDM,
    FGMRES,
}

impl From<Solvers> for config::Solvers {
    fn from(s: Solvers) -> config::Solvers {
        match s {
            Solvers::DDM => config::Solvers::DDM,
            Solvers::FGMRES => config::Solvers::FGMRES,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DDMParams {
    inner: config::DDMParams,
}
#[pymethods]
impl DDMParams {
    #[new]
    #[pyo3(signature=(leaf_threshold, overlap_quota, coarse_ratio, coarse_threshold))]
    fn new(
        leaf_threshold: usize,
        overlap_quota: f64,
        coarse_ratio: f64,
        coarse_threshold: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: config::DDMParams {
                leaf_threshold,
                overlap_quota,
                coarse_ratio,
                coarse_threshold,
            },
        })
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FmmCompressionType {
    #[pyo3(name = "None_")]
    None,
    SVD,
    ACA,
}

impl From<FmmCompressionType> for config::FmmCompressionType {
    fn from(s: FmmCompressionType) -> config::FmmCompressionType {
        match s {
            FmmCompressionType::None => config::FmmCompressionType::None,
            FmmCompressionType::SVD => config::FmmCompressionType::SVD,
            FmmCompressionType::ACA => config::FmmCompressionType::ACA,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FmmParams {
    inner: config::FmmParams,
}
#[pymethods]
impl FmmParams {
    #[new]
    #[pyo3(signature=(interpolation_order, max_points_per_cell, compression_type, epsilon, eval_chunk_size))]
    fn new(
        interpolation_order: usize,
        max_points_per_cell: usize,
        compression_type: FmmCompressionType,
        epsilon: f64,
        eval_chunk_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: config::FmmParams {
                interpolation_order,
                max_points_per_cell,
                compression_type: compression_type.into(),
                epsilon,
                eval_chunk_size,
            },
        })
    }
}

#[pyclass]
pub struct ProgressEvent;

#[pyclass]
pub struct SolverIteration {
    #[pyo3(get)]
    pub iter: usize,
    #[pyo3(get)]
    pub residual: f64,
    #[pyo3(get)]
    pub progress: f64,
}

#[pyclass]
pub struct SurfacingProgress {
    #[pyo3(get)]
    pub isovalue: f64,
    #[pyo3(get)]
    pub stage: String,
    #[pyo3(get)]
    pub progress: f64,
}

#[pyclass]
pub struct DuplicatesRemoved {
    #[pyo3(get)]
    pub num_duplicates: usize,
}

#[pyclass]
pub struct Message {
    #[pyo3(get)]
    pub message: String,
}

fn map_msg_to_py(py: Python<'_>, msg: ferreus_rbf::progress::ProgressMsg) -> Py<PyAny> {
    match msg {
        ferreus_rbf::progress::ProgressMsg::SolverIteration { iter, residual, progress } => {
            Py::new(py, SolverIteration { iter, residual, progress })
                .expect("alloc SolverIteration")
                .into()
        }
        ferreus_rbf::progress::ProgressMsg::SurfacingProgress { isovalue, stage, progress } => {
            Py::new(py, SurfacingProgress { isovalue, stage, progress })
                .expect("alloc SurfacingProgress")
                .into()
        }
        ferreus_rbf::progress::ProgressMsg::DuplicatesRemoved { num_duplicates } => {
            Py::new(py, DuplicatesRemoved { num_duplicates })
                .expect("alloc DuplicatesRemoved")
                .into()
        }
        ferreus_rbf::progress::ProgressMsg::Message { message } => {
            Py::new(py, Message { message })
                .expect("alloc Message")
                .into()
        }
    }
}

#[derive(Debug)]
struct PyProgressSink {
    callback: Option<Py<PyAny>>,
}

impl ferreus_rbf::progress::ProgressSink for PyProgressSink {
    fn emit(&self, msg: ferreus_rbf::progress::ProgressMsg) {
        if let Some(cb) = self.callback.as_ref() {
            Python::attach(|py| {
                let obj = map_msg_to_py(py, msg);
                if let Err(e) = cb.call1(py, (obj,)) {
                    e.print(py); // don’t crash on callback exceptions
                }
            });
        }
    }
}


#[pyclass]
pub struct Progress {
    sink: Arc<dyn ferreus_rbf::progress::ProgressSink>,
}

#[pymethods]
impl Progress {
    /// Create a synchronous progress sink.
    #[new]
    #[pyo3(signature=(callback=None))]
    fn new(callback: Option<Py<PyAny>>) -> PyResult<Self> {
        let sink: Arc<dyn ferreus_rbf::progress::ProgressSink> = Arc::new(PyProgressSink { callback });
        Ok(Self { sink })
    }
}

impl Progress {
    pub fn __clone_sink__(&self) -> Arc<dyn ferreus_rbf::progress::ProgressSink> {
        self.sink.clone()
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RBFKernelType {
    Linear,
    ThinPlateSpline,
    Cubic,
    Spheroidal,
}

impl From<RBFKernelType> for interpolant_config::RBFKernelType {
    fn from(v: RBFKernelType) -> Self {
        match v {
            RBFKernelType::Linear => interpolant_config::RBFKernelType::Linear,
            RBFKernelType::ThinPlateSpline => interpolant_config::RBFKernelType::ThinPlateSpline,
            RBFKernelType::Cubic => interpolant_config::RBFKernelType::Cubic,
            RBFKernelType::Spheroidal => interpolant_config::RBFKernelType::Spheroidal,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Drift {
    #[pyo3(name = "None_")]
    None = 0,
    Constant = 1,
    Linear = 2,
    Quadratic = 3,
}

impl From<Drift> for ferreus_rbf::interpolant_config::Drift {
    fn from(d: Drift) -> Self {
        match d {
            Drift::None => ferreus_rbf::interpolant_config::Drift::None,
            Drift::Constant => ferreus_rbf::interpolant_config::Drift::Constant,
            Drift::Linear => ferreus_rbf::interpolant_config::Drift::Linear,
            Drift::Quadratic => ferreus_rbf::interpolant_config::Drift::Quadratic,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpheroidalOrder {
    Three = 3,
    Five = 5,
    Seven = 7,
    Nine = 9,
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct InterpolantSettings {
    inner: ferreus_rbf::interpolant_config::InterpolantSettings
}

#[pymethods]
impl InterpolantSettings {
    #[new]
    #[pyo3(signature=(
        kernel_type,
        *,
        drift = None,
        nugget = None,
        spheroidal_order = None,
        base_range = None,
        total_sill = None,
        fitting_accuracy = None,
    ))]
    fn new(
        kernel_type: RBFKernelType,
        drift: Option<Drift>,
        nugget: Option<f64>,
        spheroidal_order: Option<SpheroidalOrder>,
        base_range: Option<f64>,
        total_sill: Option<f64>,
        fitting_accuracy: Option<FittingAccuracy>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ferreus_rbf::interpolant_config::InterpolantSettings { 
                kernel_type: kernel_type.into(),
                spheroidal_order: match spheroidal_order.is_some() {
                    true => match spheroidal_order.unwrap() {
                        SpheroidalOrder::Three => ferreus_rbf::interpolant_config::SpheroidalOrder::Three,
                        SpheroidalOrder::Five => ferreus_rbf::interpolant_config::SpheroidalOrder::Five,
                        SpheroidalOrder::Seven => ferreus_rbf::interpolant_config::SpheroidalOrder::Seven,
                        SpheroidalOrder::Nine => ferreus_rbf::interpolant_config::SpheroidalOrder::Nine,
                    },
                    false => ferreus_rbf::interpolant_config::SpheroidalOrder::Three,
                },
                drift: {
                    match drift.is_some() {
                        true => drift.unwrap().into(),
                        false => ferreus_rbf::interpolant_config::get_min_drift(kernel_type.into()),
                    }
                }, 
                nugget: nugget.unwrap_or(0.0), 
                base_range: base_range.unwrap_or(1.0), 
                total_sill: total_sill.unwrap_or(1.0), 
                basis_size: 0, 
                polynomial_degree: -1, 
                fitting_accuracy: {
                    match fitting_accuracy.is_some() {
                        true => fitting_accuracy.unwrap().inner,
                        false => FittingAccuracy::default().inner,
                    }
                },
            }
        })
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FittingAccuracyType {
    Absolute,
    Relative,
}

impl From<FittingAccuracyType> for interpolant_config::FittingAccuracyType {
    fn from(f: FittingAccuracyType) -> Self {
        match f {
            FittingAccuracyType::Absolute => interpolant_config::FittingAccuracyType::Absolute,
            FittingAccuracyType::Relative => interpolant_config::FittingAccuracyType::Relative,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct FittingAccuracy {
    inner: interpolant_config::FittingAccuracy,
}

impl Default for FittingAccuracy {
    fn default() -> Self {
        Self {
            inner: interpolant_config::FittingAccuracy::default(),
        }
    }
}

#[pymethods]
impl FittingAccuracy {
    #[new]
    #[pyo3(signature=(tolerance, tolerance_type))]
    fn new(
        tolerance: f64,
        tolerance_type: FittingAccuracyType,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: interpolant_config::FittingAccuracy {
                tolerance,
                tolerance_type: tolerance_type.into(),
            },
        })
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Params {
    inner: config::Params,
}

#[pymethods]
impl Params {
    #[new]
    #[pyo3(signature=(
        kernel_type,
        *,
        solver_type = None,
        ddm_params = None,
        fmm_params = None,
        naive_solve_threshold = None,
        test_unique = None,
    ))]
    fn new(
        kernel_type: RBFKernelType,
        solver_type: Option<Solvers>,
        ddm_params: Option<DDMParams>,
        fmm_params: Option<FmmParams>,
        naive_solve_threshold: Option<usize>,
        test_unique: Option<bool>,
    ) -> Self {
            Self {
                inner: config::Params {
                    solver_type: solver_type.unwrap_or(Solvers::FGMRES).into(),
                    ddm_params: {
                        match ddm_params.is_some() {
                            true => ddm_params.unwrap().inner,
                            false => config::DDMParams::default(),
                        }
                    },
                    fmm_params: {
                        match fmm_params.is_some() {
                            true => fmm_params.unwrap().inner,
                            false => config::FmmParams::new_defaults(kernel_type.into()),
                        }
                    },
                    naive_solve_threshold: naive_solve_threshold.unwrap_or(4096),
                    test_unique: test_unique.unwrap_or(true),
            }
        }
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct GlobalTrend {
    inner: ferreus_rbf::GlobalTrend,
}
#[pymethods]
impl GlobalTrend {
    #[classmethod]
    pub fn one(_cls: &Bound<PyType>, major_ratio: f64) -> PyResult<Self> {
        Ok(Self {
            inner: ferreus_rbf::GlobalTrend::One { major_ratio },
        })
    }
    #[classmethod]
    pub fn two(
        _cls: &Bound<PyType>,
        rotation_angle: f64,
        major_ratio: f64,
        minor_ratio: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ferreus_rbf::GlobalTrend::Two {
                rotation_angle,
                major_ratio,
                minor_ratio,
            },
        })
    }
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn three(
        _cls: &Bound<PyType>,
        dip: f64,
        dip_direction: f64,
        pitch: f64,
        major_ratio: f64,
        semi_major_ratio: f64,
        minor_ratio: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ferreus_rbf::GlobalTrend::Three {
                dip,
                dip_direction,
                pitch,
                major_ratio,
                semi_major_ratio,
                minor_ratio,
            },
        })
    }
}
impl From<GlobalTrend> for ferreus_rbf::GlobalTrend {
    fn from(pygt: GlobalTrend) -> Self {
        pygt.inner
    }
}
impl From<&GlobalTrend> for ferreus_rbf::GlobalTrend {
    fn from(pygt: &GlobalTrend) -> Self {
        pygt.inner.clone()
    }
}

#[pyclass]
pub struct RBFInterpolator {
    inner: ferreus_rbf::RBFInterpolator,
}

#[pymethods]
impl RBFInterpolator {
    #[new]
    #[pyo3(signature = (
        points,
        values,
        interpolant_settings,
        *,
        params = None,
        global_trend = None,
        progress_callback = None,
    ))]
    fn new(
        py: Python<'_>,
        points: PyReadonlyArray2<'_, f64>,
        values: PyReadonlyArray2<'_, f64>,
        interpolant_settings: InterpolantSettings,
        params: Option<Params>,
        global_trend: Option<PyRef<GlobalTrend>>,
        progress_callback: Option<Py<Progress>>,
    ) -> PyResult<Self> {
        let points: Mat<f64> = points.into_faer().to_owned();
        let values: Mat<f64> = values.into_faer().to_owned();

        let rbfib = ferreus_rbf::RBFInterpolator::builder(
            points, 
            values, 
            interpolant_settings.inner
        );

        let rbfib = match params.is_some() {
            true => rbfib.params(params.unwrap().inner),
            false => rbfib,
        };

        let rbfib =  match global_trend.is_some() {
            true => rbfib.global_trend(global_trend.unwrap().inner),
            false => rbfib,
        };

        let rbfib = match progress_callback.is_some() {
            true => {
                let sink_arc = Python::attach(|py| {
                    progress_callback.unwrap().borrow(py).__clone_sink__()
                });
                rbfib.progress_callback(sink_arc)
            },
            false => rbfib,
        };

        let inner = py.detach(|| rbfib.build());

        Ok(Self {inner} )
    }

    /// Evaluates the interpolator at the given target points.
    #[pyo3(signature = (targets))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        targets: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let target_mat = targets.into_faer().to_owned();

        // Run the heavy work without the GIL
        let result: Mat<f64> = py.detach(|| self.inner.evaluate(&target_mat));

        // Convert after we’ve got the GIL again
        mat_to_numpy(&result, py)
    }

    /// Evaluates the interpolator at the source points.
    #[pyo3(signature = (*, add_nugget=false))]
    fn evaluate_at_source<'py>(
        &self,
        py: Python<'py>,
        add_nugget: bool,
    ) -> Bound<'py, PyArray2<f64>> {
        // Run the heavy work without the GIL
        let result: Mat<f64> = py.detach(|| self.inner.evaluate_at_source(add_nugget));

        // Convert after we’ve got the GIL again
        mat_to_numpy(&result, py)
    }

    /// Builds the internal evaluator using an optional list of extents.
    #[pyo3(signature = (extents=None))]
    fn build_evaluator<'py>(&mut self, py: Python<'py>, extents: Option<PyReadonlyArray1<'py, f64>>) {
        let extents_vec = extents.map(|e| e.to_vec().unwrap());
        py.detach(|| {
            self.inner.build_evaluator(extents_vec);
        });
    }

    /// Evaluates the interpolator at the given target points using the pre-built evaluator.
    fn evaluate_targets<'py>(
        &mut self,
        py: Python<'py>,
        targets: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let target_mat = targets.into_faer().to_owned();
        let result: Mat<f64> = py.detach(|| self.inner.evaluate_targets(&target_mat));
        mat_to_numpy(&result, py)
    }

    fn build_isosurfaces<'py>(
        &mut self,
        py: Python<'py>,
        extents: PyReadonlyArray1<'py, f64>,
        resolution: f64,
        isovalues: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let extents_vec = extents.as_slice()?.to_vec();

        // Heavy compute without GIL
        let (all_pts, all_faces) = py.detach(|| {
            self.inner.build_isosurfaces(&extents_vec, &resolution, &isovalues)
        });

        // Convert once we’re back with the GIL
        let vertex_arrays: Vec<_> = all_pts.iter().map(|m| mat_to_numpy(m, py)).collect();
        let face_arrays: Vec<_> = all_faces.iter().map(|m| mat_to_numpy(m, py)).collect();
        let py_vertices = PyList::new(py, &vertex_arrays)?;
        let py_faces = PyList::new(py, &face_arrays)?;
        Ok((py_vertices, py_faces))
    }

    /// Save this interpolator to a JSON file.
    #[pyo3(signature = (path))]
    fn save_model(&self, path: &str) -> PyResult<()> {
        self.inner.save_model(path).map_err(model_error_to_py)?;
        Ok(())
    }

    /// Load a model saved by `save_model`, validating format & version.
    #[staticmethod]
    #[pyo3(signature = (path, progress_callback=None))]
    fn load_model(py: Python<'_>, path: &str, progress_callback: Option<Py<Progress>>) -> PyResult<Self> {
        let sink_arc = match progress_callback.is_some() {
            true => Some(progress_callback.unwrap().borrow(py).__clone_sink__()),
            false => None,
        };
        let inner = ferreus_rbf::RBFInterpolator::load_model(path, sink_arc)
            .map_err(model_error_to_py)?;

        Ok(Self { inner })
    }

    /// Access the stored source points from the interpolator
    fn source_points<'py>(
        &self,
        py: Python<'py>
    ) -> Bound<'py, PyArray2<f64>> {
        let points = &self.inner.points;
        mat_to_numpy(&points, py)
    }

    /// Access the stored source values from the interpolator
    fn source_values<'py>(
        &self,
        py: Python<'py>
    ) -> Bound<'py, PyArray2<f64>> {
        let points = &self.inner.points;
        mat_to_numpy(&points, py)
    }

}

fn model_error_to_py(err: ferreus_rbf::ModelIOError) -> PyErr {
    use ferreus_rbf::ModelIOError::*;
    let io_to_py = |path: &std::path::PathBuf, e: &std::io::Error, action: &str| {
        let msg = format!("{} {}: {}", action, path.display(), e);
        if let Some(code) = e.raw_os_error() {
            PyOSError::new_err((code, msg))
        } else {
            PyOSError::new_err(msg)
        }
    };

    match err {
        Create { path, source } => io_to_py(&path, &source, "creating"),
        Open   { path, source } => io_to_py(&path, &source, "opening"),
        Write  { path, source } => io_to_py(&path, &source, "writing"),
        Flush  { path, source } => io_to_py(&path, &source, "flushing"),

        Serialize { path, source } =>
            PyOSError::new_err(format!("serializing JSON to {}: {}", path.display(), source)),
        Parse { path, source } =>
            PyOSError::new_err(format!("parsing JSON in {}: {}", path.display(), source)),

        FormatMismatch { path, found, expected } =>
            PyOSError::new_err(format!(
                "unsupported format {:?} (expected {:?}) in {}",
                found, expected, path.display()
            )),
        VersionMismatch { path, found, expected } =>
            PyOSError::new_err(format!(
                "unsupported version {} (expected {}) in {}",
                found, expected, path.display()
            )),
    }
}

/// Save an isosurface to an obj
#[pyfunction]
pub fn save_obj(py: Python, path: &str, name: &str, verts: Py<PyAny>, faces: Py<PyAny>,) -> PyResult<()> {
    let verts_mat = numpy_to_matref::<f64>(py, &verts)
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for verts"))?;

    let faces_mat = numpy_to_matref::<usize>(py, &faces)
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected a 1D/2D int array for faces"))?;

    ferreus_rbf::save_obj(path, name, verts_mat, faces_mat)?;

    Ok(())
}

#[pyclass]
pub struct RBFTestFunctions;

#[pymethods]
impl RBFTestFunctions {
    /// Franke's function on R^2.
    #[staticmethod]
    pub fn franke_2d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::franke_2d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    pub fn f1_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f1_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f2_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f2_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f3_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f3_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f4_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f4_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f5_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f5_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f6_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f6_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f7_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f7_3d(&mat_points);

        mat_to_numpy(&values, py)
    }

    #[staticmethod]
    fn f8_3d<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let mat_points = points.into_faer().to_owned();
        let values = ferreus_rbf::RBFTestFunctions::f8_3d(&mat_points);

        mat_to_numpy(&values, py)
    }
}
