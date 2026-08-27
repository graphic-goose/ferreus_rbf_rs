/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements PyO3 bindings and NumPy conversion helpers for the regularised marching tetrahedra
// Python API.
//
// Created on: 08 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef};
use faer_ext::IntoFaer;
use ferreus_rmt;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::{io, sync::Arc};

/// Convert a NumPy array into a 'faer::MatRef<T>'.
fn numpy_to_matref<'py, T>(
    py: Python<'py>,
    obj: &'py Py<PyAny>,
) -> Result<MatRef<'py, T>, &'static str>
where
    T: numpy::Element + Copy,
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

/// Convert a `faer::MatRef<T>` to a NumPy array.
pub fn matref_to_numpy<'py, T>(mat: MatRef<'_, T>, py: Python<'py>) -> Bound<'py, PyArray2<T>>
where
    T: numpy::Element + Copy,
{
    let (nrows, ncols) = mat.shape();

    let array = PyArray2::<T>::zeros(py, [nrows, ncols], false);
    let mut slice_mut = unsafe { array.as_array_mut() };

    for i in 0..nrows {
        for j in 0..ncols {
            slice_mut[[i, j]] = mat[(i, j)];
        }
    }

    array
}

fn numpy_to_scalar_column<'py>(py: Python<'py>, obj: &'py Py<PyAny>, name: &str) -> Mat<f64> {
    let result = numpy_to_matref::<f64>(py, obj)
        .unwrap_or_else(|_| panic!("{name} must return a 1D or 2D float64 numpy array"))
        .to_owned();

    if result.ncols() != 1 {
        panic!(
            "{name} must return shape (N,) or (N, 1); got ({}, {})",
            result.nrows(),
            result.ncols()
        );
    }

    result
}

fn io_error_to_py(err: io::Error) -> PyErr {
    let msg = err.to_string();
    if let Some(code) = err.raw_os_error() {
        PyOSError::new_err((code, msg))
    } else {
        PyOSError::new_err(msg)
    }
}

fn validate_mat_shape<T>(mat: MatRef<'_, T>, name: &str, ncols: usize) -> PyResult<()>
where
    T: Copy,
{
    if mat.ncols() != ncols {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {ncols})"
        )));
    }
    Ok(())
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterMethod {
    #[pyo3(name = "None_")]
    None,
    Average,
    CurvatureWeighted,
}

impl From<ClusterMethod> for ferreus_rmt::ClusterMethod {
    fn from(cluster_method: ClusterMethod) -> Self {
        match cluster_method {
            ClusterMethod::None => ferreus_rmt::ClusterMethod::None,
            ClusterMethod::Average => ferreus_rmt::ClusterMethod::Average,
            ClusterMethod::CurvatureWeighted => ferreus_rmt::ClusterMethod::CurvatureWeighted,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryClosure {
    #[pyo3(name = "None_")]
    None,
    ClosePositive,
    CloseNegative,
}

impl From<BoundaryClosure> for ferreus_rmt::BoundaryClosure {
    fn from(boundary_closure: BoundaryClosure) -> Self {
        match boundary_closure {
            BoundaryClosure::None => ferreus_rmt::BoundaryClosure::None,
            BoundaryClosure::ClosePositive => ferreus_rmt::BoundaryClosure::ClosePositive,
            BoundaryClosure::CloseNegative => ferreus_rmt::BoundaryClosure::CloseNegative,
        }
    }
}

#[pyclass]
pub struct ProgressEvent;

#[pyclass]
pub struct IsosurfaceProgress {
    #[pyo3(get)]
    pub isovalue: f64,
    #[pyo3(get)]
    pub stage: String,
    #[pyo3(get)]
    pub progress: f64,
}

#[pyclass]
pub struct Message {
    #[pyo3(get)]
    pub message: String,
}

fn map_msg_to_py(py: Python<'_>, msg: ferreus_rmt::progress::ProgressMsg) -> Py<PyAny> {
    match msg {
        ferreus_rmt::progress::ProgressMsg::IsosurfaceProgress {
            isovalue,
            stage,
            progress,
        } => Py::new(
            py,
            IsosurfaceProgress {
                isovalue,
                stage: stage.to_string(),
                progress,
            },
        )
        .expect("alloc IsosurfaceProgress")
        .into(),
        ferreus_rmt::progress::ProgressMsg::Message { message } => Py::new(py, Message { message })
            .expect("alloc Message")
            .into(),
    }
}

#[derive(Debug)]
struct PyProgressSink {
    callback: Option<Py<PyAny>>,
}

impl ferreus_rmt::progress::ProgressSink for PyProgressSink {
    fn emit(&self, msg: ferreus_rmt::progress::ProgressMsg) {
        if let Some(cb) = self.callback.as_ref() {
            Python::attach(|py| {
                let obj = map_msg_to_py(py, msg);
                if let Err(err) = cb.call1(py, (obj,)) {
                    err.print(py);
                }
            });
        }
    }
}

#[pyclass]
pub struct Progress {
    sink: Arc<dyn ferreus_rmt::progress::ProgressSink>,
}

#[pymethods]
impl Progress {
    #[new]
    #[pyo3(signature=(callback=None))]
    fn new(callback: Option<Py<PyAny>>) -> PyResult<Self> {
        let sink: Arc<dyn ferreus_rmt::progress::ProgressSink> =
            Arc::new(PyProgressSink { callback });
        Ok(Self { sink })
    }
}

impl Progress {
    pub fn __clone_sink__(&self) -> Arc<dyn ferreus_rmt::progress::ProgressSink> {
        self.sink.clone()
    }
}

#[pyclass]
pub struct Mesh {
    inner: ferreus_rmt::Mesh,
}

#[pymethods]
impl Mesh {
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat_to_numpy(&self.inner.vertices, py)
    }

    #[getter]
    fn facets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<usize>> {
        mat_to_numpy(&self.inner.facets, py)
    }

    #[pyo3(signature=(path, name))]
    fn save_obj(&self, path: &str, name: &str) -> PyResult<()> {
        self.inner.save_obj(path, name).map_err(io_error_to_py)?;
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature=(
    seed_points,
    extents,
    resolution,
    isovalue,
    surface_fn,
    *,
    gradient_fn=None,
    cluster_method=ClusterMethod::CurvatureWeighted,
    boundary_closure=BoundaryClosure::None,
    progress_callback=None,
))]
pub fn build_isosurface(
    py: Python<'_>,
    seed_points: PyReadonlyArray2<'_, f64>,
    extents: PyReadonlyArray1<'_, f64>,
    resolution: f64,
    isovalue: f64,
    surface_fn: Py<PyAny>,
    gradient_fn: Option<Py<PyAny>>,
    cluster_method: ClusterMethod,
    boundary_closure: BoundaryClosure,
    progress_callback: Option<Py<Progress>>,
) -> PyResult<Mesh> {
    let seed_points_mat = seed_points.into_faer();
    validate_mat_shape(seed_points_mat, "seed_points", 3)?;
    let extents_slice = extents.as_slice()?;
    if extents_slice.len() != 6 {
        return Err(PyValueError::new_err("extents must have shape (6,)"));
    }
    let progress_sink = progress_callback.map(|p| p.borrow(py).__clone_sink__());

    let mut py_surface_fn = move |targets: MatRef<f64>| -> Mat<f64> {
        Python::attach(|py| {
            let targets_np = matref_to_numpy(targets, py);

            let result_obj: Py<PyAny> = match surface_fn.call1(py, (targets_np,)) {
                Ok(obj) => obj.into(),
                Err(err) => {
                    err.print(py);
                    panic!("surface_fn callback raised an exception")
                }
            };

            let result = numpy_to_scalar_column(py, &result_obj, "surface_fn");

            if result.nrows() != targets.nrows() {
                panic!(
                    "surface_fn must return shape (N,) or (N, 1); got {} rows for {} targets",
                    result.nrows(),
                    targets.nrows()
                );
            }

            result
        })
    };

    let mut py_gradient_fn = gradient_fn.map(|gradient_fn| {
        move |targets: MatRef<f64>| -> (Mat<f64>, Mat<f64>) {
            Python::attach(|py| {
                let targets_np = matref_to_numpy(targets, py);

                let result_obj: Py<PyAny> = match gradient_fn.call1(py, (targets_np,)) {
                    Ok(obj) => obj.into(),
                    Err(err) => {
                        err.print(py);
                        panic!("gradient_fn callback raised an exception")
                    }
                };

                let (values_obj, gradients_obj): (Py<PyAny>, Py<PyAny>) = result_obj
                    .extract(py)
                    .expect("gradient_fn must return a tuple of (values, gradients)");

                let values = numpy_to_scalar_column(py, &values_obj, "gradient_fn values");
                let gradients = numpy_to_matref::<f64>(py, &gradients_obj)
                    .expect("gradient_fn gradients must be a 2D float64 numpy array")
                    .to_owned();

                if values.nrows() != targets.nrows() {
                    panic!(
                        "gradient_fn values must return shape (N,) or (N, 1); got {} rows for {} targets",
                        values.nrows(),
                        targets.nrows()
                    );
                }
                if gradients.nrows() != targets.nrows() || gradients.ncols() != 3 {
                    panic!(
                        "gradient_fn gradients must return shape (N, 3); got ({}, {})",
                        gradients.nrows(),
                        gradients.ncols()
                    );
                }

                (values, gradients)
            })
        }
    });

    let gradient_fn_ref = py_gradient_fn
        .as_mut()
        .map(|f| f as &mut dyn FnMut(MatRef<f64>) -> (Mat<f64>, Mat<f64>));

    let inner = ferreus_rmt::build_isosurface(
        seed_points_mat,
        extents_slice,
        resolution,
        isovalue,
        &mut py_surface_fn,
        gradient_fn_ref,
        cluster_method.into(),
        boundary_closure.into(),
        progress_sink.as_deref(),
    );

    Ok(Mesh { inner })
}

#[pyfunction]
#[pyo3(signature = (
    seed_points,
    extents,
    resolution,
    isovalues,
    isosurface_fn,
    *,
    gradient_fn=None,
    cluster_method=ClusterMethod::CurvatureWeighted,
    boundary_closure=BoundaryClosure::None,
    progress_callback=None
))]
pub fn build_isosurfaces<'py>(
    py: Python<'py>,
    seed_points: PyReadonlyArray2<'_, f64>,
    extents: PyReadonlyArray1<'py, f64>,
    resolution: f64,
    isovalues: Vec<f64>,
    isosurface_fn: Py<PyAny>,
    gradient_fn: Option<Py<PyAny>>,
    cluster_method: Option<ClusterMethod>,
    boundary_closure: Option<BoundaryClosure>,
    progress_callback: Option<Py<Progress>>,
) -> PyResult<Bound<'py, PyList>> {
    let extents_slice = extents.as_slice()?;
    if extents_slice.len() != 6 {
        return Err(PyValueError::new_err(
            "extents must be length 6: [minx, miny, minz, maxx, maxy, maxz]",
        ));
    }

    let seed_points_mat = seed_points.into_faer();

    if seed_points_mat.ncols() != 3 {
        return Err(PyValueError::new_err("seed_points must have shape (N, 3)"));
    }

    let progress_sink = progress_callback.map(|p| p.borrow(py).__clone_sink__());

    let mut py_surface_fn = move |targets: MatRef<f64>| -> Mat<f64> {
        Python::attach(|py| {
            let targets_owned = targets.to_owned();
            let targets_np = mat_to_numpy(&targets_owned, py);

            let result_obj: Py<PyAny> = match isosurface_fn.call1(py, (targets_np,)) {
                Ok(obj) => obj.into(),
                Err(err) => {
                    err.print(py);
                    panic!("surface_fn callback raised an exception")
                }
            };

            let result = numpy_to_matref::<f64>(py, &result_obj)
                .expect("surface_fn must return a 1D or 2D float64 numpy array")
                .to_owned();

            if result.nrows() != targets.nrows() || result.ncols() != 1 {
                panic!(
                    "surface_fn must return shape (N,) or (N, 1); got ({}, {})",
                    result.nrows(),
                    result.ncols()
                );
            }

            result
        })
    };

    let mut py_gradient_fn = gradient_fn.map(|gradient_fn| {
        move |targets: MatRef<f64>| -> (Mat<f64>, Mat<f64>) {
            Python::attach(|py| {
                let targets_np = matref_to_numpy(targets, py);

                let result_obj: Py<PyAny> = match gradient_fn.call1(py, (targets_np,)) {
                    Ok(obj) => obj.into(),
                    Err(err) => {
                        err.print(py);
                        panic!("gradient_fn callback raised an exception")
                    }
                };

                let (values_obj, gradients_obj): (Py<PyAny>, Py<PyAny>) = result_obj
                    .extract(py)
                    .expect("gradient_fn must return a tuple of (values, gradients)");

                let values = numpy_to_scalar_column(py, &values_obj, "gradient_fn values");
                let gradients = numpy_to_matref::<f64>(py, &gradients_obj)
                    .expect("gradient_fn gradients must be a 2D float64 numpy array")
                    .to_owned();

                if values.nrows() != targets.nrows() {
                    panic!(
                        "gradient_fn values must return shape (N,) or (N, 1); got {} rows for {} targets",
                        values.nrows(),
                        targets.nrows()
                    );
                }
                if gradients.nrows() != targets.nrows() || gradients.ncols() != 3 {
                    panic!(
                        "gradient_fn gradients must return shape (N, 3); got ({}, {})",
                        gradients.nrows(),
                        gradients.ncols()
                    );
                }

                (values, gradients)
            })
        }
    });

    let gradient_fn_ref = py_gradient_fn
        .as_mut()
        .map(|f| f as &mut dyn FnMut(MatRef<f64>) -> (Mat<f64>, Mat<f64>));

    let meshes = ferreus_rmt::build_isosurfaces(
        seed_points_mat,
        extents_slice,
        resolution,
        isovalues,
        &mut py_surface_fn,
        gradient_fn_ref,
        cluster_method.unwrap().into(),
        boundary_closure.unwrap().into(),
        progress_sink.as_deref(),
    );

    let meshes: Vec<Mesh> = meshes
        .into_iter()
        .map(|mesh| Mesh { inner: mesh })
        .collect();

    PyList::new(py, meshes)
}
