/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements PyO3 bindings and NumPy conversion helpers for the ferreus_bbfmm Python API.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef, mat::AsMatRef};
use faer_ext::IntoFaer;
use ferreus_bbfmm::{EvaluationTargets, FmmError, TargetGrid as RustTargetGrid};
use ferreus_rbf_utils;
use ferreus_rbf_utils::KernelType;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
};
use pyo3::prelude::*;
use std::sync::Arc;

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

/// Convert a `faer::Mat<f64>` to a NumPy array.
pub fn mat_to_numpy<'py>(mat: &Mat<f64>, py: Python<'py>) -> Py<PyAny> {
    let (nrows, ncols) = mat.shape();

    if ncols == 1 {
        let array = unsafe {
            let arr = PyArray1::<f64>::zeros(py, nrows, false);
            for i in 0..nrows {
                arr.uget_raw([i]).write(*mat.get(i, 0));
            }
            arr
        };

        array.into_any().into()
    } else {
        let array = unsafe {
            let arr = PyArray2::<f64>::zeros(py, [nrows, ncols], false);
            for i in 0..nrows {
                for j in 0..ncols {
                    arr.uget_raw([i, j]).write(*mat.get(i, j));
                }
            }
            arr
        };
        array.into_any().into()
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FmmKernelType {
    LinearRbf,
    ThinPlateSplineRbf,
    CubicRbf,
    SpheroidalRbf,
    WendlandsC2Rbf,
    SphericalRbf,
    ExponentialRbf,
    GaussianRbf,
    Cubic2Rbf,
    InverseMultiquadraticRbf,
    Laplacian,
    OneOverR2,
    OneOverR4,
}

/// The implemented orders for the spheroidal kernel.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpheroidalOrder {
    Three,
    Five,
    Seven,
    Nine,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum M2LCompressionType {
    #[pyo3(name = "None_")]
    None,
    SVD,
    ACA,
}

impl From<M2LCompressionType> for ferreus_bbfmm::M2LCompressionType {
    fn from(c: M2LCompressionType) -> Self {
        match c {
            M2LCompressionType::None => ferreus_bbfmm::M2LCompressionType::None,
            M2LCompressionType::SVD => ferreus_bbfmm::M2LCompressionType::SVD,
            M2LCompressionType::ACA => ferreus_bbfmm::M2LCompressionType::ACA,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FmmParams {
    inner: ferreus_bbfmm::FmmParams,
}

#[pymethods]
impl FmmParams {
    #[new]
    #[pyo3(signature=(max_points_per_cell, compression_type, epsilon, eval_chunk_size))]
    fn new(
        max_points_per_cell: usize,
        compression_type: M2LCompressionType,
        epsilon: f64,
        eval_chunk_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ferreus_bbfmm::FmmParams {
                max_points_per_cell,
                compression_type: compression_type.into(),
                epsilon,
                eval_chunk_size,
            },
        })
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct KernelParams {
    inner: ferreus_rbf_utils::KernelParams,
}

#[pymethods]
impl KernelParams {
    #[new]
    #[pyo3(signature=(
        kernel_type,
        *,
        spheroidal_order=None,
        base_range=None,
        total_sill=None,
    ))]
    fn new(
        kernel_type: FmmKernelType,
        spheroidal_order: Option<SpheroidalOrder>,
        base_range: Option<f64>,
        total_sill: Option<f64>,
    ) -> PyResult<Self> {
        let fmm_kt = match kernel_type {
            FmmKernelType::LinearRbf => KernelType::LinearRbf,
            FmmKernelType::ThinPlateSplineRbf => KernelType::ThinPlateSplineRbf,
            FmmKernelType::CubicRbf => KernelType::CubicRbf,
            FmmKernelType::SpheroidalRbf => {
                if let Some(order) = spheroidal_order {
                    match order {
                        SpheroidalOrder::Three => KernelType::Spheroidal3Rbf,
                        SpheroidalOrder::Five => KernelType::Spheroidal5Rbf,
                        SpheroidalOrder::Seven => KernelType::Spheroidal7Rbf,
                        SpheroidalOrder::Nine => KernelType::Spheroidal9Rbf,
                    }
                } else {
                    KernelType::Spheroidal3Rbf
                }
            }
            FmmKernelType::WendlandsC2Rbf => KernelType::WendlandsC2Rbf,
            FmmKernelType::SphericalRbf => KernelType::SphericalRbf,
            FmmKernelType::ExponentialRbf => KernelType::ExponentialRbf,
            FmmKernelType::GaussianRbf => KernelType::GaussianRbf,
            FmmKernelType::Cubic2Rbf => KernelType::Cubic2Rbf,
            FmmKernelType::InverseMultiquadraticRbf => KernelType::InverseMultiquadraticRbf,
            FmmKernelType::Laplacian => KernelType::Laplacian,
            FmmKernelType::OneOverR2 => KernelType::OneOverR2,
            FmmKernelType::OneOverR4 => KernelType::OneOverR4,
        };
        let mut builder = ferreus_rbf_utils::KernelParams::builder(fmm_kt);

        if let Some(v) = base_range {
            builder = builder.base_range(v);
        }
        if let Some(v) = total_sill {
            builder = builder.total_sill(v);
        }
        let inner = builder.build();

        Ok(Self { inner })
    }
}

/// Represents a regular grid of target points in one, two, or three dimensions.
///
/// Stores only the origin, spacing, and number of samples on each axis.
/// Target coordinates are calculated in C order, with the last axis varying fastest.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct TargetGrid {
    inner: RustTargetGrid,
}

#[pymethods]
impl TargetGrid {
    /// Constructs a regular target grid from axis origins, spacing, and sample counts.
    #[new]
    #[pyo3(signature = (origin, spacing, shape))]
    fn new(origin: Vec<f64>, spacing: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let inner = RustTargetGrid::try_new(&origin, &spacing, &shape)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Constructs a regular target grid from bounding extents and axis spacing.
    #[staticmethod]
    #[pyo3(signature = (extents, spacing))]
    fn from_spacing(extents: Vec<f64>, spacing: Vec<f64>) -> PyResult<Self> {
        let inner = RustTargetGrid::from_spacing(&extents, &spacing)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Gets the number of grid axes.
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Gets the number of samples on each grid axis.
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Calculates the total number of grid targets as the product of the axis sample counts.
    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// Calculates a single coordinate of a grid target for the given axis.
    fn coordinate(&self, target_index: usize, axis: usize) -> PyResult<f64> {
        if target_index >= self.inner.point_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "target index out of bounds",
            ));
        }
        if axis >= self.inner.dimensions() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "axis out of bounds",
            ));
        }
        Ok(self.inner.coordinate(target_index, axis))
    }

    /// Calculates the coordinates of a grid target and writes them to the given output array.
    fn write_target(
        &self,
        target_index: usize,
        mut output: PyReadwriteArray1<'_, f64>,
    ) -> PyResult<()> {
        if target_index >= self.inner.point_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "target index out of bounds",
            ));
        }
        let dims = self.inner.dimensions();
        if output.as_array().len() != dims {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "output dimension mismatch",
            ));
        }
        let mut coordinates = [0.0; 3];
        self.inner
            .write_target(target_index, &mut coordinates[..dims]);
        for (slot, value) in output.as_array_mut().iter_mut().zip(&coordinates[..dims]) {
            *slot = *value;
        }
        Ok(())
    }

    /// Gets the bounding extents of the sampled grid.
    fn extents(&self) -> Vec<f64> {
        self.inner.extents()
    }

    /// Creates a matrix containing a contiguous range of grid target coordinates.
    fn points(&self, py: Python<'_>, start: usize, count: usize) -> PyResult<Py<PyArray2<f64>>> {
        let total = self.inner.point_count();
        if start > total || count > total - start {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "target range out of bounds",
            ));
        }
        let points = self.inner.points(start, count);
        let array = PyArray2::<f64>::zeros(py, [count, self.inner.dimensions()], false);
        let mut output = unsafe { array.as_array_mut() };
        for i in 0..points.nrows() {
            for d in 0..points.ncols() {
                output[[i, d]] = points[(i, d)];
            }
        }
        Ok(array.unbind())
    }
}

#[pyclass]
pub struct FmmTree {
    inner: ferreus_rbf_utils::FmmTree,
}

#[pymethods]
impl FmmTree {
    #[new]
    #[pyo3(signature=(
        source_points,
        interpolation_order,
        kernel_params,
        sparse,
        *,
        extents=None,
        params=None,
        target_points=None,
        target_grid=None,
    ))]
    fn new(
        py: Python<'_>,
        source_points: Py<PyAny>,
        interpolation_order: usize,
        kernel_params: KernelParams,
        sparse: bool,
        extents: Option<PyReadonlyArray1<'_, f64>>,
        params: Option<FmmParams>,
        target_points: Option<Py<PyAny>>,
        target_grid: Option<PyRef<'_, TargetGrid>>,
    ) -> PyResult<Self> {
        let source_points_mat = Arc::new(
            numpy_to_matref::<f64>(py, &source_points)
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Expected a 1D/2D float64 array for source_points",
                    )
                })?
                .to_owned(),
        );
        let mut extents_vec = extents.map(|e| e.to_vec().unwrap().clone());
        let p = params.map(|p| p.inner);

        if target_points.is_some() && target_grid.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "provide target_points or target_grid, not both",
            ));
        }
        let points = target_points
            .as_ref()
            .map(|obj| {
                numpy_to_matref::<f64>(py, obj)
                    .map(|points| points.to_owned())
                    .map_err(pyo3::exceptions::PyTypeError::new_err)
            })
            .transpose()?;
        let grid = target_grid.as_ref().map(|grid| grid.inner.clone());
        let targets = grid
            .as_ref()
            .map(|grid| EvaluationTargets::Grid(grid))
            .or_else(|| {
                points
                    .as_ref()
                    .map(|points| EvaluationTargets::Points(points.as_ref()))
            });
        if let Some(targets) = targets {
            if targets.dimensions() != source_points_mat.ncols() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "target and source dimensions differ",
                ));
            }
            let bounds = match targets {
                EvaluationTargets::Grid(grid) => grid.extents(),
                EvaluationTargets::Points(points) => {
                    ferreus_rbf_utils::get_pointarray_extents(points)
                }
            };
            let mut extent = extents_vec.unwrap_or_else(|| {
                ferreus_rbf_utils::get_pointarray_extents(source_points_mat.as_mat_ref())
            });
            let dims = source_points_mat.ncols();
            if extent.len() != 2 * dims {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "expected two extent bounds per axis",
                ));
            }
            for d in 0..dims {
                extent[d] = extent[d].min(bounds[d]);
                extent[d + dims] = extent[d + dims].max(bounds[d + dims]);
            }
            extents_vec = Some(extent);
        }
        let inner = py.detach(|| {
            ferreus_rbf_utils::FmmTree::new_with_targets(
                source_points_mat,
                interpolation_order,
                kernel_params.inner,
                sparse,
                extents_vec,
                p,
                targets,
            )
        });
        Ok(Self { inner })
    }

    #[pyo3(signature=(weights))]
    fn set_weights(&mut self, py: Python<'_>, weights: Py<PyAny>) -> PyResult<()> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        self.inner.set_weights(w);
        Ok(())
    }

    #[pyo3()]
    fn set_local_coefficients(&mut self) -> PyResult<()> {
        self.inner.set_local_coefficients();
        Ok(())
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        let target_values = self.inner.evaluate(w, x.as_mat_ref()).map_err(|err| {
            let msg = match err {
                FmmError::PointOutsideTree { point_index } => format!(
                    "FMM evaluation failed: target point at row {} lies outside the tree extents",
                    point_index
                ),
                FmmError::InvalidTargets(reason) => reason.to_owned(),
                FmmError::KernelDoesNotSupportGradients => {
                    "FMM evaluation failed: gradient evaluation requested but kernel does not support gradients".to_string()
                }
            };
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        Ok(mat_to_numpy(&target_values, py))
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate_with_gradients(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        let (target_values, gradients) = self.inner.evaluate_with_gradients(w, x.as_mat_ref()).map_err(|err| {
            let msg = match err {
                FmmError::PointOutsideTree { point_index } => format!(
                    "FMM evaluation failed: target point at row {} lies outside the tree extents",
                    point_index
                ),
                FmmError::InvalidTargets(reason) => reason.to_owned(),
                FmmError::KernelDoesNotSupportGradients => {
                    "FMM evaluation failed: gradient evaluation requested but kernel does not support gradients".to_string()
                }
            };
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        Ok((
            mat_to_numpy(&target_values, py),
            mat_to_numpy(&gradients, py),
        ))
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate_leaves(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        let target_points = self.inner
            .evaluate_leaves(w, x.as_mat_ref())
            .map_err(|err| {
                let msg = match err {
                    FmmError::PointOutsideTree { point_index } => format!(
                        "FMM leaf evaluation failed: target point at row {} lies outside the tree extents",
                        point_index
                    ),
                    FmmError::InvalidTargets(reason) => reason.to_owned(),
                FmmError::KernelDoesNotSupportGradients => {
                        "FMM leaf evaluation failed: gradient evaluation requested but kernel does not support gradients".to_string()
                    }
                };
                pyo3::exceptions::PyValueError::new_err(msg)
            })?;
        Ok(mat_to_numpy(&target_points, py))
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate_leaves_with_gradients(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        let (target_values, gradients) = self.inner.evaluate_leaves_with_gradients(w, x.as_mat_ref()).map_err(|err| {
            let msg = match err {
                FmmError::PointOutsideTree { point_index } => format!(
                    "FMM evaluation failed: target point at row {} lies outside the tree extents",
                    point_index
                ),
                FmmError::InvalidTargets(reason) => reason.to_owned(),
                FmmError::KernelDoesNotSupportGradients => {
                    "FMM evaluation failed: gradient evaluation requested but kernel does not support gradients".to_string()
                }
            };
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        Ok((
            mat_to_numpy(&target_values, py),
            mat_to_numpy(&gradients, py),
        ))
    }

    /// Evaluates a regular target grid without storing a matrix of all target points.
    #[pyo3(signature=(weights, grid))]
    fn evaluate_grid(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        grid: PyRef<'_, TargetGrid>,
    ) -> PyResult<Py<PyAny>> {
        let definition = &grid.inner;
        let w = numpy_to_matref(py, &weights).map_err(pyo3::exceptions::PyTypeError::new_err)?;
        let values = py
            .detach(|| self.inner.evaluate_grid(w, definition))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        grid_values_to_numpy(py, &values, definition.shape())
    }

    /// Evaluates a regular target grid without storing a matrix of all target points.
    #[pyo3(signature=(weights, grid))]
    fn evaluate_grid_with_gradients(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        grid: PyRef<'_, TargetGrid>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let definition = &grid.inner;
        let w = numpy_to_matref(py, &weights).map_err(pyo3::exceptions::PyTypeError::new_err)?;
        let (values, gradients) = py
            .detach(|| self.inner.evaluate_grid_with_gradients(w, definition))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((
            grid_values_to_numpy(py, &values, definition.shape())?,
            grid_gradients_to_numpy(py, &gradients, definition.shape(), values.ncols())?,
        ))
    }

    /// Evaluates a regular target grid without storing a matrix of all target points.
    #[pyo3(signature=(weights, grid))]
    fn evaluate_grid_leaves(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        grid: PyRef<'_, TargetGrid>,
    ) -> PyResult<Py<PyAny>> {
        let definition = &grid.inner;
        let w = numpy_to_matref(py, &weights).map_err(pyo3::exceptions::PyTypeError::new_err)?;
        let values = py
            .detach(|| self.inner.evaluate_grid_leaves(w, definition))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        grid_values_to_numpy(py, &values, definition.shape())
    }

    /// Evaluates a regular target grid without storing a matrix of all target points.
    #[pyo3(signature=(weights, grid))]
    fn evaluate_grid_leaves_with_gradients(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        grid: PyRef<'_, TargetGrid>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let definition = &grid.inner;
        let w = numpy_to_matref(py, &weights).map_err(pyo3::exceptions::PyTypeError::new_err)?;
        let (values, gradients) = py
            .detach(|| {
                self.inner
                    .evaluate_grid_leaves_with_gradients(w, definition)
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((
            grid_values_to_numpy(py, &values, definition.shape())?,
            grid_gradients_to_numpy(py, &gradients, definition.shape(), values.ncols())?,
        ))
    }

    /// Returns the source points matrix as a NumPy array.
    fn source_points(&self, py: Python<'_>) -> Py<PyAny> {
        mat_to_numpy(&self.inner.source_points(), py)
    }
}

/// Grid outputs use C order; a RHS axis is appended only for multiple fields.
fn grid_values_to_numpy(py: Python<'_>, values: &Mat<f64>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    let mut output_shape = shape.to_vec();
    if values.ncols() > 1 {
        output_shape.push(values.ncols());
    }
    mat_to_numpy(values, py).call_method1(py, "reshape", (output_shape,))
}

fn grid_gradients_to_numpy(
    py: Python<'_>,
    gradients: &Mat<f64>,
    shape: &[usize],
    nrhs: usize,
) -> PyResult<Py<PyAny>> {
    let mut output_shape = shape.to_vec();
    if nrhs > 1 {
        output_shape.push(nrhs);
    }
    output_shape.push(shape.len());
    mat_to_numpy(gradients, py).call_method1(py, "reshape", (output_shape,))
}
