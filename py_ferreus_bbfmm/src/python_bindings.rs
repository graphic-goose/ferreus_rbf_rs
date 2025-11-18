/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements PyO3 bindings and NumPy conversion helpers for the ferreus_bbfmm Python API.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{Mat, MatRef};
use faer_ext::IntoFaer;
use ferreus_bbfmm::FmmError;
use ferreus_rbf_utils;
use ferreus_rbf_utils::KernelType;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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
    Laplacian,
    OneOverR2,
    OneOverR4
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
                        SpheroidalOrder::Five  => KernelType::Spheroidal5Rbf,
                        SpheroidalOrder::Seven => KernelType::Spheroidal7Rbf,
                        SpheroidalOrder::Nine  => KernelType::Spheroidal9Rbf,
                    }
                } else {
                    KernelType::Spheroidal3Rbf
                }
            }
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
        adaptive_tree,
        sparse,
        *,
        extents=None,
        params=None,
    ))]
    fn new(
        py: Python<'_>,
        source_points: Py<PyAny>,
        interpolation_order: usize,
        kernel_params: KernelParams,
        adaptive_tree: bool,
        sparse: bool,
        extents: Option<PyReadonlyArray1<'_, f64>>,
        params: Option<FmmParams>,
    ) -> PyResult<Self> {
        let source_points_mat = numpy_to_matref::<f64>(py, &source_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "Expected a 1D/2D float64 array for source_points",
                )
            })?
            .to_owned();
        let extents_vec = extents.map(|e| e.to_vec().unwrap().clone());
        let p = params.map(|p| p.inner);

        let inner = ferreus_rbf_utils::FmmTree::new(
            source_points_mat,
            interpolation_order,
            kernel_params.inner,
            adaptive_tree,
            sparse,
            extents_vec,
            p,
        );
        Ok(Self { inner })
    }

    #[pyo3(signature=(weights))]
    fn set_weights(&mut self, py: Python<'_>, weights: Py<PyAny>) -> PyResult<()> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        self.inner.set_weights(&w);
        Ok(())
    }

    #[pyo3(signature=(weights))]
    fn set_local_coefficients(&mut self, py: Python<'_>, weights: Py<PyAny>) -> PyResult<()> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        self.inner.set_local_coefficients(&w);
        Ok(())
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<()> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        self.inner.evaluate(&w, &x).map_err(|err| {
            let msg = match err {
                FmmError::PointOutsideTree { point_index } => format!(
                    "FMM evaluation failed: target point at row {} lies outside the tree extents",
                    point_index
                ),
            };
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        Ok(())
    }

    #[pyo3(signature=(weights, target_points))]
    fn evaluate_leaves(
        &mut self,
        py: Python<'_>,
        weights: Py<PyAny>,
        target_points: Py<PyAny>,
    ) -> PyResult<()> {
        let w = numpy_to_matref(py, &weights).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for weights")
        })?;
        let x = numpy_to_matref::<f64>(py, &target_points)
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("Expected 1D/2D float64 for target_points")
            })?
            .to_owned();
        self.inner
            .evaluate_leaves(&w, &x)
            .map_err(|err| {
                let msg = match err {
                    FmmError::PointOutsideTree { point_index } => format!(
                        "FMM leaf evaluation failed: target point at row {} lies outside the tree extents",
                        point_index
                    ),
                };
                pyo3::exceptions::PyValueError::new_err(msg)
            })?;
        Ok(())
    }

    /// Returns the last computed target values as a NumPy array.
    fn target_values(&self, py: Python<'_>) -> Py<PyAny> {
        mat_to_numpy(&self.inner.target_values(), py)
    }

    /// Returns the source points matrix as a NumPy array.
    fn source_points(&self, py: Python<'_>) -> Py<PyAny> {
        mat_to_numpy(&self.inner.source_points(), py)
    }
}
