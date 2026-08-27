/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the Python extension module for exposing the regularised marching tetrahedra API.
//
// Created on: 08 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use pyo3::prelude::*;

mod python_bindings;

#[pymodule]
pub fn ferreus_rmt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let prog = PyModule::new(m.py(), "progress")?;
    prog.add_class::<python_bindings::IsosurfaceProgress>()?;
    prog.add_class::<python_bindings::Message>()?;
    prog.add_class::<python_bindings::ProgressEvent>()?;
    prog.add_class::<python_bindings::Progress>()?;

    m.add_submodule(&prog)?;
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("ferreus_rmt.progress", prog)?;

    m.add_class::<python_bindings::BoundaryClosure>()?;
    m.add_class::<python_bindings::ClusterMethod>()?;
    m.add_class::<python_bindings::Mesh>()?;
    m.add_function(wrap_pyfunction!(python_bindings::build_isosurface, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::build_isosurfaces, m)?)?;
    Ok(())
}
