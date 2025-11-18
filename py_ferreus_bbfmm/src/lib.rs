/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the Python extension module for exposing the ferreus_bbfmm API to Python.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use pyo3::prelude::*;

mod python_bindings;

#[pymodule]
pub fn ferreus_bbfmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_bindings::FmmTree>()?;
    m.add_class::<python_bindings::FmmKernelType>()?;
    m.add_class::<python_bindings::FmmParams>()?;
    m.add_class::<python_bindings::KernelParams>()?;
    m.add_class::<python_bindings::M2LCompressionType>()?;
    m.add_class::<python_bindings::SpheroidalOrder>()?;
    Ok(())
}
