/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the Python extension module and submodules for the ferreus_rbf interpolation API.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use pyo3::prelude::*;

mod python_bindings;

#[pymodule]
pub fn ferreus_rbf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    faer::set_global_parallelism(faer::Par::Seq);

    let cfg = PyModule::new(m.py(), "config")?;
    cfg.add_class::<python_bindings::DDMParams>()?;
    cfg.add_class::<python_bindings::FmmParams>()?;
    cfg.add_class::<python_bindings::Solvers>()?;
    cfg.add_class::<python_bindings::FmmCompressionType>()?;
    cfg.add_class::<python_bindings::Params>()?;

    m.add_submodule(&cfg)?;
    m.py().import("sys")?.getattr("modules")?.set_item("ferreus_rbf.config", cfg)?;

    let prog = PyModule::new(m.py(), "progress")?;
    prog.add_class::<python_bindings::DuplicatesRemoved>()?;
    prog.add_class::<python_bindings::SolverIteration>()?;
    prog.add_class::<python_bindings::SurfacingProgress>()?;
    prog.add_class::<python_bindings::Message>()?;
    prog.add_class::<python_bindings::ProgressEvent>()?;
    prog.add_class::<python_bindings::Progress>()?;

    m.add_submodule(&prog)?;
    m.py().import("sys")?.getattr("modules")?.set_item("ferreus_rbf.progress", prog)?;

    let interp = PyModule::new(m.py(), "interpolant_config")?;
    interp.add_class::<python_bindings::RBFKernelType>()?;
    interp.add_class::<python_bindings::InterpolantSettings>()?;
    interp.add_class::<python_bindings::Drift>()?;
    interp.add_class::<python_bindings::SpheroidalOrder>()?;
    interp.add_class::<python_bindings::FittingAccuracy>()?;
    interp.add_class::<python_bindings::FittingAccuracyType>()?;

    m.add_submodule(&interp)?;
    m.py().import("sys")?.getattr("modules")?.set_item("ferreus_rbf.interpolant_config", interp)?;    

    m.add_class::<python_bindings::RBFInterpolator>()?;
    m.add_class::<python_bindings::GlobalTrend>()?;
    m.add_class::<python_bindings::RBFTestFunctions>()?;

    m.add_function(wrap_pyfunction!(python_bindings::save_obj, m)?)?;
    Ok(())
}
