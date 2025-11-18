/////////////////////////////////////////////////////////////////////////////////////////////
//
// Declares trait for shared kernel parameter sets.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::kernel_helpers::KernelParams;

/// Converts a shared [`KernelParams`] configuration into a concrete kernel type.
pub trait KernelFromParams: Sized {
    /// Constructs `Self` from a set of uniform kernel parameters.
    fn from_params(p: &KernelParams) -> Self;
}
