/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines constants for spheroidal RBF kernels for the implemented orders.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

/// Tunable parameters that define a particular spheroidal RBF kernel family.
#[derive(Clone, Debug, Copy)]
pub struct SpheroidalConstants {
    pub inflexion_point: f64,
    pub linear_slope: f64,
    pub range_scaling: f64,
    pub inv_y_intercept: f64,
}

/// Calibrated constants for the order-3 spheroidal RBF kernel.
pub const SPHEROIDAL_CONSTANTS_THREE: SpheroidalConstants = SpheroidalConstants {
    inflexion_point: 0.5000000000,
    linear_slope: 0.7500000000,
    range_scaling: 2.6798340586,
    inv_y_intercept: 0.8734640537,
};

/// Calibrated constants for the order-5 spheroidal RBF kernel.
pub const SPHEROIDAL_CONSTANTS_FIVE: SpheroidalConstants = SpheroidalConstants {
    inflexion_point: 0.4082482905,
    linear_slope: 1.0206207262,
    range_scaling: 1.5822795750,
    inv_y_intercept: 0.8575980168,
};

/// Calibrated constants for the order-7 spheroidal RBF kernel.
pub const SPHEROIDAL_CONSTANTS_SEVEN: SpheroidalConstants = SpheroidalConstants {
    inflexion_point: 0.3535533906,
    linear_slope: 1.2374368671,
    range_scaling: 1.2008676644,
    inv_y_intercept: 0.8494862533,
};

/// Calibrated constants for the order-9 spheroidal RBF kernel.
pub const SPHEROIDAL_CONSTANTS_NINE: SpheroidalConstants = SpheroidalConstants {
    inflexion_point: 0.3162277660,
    linear_slope: 1.4230249471,
    range_scaling: 1.0000000000,
    inv_y_intercept: 0.8445585690,
};
