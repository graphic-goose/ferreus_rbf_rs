/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines global anisotropy and trend transforms for shaping RBF interpolation behaviour.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use faer::{concat, linalg::solvers::DenseSolveCore, mat, Mat, Row};
use serde::{Serialize, Deserialize};

/// Defines an anisotropy transform for an RBF problem by specifying
/// principal directions and scaling ratios.
/// 
/// The variant to use depends on the dimensionality of the RBF problem:
/// 
/// - [`GlobalTrend::One`] - for **1D problems**, with a single principal axis.
/// - [`GlobalTrend::Two`] - for **2D problems**, with two axes lying in a plane,
///   oriented by a rotation angle.
/// - [`GlobalTrend::Three`] - for **3D problems**, with a full orientation
///   defined by sequential rotations.
/// 
/// Each variant encodes the relative scaling (ratios) along its principal axes,
/// providing a compact way to represent anisotropy and directional stretching
/// appropriate for the problem dimension.
/// 
/// This is particularly useful when the input data shows a clear
/// directional continuity or trend: by increasing the relative
/// weighting along that direction, interpolation can better reflect
/// the structure present in the data.
/// 
/// **Note:** All angles are specified in **degrees**.
#[derive(Clone, Copy, Debug)]
pub enum GlobalTrend {
    /// A 1D global trend.
    ///
    /// Represents anisotropy with a single scaling ratio
    /// along one principal axis.
    One {
        /// Scaling ratio along the principal axis.
        major_ratio: f64,
    },

    /// A 2D global trend.
    ///
    /// Defined within the XY plane, oriented by a rotation angle.
    /// Two scaling ratios describe the major and minor
    /// axes lying in the rotated space.
    Two {
        /// Rotation angle in degrees (positive = clockwise).
        rotation_angle: f64,

        /// Scaling ratio along the major axis, aligned with the
        /// rotation direction.
        major_ratio: f64,

        /// Scaling ratio along the minor axis, perpendicular
        /// to the rotation direction within the plane.
        minor_ratio: f64,
    },

    /// A 3D global trend.
    ///
    /// Rotation conventions:
    ///  - Left-hand rule for rotations (positive = clockwise).
    ///  - Rotation sequence is Z-X-Z′.
    ///
    /// ```text
    ///      +Z     +Y
    ///      ^      ^
    ///      |     /
    ///      |    /
    ///      |   /
    ///      |  /
    ///      | /
    ///      |/
    ///      o— — — — — — —> +X
    /// ```
    ///
    /// Terminology:
    /// - `dip_direction`: azimuth angle in the XY plane.
    /// - `dip`: tilt angle from horizontal toward the dip direction.
    /// - `strike`: dip_direction − 90° (perpendicular to dip direction).
    /// - `pitch`: rotation within the tilted plane, measured from strike.
    ///
    /// After the Z and X rotations, the plane is tilted; `pitch`
    /// rotates within that plane about the new Z′ axis.
    ///
    /// ```text
    ///           +Z′
    ///           ^
    ///            \
    ///             \       strike 
    ///              o — — — — — — — — —> -X′
    ///             /                  /
    ///            /                  /
    ///           /— — — — — — — — — /
    ///       dipdir        \ pitch / 
    ///         /            \     /  
    ///        v              \   /   
    ///       +Y′— — — — — — — — — 
    /// ```
    Three {
        /// Tilt angle in degrees from horizontal toward `dip_direction`.
        dip: f64,

        /// Azimuth angle in degrees in the XY plane, defining tilt direction.
        dip_direction: f64,

        /// Rotation in degrees within the tilted plane, measured from strike.
        pitch: f64,

        /// Scaling ratio along the major axis (aligned with pitch).
        major_ratio: f64,

        /// Scaling ratio along the semi-major axis (perpendicular
        /// to pitch within the plane).
        semi_major_ratio: f64,

        /// Scaling ratio along the minor axis (aligned with
        /// the plane normal).
        minor_ratio: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalTrendTransform {
    affine_transform: Mat<f64>,
    inverse_transform: Mat<f64>,
}

impl GlobalTrendTransform {
    pub fn new(center: Row<f64>, global_trend: GlobalTrend) -> Self {
        let affine_transform = match global_trend {
            GlobalTrend::One { major_ratio } => {
                let transform = mat![[1.0, -center[0]], [0.0, 1.0],];

                let transform_back = mat![[1.0, center[0]], [0.0, 1.0],];

                let scale = mat![[1.0 / major_ratio, 0.0], [0.0, 1.0],];

                let affine_transform = transform_back * scale * transform;

                affine_transform.transpose().to_owned()
            }
            GlobalTrend::Two {
                rotation_angle,
                major_ratio,
                minor_ratio,
            } => {
                let transform = mat![
                    [1.0, 0.0, -center[0]],
                    [0.0, 1.0, -center[1]],
                    [0.0, 0.0, 1.0],
                ];

                let transform_back = mat![
                    [1.0, 0.0, center[0]],
                    [0.0, 1.0, center[1]],
                    [0.0, 0.0, 1.0],
                ];

                // Because we want to unwind the points from world
                // space into a local coordinate space we negate the
                // angle to effectively rotate counter-clockwise.
                let pitchr = -rotation_angle.to_radians();

                let rotation = mat![
                    [pitchr.cos(), pitchr.sin(), 0.0],
                    [-pitchr.sin(), pitchr.cos(), 0.0],
                    [0.0, 0.0, 1.0],
                ];

                let scale = mat![
                    [1.0 / major_ratio, 0.0, 0.0],
                    [0.0, 1.0 / minor_ratio, 0.0],
                    [0.0, 0.0, 1.0],
                ];

                let affine_transform = transform_back * scale * rotation * transform;

                affine_transform.transpose().to_owned()
            }
            GlobalTrend::Three {
                dip,
                dip_direction,
                pitch,
                major_ratio,
                semi_major_ratio,
                minor_ratio,
            } => {
                let transform = mat![
                    [1.0, 0.0, 0.0, -center[0]],
                    [0.0, 1.0, 0.0, -center[1]],
                    [0.0, 0.0, 1.0, -center[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                let transform_back = mat![
                    [1.0, 0.0, 0.0, center[0]],
                    [0.0, 1.0, 0.0, center[1]],
                    [0.0, 0.0, 1.0, center[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                // Because we want to unwind the points from world
                // space into a local coordinate space we negate the
                // angles to effectively rotate counter-clockwise.
                let dipr = -dip.to_radians();
                let dipdirr = -dip_direction.to_radians();
                let pitchr = -pitch.to_radians();
            
                // Rotate about global Z so that local Y′ aligns
                // with the dip direction.
                let rot_z = mat![
                    [dipdirr.cos(), dipdirr.sin(), 0.0, 0.0],
                    [-dipdirr.sin(), dipdirr.cos(), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                // Tilt about X′ by dip angle.
                let rot_x = mat![
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, dipr.cos(), dipr.sin(), 0.0],
                    [0.0, -dipr.sin(), dipr.cos(), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                // Rotate within the dip plane by pitch.
                let rot_z_2 = mat![
                    [pitchr.cos(), pitchr.sin(), 0.0, 0.0],
                    [-pitchr.sin(), pitchr.cos(), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                // Final: ZXZ′ sequence (dipdir -> dip -> pitch),
                // then scale in rotated frame.
                let rotation = rot_z_2 * rot_x * rot_z;

                let scale = mat![
                    [1.0 / major_ratio, 0.0, 0.0, 0.0],
                    [0.0, 1.0 / semi_major_ratio, 0.0, 0.0],
                    [0.0, 0.0, 1.0 / minor_ratio, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ];

                let affine_transform = transform_back * scale * rotation * transform;

                affine_transform.transpose().to_owned()
            }    
        };

        let lu = &affine_transform.partial_piv_lu();
        let inv_affine_transform = lu.inverse();

        GlobalTrendTransform {
            affine_transform: affine_transform,
            inverse_transform: inv_affine_transform,
        }
    }

    pub fn transform_points(&self, points: &Mat<f64>) -> Mat<f64> {
        let homogenous_points = concat![[points, Mat::<f64>::ones(points.nrows(), 1)],];

        let translated = homogenous_points * &self.affine_transform;

        translated.subcols(0, points.ncols()).to_owned()
    }

    pub fn inverse_transform_points(&self, points: &Mat<f64>) -> Mat<f64> {
        let homogenous_points = concat![[points, Mat::<f64>::ones(points.nrows(), 1)],];

        let translated_back = homogenous_points * &self.inverse_transform;

        translated_back.subcols(0, points.ncols()).to_owned()
    }
}
