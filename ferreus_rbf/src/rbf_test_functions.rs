/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides benchmark test functions for validating and demonstrating RBF interpolation quality.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! 3D test functions f1_3d - f8_3d are implemented from [1].
//!
//! # References
//! 1. Bozzini, Mira & Rossini, Milvia. (2002). Testing methods for 3D scattered data 
//!    interpolation. 20. 111-135.
use faer::Mat;

/// Struct that implements various 2D and 3D functions to generate values for testing RBF
/// inteprolation.
pub struct RBFTestFunctions;

impl RBFTestFunctions {
    /// Franke's two-dimensional test function:
    /// <div>
    /// $$
    /// \begin{aligned}
    /// F(x,y) &= 
    /// \tfrac{3}{4}\exp\!\left[
    ///     -\frac{(9x-2)^2 + (9y-2)^2}{4}
    /// \right] \\[6pt]
    /// &\quad+ \tfrac{3}{4}\exp\!\left[
    ///     -\frac{(9x+1)^2}{49}
    ///     -\frac{(9y+1)^2}{10}
    /// \right] \\[6pt]
    /// &\quad+ \tfrac{1}{2}\exp\!\left[
    ///     -\frac{(9x-7)^2 + (9y-3)^2}{4}
    /// \right] \\[6pt]
    /// &\quad- \tfrac{1}{5}\exp\!\left[
    ///     -(9x-4)^2 - (9y-7)^2
    /// \right]
    /// \end{aligned}
    /// $$
    /// </div>
    pub fn franke_2d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 2);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];

            let nx = 9.0 * x;
            let ny = 9.0 * y;

            // 3/4 * exp(-((9x-2)^2 + (9y-2)^2)/4)
            let dx1 = nx - 2.0;
            let dy1 = ny - 2.0;
            let term1 = 0.75 * (-(dx1.powi(2) + dy1.powi(2)) / 4.0).exp();

            // 3/4 * exp(-(9x+1)^2/49 - (9y+1)^2/10)
            let dx2 = nx + 1.0;
            let dy2 = ny + 1.0;
            let term2 = 0.75 * (-(dx2.powi(2)) / 49.0 - (dy2.powi(2)) / 10.0).exp();

            // 1/2 * exp(-((9x-7)^2 + (9y-3)^2)/4)
            let dx3 = nx - 7.0;
            let dy3 = ny - 3.0;
            let term3 = 0.5 * (-(dx3.powi(2) + dy3.powi(2)) / 4.0).exp();

            // -(1/5) * exp(-((9x-4)^2 + (9y-7)^2))
            let dx4 = nx - 4.0;
            let dy4 = ny - 7.0;
            let term4 = -0.2 * (-(dx4.powi(2) + dy4.powi(2))).exp();

            term1 + term2 + term3 + term4
        })
    }

    /// 3D Franke-like test function:
    ///
    /// <div> 
    /// $$
    /// \begin{aligned}
    /// F(x,y,z) &= 
    /// \tfrac{3}{4}\exp\!\left[
    ///     -\frac{(9x-2)^2 + (9y-2)^2 + (9z-2)^2}{4}
    /// \right] \\[6pt]
    /// &\quad+ \tfrac{3}{4}\exp\!\left[
    ///     -\frac{(9x+1)^2}{49}
    ///     -\frac{(9y+1)^2}{10}
    ///     -\frac{(9z+1)^2}{10}
    /// \right] \\[6pt]
    /// &\quad+ \tfrac{1}{2}\exp\!\left[
    ///     -\frac{(9x-7)^2 + (9y-3)^2 + (9z-5)^2}{4}
    /// \right] \\[6pt]
    /// &\quad- \tfrac{1}{5}\exp\!\left[
    ///     -(9x-4)^2 - (9y-7)^2 - (9z-5)^2
    /// \right]
    /// \end{aligned}
    /// $$
    /// </div>
    pub fn f1_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            let nx = 9.0 * x;
            let ny = 9.0 * y;
            let nz = 9.0 * z;

            // 3/4 * exp(-((9x-2)^2 + (9y-2)^2 + (9z-2)^2)/4)
            let dx1 = nx - 2.0;
            let dy1 = ny - 2.0;
            let dz1 = nz - 2.0;
            let term1 = 0.75 * (-(dx1.powi(2) + dy1.powi(2) + dz1.powi(2)) / 4.0).exp();

            // 3/4 * exp(-(9x+1)^2/49 - (9y+1)^2/10 - (9z+1)^2/10)
            let dx2 = nx + 1.0;
            let dy2 = ny + 1.0;
            let dz2 = nz + 1.0;
            let term2 =
                0.75 * (-(dx2.powi(2)) / 49.0 - (dy2.powi(2)) / 10.0 - (dz2.powi(2) / 10.0)).exp();

            // 1/2 * exp(-((9x-7)^2 + (9y-3)^2 + (9z-5)^2)/4)
            let dx3 = nx - 7.0;
            let dy3 = ny - 3.0;
            let dz3 = nz - 5.0;
            let term3 = 0.5 * (-(dx3.powi(2) + dy3.powi(2) + dz3.powi(2)) / 4.0).exp();

            // -0.2 * exp(-(9x-4)^2 - (9y-7)^2 - (9z-5)^2)
            let dx4 = nx - 4.0;
            let dy4 = ny - 7.0;
            let dz4 = nz - 5.0;
            let term4 = -0.2 * (-(dx4.powi(2) + dy4.powi(2) + dz4.powi(2))).exp();

            term1 + term2 + term3 + term4
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) = 
    /// \frac{
    ///     \tanh(9z - 9x - 9y) + 1
    /// }{
    ///     9
    /// }
    /// $$
    /// </div>
    pub fn f2_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            ((9.0 * z - 9.0 * x - 9.0 * y).tanh() + 1.0) / 9.0
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) =
    /// \frac{
    ///     \cos(6z)\,\bigl(1.25 + \cos(5.4y)\bigr)
    /// }{
    ///     6 + 6(3x - 1)^2
    /// }
    /// $$
    /// </div>
    pub fn f3_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            (6.0 * z).cos() * (1.25 + (5.4 * y).cos()) / (6.0 + 6.0 * (3.0 * x - 1.0).powi(2))
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) =
    /// \frac{1}{3}\,
    /// \exp\!\left[
    ///     -\frac{81}{16}
    ///     \bigl(
    ///         (x-\tfrac{1}{2})^2 +
    ///         (y-\tfrac{1}{2})^2 +
    ///         (z-\tfrac{1}{2})^2
    ///     \bigr)
    /// \right]
    /// $$
    /// </div>
    pub fn f4_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            (-81.0 / 16.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2))).exp() / 3.0
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) =
    /// \frac{1}{3}\,
    /// \exp\!\left[
    ///     -\frac{81}{4}
    ///     \bigl(
    ///         (x-\tfrac{1}{2})^2 +
    ///         (y-\tfrac{1}{2})^2 +
    ///         (z-\tfrac{1}{2})^2
    ///     \bigr)
    /// \right]
    /// $$
    /// </div>
    pub fn f5_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            (-81.0 / 4.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2))).exp() / 3.0
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) =
    /// \frac{
    ///     \sqrt{
    ///         64 -
    ///         81\bigl[
    ///             (x-\tfrac{1}{2})^2 +
    ///             (y-\tfrac{1}{2})^2 +
    ///             (z-\tfrac{1}{2})^2
    ///         \bigr]
    ///     }
    /// }{
    ///     9
    /// }
    /// - \tfrac{1}{2}
    /// $$
    /// </div>
    pub fn f6_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            (64.0 - 81.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2))).sqrt() / 9.0
                - 0.5
        })
    }

    /// <div>
    /// $$
    /// F(x,y,z) =
    /// \frac{
    ///     1
    /// }{
    ///     \sqrt{
    ///         1 + 2\exp\!\bigl(
    ///             -3\bigl(\sqrt{x^2 + y^2 + z^2} - 6.7\bigr)
    ///         \bigr)
    ///     }
    /// }
    /// $$
    /// </div>
    pub fn f7_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let z = points[(i, 2)];

            1.0 / (1.0 + 2.0 * (-3.0 * ((x.powi(2) + y.powi(2) + z.powi(2)).sqrt() - 6.7)).exp()).sqrt()
        })
    }
    /// Peak function (independent of ``z``):
    /// 
    /// <div>
    /// $$
    /// \begin{aligned}
    /// F(x,y,z) &= 
    /// 50\,\exp\!\left[
    ///     -200\bigl((x-0.3)^2 + (y-0.3)^2\bigr)
    /// \right] \\[6pt]
    /// &\quad+ \exp\!\left[
    ///     -50\bigl((x-0.5)^2 + (y-0.5)^2\bigr)
    /// \right]
    /// \end{aligned}
    /// $$
    /// </div>
    pub fn f8_3d(points: &Mat<f64>) -> Mat<f64> {
        assert_eq!(points.ncols(), 3);
        let n = points.nrows();

        Mat::from_fn(n, 1, |i, _| {
            let x = points[(i, 0)];
            let y = points[(i, 1)];
            let _z = points[(i, 2)];

            50.0 * (-200.0 * ((x - 0.3).powi(2) + (y - 0.3).powi(2))).exp()
                + (-50.0 * ((x - 0.5).powi(2) + (y - 0.5).powi(2))).exp()
        })
    }
}
