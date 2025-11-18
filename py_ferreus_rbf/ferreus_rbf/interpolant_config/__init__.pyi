'''
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Stubs file for Python bindings of the interpolant_config module that enables typehints in IDE's.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////
'''

from enum import Enum
from typing import Optional

class Drift(Enum):
    """
    The name of the polynomial order to add to the RBF system.

    The drift affects the interpolant away from data locations.
    
    To ensure a unique solution to the RBF system of equations,
    some kernels have a minimum required polynomial that must
    be added, as shown below.

    |    Kernel       |  Minimum drift  |  Default drift  |
    |---------------- |-----------------|-----------------|
    | Linear          | Constant        | Constant        |
    | ThinPlateSpline | Linear          | Linear          |
    | Cubic           | Linear          | Linear          |
    | Spheroidal      | None            | None            |    
    """
    None_ = 0
    Constant = 1
    Linear = 2
    Quadratic = 3

class RBFKernelType(Enum):
    """Implemented kernel functions.
    """

    Linear = 0
    r"""
    $$
    \varphi(r) = -r
    $$
    """

    ThinPlateSpline = 1
    r"""
    $$
    \varphi(r) =
    \begin{cases}
        0, & r=0,\\
        r^2 \log r, & r>0 .
    \end{cases}
    $$
    """

    Cubic = 2
    r"""
    $$
    \varphi(r) = r^3
    $$
    """

    Spheroidal = 3
    r""" 
    $$
    \varphi(r) = s
    \begin{cases}
        1 - \lambda_{m}r_{s}, & r_{s} \le x^{*}_{m},\\
        c_{m}^{-1}(1 + r_{s}^2)^{-m / 2}, & r_{s} \ge x^{*}_{m}
    \end{cases}
    $$
    
    where   
    $$
    r_{s} = \kappa_{m}{r / R}
    $$

    with

    - s = total sill
    - R = base range
    
    !!! info "Spheroidal RBF Functions"
        The Spheroidal family of covariance functions have the same
        definition, with varying constant parameters based on the selected
        order.

        The order determines how steeply the interpolant asymptotically approaches `0.0`.
        A higher order value gives more weighting to points at intermediate distances,
        compared with lower orders.

        The Spheroidal covariance function is a piecewise function that combines the linear
        RBF function up to the inflexion point, and a scaled inverse multiquadric function
        after that.

        More information can be found [here](https://www.seequent.com/the-spheroidal-family-of-variograms-explained/).

        <div style="width: 100%;">
            <table style="width: 100%; border-collapse: collapse;">
                <caption style="
                    text-align: left;
                    font-family: var(--md-text-font--heading);
                    font-size: 1.25em;
                    font-weight: 600;">
                    Constant parameters for each supported spheroidal order
                </caption>
                <thead>
                    <tr>
                    <th style="text-align:left;">Order (<span>\(m\)</span>)</th>
                    <th style="text-align:right;">3</th>
                    <th style="text-align:right;">5</th>
                    <th style="text-align:right;">7</th>
                    <th style="text-align:right;">9</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td>Inflexion point (<span>\(x^{*}_{m}\)</span>)</td>
                    <td style="text-align:right;">0.5000000000</td>
                    <td style="text-align:right;">0.4082482905</td>
                    <td style="text-align:right;">0.3535533906</td>
                    <td style="text-align:right;">0.3162277660</td>
                    </tr>
                    <tr>
                    <td>Y-intercept (<span>\(c_{m}\)</span>)</td>
                    <td style="text-align:right;">1.1448668044</td>
                    <td style="text-align:right;">1.1660474725</td>
                    <td style="text-align:right;">1.1771820863</td>
                    <td style="text-align:right;">1.1840505048</td>
                    </tr>    
                    <tr>
                    <td>Linear slope (<span>\(\lambda_{m}\)</span>)</td>
                    <td style="text-align:right;">0.7500000000</td>
                    <td style="text-align:right;">1.0206207262</td>
                    <td style="text-align:right;">1.2374368671</td>
                    <td style="text-align:right;">1.4230249471</td>
                    </tr>
                    <tr>
                    <td>Range scaling (<span>\(\kappa_{m}\)</span>)</td>
                    <td style="text-align:right;">2.6798340586</td>
                    <td style="text-align:right;">1.5822795750</td>
                    <td style="text-align:right;">1.2008676644</td>
                    <td style="text-align:right;">1.0000000000</td>
                    </tr>
                </tbody>
            </table>
        </div>

    """

class SpheroidalOrder(Enum):
    """The implemented orders (alpha) for the spheroidal kernel.
    """
    Three = 3
    Five = 5
    Seven = 7
    Nine = 9

class FittingAccuracyType(Enum):
    """Defines whether to use relative or absolute stopping criteria for the solver."""
    Absolute = 0
    """The mismatch must be reduced by this factor compared to the initial mismatch."""

    Relative = 1
    """The mismatch must be less than this fixed amount in the same units as the
        data values.
    """

class FittingAccuracy:
    """
    Defines how closely the interpolated RBF solution should match the input data.

    When solving an RBF system, the algorithm iteratively refines the coefficients
    until the predicted values at the data locations are sufficiently close to the
    given sample values. ``FittingAccuracy`` tells the solver *when to stop refining*.
    
    Parameters
    ----------
    tolerance : float
        Sets the acceptable mismatch between the model and the input data.
        Smaller values mean the solution will track the data more tightly,
        but may require more iterations and time to compute.
    tolerance_type : FittingAccuracyType
        Sets the type of stopping criteria.

    """    
    def __init__(
        self,
        tolerance: float,
        tolerance_type: FittingAccuracyType,
    ) -> None: ...

class InterpolantSettings:
    """
    Holds the configuration parameters for an RBF kernel.

    The only required input is the ``RBFKernelType``.  

    If no additional options are provided, defaults are applied: 

    - ``Drift`` is set to the minimum valid choice for the kernel.  
    - For spheroidal kernels, `base_range` and `total_sill` default to `1.0`.  
    - The nugget defaults to `0.0` but may be specified for any kernel.
    - `fitting_accuracy`: [`FittingAccuracy`][ferreus_rbf.interpolant_config.FittingAccuracy](tolerance=1E-6, tolerance_type=[`FittingAccuracyType.Relative`][ferreus_rbf.interpolant_config.FittingAccuracyType.Relative]) 

    Parameters
    ----------
    kernel_type : RBFKernelType
        The RBF kernel to use for interpolation.
    spheroidal_order: SpheroidalOrder, optional
        The order (alpha) of spheroidal kernel. If ``None``, SpheroidalOrder.Three will be used.
    drift : Drift, optional
        The polynomial drift term added to the RBF system. If ``None``,
        the implementation will infer a default based on ``kernel_type``.
    nugget : float, optional
        Optional smoothing parameter. A value of `0.0` (default) enforces an exact
        fit to all input data. Larger values soften the fit, which can reduce
        sensitivity to noisy data.
    base_range : float, optional
        Controls how quickly the interpolant decays with distance from each point. Smaller
        values restrict influence to a local neighborhood, while larger values produce
        smoother, broader effects.  
    
        Typically chosen based on the spacing of your data.  
        Only used in spheroidal kernels.
    total_sill : float, optional
        Sets the overall strength of influence each point exerts. Higher values give
        points more weight and stronger local effects. Lower values yield smoother,
        less pronounced variation.  
    
        Works in combination with `base_range` and the spheroidal_order.  
        Only used in spheroidal kernels.
    fitting_accuracy : FittingAccuracy, optional
        Desired fitting accuracy and tolerance criteria.        
    """
    def __init__(
        self,
        kernel_type: RBFKernelType,
        spheroidal_order: Optional[SpheroidalOrder] = None,
        drift: Optional[Drift] = None,
        nugget: Optional[float] = None,
        base_range: Optional[float] = None,
        total_sill: Optional[float] = None,
        fitting_accuracy: Optional[FittingAccuracy] = None,
    ) -> None: ...