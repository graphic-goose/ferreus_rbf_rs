"""
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Stubs file that enables type hints and intellisense for the ferreus_bbfmm Python API.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////
"""

from enum import Enum
from typing import Optional
import numpy as np
import numpy.typing as npt

class FmmKernelType(Enum):
    """Implemented kernel functions.
    """

    Laplacian = 0
    r"""
    $$
    \varphi(r) = 1 / r
    $$
    """

    OneOverR2 = 1
    r"""
    $$
    \varphi(r) = 1 / r^2
    $$
    """

    OneOverR4 = 2
    r"""
    $$
    \varphi(r) = 1 / r^4
    $$
    """

    LinearRbf = 3
    r"""
    $$
    \varphi(r) = -r
    $$
    """

    ThinPlateSplineRbf = 4
    r"""
    $$
    \varphi(r) =
    \begin{cases}
        0, & r=0,\\
        r^2 \log r, & r>0 .
    \end{cases}
    $$
    """

    CubicRbf = 5
    r"""
    $$
    \varphi(r) = r^3
    $$
    """

    SpheroidalRbf = 6
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

class M2LCompressionType(Enum):
    """
    Enum for the available compression methods for the M2L operators.

    """    
    None_ = 0
    """No compression applied to M2L operators"""

    SVD = 1
    """A truncated Singular Value Decompositio (SVD) is performed on the M2L operators."""

    ACA = 2
    """Adaptive cross approximation (ACA) is performed on the M2L operators, followed by SVD recompression."""

class FmmParams:
    """Optional parameters for tuning the FMM performance.

    Parameters
    ----------
    max_points_per_cell : int
        Maximum number of points per cell before it must be subdivided.
        When FmmParams is not provided the default value is 256.
    compression_type : M2LCompressionType
        The type of compression to apply to the M2L operators.
        When FmmParams is not provided the default value is ACA.
    epsilon : float
        Tolerance threshold for M2L compression.
        When FmmParams is not provided the default value is 10^-interpolation_order
    eval_chunk_size : int
        Number of target points to evaluate in each chunk.
        When FmmParams is not provided the default value is 1024.
    """
    def __init__(
        self,
        max_points_per_cell: int,
        compression_type: M2LCompressionType,
        epsilon: float,
        eval_chunk_size: int,
    ) -> None: ...

class KernelParams:
    """Defines the KernelType to use, along with parameter
    values for Spheroidal kernels.

    Parameters
    ----------
    kernel_type : FmmKernelType
        FmmKernelType enum variant to use.
    spheroidal_order : Optional[SpheroidalOrder]
        SpheroidalOrder enum variant to use.
        Only applicable when using the spheroidal kernel.
        If spheroidal kernel is used and an order isn't provided
        then the default is SpheroidalOrder.Three.
    base_range : Optional[float]
        Controls how quickly the interpolant decays with distance from each point. 
        Smaller values restrict influence to a local neighborhood, while larger values
        produce smoother, broader effects.

        Typically chosen based on the spacing of your data.
        Only used in spheroidal kernels.
    total_sill : Optional[float]
        Sets the overall strength of influence each point exerts. Higher values give
        points more weight and stronger local effects. Lower values yield smoother, 
        less pronounced variation.

        Works in combination with base_range and the kernel degree.
        Only used in spheroidal kernels.
    """    
    def __init__(
        self,
        kernel_type: FmmKernelType,
        spheroidal_order: Optional[SpheroidalOrder],
        base_range: Optional[float],
        total_sill: Optional[float],
    ) -> None: ...

class FmmTree:
    """A Fast Multipole Method (FMM) tree that organises source points into a hierarchical spatial
    structure to accelerate kernel summation tasks.

    The tree supports both adaptive and uniform refinement, with optional sparse leaf pruning.
    It efficiently precomputes all operators (M2M and M2L) required for far-field approximation.

    Parameters
    ----------
    source_points : npt.NDArray[np.float64]
        Source point locations used to build the tree.
        Expected to be a numpy array with shape (N, D), where N is the number of points and D is 
        the dimensionality.
    interpolation_order : int
        Number of Chebyshev interpolation nodes per dimension.
    kernel_params : FmmParams
        KernelParams that define the kernel function used for interaction computations.
    adaptive_tree : bool
        Whether the tree uses adaptive or uniform subdivision.
    sparse : bool
        If `True`, constructs a sparse tree that omits empty leaves.
    extents : Optional[npt.NDArray[np.float64]]
        Optional bounding box `[xmin, xmax, ymin, ymax, ...]`; if `None`, computed from data.
    params : Optional[FmmParams]
        Optional parameters for tuning the FMM performance.
    """    
    def __init__(
        self,
        source_points: npt.NDArray[np.float64],
        interpolation_order: int,
        kernel_params: KernelParams,
        adaptive_tree: bool,
        sparse: bool,
        extents: Optional[npt.NDArray[np.float64]],
        params: Optional[FmmParams],
    ) -> None: ...

    def set_weights(
        self,
        weights: npt.NDArray[np.float64],
    ) -> None: 
        """Performs an upward pass of the tree to set the multipole coefficients.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number
            of right-hand sides to evaluate, containing source point weights (values)
        """
        ...

    def evaluate(
        self,
        weights: npt.NDArray[np.float64],
        target_points: npt.NDArray[np.float64],
    ) -> None: 
        """Performs a downward pass of the tree to set the local coefficients and
        then performs a leaf evaluation pass to evaluate the values at the
        target locations.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number
            of right-hand sides to evaluate, containing source point weights (values)
        target_points : npt.NDArray[np.float64]
            Numpy array of shape (N, D), where N is the number of target points and D is the
            dimensionality.
        """
        ...

    def set_local_coefficients(
        self,
        weights: npt.NDArray[np.float64],
    ) -> None: 
        """Performs a downward pass of the tree to set the local coefficients. Intended to be
        used before calling ['FmmTree.evaluate_leaves`].

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number
            of right-hand sides to evaluate, containing source point weights (values)
        """
        ...

    def evaluate_leaves(
        self,
        weights: npt.NDArray[np.float64],
        target_points: npt.NDArray[np.float64],
    ) -> None: 
        """Performs a leaf evaluation pass to calculate the values at the target locations. 
        Intended to be used after [`set_local_coefficients`][ferreus_bbfmm.FmmTree.set_local_coefficients],
        for when repeated calls to this function are desired, such as when using 'surface following'
        isosurface generation algorithms.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number
            of right-hand sides to evaluate, containing source point weights (values)
        target_points : npt.NDArray[np.float64]
            Numpy array of shape (N, D), where N is the number of target points and D is the
            dimensionality.
        """
        ...

    def target_values(
        self,
    ) -> npt.NDArray[np.float64]: 
        """Values at target locations after [evaluate][ferreus_bbfmm.FmmTree.evaluate] or
        [evaluate_leaves][ferreus_bbfmm.FmmTree.evaluate_leaves]  is called.

        Returns
        -------
        npt.NDArray[np.float64]
            Returns a numpy array with shape (N, K), where N is the number of target points and K
            is the number of right-hand-sides evaluated.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
