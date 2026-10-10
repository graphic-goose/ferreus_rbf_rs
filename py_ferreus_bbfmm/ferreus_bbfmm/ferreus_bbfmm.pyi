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
    WendlandsC2Rbf = 7
    r"""
    $$
    \varphi(r) =
    \begin{cases}
        (1 - r)^4 (4r + 1), & r < 1,\\
        0, & r \ge 1
    \end{cases}
    $$
    """

    SphericalRbf = 8
    r"""
    $$
    \varphi(r) =
    \begin{cases}
        1 - r\,(1.5 - 0.5\,r^2), & r < 1,\\
        0, & r \ge 1
    \end{cases}
    $$
    """

    ExponentialRbf = 9
    r"""
    $$
    \varphi(r) = e^{-3r}
    $$
    """

    GaussianRbf = 10
    r"""
    $$
    \varphi(r) = e^{-3r^2}
    $$
    """

    Cubic2Rbf = 11
    r"""
    Cubic RBF kernel as defined by Chiles, Delfiner (1999).

    $$
    \varphi(r) =
    \begin{cases}
        1 - 7r^2 + 8.75\,r^3 - 3.5\,r^5 + 0.75\,r^7, & r < 1,\\
        0, & r \ge 1
    \end{cases}
    $$
    """

    InverseMultiquadraticRbf = 12
    r"""
    Inverse Multiquadratic RBF kernel. Kernel decay is scaled to
    approximately align with the other kernels via $\kappa_m = 6.5$.

    $$
    \varphi(r) = \frac{1}{\sqrt{1 + \kappa_m^2\,r^2}}
    $$
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

class TargetGrid:
    """Represents a regular grid of target points in one, two, or three dimensions.

    Stores only the origin, spacing, and number of samples on each axis.
    Target coordinates are calculated in C order, with the last axis varying fastest.
    """

    def __init__(self, origin: list[float], spacing: list[float], shape: list[int]) -> None:
        """Constructs a regular target grid from axis origins, spacing, and sample counts.

        Parameters
        ----------
        origin : list[float]
            Coordinate of the first sample on each axis.
        spacing : list[float]
            Non-negative spacing on each axis. Zero spacing is allowed only for axes with one sample.
        shape : list[int]
            Positive number of samples on each axis.

        Raises
        ------
        ValueError
            If the arrays do not describe one to three axes, the coordinates or spacing
            are invalid, or the total number of target points exceeds the supported index range.
        OverflowError
            If a sample count is negative or exceeds the supported index range.
        """
        ...

    @staticmethod
    def from_spacing(extents: list[float], spacing: list[float]) -> TargetGrid:
        """Constructs a regular target grid from bounding extents and axis spacing.

        Parameters
        ----------
        extents : list[float]
            Bounding box `[xmin, ymin, ..., xmax, ymax, ...]` for one to three dimensions.
        spacing : list[float]
            Positive spacing on each axis.

        Returns
        -------
        TargetGrid
            A grid beginning at the lower bounds and containing all samples up to the upper bounds.
            The given spacing is preserved, so the final sample may fall short of the upper bound.

        Raises
        ------
        ValueError
            If the bounds, spacing, or resulting sample counts are invalid.
        """
        ...

    def dimensions(self) -> int:
        """Gets the number of grid axes."""
        ...

    def shape(self) -> list[int]:
        """Gets the number of samples on each grid axis."""
        ...

    def point_count(self) -> int:
        """Calculates the total number of grid targets as the product of the axis sample counts."""
        ...

    def coordinate(self, target_index: int, axis: int) -> float:
        """Calculates a single coordinate of a grid target for the given axis.

        The target index uses C order, with the last axis varying fastest.

        Raises
        ------
        IndexError
            If the target index or axis is out of bounds.
        OverflowError
            If the target index or axis is negative or exceeds the supported index range.
        """
        ...

    def write_target(self, target_index: int, output: npt.NDArray[np.float64]) -> None:
        """Calculates the coordinates of a grid target and writes them to the given output array.

        The target index uses C order, with the last axis varying fastest.
        The output must be a writable one-dimensional float64 array.

        Raises
        ------
        IndexError
            If the target index is out of bounds.
        ValueError
            If the output length differs from the dimensionality.
        OverflowError
            If the target index is negative or exceeds the supported index range.
        """
        ...

    def extents(self) -> list[float]:
        """Gets the bounding extents of the sampled grid.

        Returns `[xmin, ymin, ..., xmax, ymax, ...]` using the first and last samples on each axis,
        before any tree padding or coordinate transforms.
        """
        ...

    def points(self, start: int, count: int) -> npt.NDArray[np.float64]:
        """Creates a matrix containing a contiguous range of grid target coordinates.

        Parameters
        ----------
        start : int
            Global index of the first target in C order.
        count : int
            Number of consecutive targets to include.

        Returns
        -------
        npt.NDArray[np.float64]
            A target point array of shape (count, D), where D is the dimensionality.

        Raises
        ------
        IndexError
            If the requested range extends beyond the grid targets.
        OverflowError
            If start or count is negative or exceeds the supported index range.
        """
        ...

class FmmTree:
    """A Fast Multipole Method (FMM) tree that organises source points into a hierarchical spatial
    structure to accelerate kernel summation tasks.

    The tree is adaptively and refined, with optional sparse leaf pruning.
    It efficiently precomputes all operators (M2M and M2L) required for far-field approximation.

    Parameters
    ----------
    source_points : npt.NDArray[np.float64]
        Source point locations used to build the tree.
        Expected to be a numpy array with shape (N, D), where N is the number of points and D is 
        the dimensionality.
    interpolation_order : int
        Number of Chebyshev interpolation nodes per dimension.
    kernel_params : KernelParams
        KernelParams that define the kernel function used for interaction computations.
    sparse : bool
        If `True`, constructs a sparse tree that omits empty leaves.
    extents : Optional[npt.NDArray[np.float64]]
        Optional bounding box `[xmin, ymin, ..., xmax, ymax, ...]`; if `None`, computed from data.
    params : Optional[FmmParams]
        Optional parameters for tuning the FMM performance.
    target_points : Optional[npt.NDArray[np.float64]]
        Optional target locations used to guide adaptive refinement when `sparse`=False.
    target_grid : Optional[TargetGrid]
        Optional regular target grid used to guide adaptive refinement, without target coordinates.
        Provide either target_points or target_grid.
    """    
    def __init__(
        self,
        source_points: npt.NDArray[np.float64],
        interpolation_order: int,
        kernel_params: KernelParams,
        sparse: bool,
        *,
        extents: Optional[npt.NDArray[np.float64]] = None,
        params: Optional[FmmParams] = None,
        target_points: npt.NDArray[np.float64] | None = None,
        target_grid: TargetGrid | None = None,
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
    ) -> npt.NDArray[np.float64]: 
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

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape (N, K), where N is the number of target points and K
            is the number of right-hand-sides evaluated.                  
        """
        ...

    def evaluate_with_gradients(
        self,
        weights: npt.NDArray[np.float64],
        target_points: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
        """Performs a downward pass of the tree to set the local coefficients and
        then performs a leaf evaluation pass to evaluate the values and gradients at the
        target locations.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number
            of right-hand sides to evaluate, containing source point weights (values)
        target_points : npt.NDArray[np.float64]
            Numpy array of shape (N, D), where N is the number of target points and D is the
            dimensionality.

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape (N, K), where N is the number of target points and K
            is the number of right-hand-sides evaluated.
        gradients : npt.NDArray[np.float64]
            Array of evaluated gradients with shape (N, D x M), where N is the number of target points,
            D is the dimensionality and M is the number of columns of values interpolated.    
            The gradient values are stored in batches of D columns, so the first D columns are for each dimension
            of the first column of values evaluated, the second D columns are for each dimension of the second column
            of values evaluated etc.                                       
        """
        ...

    def set_local_coefficients(
        self,
    ) -> None: 
        """Performs a downward pass of the tree to set the local coefficients. Intended to be
        used after calling [`set_weights`][ferreus_bbfmm.FmmTree.set_weights] and before calling
        [`evaluate_leaves`][ferreus_bbfmm.FmmTree.evaluate_leaves].
        """
        ...

    def evaluate_leaves(
        self,
        weights: npt.NDArray[np.float64],
        target_points: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: 
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

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape (N, K), where N is the number of target points and K
            is the number of right-hand-sides evaluated.    
        """
        ...

    def evaluate_leaves_with_gradients(
        self,
        weights: npt.NDArray[np.float64],
        target_points: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
        """Performs a leaf evaluation pass to calculate the values and gradients at the target locations. 
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
            
        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape (N, K), where N is the number of target points and K
            is the number of right-hand-sides evaluated.
        gradients : npt.NDArray[np.float64]
            Array of evaluated gradients with shape (N, D x M), where N is the number of target points,
            D is the dimensionality and M is the number of columns of values interpolated.    
            The gradient values are stored in batches of D columns, so the first D columns are for each dimension
            of the first column of values evaluated, the second D columns are for each dimension of the second column
            of values evaluated etc.                  
        """
        ...

    def evaluate_grid(self, weights: npt.NDArray[np.float64], grid: TargetGrid) -> npt.NDArray[np.float64]:
        """Performs a downward pass of the tree to set the local coefficients and
        then performs a leaf evaluation pass to evaluate the values at the
        grid target locations.

        Call [`set_weights`][ferreus_bbfmm.FmmTree.set_weights] before evaluating, and repeat that step whenever the weights change.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number of right-hand sides
            to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
        grid : TargetGrid
            Regular target grid with the same dimensionality as the tree.

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape `grid_shape` for a single right-hand side or
            `grid_shape + (K,)` for multiple right-hand sides, where K is the number of
            right-hand sides evaluated. The spatial axes follow the order of the grid axes,
            with the last grid axis varying fastest (C order).

        Raises
        ------
        ValueError
            If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
            Sparse trees may omit cells containing grid targets.

        Notes
        -----
        Target coordinates are generated as needed, without storing the complete coordinate matrix.
        `grid_shape` is given by [`TargetGrid.shape`][ferreus_bbfmm.TargetGrid.shape].
        """
        ...

    def evaluate_grid_with_gradients(self, weights: npt.NDArray[np.float64], grid: TargetGrid) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Performs a downward pass of the tree to set the local coefficients and
        then performs a leaf evaluation pass to evaluate the values and gradients at the
        grid target locations.

        Call [`set_weights`][ferreus_bbfmm.FmmTree.set_weights] before evaluating, and repeat that step whenever the weights change.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number of right-hand sides
            to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
        grid : TargetGrid
            Regular target grid with the same dimensionality as the tree.

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape `grid_shape` for a single right-hand side or
            `grid_shape + (K,)` for multiple right-hand sides, where K is the number of
            right-hand sides evaluated. The spatial axes follow the order of the grid axes,
            with the last grid axis varying fastest (C order).
        gradients : npt.NDArray[np.float64]
            Array of evaluated gradients with shape `grid_shape + (D,)` for a single right-hand
            side or `grid_shape + (K, D)` for multiple right-hand sides, where D is the
            dimensionality and K is the number of right-hand sides evaluated.
            The final axis contains the gradient components for each dimension, and the
            preceding axis selects the right-hand side when K is greater than one.

        Raises
        ------
        ValueError
            If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
            Sparse trees may omit cells containing grid targets.
            Also raised if the kernel does not support gradient evaluation.

        Notes
        -----
        Target coordinates are generated as needed, without storing the complete coordinate matrix.
        `grid_shape` is given by [`TargetGrid.shape`][ferreus_bbfmm.TargetGrid.shape].
        """
        ...

    def evaluate_grid_leaves(self, weights: npt.NDArray[np.float64], grid: TargetGrid) -> npt.NDArray[np.float64]:
        """Performs a leaf evaluation pass to calculate the values at the grid target locations. Intended to be
        used after [`set_local_coefficients`][ferreus_bbfmm.FmmTree.set_local_coefficients], for when repeated calls to this function are desired,
        such as when using 'surface following' isosurface generation algorithms.

        Call [`set_weights`][ferreus_bbfmm.FmmTree.set_weights] followed by [`set_local_coefficients`][ferreus_bbfmm.FmmTree.set_local_coefficients] before evaluating.
        If the weights change, repeat both steps before calling this method again.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number of right-hand sides
            to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
        grid : TargetGrid
            Regular target grid with the same dimensionality as the tree.

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape `grid_shape` for a single right-hand side or
            `grid_shape + (K,)` for multiple right-hand sides, where K is the number of
            right-hand sides evaluated. The spatial axes follow the order of the grid axes,
            with the last grid axis varying fastest (C order).

        Raises
        ------
        ValueError
            If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
            Sparse trees may omit cells containing grid targets.

        Notes
        -----
        Target coordinates are generated as needed, without storing the complete coordinate matrix.
        `grid_shape` is given by [`TargetGrid.shape`][ferreus_bbfmm.TargetGrid.shape].
        """
        ...

    def evaluate_grid_leaves_with_gradients(self, weights: npt.NDArray[np.float64], grid: TargetGrid) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Performs a leaf evaluation pass to calculate the values and gradients at the grid target locations. Intended to be
        used after [`set_local_coefficients`][ferreus_bbfmm.FmmTree.set_local_coefficients], for when repeated calls to this function are desired,
        such as when using 'surface following' isosurface generation algorithms.

        Call [`set_weights`][ferreus_bbfmm.FmmTree.set_weights] followed by [`set_local_coefficients`][ferreus_bbfmm.FmmTree.set_local_coefficients] before evaluating.
        If the weights change, repeat both steps before calling this method again.

        Parameters
        ----------
        weights : npt.NDArray[np.float64]
            Numpy array of shape (N, K), where N is the number of source points and K is the number of right-hand sides
            to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
        grid : TargetGrid
            Regular target grid with the same dimensionality as the tree.

        Returns
        -------
        values : npt.NDArray[np.float64]
            Array of evaluated values with shape `grid_shape` for a single right-hand side or
            `grid_shape + (K,)` for multiple right-hand sides, where K is the number of
            right-hand sides evaluated. The spatial axes follow the order of the grid axes,
            with the last grid axis varying fastest (C order).
        gradients : npt.NDArray[np.float64]
            Array of evaluated gradients with shape `grid_shape + (D,)` for a single right-hand
            side or `grid_shape + (K, D)` for multiple right-hand sides, where D is the
            dimensionality and K is the number of right-hand sides evaluated.
            The final axis contains the gradient components for each dimension, and the
            preceding axis selects the right-hand side when K is greater than one.

        Raises
        ------
        ValueError
            If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
            Sparse trees may omit cells containing grid targets.
            Also raised if the kernel does not support gradient evaluation.

        Notes
        -----
        Target coordinates are generated as needed, without storing the complete coordinate matrix.
        `grid_shape` is given by [`TargetGrid.shape`][ferreus_bbfmm.TargetGrid.shape].
        """
        ...

    def source_points(
        self,
    ) -> npt.NDArray[np.float64]: 
        """Source point locations used to build the tree.

        Returns
        -------
        source_points : npt.NDArray[np.float64]
            Numpy array of shape (N, D), where N is the number of source points and D is the
            dimensionality.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
