'''
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Stubs file for Python bindings of the RBFInterpolator module that enables typehints in IDE's.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////
'''

from typing import Optional, Union
import numpy as np
import numpy.typing as npt
from ferreus_rbf.config import Params
from ferreus_rbf.progress import Progress
from ferreus_rbf.interpolant_config import InterpolantSettings

class GlobalTrend:
    """
    Defines an anisotropy transform for an RBF problem by specifying
    principal directions and scaling ratios.
     
    The variant to use depends on the dimensionality of the RBF problem:  
     
    - ``GlobalTrend.one`` - for **1D problems**, with a single principal axis.  
    - ``GlobalTrend.two`` - for **2D problems**, with two axes lying in a plane,
      oriented by a rotation angle.  
    - ``GlobalTrend.three`` - for **3D problems**, with a full orientation
      defined by sequential rotations.
     
    Each variant encodes the relative scaling (ratios) along its principal axes,
    providing a compact way to represent anisotropy and directional stretching
    appropriate for the problem dimension.
     
    This is particularly useful when the input data shows a clear
    directional continuity or trend: by increasing the relative
    weighting along that direction, interpolation can better reflect
    the structure present in the data.
     
    **Note:** All angles are specified in **degrees**.    
    """    
    @classmethod
    def one(major_ratio: float) -> "GlobalTrend": 
        """
        A 1D global trend.

        Represents anisotropy with a single scaling ratio along one principal axis.

        Parameters
        ----------
        major_ratio : float
            Scaling ratio along the principal axis.

        Returns
        -------
        GlobalTrend
        """
        ...
    
    @classmethod
    def two(rotation_angle: float, major_ratio: float, minor_ratio: float) -> "GlobalTrend": 
        """
        A 2D global trend.

        Defined within the XY plane, oriented by a rotation angle.
        Two scaling ratios describe the major and minor axes lying in the rotated space.

        Parameters
        ----------
        rotation_angle : float
            Rotation angle in degrees (positive = clockwise).
        major_ratio : float
            Scaling ratio along the major axis, aligned with the rotation direction.
        minor_ratio : float
            Scaling ratio along the minor axis, perpendicular to the rotation direction within the plane.

        Returns
        -------
        GlobalTrend
        """
        ...
    
    @classmethod
    def three(
        dip: float,
        dip_direction: float,
        pitch: float,
        major_ratio: float,
        semi_major_ratio: float,
        minor_ratio: float
    ) -> "GlobalTrend": 
        r"""
        A 3D global trend.

        Rotation conventions:  

         - Left-hand rule for rotations (positive = clockwise).  
         - Rotation sequence is Z-X-Z'.
        
        ```text
             +Z     +Y
             ^      ^
             |     /
             |    /
             |   /
             |  /
             | /
             |/
             o- - - - - - - -> +X
        ```
        
        Terminology:  

        - `dip_direction`: azimuth angle in the XY plane.  
        - `dip`: tilt angle from horizontal toward the dip direction.  
        - `strike`: dip_direction - 90 (perpendicular to dip direction).  
        - `pitch`: rotation within the tilted plane, measured from strike.  

        After the Z and X rotations, the plane is tilted; `pitch`
        rotates within that plane about the new Z' axis.
        
        ```text
                  +Z'
                  ^
                   \
                    \       strike
                     o - - - - - - - - - > -X'
                    /                   /
                   /                   /
                  / - - - - - - - - - /
              dipdir        \ pitch /
                /            \     /
               v              \   /
              +Y' - - - - - - - - -
        ```

        Parameters
        ----------
        dip : float
            Tilt angle in degrees from horizontal toward `dip_direction`.
        dip_direction : float
            Azimuth angle in degrees in the XY plane, defining tilt direction.
        pitch : float
            Rotation in degrees within the tilted plane, measured from strike.
        major_ratio : float
            Scaling ratio along the major axis (aligned with pitch).
        semi_major_ratio : float
            Scaling ratio along the semi-major axis (perpendicular to pitch within the plane).
        minor_ratio : float
            Scaling ratio along the minor axis (aligned with the plane normal).

        Returns
        -------
        GlobalTrend
        """
        ...

class RBFTestFunctions:
    r"""
    Various RBF test functions.

    3D test functions ``f1_3d``-``f8_3d`` are implemented from [1].

    # References
    1. Bozzini, Mira & Rossini, Milvia. (2002). *Testing methods for 3D scattered data interpolation.* 20, 111-135.
    """

    @classmethod
    def franke_2d(self, xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Franke's two-dimensional test function:

        $$
        \begin{aligned}
        F(x,y) &= 
        \tfrac{3}{4}\exp\!\left[
            -\frac{(9x-2)^2 + (9y-2)^2}{4}
        \right] \\[6pt]
        &\quad+ \tfrac{3}{4}\exp\!\left[
            -\frac{(9x+1)^2}{49}
            -\frac{(9y+1)^2}{10}
        \right] \\[6pt]
        &\quad+ \tfrac{1}{2}\exp\!\left[
            -\frac{(9x-7)^2 + (9y-3)^2}{4}
        \right] \\[6pt]
        &\quad- \tfrac{1}{5}\exp\!\left[
            -(9x-4)^2 - (9y-7)^2
        \right]
        \end{aligned}
        $$

        Parameters
        ----------
        xy : (N, 2) float64 ndarray
            Points in the unit square ``[0, 1]^2``.

        Returns
        -------
        (N, 1) float64 ndarray
            Function values at the input points.
        """
        ...

    @classmethod
    def f1_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        3D Franke-like test function:

        $$
        \begin{aligned}
        F(x,y,z) &= 
        \tfrac{3}{4}\exp\!\left[
            -\frac{(9x-2)^2 + (9y-2)^2 + (9z-2)^2}{4}
        \right] \\[6pt]
        &\quad+ \tfrac{3}{4}\exp\!\left[
            -\frac{(9x+1)^2}{49}
            -\frac{(9y+1)^2}{10}
            -\frac{(9z+1)^2}{10}
        \right] \\[6pt]
        &\quad+ \tfrac{1}{2}\exp\!\left[
            -\frac{(9x-7)^2 + (9y-3)^2 + (9z-5)^2}{4}
        \right] \\[6pt]
        &\quad- \tfrac{1}{5}\exp\!\left[
            -(9x-4)^2 - (9y-7)^2 - (9z-5)^2
        \right]
        \end{aligned}
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
            Function values at the input points.
        """
        ...

    @classmethod
    def f2_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        $$
        F(x,y,z) = 
        \frac{
            \tanh(9z - 9x - 9y) + 1
        }{
            9
        }
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f3_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        $$
        F(x,y,z) =
        \frac{
            \cos(6z)\,\bigl(1.25 + \cos(5.4y)\bigr)
        }{
            6 + 6(3x - 1)^2
        }
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f4_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        $$
        F(x,y,z) =
        \frac{1}{3}\,
        \exp\!\left[
            -\frac{81}{16}
            \bigl(
                (x-\tfrac{1}{2})^2 +
                (y-\tfrac{1}{2})^2 +
                (z-\tfrac{1}{2})^2
            \bigr)
        \right]
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f5_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        $$
        F(x,y,z) =
        \frac{1}{3}\,
        \exp\!\left[
            -\frac{81}{4}
            \bigl(
                (x-\tfrac{1}{2})^2 +
                (y-\tfrac{1}{2})^2 +
                (z-\tfrac{1}{2})^2
            \bigr)
        \right]
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f6_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        $$
        F(x,y,z) =
        \frac{
            \sqrt{
                64 -
                81\bigl[
                    (x-\tfrac{1}{2})^2 +
                    (y-\tfrac{1}{2})^2 +
                    (z-\tfrac{1}{2})^2
                \bigr]
            }
        }{
            9
        }
        - \tfrac{1}{2}
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f7_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Sigmoidal test function:

        $$
        F(x,y,z) =
        \frac{
            1
        }{
            \sqrt{
                1 + 2\exp\!\bigl(
                    -3\bigl(\sqrt{x^2 + y^2 + z^2} - 6.7\bigr)
                \bigr)
            }
        }
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

    @classmethod
    def f8_3d(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""
        Peak function (independent of ``z``):

        $$
        \begin{aligned}
        F(x,y,z) &= 
        50\,\exp\!\left[
            -200\bigl((x-0.3)^2 + (y-0.3)^2\bigr)
        \right] \\[6pt]
        &\quad+ \exp\!\left[
            -50\bigl((x-0.5)^2 + (y-0.5)^2\bigr)
        \right]
        \end{aligned}
        $$

        Parameters
        ----------
        xyz : (N, 3) float64 ndarray
            Points in the unit cube ``[0, 1]^3``.

        Returns
        -------
        (N, 1) float64 ndarray
        """
        ...

class RBFInterpolator:
    """
    Radial basis function (RBF) interpolator.

    An ``RBFInterpolator`` represents a fitted RBF model built from input
    data points, their associated values, and a chosen kernel. Once
    constructed, it can be used to evaluate interpolated values at new
    locations, or serialized for later reuse.

    The interpolator stores:

    - The original input points and values.  
    - The solved RBF and polynomial coefficients.  
    - Kernel settings and solver parameters used during fitting.  
    - Optional global trend transforms (e.g. anisotropy/scaling/rotation).  
    - An optional Fast Multipole Method (FMM) tree evaluator for efficient queries.  
    """
    def __init__(
        self,
        points: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        interpolant_settings: InterpolantSettings,
        params: Optional[Params] = None,
        global_trend: Optional[GlobalTrend] = None,
        progress_callback: Optional[Progress] = None,
    ) -> RBFInterpolator:
        """
        Parameters
        ----------
        points : npt.NDArray[np.float64]
            Coordinates of the input data points with shape (N, D), where N is the number of points
            and D is the dimensionality.
        values : npt.NDArray[np.float64]
            Observed values at the input source point locations with shape (N, M), where N
            is the number of points and M is the number of columns of observed values to solve for.
        interpolant_settings : InterpolantSettings
            Settings used to configure the interpolator.
        params : Optional[Params], optional
            Solver and algorithm parameters, by default None
        global_trend : Optional[GlobalTrend], optional
            Optional global trend transform (anisotropy / rotation), by default None
        progress_callback : Optional[Progress]
            Optional callback for reporting solver progress, by default None            
        """
        ...

    def evaluate(
        self,
        targets: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the interpolant at `target_points` using a **one-shot** FMM evaluator.

        This is the most convenient way to evaluate a single batch: it builds a
        temporary FMM tree, evaluates, and discards the evaluator. If a
        `global_trend` is present, the target points are transformed for evaluation.

        Extents are computed as the **union** of the source and target point
        bounding boxes to ensure all targets can be assigned to tree boxes.

        Parameters
        ----------
        targets : npt.NDArray[np.float64]
            Coordinates of the target data points with shape (N, D), where N is the number of points
            and D is the dimensionality.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of interpolated values with shape (N, M), where N is the number of target points
            and M is the number of columns of values interpolated.
        """
        ...

    def evaluate_at_source(self, add_nugget: Optional[bool] = False) -> npt.NDArray[np.float64]:
        """Evaluate the interpolant **at the original source points**.

        Useful for **convergence checks** and diagnostics.

        Parameters
        ----------
        add_nugget : Optional[bool], optional
            Whether to add the nugget effect back to the result, by default False
            
            When `add_nugget = True`, the diagonal “nugget” term is added back so the evaluated
            values should match the input samples to within the solver's tolerance (undoing any
            smoothing from the nugget).
            
            When `add_nugget = False`, you observe the smoothed/regularised fit.
            
        Returns
        -------
        npt.NDArray[np.float64]
            Array of interpolated values with shape (N, M), where N is the number of source points
            and M is the number of columns of values interpolated.
        
        Notes
        -----
        This path uses a sparse/leaf-only evaluation strategy optimized for source-point queries.
        """
        ...

    def build_evaluator(self, extents: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Build and store an FMM evaluator for **repeated evaluations**.

        Use this when you'll call [`evaluate_targets`][ferreus_rbf.RBFInterpolator.evaluate_targets] many times (e.g. during
        isosurfacing or interactive probing). The evaluator is constructed once and
        saved inside the interpolator.

        Parameters
        ----------
        extents : Optional[npt.NDArray[np.float64]], optional
            AABB extents to build the evaluator `[min_0.., max_0..]`, by default None.
            If None, extents are derived from the (transformed, if applicable) source points.
        """
        ...

    def evaluate_targets(
        self,
        targets: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate using the **stored** evaluator built by [`build_evaluator`][ferreus_rbf.RBFInterpolator.build_evaluator].

        This is the fast path for repeated calls. If a `global_trend` is present,
        target points are transformed consistently with the stored evaluator.

        Panics
        ------

        - If called before [`build_evaluator`][ferreus_rbf.RBFInterpolator.build_evaluator].
        - If any `target_points` lie **outside** the extents used to build the evaluator.

        Parameters
        ----------
        targets : npt.NDArray[np.float64]
            Coordinates of the target data points with shape (N, D), where N is the number of points
            and D is the dimensionality.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of interpolated values with shape (N, M), where N is the number of target points
            and M is the number of columns of values interpolated.
        """
        ...

    def build_isosurfaces(
        self,
        extents: npt.NDArray[np.float64],
        resolution: float,
        isovalues: list[float],
    ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.uintp]]]:
        """Build 3D isosurfaces using a **surface-following, non-adaptive Surface Nets** method.

        The sampling `resolution` controls grid density; choose it relative to the
        data scale and desired detail. Multiple `isovalues` may be provided; each
        produces a separate surface.

        Seed cells are selected from samples within `resolution` of an isovalue.
        If no seeds are found for a given isovalue, the corresponding entry is
        empty.

        !!! warning "Surface quality"
            The current isosurface extraction method does **not** guarantee
            manifold or valid meshes; surfaces may contain trifurcations or self-intersections.

            Surfaces may therefore not be suitable for downstream boolean operations.

        Parameters
        ----------
        extents : npt.NDArray[np.float64]
            evaluation domain `[minx, miny, minz, maxx, maxy, maxz]`
        resolution : float
            grid step in world units.
        isovalues : list[float]
            list of scalar levels to extract.

        Returns
        -------
        tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.uintp]]]
            `(points_per_iso, faces_per_iso)` where:  

            - `points_per_iso[i]` is a `(V_i, 3)` array of vertex positions for the
              `i`-th isosurface.  
            - `faces_per_iso[i]` is an `(F_i, 3)` integer array of triangle vertex indices.
        """
        ...

    def save_model(
        self,
        path: str,
    ) -> None:
        """Save this interpolator to a **JSON envelope** `{ format, version, model }`.

        The on-disk format is versioned via `JSON_FORMAT_NAME` and `JSON_VERSION`.
        Files produced here are intended to be read back with [`load_model`][ferreus_rbf.RBFInterpolator.load_model].

        Parameters
        ----------
        path : str
            file path to save the model to.

        """
        ...

    @staticmethod
    def load_model(
        path: str,
        progress_callback: Optional[Progress] = None
    ) -> RBFInterpolator: 
        """Load an interpolator from a versioned **JSON envelope**, validating format & version,
        saved using [save_model][ferreus_rbf.RBFInterpolator.save_model].

        If `progress` is `Some`, installs the sink into `self.params.progress_callback`
        on the returned model so subsequent long-running operations (evaluation, surface
        extraction, etc.) can report progress.

        Parameters
        ----------
        path : str
            file path to load the model from.
        progress_callback : Optional[Progress], optional
            Progress callback operator, by default None

        Returns
        -------
        RBFInterpolator
            A solved RBFInterpolator that can be used for evaluations and surfacing.
        """
        ...

    def source_points(
        self,
    ) -> npt.NDArray[np.float64]:
        """Access the stored source points from the interpolator

        Returns
        -------
        npt.NDArray[np.float64]
            Array of the source points with shape (N, D), where N is the number of source points
            and D is the dimenstionality.
        """
        ...

    def source_values(
        self,
    ) -> npt.NDArray[np.float64]:
        """Access the stored source values from the interpolator

        Returns
        -------
        npt.NDArray[np.float64]
            Array of interpolated values with shape (N, M), where N is the number of source points
            and M is the number of columns of values.
        """
        ...

def save_obj(
    path: str,
    name: str,
    verts: npt.NDArray[np.float64],
    faces: Union[npt.NDArray[np.uintp], npt.NDArray[np.int64]],
) -> None:
    """
    Save an isosurface to an OBJ file.

    Parameters
    ----------
    path : str
        Output `.obj` path.
    name : str
        Object name written as `o <name>` inside the OBJ.
    verts : (V, 3) float64 ndarray
        Vertex positions.
    faces : (F, 3) uintp or int64 ndarray (0-based)
        Triangle indices (0-based). Converted to OBJ's 1-based indices.
    """
    ...
