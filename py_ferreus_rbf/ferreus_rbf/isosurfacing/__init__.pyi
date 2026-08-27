'''
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Stubs file for Python bindings of the isosurfacing module that enables typehints in IDE's.
//
// Created on: 07 April 2026     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////
'''

from typing import Callable, Optional
from enum import Enum
import numpy as np
import numpy.typing as npt
from ferreus_rbf.progress import Progress

class ClusterMethod(Enum):
    """Vertex clustering mode."""

    None_ = 0
    """No vertex clustering is performed. Raw marching tetrahedra triangles are produced."""
    Average = 1
    """Cluster topology-compatible intersections to their arithmetic mean."""

    CurvatureWeighted = 2
    """Cluster topology-compatible intersections using the local curvature estimate."""

class BoundaryClosure(Enum):
    """Boundary closure mode for isosurfaces."""

    None_ = 0
    """Do not add cap triangles; leave the clipped surface open."""
    ClosePositive = 1
    """Close the surface as if values outside the AABB are above the isovalue."""
    CloseNegative = 2
    """Close the surface as if values outside the AABB are below the isovalue."""

class Mesh:
    """Triangle mesh returned by isosurface extraction."""

    @staticmethod
    def save_obj(self, path: str, name: str) -> None: 
        """Save this mesh to a Wavefront OBJ file.

        Parameters
        ----------
        path : str
            File path to save the obj to.
        name : str
            Object name for the mesh.
        """
        ...

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Vertex coordinates with shape (V, 3).

        Returns
        -------
        npt.NDArray[np.float64]
            Array of the vertices.
        """
        ...

    @property
    def facets(self) -> npt.NDArray[np.uintp]:
        """Triangle vertex indices with shape (F, 3).

        Returns
        -------
        npt.NDArray[np.uintp]
            Array of triangles.
        """
        ...


def build_isosurface(
    seed_points: npt.NDArray[np.float64],
    extents: npt.NDArray[np.float64],
    resolution: float,
    isovalue: float,
    isosurface_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    *,
    gradient_fn: Optional[
        Callable[
            [npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ]
    ] = None,
    cluster_method: Optional[ClusterMethod] = ClusterMethod.CurvatureWeighted,
    boundary_closure: Optional[BoundaryClosure] = BoundaryClosure.None_,
    progress_callback: Optional[Progress] = None,
) -> Mesh:
    """Extract an isosurface using regularised marching tetrahedra.

    Parameters
    ----------
    seed_points : npt.NDArray[np.float64]
        Numpy array of points of shape (N, 3), where N is the number of seed points, to seed the
        isosurface extraction. The algorithm is most efficient when these points are on, or close
        to, the surface to be extacted, as it reduces the number of cells that need to be evaluated
        in order to extract the surface.    
    extents : npt.NDArray[np.float64]
        AABB extents as a 1D numpy array in order of [minx, miny, minz, maxx, maxy, maxz].
    resolution : float
        The isosurfacing resolution.
    isovalue : float
        The value at which to extract the isosurface.
    isosurface_fn : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        Callable function to evaluate the cell points at isosurface extraction.
        Must take in a 2D numpy array of float64 3D point coordinates of shape (N, 3) and
        return a float64 values array of shape (N,) or (N, 1), where N is the number of points
        being evaluated.
    gradient_fn : Callable | None, optional
        Optional callable for evaluating values and gradients during seed projection. It must
        take a float64 array of point coordinates with shape (N, 3) and return a tuple of
        ``(values, gradients)``. Values must have shape (N,) or (N, 1), and gradients must have
        shape (N, 3).
        If not provided central-differences will be used to calculate gradients.
    cluster_method: ClusterMethod | ClusterMethod.CurvatureWeighted, optional
        Vertex clustering method to use. If not provided, CurvatureWeighted will be used.
    boundary_closure : BoundaryClosure | BoundaryClosure.None_, optional
        Boundary closure method to use. If None_, leaves clipped boundaries open.
    progress_callback : Progress | None, optional
        Progress callback operator, by default None

    Returns
    -------
    Mesh
        Extracted triangle mesh.
    """
    ...

def build_isosurfaces(
    seed_points: npt.NDArray[np.float64],
    extents: npt.NDArray[np.float64],
    resolution: float,
    isovalues: list[float],
    isosurface_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    *,
    gradient_fn: Optional[
        Callable[
            [npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ]
    ] = None,
    cluster_method: Optional[ClusterMethod] = ClusterMethod.CurvatureWeighted,
    boundary_closure: Optional[BoundaryClosure] = BoundaryClosure.None_,
    progress_callback: Optional[Progress] = None,
) -> list[Mesh]:
    """Convenience wrapper for [`build_isosurface`][ferreus_rbf.isosurfacing.build_isosurface] that
    can extract multiple meshes from a list of isovalues at once.

    Parameters
    ----------
    seed_points : npt.NDArray[np.float64]
        Numpy array of points of shape (N, 3), where N is the number of seed points, to seed the
        isosurface extraction. The algorithm is most efficient when these points are on, or close
        to, the surface to be extacted, as it reduces the number of cells that need to be evaluated
        in order to extract the surface.    
    extents : npt.NDArray[np.float64]
        AABB extents as a 1D numpy array in order of [minx, miny, minz, maxx, maxy, maxz].
    resolution : float
        The isosurfacing resolution.
    isovalues : list[float]
        List of values at which to extract isosurfaces.
    isosurface_fn : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        Callable function to evaluate the cell points at isosurface extraction.
        Must take in a 2D numpy array of float64 3D point coordinates of shape (N, 3) and
        return a float64 values array of shape (N,) or (N, 1), where N is the number of points
        being evaluated.
    gradient_fn : Callable | None, optional
        Optional callable for evaluating values and gradients during seed projection. It must
        take a float64 array of point coordinates with shape (N, 3) and return a tuple of
        ``(values, gradients)``. Values must have shape (N,) or (N, 1), and gradients must have
        shape (N, 3).
        If not provided central-differences will be used to calculate gradients.
    cluster_method: ClusterMethod | ClusterMethod.CurvatureWeighted, optional
        Vertex clustering method to use. If not provided, CurvatureWeighted will be used.
    boundary_closure : BoundaryClosure | BoundaryClosure.None_, optional
        Boundary closure method to use. If None_, leaves clipped boundaries open.
    progress_callback : Progress | None, optional
        Progress callback operator, by default None

    Returns
    -------
    list[Mesh]
        Extracted triangle meshes for each isovalue.
    """
    ...    
