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

from typing import Union, Callable, Optional
import numpy as np
import numpy.typing as npt
from ferreus_rbf.progress import Progress

def surface_nets(
    extents: npt.NDArray[np.float64],
    resolution: float,
    isovalue: float,
    surface_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    seed_points: npt.NDArray[np.float64],
    seed_values: npt.NDArray[np.float64],
    progress_callback: Optional[Progress] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uintp]]:
    """Extract an isosurface using a surface-following surface nets implementation.

    Parameters
    ----------
    extents : npt.NDArray[np.float64]
        AABB extents as a 1D numpy array in order of [minx, miny, minz, maxx, maxy, maxz].
    resolution : float
        The isosurfacing resolution.
    isovalue : float
        The value at which to extract the isosurface.
    surface_fn : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        Callable function to evaluate the cell points at isosurface extraction.
        Must take in a 2D numpy array of float64 3D point coordinates of shape (N, 3) and
        return a 2D numpy array of float64 values of shape (N, 1), where N is the number of points
        being evaluated.
    seed_points : npt.NDArray[np.float64]
        Numpy array of points of shape (N, 3), where N is the number of seed points, to seed the
        isosurface extraction. The algorithm is most efficient when these points are on, or close
        to, the surface to be extacted, as it reduces the number of cells that need to be evaluated
        in order to extract the surface.
    seed_values : npt.NDArray[np.float64]
        Numpy array of values of shape (N, 1), where N is the number of seed points, to seed the
        isosurface extraction. Seed points are filtered to only use seed points that have values that
        are within resolution of the isosurface value. 
    progress_callback : Progress | None, optional
        Progress callback operator, by default None

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.uintp]]
        Numpy arrays of shape (N, 3) representing the verticesa and faces of isosurface.
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