'''
/////////////////////////////////////////////////////////////////////////////////////////////
//
// Stubs file for Python bindings of the config module that enables typehints in IDE's.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////
'''

from enum import Enum
from typing import Optional
from ferreus_rbf.interpolant_config import RBFKernelType

class Solvers(Enum):
    """
    Enum for the available iterative solvers.
    """
    DDM = 0
    """Domain Decomposition solver."""
    
    FGMRES = 1
    """Flexible generalised minimal residual method (FGMRES) solver."""

class FmmCompressionType(Enum):
    """
    Enum for the available compression methods for the M2L operators in the
    FMM evaluator.

    """
    None_ = 0
    """No compression applied to M2L operators"""

    SVD = 1
    """A truncated Singular Value Decompositio (SVD) is performed on the M2L operators."""

    ACA = 2
    """Adaptive cross approximation (ACA) is performed on the M2L operators, followed by SVD recompression."""

class DDMParams:
    """
    Parameters controlling construction of the **domain decomposition hierarchy**.
    
    `ferreus_rbf` employs a *domain decomposition preconditioner* to accelerate
    convergence of the iterative RBF solver. The algorithm recursively partitions
    the input point cloud into a hierarchy of overlapping subdomains, within which
    local RBF systems are solved directly and combined to form a global preconditioner.
    
    This class defines the key thresholds and ratios governing how that
    hierarchy is generated - for example, the number of points permitted per
    leaf domain, how much overlap occurs between neighboring subdomains, and
    the scale at which coarse levels are formed.
    
    ### Intended Usage
    This configuration is part of the public API mainly for **developers and
    advanced users** who wish to experiment with or tune the decomposition
    process. For example, increasing subdomain overlap and coarse ratio can
    improve convergence, but at the cost of higher memory usage.
    In general, the default values have been selected to provide
    a robust trade-off between memory usage and solver performance across
    a wide range of problem sizes.
    
    Default values when [`DDMParams`][ferreus_rbf.config.DDMParams] isn't provided to [`RBFInterpolator`][ferreus_rbf.RBFInterpolator]: 

    - `leaf_threshold`: `1024`  
    - `overlap_quota`: `0.5`  
    - `coarse_ratio`: `0.125`  
    - `coarse_threshold`: `4096`     

    Parameters
    ----------
    leaf_threshold : int
        Target maximum number of points (internal + overlapping) within a leaf domain.
    overlap_quota : float
        Overlap fraction. Larger fraction will add more overlapping points to each leaf domain.
    coarse_ratio : float
        Fraction of **internal** points per leaf promoted to the next coarser level.
    coarse_threshold : int
        Maximum number of points in the coarsest level.
    """
    def __init__(
        self,
        leaf_threshold: int,
        overlap_quota: float,
        coarse_ratio: float,
        coarse_threshold: int,
    ) -> None: ...

class FmmParams:
    """
    Parameters controlling the **Fast Multipole Method (FMM)** evaluator.
    
    These settings configure the ``ferreus_bbfmm`` backend, which performs
    fast evaluation of RBF interpolants by hierarchically partitioning space
    and approximating long-range interactions through low-rank interpolation
    and optional M2L operator compression.
    
    ### Intended Usage
    This configuration is primarily exposed for **developers and advanced users**
    who wish to experiment with or tune FMM performance. In general, the
    default values have been selected to provide a rubust balance between accuracy,
    memory usage, and computation time across a broad range of problems.
    
    Increasing the interpolation order improves accuracy but also increases
    computational cost. Orders that are too low may stall solver convergence.  

    Default interpolation order:

    - Linear and Spheroidal kernels -> `7`
    - ThinPlateSpline kernel -> `9`
    - Cubic kernel -> `11`  
    
    Default values when [`FmmParams`][ferreus_rbf.config.FmmParams] isn't provided to [`RBFInterpolator`][ferreus_rbf.RBFInterpolator]: 
    
    - `interpolation_order`: *kernel dependent*  
    - `max_points_per_cell`: `256`  
    - `compression_type`: [`FmmCompressionType.ACA`][ferreus_rbf.config.FmmCompressionType.ACA]  
    - `epsilon`: `10^(-interpolation_order)`  
    - `eval_chunk_size`: `1024`  

    Parameters
    ----------
    interpolation_order : int
        Number of Chebyshev interpolation nodes per dimension.
    max_points_per_cell : int
        Maximum number of points per cell before it is subdivided.
    compression_type : FmmCompressionType
        What type of compression to apply to the M2L operators.
    epsilon : float
        Tolerance threshold for M2L compression.
    eval_chunk_size : int
        Number of target points to evaluate in each chunk.
    """
    def __init__(
        self,
        interpolation_order: int,
        max_points_per_cell: int,
        compression_type: FmmCompressionType,
        epsilon: float,
        eval_chunk_size: int,
    ) -> None: ...

class Params:
    """
    Configuration parameters controlling how an RBF system is solved.

    A ``Params`` instance specifies solver options, accuracy targets,
    domain decomposition behaviour, fast multipole settings, and other
    controls for model fitting and evaluation.

    Default values when [`Params`][ferreus_rbf.config.Params] isn't provided to [`RBFInterpolator`][ferreus_rbf.RBFInterpolator]: 

    - `solver_type`: [`Solvers.FGMRES`][ferreus_rbf.config.Solvers.FGMRES]   
    - `ddm_params`: Default DDMParams  
    - `fmm_params`: Default FmmParams  
    - `naive_solve_threshold`: `4096`  
    - `test_unique`: `true`  
    """
    def __init__(
        self,
        kernel_type: RBFKernelType,
        solver_type: Optional[Solvers] = None,
        ddm_params: Optional[DDMParams] = None,
        fmm_params: Optional[FmmParams] = None,
        naive_solve_threshold: Optional[int] = None,
        test_unique: Optional[bool] = None,
    ) -> None: 
        """
        Parameters
        ----------
        kernel_type : RBFKernel
            The [`RBFKernel`][ferreus_rbf.interpolant_config.RBFKernelType] variant being used.
        solver_type : Optional[Solvers]
            Iterative solver to use when fitting the RBF system.
        ddm_params : Optional[DDMParams]
            Parameters controlling domain decomposition preconditioning.
        fmm_params : Optional[FmmParams]
            Parameters controlling the fast multipole method (FMM).
        naive_solve_threshold : Optional[int]
            Threshold below which the system is solved directly rather than using iterative methods.
        test_unique : Optional[bool]
            Whether to test for and remove duplicate source points. This is highly recommended, as in order to ensure a unique solution to the RBF, the source points must be unique.
        """
        ...