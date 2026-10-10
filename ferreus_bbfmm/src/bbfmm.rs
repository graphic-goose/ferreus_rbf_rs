/////////////////////////////////////////////////////////////////////////////////////////////
//
// Implements the core Black Box Fast Multipole Method (BBFMM) tree and evaluation routines.
//
// Created on: 15 Nov 2025     Author: Daniel Owen
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

use crate::{
    EvaluationTargets, TargetGrid, chebyshev, linear_tree, morton, traits::KernelFunction, utils,
};
use faer::mat::AsMatRef;
use faer::{Mat, MatRef};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::sync::Arc;

/// Errors that can occur during FMM tree operations.
#[derive(Debug)]
pub enum FmmError {
    /// A target point could not be assigned to any cell in the tree
    /// because it lies outside the tree extents.
    PointOutsideTree {
        point_index: usize,
    },

    /// Gradient evaluation was requested but the kernel does not provide a gradient implementation.
    KernelDoesNotSupportGradients,

    InvalidTargets(&'static str),
}

impl fmt::Display for FmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmmError::PointOutsideTree { point_index } => write!(
                f,
                "FMM evaluation failed: target point at row {} lies outside the tree extents",
                point_index
            ),
            FmmError::KernelDoesNotSupportGradients => write!(
                f,
                "FMM evaluation failed: gradient evaluation requested but kernel does not support gradients"
            ),
            FmmError::InvalidTargets(reason) => {
                write!(f, "Invalid evaluation targets: {reason}")
            }
        }
    }
}

impl std::error::Error for FmmError {}

/// Supported spatial dimensions for the `FmmTree`.
///
/// Each dimension corresponds to a specific type of tree:
/// - `One` (1D): Binary tree  
/// - `Two` (2D): Quadtree  
/// - `Three` (3D): Octree
#[derive(Debug, Copy, Clone)]
pub enum Dimensions {
    One = 1,
    Two = 2,
    Three = 3,
}

/// Supported M2L operator compression methods.
#[derive(Debug, Copy, Clone)]
pub enum M2LCompressionType {
    /// No compression is applied.
    None,

    /// A truncated Singular Value Decompositio (SVD) is
    /// performed on the M2L operators.
    SVD,

    /// Adaptive cross approximation (ACA), followed by SVD recompression
    /// is performed on the M2L operators.
    ACA,
}

/// Optional parameters for tuning the FMM performance.
#[derive(Debug, Copy, Clone)]
pub struct FmmParams {
    /// Maximum number of points per cell before it must be subdivided.
    /// When FmmParams is not provided the default value is 256.
    pub max_points_per_cell: usize,

    /// The type of compression to apply to the M2L operators.
    /// When FmmParams is not provided the default value is ACA.
    pub compression_type: M2LCompressionType,

    /// Tolerance threshold for M2L compression.
    /// When FmmParams is not provided the default value is 10^-interpolation_order
    pub epsilon: f64,

    /// Number of target points to evaluate in each chunk.
    /// When FmmParams is not provided the default value is 1024.
    pub eval_chunk_size: usize,
}

impl FmmParams {
    pub fn new_defaults(interpolation_order: usize) -> Self {
        Self {
            max_points_per_cell: 256,
            compression_type: M2LCompressionType::ACA,
            epsilon: 10f64.powi(-(interpolation_order as i32)),
            eval_chunk_size: 1024,
        }
    }
}

#[derive(Debug)]
pub enum LeafTargetIndices {
    Points(Vec<usize>),
    Grid {
        ranges: Vec<std::ops::Range<usize>>,
        count: usize,
    },
}

impl LeafTargetIndices {
    fn len(&self) -> usize {
        match self {
            Self::Points(indices) => indices.len(),
            Self::Grid { count, .. } => *count,
        }
    }

    /// Resolve a leaf-local ordinal to a global target index.
    fn target_index(&self, ordinal: usize, targets: EvaluationTargets<'_>) -> usize {
        match self {
            Self::Points(indices) => indices[ordinal],
            Self::Grid { ranges, .. } => match targets {
                EvaluationTargets::Grid(grid) => grid.target_index_in_ranges(ranges, ordinal),
                EvaluationTargets::Points(_) => {
                    unreachable!("grid leaf ranges require grid targets")
                }
            },
        }
    }

    /// Iterate global indices without allocating an index list.
    fn iter<'a>(&'a self, targets: EvaluationTargets<'a>) -> impl Iterator<Item = usize> + 'a {
        (0..self.len()).map(move |ordinal| self.target_index(ordinal, targets))
    }

    /// Borrow point chunks; generate grid indices one chunk at a time.
    fn chunks<'a>(
        &'a self,
        targets: EvaluationTargets<'a>,
        chunk_size: usize,
    ) -> impl Iterator<Item = std::borrow::Cow<'a, [usize]>> + 'a {
        assert!(chunk_size > 0, "target chunk size must be positive");

        let count = self.len();

        (0..count).step_by(chunk_size).map(move |start| {
            let end = start + (count - start).min(chunk_size);

            match self {
                Self::Points(indices) => std::borrow::Cow::Borrowed(&indices[start..end]),
                Self::Grid { .. } => std::borrow::Cow::Owned(
                    (start..end)
                        .map(|ordinal| self.target_index(ordinal, targets))
                        .collect(),
                ),
            }
        })
    }
}

/// Represents the tree and interaction lists used in the Fast Multipole Method (FMM).
///
/// Contains Morton-encoded spatial cells, their hierarchical relationships, and mappings for
/// source/target points and interaction lists used in FMM evaluations.
#[derive(Debug)]
pub struct TreeLists {
    /// All Morton codes in the tree (leaf and non-leaf cells).
    pub tree: HashSet<u64>,

    /// Morton codes for all leaf cells.
    pub leaves: HashSet<u64>,

    /// Mapping from each cell to its children.
    pub children: HashMap<u64, Vec<u64>>,

    /// Mapping from each leaf cell to adjacent leaf cells (P2P).
    pub u_lists: HashMap<u64, HashSet<u64>>,

    /// Mapping from each cell to non-adjacent children of parent's colleagues (M2L).
    pub v_lists: HashMap<u64, HashSet<u64>>,

    /// Maps tree level to Morton codes at that level.
    pub level_cells_map: HashMap<u64, Vec<u64>>,

    /// Maps Morton code to global index.
    pub key_to_index_map: HashMap<u64, usize>,

    /// Maps source-containing cells to packed multipole indices.
    pub source_key_to_index_map: HashMap<u64, usize>,

    /// Maps leaf cells to source point indices they contain.
    pub leaf_source_indices: HashMap<u64, Vec<usize>>,

    /// Maps leaf cells to target point indices they contain.
    pub leaf_target_indices: HashMap<u64, LeafTargetIndices>,
}

/// Stores precomputed operators and metadata used for fast kernel approximations in the FMM.
///
/// Includes node locations, polynomial interpolation nodes, transfer matrices, and compressed
/// low-rank representations (via ACA and truncated SVD) for M2L interactions.
#[derive(Debug)]
pub struct PrecomputeOperators {
    /// Total number of interpolation nodes in all dimensions (n^d).
    pub num_nodes_nd: usize,

    /// Subset of nodes used for polynomial projection and evaluation.
    pub polynomial_nodes: Mat<f64>,

    /// One-dimensional child-to-parent M2M transfer operators.
    ///
    /// The lower and upper child operators are stored as adjacent p × p
    /// column blocks in a p × 2p matrix: [lower | upper]
    pub m2m_transfer_operators: Mat<f64>,

    /// Left factors from SVD compression of M2L interaction matrices, indexed by level and source/target offsets.
    pub u: HashMap<usize, HashMap<usize, Mat<f64>>>,

    /// Right factors from SVD compression of M2L interaction matrices, indexed by level and source/target offsets.
    pub vt: HashMap<usize, HashMap<usize, Mat<f64>>>,

    /// Permutation indices applied during M2L reordering.
    pub permutation_indices: Vec<Vec<usize>>,

    /// Inverse permutations to restore original ordering.
    pub inverse_permutations: Vec<Vec<usize>>,

    /// Lookup indices that map the M2L vector to the required permutation indices for the relevant reference vector.
    pub permutation_lookups: Vec<usize>,

    /// Lookup indices that map M2L vector to reference vector.
    pub reference_vector_lookups: Vec<usize>,
}

/// A Fast Multipole Method (FMM) tree that organises source points into a hierarchical spatial
/// structure to accelerate kernel summation tasks.
///
/// The tree is adaptively refined, with optional sparse leaf pruning.
/// It efficiently precomputes all operators (M2M and M2L) required for far-field approximation.
///
/// The generic parameter `K` must implement [`KernelFunction`]
#[derive(Debug)]
pub struct FmmTree<K: KernelFunction> {
    /// Source point locations used to build the tree.
    ///
    /// Expected to be an Arc wrapped [`faer::Mat<f64>`](https://docs.rs/faer/latest/faer/mat/type.Mat.html)
    /// with shape (N, D), where N is the number of points and D is the dimensionality.
    pub source_points: Arc<Mat<f64>>,

    /// Number of Chebyshev interpolation nodes per dimension.
    interpolation_order: usize,

    /// The kernel function used for interaction computations.
    kernel: K,

    /// Center of the root cell’s bounding box.
    center: Vec<f64>,

    /// Half the length of the root cell’s bounding box.
    radius: f64,

    /// Number of right-hand sides to evaluate.
    nrhs: usize,

    /// Spatial dimensionality of the tree (1D, 2D, or 3D).
    dimensions: Dimensions,

    /// Maximum number of points allowed in a leaf cell before subdivision.
    max_points_per_cell: usize,

    /// Maximum tree depth (i.e., number of refinement levels).
    depth: u64,

    /// Tree structure and interaction lists (u, v) built during setup
    tree_lists: TreeLists,

    /// Precomputed interpolation and low-rank approximation operators for fast evaluation.
    precompute_operators: PrecomputeOperators,

    /// Multipole coefficients for each cell in the tree of shape (N, S x K), where N is the
    /// number of Chebyshev nodes in all dimensions, S is the number of source-containing cells in
    /// the tree and K is the number of right-hand sides of source/target values.
    multipole_coefficients: Mat<f64>,

    /// Local coefficients for each cell in the tree of shape (N, M x K), where N is the
    /// number of Chebyshev nodes in all dimensions, M is the number of cells in the tree and
    /// K is the number of right-hand sides of source/target values.
    local_coefficients: Mat<f64>,

    /// Whether to store empty leaf nodes or not.
    sparse_tree: bool,

    /// Low-rank compression type for M2L interactions.
    compression_type: M2LCompressionType,

    /// Chunk size for evaluating batches of target points during leaf pass.
    eval_chunk_size: usize,

    /// Tolerance for compression of M2L operators.
    epsilon: f64,
}

impl<K: KernelFunction + Send + Sync> FmmTree<K> {
    /// Constructs a new [`FmmTree`] from the given source points and parameters.
    ///
    /// # Arguments
    /// * `source_points`: Arc wrapped input matrix of shape (N, D), where N is the number of points, D is the dimensionality.
    /// * `interpolation_order`: Number of Chebyshev nodes per dimension.
    /// * `kernel_function`: Kernel function used for evaluating interactions.
    ///    Must implement [`KernelFunction`]
    /// * `sparse`: If `true`, constructs a sparse tree that omits empty leaves.
    /// * `extents`: Optional bounding box `[mins..., maxs...]`; if `None`, computed from data.
    /// * `params`: Optional parameters for tuning the FMM performance.
    ///
    /// # Returns
    /// * A fully initialised [`FmmTree`] with all data structures allocated and tree built.    
    pub fn new(
        source_points: Arc<Mat<f64>>,
        interpolation_order: usize,
        kernel: K,
        sparse: bool,
        extents: Option<Vec<f64>>,
        params: Option<FmmParams>,
    ) -> Self {
        Self::new_with_targets(
            source_points,
            interpolation_order,
            kernel,
            sparse,
            extents,
            params,
            None,
        )
    }
    /// Constructs a new [`FmmTree`] from the given source points and parameters.
    ///
    /// # Arguments
    /// * `source_points`: Arc wrapped input matrix of shape (N, D), where N is the number of points, D is the dimensionality.
    /// * `interpolation_order`: Number of Chebyshev nodes per dimension.
    /// * `kernel_function`: Kernel function used for evaluating interactions.
    ///    Must implement [`KernelFunction`]
    /// * `sparse`: If `true`, constructs a sparse tree that omits empty leaves.
    /// * `extents`: Optional bounding box `[mins..., maxs...]`; if `None`, computed from data.
    /// * `params`: Optional parameters for tuning the FMM performance.
    /// * `targets`: Optional targets used in refinment when building the tree when `sparse`=false.
    ///
    /// # Returns
    /// * A fully initialised [`FmmTree`] with all data structures allocated and tree built.
    pub fn new_with_targets(
        source_points: Arc<Mat<f64>>,
        interpolation_order: usize,
        kernel: K,
        sparse: bool,
        extents: Option<Vec<f64>>,
        params: Option<FmmParams>,
        targets: Option<EvaluationTargets<'_>>,
    ) -> Self {
        let tree_extents = match extents.is_some() {
            true => extents.unwrap().clone(),
            false => utils::get_pointarray_extents(&source_points),
        };

        let fmm_params = match params.is_some() {
            true => params.unwrap(),
            false => FmmParams::new_defaults(interpolation_order),
        };

        let dim = tree_extents.len() / 2;

        let dimensions = match dim {
            1 => Dimensions::One,
            2 => Dimensions::Two,
            3 => Dimensions::Three,
            _ => panic!("Unsupported number of dimensions: {}", dim),
        };

        let (center, radius) = morton::calculate_tree_center_and_radius(&tree_extents);

        let tree_lists = TreeLists {
            tree: HashSet::default(),
            leaves: HashSet::default(),
            children: HashMap::default(),
            u_lists: HashMap::default(),
            v_lists: HashMap::default(),
            level_cells_map: HashMap::default(),
            key_to_index_map: HashMap::default(),
            source_key_to_index_map: HashMap::default(),
            leaf_source_indices: HashMap::default(),
            leaf_target_indices: HashMap::default(),
        };

        let precompute_operators = PrecomputeOperators {
            num_nodes_nd: usize::default(),
            polynomial_nodes: Mat::new(),
            m2m_transfer_operators: Mat::new(),
            u: HashMap::default(),
            vt: HashMap::default(),
            permutation_indices: Vec::default(),
            inverse_permutations: Vec::default(),
            permutation_lookups: Vec::default(),
            reference_vector_lookups: Vec::default(),
        };

        let mut tree = Self {
            source_points,
            interpolation_order,
            kernel,
            center,
            radius,
            nrhs: 1usize,
            dimensions,
            max_points_per_cell: fmm_params.max_points_per_cell,
            depth: 0,
            tree_lists,
            precompute_operators,
            multipole_coefficients: Mat::<f64>::new(),
            local_coefficients: Mat::<f64>::new(),
            sparse_tree: sparse,
            compression_type: fmm_params.compression_type,
            eval_chunk_size: fmm_params.eval_chunk_size,
            epsilon: fmm_params.epsilon,
        };

        tree.build_tree(targets);

        tree
    }

    fn build_tree(&mut self, targets: Option<EvaluationTargets<'_>>) {
        self.tree_lists = linear_tree::build_tree(
            self.source_points.as_mat_ref(),
            &self.center,
            self.radius,
            self.max_points_per_cell,
            !self.sparse_tree,
            &mut self.depth,
            self.dimensions,
            self.interpolation_order,
            targets,
        );

        self.precompute_operators = chebyshev::precompute_approximation_operators(
            self.interpolation_order,
            self.dimensions as usize,
            self.radius,
            self.depth,
            &self.kernel,
            &self.compression_type,
            self.epsilon,
        );
    }

    /// Performs an upward pass of the tree to set the multipole coefficients.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values)
    pub fn set_weights(&mut self, weights: MatRef<f64>) {
        self.nrhs = weights.ncols();
        self.reset_multipole_coefficients();

        self.upward_pass(&weights);
    }

    /// Performs a downward pass of the tree to set the local coefficients and
    /// then performs a leaf evaluation pass to evaluate the values at the
    /// target locations.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values)
    /// * `target points`: Matrix of shape (N, D), where N is the number of target points and D is the dimensionality.
    pub fn evaluate(
        &mut self,
        weights: MatRef<f64>,
        target_points: MatRef<f64>,
    ) -> Result<Mat<f64>, FmmError> {
        let (target_values, _) =
            self._eval::<false>(weights, EvaluationTargets::Points(target_points))?;
        Ok(target_values)
    }

    /// Performs a downward pass of the tree to set the local coefficients and
    /// then performs a leaf evaluation pass to evaluate the values and gradients at the
    /// target locations.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values)
    /// * `target points`: Matrix of shape (N, D), where N is the number of target points and D is the dimensionality.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of target points and K is the number of right-hand sides evaluated.
    /// * `gradients` : Matrix of shape `(M, K * D)` where M is the number of target points, K is the number of right-hand-sides evaluated
    ///     and D is the dimensionality.
    ///     Gradient columns are laid out as `[rhs0_dx, rhs0_dy, rhs0_dz, rhs1_dx, ...]`, truncated to the tree dimensionality.    
    pub fn evaluate_with_gradients(
        &mut self,
        weights: MatRef<f64>,
        target_points: MatRef<f64>,
    ) -> Result<(Mat<f64>, Mat<f64>), FmmError> {
        let (target_values, gradients) =
            self._eval::<true>(weights, EvaluationTargets::Points(target_points))?;
        Ok((target_values, gradients.unwrap()))
    }

    /// Number of targets in each bounded evaluation chunk.
    pub fn target_chunk_size(&self) -> usize {
        self.eval_chunk_size
    }

    /// Performs a downward pass of the tree to set the local coefficients and
    /// then performs a leaf evaluation pass to evaluate the values at the
    /// grid target locations.
    ///
    /// Call [`FmmTree::set_weights`] before evaluating, and repeat that step whenever the weights change.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
    /// * `grid`: Regular target grid with the same dimensionality as the tree.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of grid targets and K is the number of right-hand sides evaluated.
    ///   Rows follow C order, with the last grid axis varying fastest.
    ///
    /// # Errors
    /// * [`FmmError::InvalidTargets`]: If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
    ///   Sparse trees may omit cells containing grid targets.
    ///
    /// # Notes
    /// Target coordinates are generated as needed, without storing the complete coordinate matrix.
    pub fn evaluate_grid(
        &mut self,
        weights: MatRef<f64>,
        grid: &TargetGrid,
    ) -> Result<Mat<f64>, FmmError> {
        let (values, gradients) = self._eval::<false>(weights, EvaluationTargets::Grid(grid))?;
        let _ = gradients;
        Ok(values)
    }

    /// Performs a downward pass of the tree to set the local coefficients and
    /// then performs a leaf evaluation pass to evaluate the values and gradients at the
    /// grid target locations.
    ///
    /// Call [`FmmTree::set_weights`] before evaluating, and repeat that step whenever the weights change.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
    /// * `grid`: Regular target grid with the same dimensionality as the tree.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of grid targets and K is the number of right-hand sides evaluated.
    ///   Rows follow C order, with the last grid axis varying fastest.
    /// * `gradients` : Matrix of shape `(M, K * D)` where M is the number of grid targets, K is the number of right-hand-sides evaluated
    ///     and D is the dimensionality.
    ///     Gradient columns are laid out as `[rhs0_dx, rhs0_dy, rhs0_dz, rhs1_dx, ...]`, truncated to the tree dimensionality.
    ///
    /// # Errors
    /// * [`FmmError::InvalidTargets`]: If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
    ///   Sparse trees may omit cells containing grid targets.
    /// * [`FmmError::KernelDoesNotSupportGradients`]: If the kernel does not support gradient evaluation.
    ///
    /// # Notes
    /// Target coordinates are generated as needed, without storing the complete coordinate matrix.
    pub fn evaluate_grid_with_gradients(
        &mut self,
        weights: MatRef<f64>,
        grid: &TargetGrid,
    ) -> Result<(Mat<f64>, Mat<f64>), FmmError> {
        let (values, gradients) = self._eval::<true>(weights, EvaluationTargets::Grid(grid))?;
        Ok((values, gradients.unwrap()))
    }

    /// Performs a leaf evaluation pass to calculate the values at the grid target locations. Intended to be
    /// used after [`FmmTree::set_local_coefficients`], for when repeated calls to this function are desired,
    /// such as when using 'surface following' isosurface generation algorithms.
    ///
    /// Call [`FmmTree::set_weights`] followed by [`FmmTree::set_local_coefficients`] before evaluating.
    /// If the weights change, repeat both steps before calling this method again.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
    /// * `grid`: Regular target grid with the same dimensionality as the tree.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of grid targets and K is the number of right-hand sides evaluated.
    ///   Rows follow C order, with the last grid axis varying fastest.
    ///
    /// # Errors
    /// * [`FmmError::InvalidTargets`]: If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
    ///   Sparse trees may omit cells containing grid targets.
    ///
    /// # Notes
    /// Target coordinates are generated as needed, without storing the complete coordinate matrix.
    pub fn evaluate_grid_leaves(
        &mut self,
        weights: MatRef<f64>,
        grid: &TargetGrid,
    ) -> Result<Mat<f64>, FmmError> {
        let (values, gradients) =
            self._eval_leaves::<false>(weights, EvaluationTargets::Grid(grid))?;
        let _ = gradients;
        Ok(values)
    }

    /// Performs a leaf evaluation pass to calculate the values and gradients at the grid target locations. Intended to be
    /// used after [`FmmTree::set_local_coefficients`], for when repeated calls to this function are desired,
    /// such as when using 'surface following' isosurface generation algorithms.
    ///
    /// Call [`FmmTree::set_weights`] followed by [`FmmTree::set_local_coefficients`] before evaluating.
    /// If the weights change, repeat both steps before calling this method again.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values). Must match the weights used to set the multipole coefficients.
    /// * `grid`: Regular target grid with the same dimensionality as the tree.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of grid targets and K is the number of right-hand sides evaluated.
    ///   Rows follow C order, with the last grid axis varying fastest.
    /// * `gradients` : Matrix of shape `(M, K * D)` where M is the number of grid targets, K is the number of right-hand-sides evaluated
    ///     and D is the dimensionality.
    ///     Gradient columns are laid out as `[rhs0_dx, rhs0_dy, rhs0_dz, rhs1_dx, ...]`, truncated to the tree dimensionality.
    ///
    /// # Errors
    /// * [`FmmError::InvalidTargets`]: If the grid dimensionality differs from the tree or the grid is not completely covered by its leaf cells.
    ///   Sparse trees may omit cells containing grid targets.
    /// * [`FmmError::KernelDoesNotSupportGradients`]: If the kernel does not support gradient evaluation.
    ///
    /// # Notes
    /// Target coordinates are generated as needed, without storing the complete coordinate matrix.
    pub fn evaluate_grid_leaves_with_gradients(
        &mut self,
        weights: MatRef<f64>,
        grid: &TargetGrid,
    ) -> Result<(Mat<f64>, Mat<f64>), FmmError> {
        let (values, gradients) =
            self._eval_leaves::<true>(weights, EvaluationTargets::Grid(grid))?;
        Ok((values, gradients.unwrap()))
    }

    /// Assign targets to leaves and check grid coverage before allocating output.
    /// Shared by the full evaluator and the evaluator that reuses local coefficients.
    fn _set_leaf_target_indices(
        &mut self,
        target_points: EvaluationTargets<'_>,
    ) -> Result<(), FmmError> {
        let ntarget_points = target_points.point_count();
        let dimensions = self.dimensions as usize;

        if target_points.dimensions() != dimensions {
            return Err(FmmError::InvalidTargets(
                "target and tree dimensions differ",
            ));
        }

        self.tree_lists.leaf_target_indices = match target_points {
            EvaluationTargets::Points(points) => {
                let targets_to_keys = linear_tree::points_to_keys(
                    points,
                    &self.tree_lists.leaves,
                    self.depth,
                    &self.center,
                    self.radius,
                    &self.dimensions,
                )?;

                linear_tree::get_points_to_leaves_map(&targets_to_keys)
                    .into_iter()
                    .map(|(leaf, indices)| (leaf, LeafTargetIndices::Points(indices)))
                    .collect()
            }

            EvaluationTargets::Grid(grid) => self
                .tree_lists
                .leaves
                .par_iter()
                .filter_map(|&leaf| {
                    let ranges =
                        grid.cell_ranges(leaf, &self.center, self.radius, &self.dimensions);

                    let count: usize = ranges.iter().map(|range| range.len()).product();

                    if count == 0 {
                        None
                    } else {
                        Some((leaf, LeafTargetIndices::Grid { ranges, count }))
                    }
                })
                .collect(),
        };

        // Sparse trees or grids outside the root can leave samples uncovered.
        // Detect this before allocating or evaluating the output.
        if matches!(target_points, EvaluationTargets::Grid(_)) {
            let covered: usize = self
                .tree_lists
                .leaf_target_indices
                .values()
                .map(LeafTargetIndices::len)
                .sum();

            if covered != ntarget_points {
                return Err(FmmError::InvalidTargets(
                    "grid is not completely covered by the tree",
                ));
            }
        }

        Ok(())
    }

    /// Internal helper function for evaluating with or without gradients.
    fn _eval<const WITH_GRADS: bool>(
        &mut self,
        weights: MatRef<f64>,
        target_points: EvaluationTargets<'_>,
    ) -> Result<(Mat<f64>, Option<Mat<f64>>), FmmError> {
        let ntarget_points = target_points.point_count();
        let dimensions = self.dimensions as usize;

        self._set_leaf_target_indices(target_points)?;

        if WITH_GRADS {
            self.check_kernel_supports_gradients(target_points, true)?;
        }

        let cells_with_targets: HashSet<u64> = self
            .tree_lists
            .leaf_target_indices
            .par_iter()
            .flat_map(|(&leaf, _)| morton::get_ancestors(leaf, &self.dimensions))
            .collect();

        self.reset_local_coefficients();
        self.downward_pass(&cells_with_targets);

        let target_values = Mat::<f64>::zeros(ntarget_points, self.nrhs);

        let gradients =
            WITH_GRADS.then(|| Mat::<f64>::zeros(ntarget_points, self.nrhs * dimensions));

        self.leaf_pass(
            weights,
            target_points,
            WITH_GRADS,
            target_values.as_ref(),
            gradients.as_ref().map(|matrix| matrix.as_ref()),
        );

        Ok((target_values, gradients))
    }

    /// Performs a downward pass of the tree to set the local coefficients. Intended to be used
    /// after calling [`FmmTree::set_weights`] and before calling [`FmmTree::evaluate_leaves`].
    pub fn set_local_coefficients(&mut self) {
        self.reset_local_coefficients();

        let full_tree_set: HashSet<u64> = self.tree_lists.tree.iter().cloned().collect();

        self.downward_pass(&full_tree_set);
    }

    /// Performs a leaf evaluation pass to calculate the values at the target locations. Intended to be
    /// used after [`FmmTree::set_local_coefficients`], for when repeated calls to this function are desired,
    /// such as when using 'surface following' isosurface generation algorithms.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values)
    /// * `target_points`: Matrix of shape (N, D), where N is the number of target points and D is the dimensionality.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of target points and K is the number of right-hand sides evaluated.
    pub fn evaluate_leaves(
        &mut self,
        weights: MatRef<f64>,
        target_points: MatRef<f64>,
    ) -> Result<Mat<f64>, FmmError> {
        let (target_values, _) =
            self._eval_leaves::<false>(weights, EvaluationTargets::Points(target_points))?;
        Ok(target_values)
    }

    /// Performs a leaf evaluation pass to calculate the values and gradients at the target locations. Intended to be
    /// used after [`FmmTree::set_local_coefficients`], for when repeated calls to this function are desired,
    /// such as when using 'surface following' isosurface generation algorithms.
    ///
    /// # Arguments
    /// * `weights`: Matrix of shape (N, K), where N is the number of source points and K is the number of right-hand sides
    ///              to evaluate, containing source point weights (values)
    /// * `target_points`: Matrix of shape (N, D), where N is the number of target points and D is the dimensionality.
    ///
    /// # Returns
    /// * `target_values` : Matrix of shape (M, K), where M is the number of target points and K is the number of right-hand sides evaluated.
    /// * `gradients` : Matrix of shape `(M, K * D)` where M is the number of target points, K is the number of right-hand-sides evaluated
    ///     and D is the dimensionality.
    ///     Gradient columns are laid out as `[rhs0_dx, rhs0_dy, rhs0_dz, rhs1_dx, ...]`, truncated to the tree dimensionality.   
    pub fn evaluate_leaves_with_gradients(
        &mut self,
        weights: MatRef<f64>,
        target_points: MatRef<f64>,
    ) -> Result<(Mat<f64>, Mat<f64>), FmmError> {
        let (target_values, gradients) =
            self._eval_leaves::<true>(weights, EvaluationTargets::Points(target_points))?;
        Ok((target_values, gradients.unwrap()))
    }

    /// Internal helper function for evaluating leaf cells with or without gradients.
    fn _eval_leaves<const WITH_GRADS: bool>(
        &mut self,
        weights: MatRef<f64>,
        target_points: EvaluationTargets<'_>,
    ) -> Result<(Mat<f64>, Option<Mat<f64>>), FmmError> {
        let ntarget_points = target_points.point_count();
        let dimensions = self.dimensions as usize;

        self._set_leaf_target_indices(target_points)?;

        if WITH_GRADS {
            self.check_kernel_supports_gradients(target_points, true)?;
        }

        let target_values = Mat::<f64>::zeros(ntarget_points, self.nrhs);

        let gradients =
            WITH_GRADS.then(|| Mat::<f64>::zeros(ntarget_points, self.nrhs * dimensions));

        self.leaf_pass(
            weights,
            target_points,
            WITH_GRADS,
            target_values.as_ref(),
            gradients.as_ref().map(|matrix| matrix.as_ref()),
        );

        Ok((target_values, gradients))
    }

    /// Resets the multipole coefficients to zeros.
    fn reset_multipole_coefficients(&mut self) {
        let rows = self.precompute_operators.num_nodes_nd;
        let cols = self.tree_lists.source_key_to_index_map.len() * self.nrhs;
        if self.multipole_coefficients.nrows() != rows
            || self.multipole_coefficients.ncols() != cols
        {
            self.multipole_coefficients = Mat::<f64>::zeros(rows, cols);
        } else {
            // Matvecs commonly replace weights without changing the RHS count.
            self.multipole_coefficients.fill(0.0);
        }
    }

    /// Resets the local coefficients to zeros.
    fn reset_local_coefficients(&mut self) {
        self.local_coefficients = Mat::<f64>::zeros(
            self.precompute_operators.num_nodes_nd,
            self.tree_lists.tree.len() * self.nrhs,
        );
    }

    fn check_kernel_supports_gradients(
        &self,
        target_points: EvaluationTargets<'_>,
        evaluate_gradients: bool,
    ) -> Result<(), FmmError> {
        if !evaluate_gradients
            || target_points.point_count() == 0
            || self.source_points.nrows() == 0
        {
            return Ok(());
        }

        let dims = self.dimensions as usize;
        let mut target_buffer = [0.0; 3];
        let mut gradient_buffer = [0.0; 3];

        let target = target_points.target(0, &mut target_buffer);

        if self
            .kernel
            .evaluate_value_gradient(
                target,
                self.source_points.row(0),
                &mut gradient_buffer[..dims],
            )
            .is_none()
        {
            return Err(FmmError::KernelDoesNotSupportGradients);
        }

        Ok(())
    }

    /// Performs the upward pass of the tree:
    /// * `P2M`: Maps source point values to multipole expansions at the
    ///   Chebyshev nodes of each leaf cell.
    /// * `M2M`: Recursively translates and aggregates child multipole
    ///   expansions to their parent cells, level by level, moving up the tree.
    ///
    fn upward_pass(&mut self, source_values: &MatRef<f64>) {
        let multipole_coefficients_ref = &self.multipole_coefficients;

        self.tree_lists
            .leaf_source_indices
            .par_iter()
            .for_each(|(key, _)| {
                self.particle_to_multipole(*key, &multipole_coefficients_ref, &source_values);
            });

        for level in (1..self.depth).into_iter().rev() {
            let level_keys = self
                .tree_lists
                .level_cells_map
                .get(&(level as u64))
                .unwrap();

            level_keys.par_iter().for_each(|parent| {
                if self
                    .tree_lists
                    .source_key_to_index_map
                    .contains_key(&parent)
                {
                    self.multipole_to_multipole(&parent, &multipole_coefficients_ref);
                }
            });
        }
    }

    /// Maps source points to Chebyshev nodes in the cell.
    fn particle_to_multipole(
        &self,
        key: u64,
        multipole_coefficients_ref: &Mat<f64>,
        source_values: &MatRef<f64>,
    ) {
        let (center, length) =
            morton::get_center_length(key, &self.center, self.radius, &self.dimensions);

        if let Some(cell_source_indices) = self.tree_lists.leaf_source_indices.get(&key) {
            let mut cell_point_locations =
                utils::select_mat_rows(self.source_points.as_mat_ref(), &cell_source_indices);

            let cell_source_values = Mat::<f64>::from_fn(
                cell_source_indices.len(),
                source_values.shape().1,
                |i, j| *source_values.get(cell_source_indices[i], j),
            );

            let cell_multipole_transfer = chebyshev::get_approximation_coefficients(
                self.interpolation_order,
                &mut cell_point_locations,
                &center,
                &length,
                &self.precompute_operators.polynomial_nodes,
                &(self.dimensions as usize),
                false,
            );

            for j in 0..self.nrhs {
                let coefficients =
                    cell_source_values.col(j).transpose() * &cell_multipole_transfer.values;

                let column_index = self.tree_lists.source_key_to_index_map.get(&key).unwrap()
                    + j * self.tree_lists.source_key_to_index_map.len();

                unsafe {
                    let cell_ptr =
                        multipole_coefficients_ref.col(column_index).as_ptr() as *mut f64;

                    (0..self.precompute_operators.num_nodes_nd)
                        .into_iter()
                        .for_each(|idx| {
                            *cell_ptr.add(idx) += coefficients[idx];
                        });
                }
            }
        }
    }

    /// Propogates the multipole expansions from children into their parent
    fn multipole_to_multipole(&self, parent: &u64, multipole_coefficients_ref: &Mat<f64>) {
        let parent_children = self.tree_lists.children.get(&parent).unwrap();
        let num_nodes = self.precompute_operators.num_nodes_nd;
        let dimensions = self.dimensions as usize;

        // Reuse the same tensor-product scratch space for every child and
        // right-hand side handled by this parent.
        let mut scratch = [
            Mat::<f64>::zeros(num_nodes, 1),
            Mat::<f64>::zeros(num_nodes, 1),
        ];

        for j in 0..self.nrhs {
            let parent_column_index = self
                .tree_lists
                .source_key_to_index_map
                .get(&parent)
                .unwrap()
                + j * self.tree_lists.source_key_to_index_map.len();

            unsafe {
                let parent_ptr =
                    multipole_coefficients_ref.col(parent_column_index).as_ptr() as *mut f64;

                parent_children.iter().for_each(|child_key| {
                    let Some(&child_index_in_storage) =
                        self.tree_lists.source_key_to_index_map.get(child_key)
                    else {
                        return; // Empty source children have no multipole storage.
                    };
                    let child_column_index =
                        child_index_in_storage + j * self.tree_lists.source_key_to_index_map.len();

                    let child_index = morton::get_child_index(&child_key, &self.dimensions);

                    chebyshev::apply_cheb_tensor_transfer(
                        multipole_coefficients_ref.col_as_slice(child_column_index),
                        &mut scratch,
                        self.precompute_operators.m2m_transfer_operators.as_ref(),
                        child_index,
                        self.interpolation_order,
                        dimensions,
                        false,
                    );

                    let child_coefficients = scratch[0].col_as_slice(0);

                    (0..self.precompute_operators.num_nodes_nd)
                        .into_iter()
                        .for_each(|idx| {
                            *parent_ptr.add(idx) += child_coefficients[idx];
                        });
                });
            }
        }
    }

    /// Performs a downward pass down the tree to populate local coefficients:
    /// * `M2L`: Low-rank interactions between non-adjacent same level cells.
    /// * `L2L`: Propogate local coefficients from parent to children.
    fn downward_pass(&mut self, cells_with_targets: &HashSet<u64>) {
        let local_coefficients_ref = &self.local_coefficients;

        for level in (1..self.depth + 1).into_iter() {
            let level_keys = self
                .tree_lists
                .level_cells_map
                .get(&(level as u64))
                .unwrap();

            level_keys.par_iter().for_each(|key| {
                if cells_with_targets.contains(&key) {
                    let cell_column_index = self.tree_lists.key_to_index_map.get(&key).unwrap();

                    if let Some(v_list) = self.tree_lists.v_lists.get(&key) {
                        if v_list.len() > 0 {
                            self.multipole_to_local(
                                *key,
                                level,
                                &v_list,
                                &local_coefficients_ref,
                                &cell_column_index,
                            );
                        }
                    }
                }
            });
        }

        for level in (1..self.depth + 1).into_iter() {
            let level_keys = self
                .tree_lists
                .level_cells_map
                .get(&(level as u64))
                .unwrap();

            level_keys.par_iter().for_each(|key| {
                if cells_with_targets.contains(&key) {
                    let cell_column_index = self.tree_lists.key_to_index_map.get(&key).unwrap();
                    if let Some(children) = self.tree_lists.children.get(&key) {
                        if children.len() > 0 {
                            self.local_to_local(
                                &children,
                                &local_coefficients_ref,
                                &cell_column_index,
                                &cells_with_targets,
                            );
                        }
                    }
                }
            });
        }
    }

    /// Low-rank interaction between multipoles and locals of two separated cells, using the
    /// v_list for each cell.
    ///
    /// Optimised to leverage symmetries and a blocking scheme to replace many matrix-vector
    /// operations with a few matrix-matrix operations.
    fn multipole_to_local(
        &self,
        key: u64,
        level: u64,
        v_list: &HashSet<u64>,
        local_coefficients_ref: &Mat<f64>,
        cell_column_index: &usize,
    ) {
        let (cell_center, cell_length) =
            morton::get_center_length(key, &self.center, self.radius, &self.dimensions);

        // Map the v-cells to the unique reference vectors required.
        let mut unique_reference_vectors: HashMap<usize, Vec<(u64, usize)>> = HashMap::new();

        v_list.iter().for_each(|v_cell| {
            let (v_center, _v_length) =
                morton::get_center_length(*v_cell, &self.center, self.radius, &self.dimensions);

            let vector_between_cells: Vec<i32> = cell_center
                .iter()
                .zip(v_center.iter())
                .map(|(c, v)| ((c - v) / cell_length).round() as i32)
                .collect();

            let wanted_m2l_vector = self.calculate_m2l_transfer_index(&vector_between_cells);

            let wanted_reference_vector =
                self.precompute_operators.reference_vector_lookups[wanted_m2l_vector];

            unique_reference_vectors
                .entry(wanted_reference_vector)
                .or_insert(Vec::new())
                .push((v_cell.clone(), wanted_m2l_vector));
        });

        // For each unique reference vector:
        //  1) Permute the multipole coefficients for each v-cell to align with the reference vector
        //  2) Perform matrix-matrix multiplication
        //  3) Permute back to the original order
        unique_reference_vectors
            .iter()
            .for_each(|(reference_cell, v)| {
                for j in 0..self.nrhs {
                    let mut permuted_multipole_coefficients =
                        Mat::<f64>::zeros(v.len(), self.precompute_operators.num_nodes_nd);

                    v.iter().enumerate().for_each(|(row_idx, v_cell)| {
                        let permutation_index =
                            self.precompute_operators.permutation_lookups[v_cell.1];

                        let perm_indices =
                            &self.precompute_operators.permutation_indices[permutation_index];

                        let v_cell_column_index = self
                            .tree_lists
                            .source_key_to_index_map
                            .get(&v_cell.0)
                            .unwrap()
                            + j * self.tree_lists.source_key_to_index_map.len();

                        let v_cell_multipole_coefficients =
                            self.multipole_coefficients.col(v_cell_column_index);

                        perm_indices
                            .iter()
                            .enumerate()
                            .for_each(|(col_idx, perm_idx)| {
                                permuted_multipole_coefficients[(row_idx, col_idx)] =
                                    v_cell_multipole_coefficients[*perm_idx].clone();
                            });
                    });

                    let u_lookup_values = self
                        .precompute_operators
                        .u
                        .get(&(level as usize))
                        .unwrap()
                        .get(&reference_cell)
                        .unwrap();

                    let mut permuted_local_coefficients: Mat<f64>;

                    match self.compression_type {
                        M2LCompressionType::ACA | M2LCompressionType::SVD => {
                            let vt_lookup_values = self
                                .precompute_operators
                                .vt
                                .get(&(level as usize))
                                .unwrap()
                                .get(&reference_cell)
                                .unwrap();

                            permuted_local_coefficients =
                                vt_lookup_values * permuted_multipole_coefficients.transpose();
                            permuted_local_coefficients =
                                u_lookup_values * permuted_local_coefficients;
                        }
                        M2LCompressionType::None => {
                            permuted_local_coefficients =
                                u_lookup_values * permuted_multipole_coefficients.transpose();
                        }
                    }

                    unsafe {
                        let cell_ptr = local_coefficients_ref
                            .col(*cell_column_index + j * self.tree_lists.tree.len())
                            .as_ptr() as *mut f64;

                        v.iter().enumerate().for_each(|(col_idx, v_cell)| {
                            let permutation_index =
                                self.precompute_operators.permutation_lookups[v_cell.1];
                            let inverse_perm =
                                &self.precompute_operators.inverse_permutations[permutation_index];

                            inverse_perm
                                .iter()
                                .enumerate()
                                .for_each(|(row_idx, perm_idx)| {
                                    *cell_ptr.add(row_idx) +=
                                        permuted_local_coefficients[(*perm_idx, col_idx)];
                                })
                        });
                    }
                }
            });
    }

    /// Calculates the index of the M2L vector based on the distance vector between the two cells.
    fn calculate_m2l_transfer_index(&self, vector_between_cell: &Vec<i32>) -> usize {
        let powers = (0..self.dimensions as u32).rev();
        let base: u32 = 7;

        vector_between_cell
            .iter()
            .zip(powers)
            .map(|(v, p)| (base.pow(p) * (*v + 3) as u32) as usize)
            .sum()
    }

    /// Propogates the local coefficients from the parent cell to its children.
    fn local_to_local(
        &self,
        children: &Vec<u64>,
        local_coefficients_ref: &Mat<f64>,
        cell_column_index: &usize,
        cells_with_targets: &HashSet<u64>,
    ) {
        let num_nodes = self.precompute_operators.num_nodes_nd;
        let dimensions = self.dimensions as usize;

        // Reuse the same tensor-product workspace for every relevant child
        // and right-hand side handled by this parent.
        let mut workspace = [
            Mat::<f64>::zeros(num_nodes, 1),
            Mat::<f64>::zeros(num_nodes, 1),
        ];

        for child_key in children {
            if cells_with_targets.contains(child_key) {
                for j in 0..self.nrhs {
                    let parent_column_index = *cell_column_index + j * self.tree_lists.tree.len();

                    let child_column_index =
                        self.tree_lists.key_to_index_map.get(child_key).unwrap()
                            + j * self.tree_lists.tree.len();

                    let child_index = morton::get_child_index(child_key, &self.dimensions);

                    chebyshev::apply_cheb_tensor_transfer(
                        local_coefficients_ref.col_as_slice(parent_column_index),
                        &mut workspace,
                        self.precompute_operators.m2m_transfer_operators.as_ref(),
                        child_index,
                        self.interpolation_order,
                        dimensions,
                        true,
                    );

                    let child_coefficients = workspace[0].col_as_slice(0);

                    unsafe {
                        let child_ptr =
                            local_coefficients_ref.col(child_column_index).as_ptr() as *mut f64;

                        for index in 0..num_nodes {
                            *child_ptr.add(index) += child_coefficients[index];
                        }
                    }
                }
            }
        }
    }

    /// Parallel loop through all leaf cells to evaluate targets.
    fn leaf_pass(
        &self,
        source_values: MatRef<f64>,
        target_points: EvaluationTargets,
        evaluate_gradients: bool,
        target_values_ref: MatRef<f64>,
        target_gradients_ref: Option<MatRef<f64>>,
    ) {
        match evaluate_gradients {
            false => self.leaf_pass_mode::<false>(
                source_values,
                target_points,
                target_values_ref,
                target_gradients_ref,
            ),
            true => self.leaf_pass_mode::<true>(
                source_values,
                target_points,
                target_values_ref,
                target_gradients_ref,
            ),
        }
    }

    fn leaf_pass_mode<const WITH_GRADS: bool>(
        &self,
        source_values: MatRef<f64>,
        target_points: EvaluationTargets,
        target_values_ref: MatRef<f64>,
        target_gradients_ref: Option<MatRef<f64>>,
    ) {
        self.tree_lists
            .leaf_target_indices
            .par_iter()
            .for_each(|(leaf, leaf_target_indices)| {
                if let Some(u_list) = self.tree_lists.u_lists.get(leaf) {
                    self.particle_to_particle::<WITH_GRADS>(
                        u_list,
                        target_points,
                        leaf_target_indices,
                        target_values_ref,
                        target_gradients_ref,
                        source_values,
                    );
                }

                self.local_to_particle::<WITH_GRADS>(
                    leaf,
                    leaf_target_indices,
                    target_points,
                    target_values_ref,
                    target_gradients_ref,
                );
            });
    }

    /// Direct interaction between particles of adjacent leaf cells.
    fn particle_to_particle<const WITH_GRADS: bool>(
        &self,
        u_list: &HashSet<u64>,
        target_points: EvaluationTargets,
        cell_target_indices: &LeafTargetIndices,
        target_values_ref: MatRef<f64>,
        target_gradients_ref: Option<MatRef<f64>>,
        source_values: MatRef<f64>,
    ) {
        let dims = target_points.dimensions();
        let mut target_buffer = [0.0; 3];

        u_list.iter().for_each(|u_cell| {
            if let Some(u_cell_source_indices) = self.tree_lists.leaf_source_indices.get(u_cell) {
                let u_cell_values = Mat::<f64>::from_fn(
                    u_cell_source_indices.len(),
                    source_values.shape().1,
                    |i, j| *source_values.get(u_cell_source_indices[i], j),
                );

                let u_cell_points =
                    utils::select_mat_rows(self.source_points.as_mat_ref(), u_cell_source_indices);

                for rhs in 0..self.nrhs {
                    let rhs_vals = u_cell_values.col(rhs);
                    let value_ptr = target_values_ref.col(rhs).as_ptr() as *mut f64;

                    let (grad_col0_ptr, grad_col1_ptr, grad_col2_ptr) = if WITH_GRADS {
                        let target_gradients_ref = target_gradients_ref
                            .expect("ValuesAndGradients mode requires gradient storage");
                        let grad_start_col = rhs * dims;
                        let grad_1_col = grad_start_col + 1;
                        let grad_2_col = grad_start_col + 2;

                        let grad_col0_ptr =
                            target_gradients_ref.col(grad_start_col).as_ptr() as *mut f64;

                        let grad_col1_ptr = if dims > 1 {
                            Some(target_gradients_ref.col(grad_1_col).as_ptr() as *mut f64)
                        } else {
                            None
                        };

                        let grad_col2_ptr = if dims > 2 {
                            Some(target_gradients_ref.col(grad_2_col).as_ptr() as *mut f64)
                        } else {
                            None
                        };

                        (grad_col0_ptr, grad_col1_ptr, grad_col2_ptr)
                    } else {
                        (std::ptr::null_mut(), None, None)
                    };

                    for target_idx in cell_target_indices.iter(target_points) {
                        let target = target_points.target(target_idx, &mut target_buffer);

                        for s in 0..u_cell_source_indices.len() as usize {
                            let source = u_cell_points.row(s);
                            let w = rhs_vals[s];

                            if WITH_GRADS {
                                let mut grad_buf = [0.0_f64; 3];
                                let value = self
                                    .kernel
                                    .evaluate_value_gradient(target, source, &mut grad_buf[..dims])
                                    .expect("Kernel gradient support was pre-checked");

                                unsafe {
                                    *value_ptr.add(target_idx) += value * w;
                                    *grad_col0_ptr.add(target_idx) += grad_buf[0] * w;
                                    if let Some(ptr) = grad_col1_ptr {
                                        *ptr.add(target_idx) += grad_buf[1] * w;
                                    }
                                    if let Some(ptr) = grad_col2_ptr {
                                        *ptr.add(target_idx) += grad_buf[2] * w;
                                    }
                                }
                            } else {
                                let value = self.kernel.evaluate(target, source);

                                unsafe {
                                    *value_ptr.add(target_idx) += value * w;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Maps the local coefficients of a cell to the targets in the cell.
    fn local_to_particle<const WITH_GRADS: bool>(
        &self,
        leaf: &u64,
        leaf_target_indices: &LeafTargetIndices,
        target_points: EvaluationTargets,
        target_values_ref: MatRef<f64>,
        target_gradients_ref: Option<MatRef<f64>>,
    ) {
        let dims = self.dimensions as usize;
        let ngrad_cols = self.interpolation_order.pow(dims as u32);

        leaf_target_indices
            .chunks(target_points, self.eval_chunk_size)
            .for_each(|chunk_target_indices| {
                let mut point_locations_mat = Mat::<f64>::from_fn(
                    chunk_target_indices.len(),
                    target_points.dimensions(),
                    |i, j| target_points.coordinate(chunk_target_indices[i], j),
                );

                let (cell_center, cell_length) =
                    morton::get_center_length(*leaf, &self.center, self.radius, &self.dimensions);

                let cell_local_transfer = chebyshev::get_approximation_coefficients(
                    self.interpolation_order,
                    &mut point_locations_mat,
                    &cell_center,
                    &cell_length,
                    &self.precompute_operators.polynomial_nodes,
                    &(self.dimensions as usize),
                    WITH_GRADS,
                );

                let grads = if WITH_GRADS {
                    Some(
                        cell_local_transfer
                            .gradients
                            .as_ref()
                            .expect("ValuesAndGradients mode requires gradient coefficients"),
                    )
                } else {
                    None
                };

                for j in 0..self.nrhs {
                    let cell_column_index = self.tree_lists.key_to_index_map.get(leaf).unwrap()
                        + j * self.tree_lists.tree.len();
                    let local_coefficients = self.local_coefficients.col(cell_column_index);
                    let local_target_values = &cell_local_transfer.values * &local_coefficients;

                    unsafe {
                        let value_ptr = target_values_ref.col(j).as_ptr() as *mut f64;
                        chunk_target_indices.iter().enumerate().for_each(
                            |(coeff_idx, target_idx)| {
                                *value_ptr.add(*target_idx) += local_target_values[coeff_idx];
                            },
                        );
                    }

                    if WITH_GRADS {
                        let grads = grads.expect("ValuesAndGradients mode requires gradients");
                        let target_gradients_ref = target_gradients_ref
                            .expect("ValuesAndGradients mode requires gradient storage");

                        for d in 0..dims {
                            let col_start = ngrad_cols * d;
                            let local_grads = grads.subcols(col_start, ngrad_cols);
                            let local_target_grad = &local_grads * &local_coefficients;
                            let col = j * dims + d;

                            unsafe {
                                let grad_ptr = target_gradients_ref.col(col).as_ptr() as *mut f64;
                                chunk_target_indices.iter().enumerate().for_each(
                                    |(coeff_idx, target_idx)| {
                                        *grad_ptr.add(*target_idx) += local_target_grad[coeff_idx];
                                    },
                                );
                            }
                        }
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::{FmmError, FmmTree, KernelFunction};
    use faer::{RowRef, mat};
    use std::sync::Arc;

    struct TestKernel;

    impl KernelFunction for TestKernel {
        #[inline(always)]
        fn evaluate(&self, target: RowRef<f64>, source: RowRef<f64>) -> f64 {
            let mut dist = 0.0;
            for (t, s) in target.iter().zip(source.iter()) {
                let diff = t - s;
                dist += diff * diff;
            }
            -dist.sqrt()
        }
    }

    /// Ensure that evaluating at a target point outside the tree extents
    /// returns `FmmError::PointOutsideTree` instead of panicking.
    #[test]
    fn evaluate_returns_error_for_target_outside_extents() {
        // Single 1D source point inside [0.0, 1.0].
        let source_points = Arc::new(mat![[0.5]]);
        let interpolation_order = 3usize;
        let kernel = TestKernel;
        let sparse_tree = false;
        // Explicit 1D extents: [xmin, xmax].
        let extents = Some(vec![0.0_f64, 1.0_f64]);

        let mut tree = FmmTree::new(
            source_points,
            interpolation_order,
            kernel,
            sparse_tree,
            extents,
            None,
        );

        // Single RHS weight.
        let weights = mat![[1.0]];
        tree.set_weights(weights.as_ref());

        // Two targets: one inside extents, one clearly outside.
        let target_points = mat![[0.5], [10.0]];

        let result = tree.evaluate(weights.as_ref(), target_points.as_ref());

        match result {
            Err(FmmError::PointOutsideTree { point_index }) => {
                assert_eq!(point_index, 1);
            }
            other => panic!("Expected PointOutsideTree error, got {:?}", other),
        }
    }
}
