import numpy as np
from ferreus_bbfmm import FmmTree, FmmKernelType, KernelParams, FmmParams, M2LCompressionType, TargetGrid

# Choose a kernel
kernel_params = KernelParams(FmmKernelType.LinearRbf)

# Define input source points in a 3D grid within [-1, 1]^3
np.random.seed(42)
dim = 3
num_rhs = 1
num_points = 10000
source_points = np.random.random((num_points, dim)) * 2 - 1
weights = np.random.random((num_points, num_rhs))

# Interpolation order defines the number of Chebyshev nodes in each dimension
# used in the far-field approximation
# A higher interpolation order is more accurate, but takes longer to compute
interpolation_order = 7

# Store empty leaves for general RBF evaluation
sparse_tree = False

# For the evaluator we may wish to evaluate over a larger region than the source points cover
extents = np.array([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])

# Define a regular target grid to guide tree refinement
grid = TargetGrid.from_spacing(extents.tolist(), [0.5] * dim)

# Optionally define some tuning parameters
params = FmmParams(
    256,
    M2LCompressionType.ACA,
    10**(-interpolation_order),
    1024,
)

# Create a new tree
tree = FmmTree(
    source_points,
    interpolation_order,
    kernel_params,
    sparse_tree,
    extents=extents,
    params=params,
    target_grid=grid,
)

# Set the weights - this performs an upward pass through the tree
# and sets the multipole coefficients
tree.set_weights(weights)

# For implicit modelling where a 'surface following' method of generating an isosurface
# is used, the evaluator may be called many times. In this case it's more efficient to
# perform a single downward pass to set all the local coefficients, then call the evaluator
# on the relevant leaves for each evaluation
tree.set_local_coefficients()

# Create some arbritrary target points
num_target_points = 100
target_points = np.random.random((num_target_points, dim)) * 4 - 2

# Perform a leaf evaluation
target_values = tree.evaluate_leaves(weights, target_points)

print(f"Evaluated values at target locations: {target_values}")

# Create some more target points
num_target_points = 1000
target_points = np.random.random((num_target_points, dim)) * 4 - 2

# Perform another leaf evaluation
target_values = tree.evaluate_leaves(weights, target_points)

print(f"Evaluated values at target locations: {target_values}")

# Reuse the same evaluator for the grid without creating all target coordinates
grid_values = tree.evaluate_grid_leaves(weights, grid)
print(f"Evaluated grid shape: {grid_values.shape}")
