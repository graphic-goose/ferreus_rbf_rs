import numpy as np
from ferreus_bbfmm import FmmTree, FmmKernelType, KernelParams, TargetGrid

# Choose a kernel
kernel_params = KernelParams(FmmKernelType.LinearRbf)

# Define input source points in a 3D grid within [-1, 1]^3
np.random.seed(42)
dim = 3
num_rhs = 2
num_points = 10000
source_points = np.random.random((num_points, dim)) * 2 - 1
weights = np.random.random((num_points, num_rhs))

# Interpolation order defines the number of Chebyshev nodes in each dimension
# used in the far-field approximation
# A higher interpolation order is more accurate, but takes longer to compute
interpolation_order = 7

# Need to store empty leaves for a general evaluator.
sparse_tree = False

# Store axis metadata instead of a matrix of all target coordinates.
grid = TargetGrid.from_spacing([-1.0, -1.0, 1.0, 1.0], [0.1, 0.1])
# Equivalently: TargetGrid(origin=[-1.0, -1.0], spacing=[0.1, 0.1], shape=[21, 21])
print(f"Grid shape: {grid.shape()}, target count: {grid.point_count()}")

tree = FmmTree(
    source_points,
    interpolation_order,
    kernel_params,
    sparse_tree,
    target_grid=grid,
)

tree.set_weights(weights)
values = tree.evaluate_grid(weights, grid)

# Prepare local coefficients once for repeated leaf evaluations.
tree.set_local_coefficients()
reused_values = tree.evaluate_grid_leaves(weights, grid)
np.testing.assert_allclose(reused_values, values)

# Materialise a small coordinate chunk only when needed.
print(f"First three target points:\n{grid.points(0, 3)}")
