import numpy as np
from ferreus_bbfmm import FmmTree, FmmKernelType, KernelParams

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

# Create an adaptive tree
adaptive_tree = True

# No need to store empty leaves for fast matrix-vector product
sparse_tree = True

# Create a new tree
tree = FmmTree(
    source_points,
    interpolation_order,
    kernel_params,
    adaptive_tree,
    sparse_tree,
)

# Set the weights - this performs an upward pass through the tree
# and sets the multipole coefficients
tree.set_weights(weights)

# Evaluate at the source points
target_points = source_points.copy()

# Perform a downward pass to set the local coefficients, then perform a leaf evaluation
tree.evaluate(weights, target_points)

print(f"Evaluated values at source locations: {tree.target_values()}")
