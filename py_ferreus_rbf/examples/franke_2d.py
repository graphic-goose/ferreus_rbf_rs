import numpy as np
import matplotlib.pyplot as plt
from ferreus_rbf import RBFInterpolator, RBFTestFunctions
from ferreus_rbf.interpolant_config import (
    RBFKernelType,
    InterpolantSettings,
)

np.random.seed(42)

# Define input source points in a 2D grid within [0, 1]^2
points = np.random.random((100, 2))

# Define some values at the source points using Franke's function
point_values = RBFTestFunctions.franke_2d(points)

# Select the thin plate spline RBF kernel
interpolant_settings = InterpolantSettings(RBFKernelType.ThinPlateSpline)

# Setup and solve the RBF
rbfi = RBFInterpolator(
    points,
    point_values,
    interpolant_settings,
)

# Build a 2D grid of target points in [0, 1]^2 to evaluate the RBF at
n = 50
x_coords = np.linspace(0.0, 1.0, n)
y_coords = np.linspace(0.0, 1.0, n)
X, Y = np.meshgrid(x_coords, y_coords)
target_points = np.column_stack((X.ravel(), Y.ravel()))

# Evaluate the RBF at the target points
Z = rbfi.evaluate(target_points)[:, 0].reshape(n, n)

# Plot a surface of the evaluated values
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="jet", linewidth=1, antialiased=True, edgecolor="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.set_title("RBF-interpolated Franke surface")
ax.view_init(elev=15, azim=55)
plt.tight_layout()
plt.show()
