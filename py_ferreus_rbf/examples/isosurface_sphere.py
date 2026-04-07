import numpy as np
from pathlib import Path
from ferreus_rbf.isosurfacing import surface_nets, save_obj


# Define a function to evaluate an isosurface from
# In this example we'll create a unit sphere
def sphere(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pts, axis=1) - 1.0


# Define the extents for the isosurfacer
extents = np.array([-1.1, -1.1, -1.1, 1.1, 1.1, 1.1])

# Define the resolution to evaluate at
resolution = 0.1

# Define some seed points on the isosurface
seed_points = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
)

# Define the function values of the seed points
seed_values = np.array([[0.0], [0.0]])

# Define the isovalue at which to surface
isovalue = 0.0

# Extract the isosurface
verts, faces = surface_nets(
    extents,
    resolution,
    isovalue,
    sphere,
    seed_points,
    seed_values,
)

# Save the isosurface out to an obj file
surface_name = f"isosurface_sphere_{resolution}m"
outpath = Path(f"{surface_name}.obj")
save_obj(str(outpath), surface_name, verts, faces)
