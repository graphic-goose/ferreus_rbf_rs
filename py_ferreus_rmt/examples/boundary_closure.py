import numpy as np
from pathlib import Path
from ferreus_rbf.isosurfacing import build_isosurface, BoundaryClosure

# Define a function to evaluate an isosurface from
# In this example we'll create a unit sphere
def sphere(pts: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pts, axis=1) - 1.0


# Define the extents for the isosurfacer
extents = np.array([-1.1, -1.1, -1.1, 0.0, 1.1, 1.1])

# Define the resolution to evaluate at
resolution = 0.2

# Define some seed points on the isosurface
seed_points = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
)

# Define the function values of the seed points
seed_values = np.array([0.0, 0.0])

# Define the isovalue at which to surface
isovalue = 0.0

# Create an isosurface for each closure mode
for closure in [BoundaryClosure.None_, BoundaryClosure.ClosePositive, BoundaryClosure.CloseNegative]:
    mesh = build_isosurface(
        seed_points,
        extents,
        resolution,
        isovalue,
        sphere,
        boundary_closure=closure,
    )

    # Save the isosurface out to an obj file
    surface_name = f"isosurface_sphere_{resolution}m {closure}"
    outpath = Path(f"{surface_name}.obj")
    mesh.save_obj(str(outpath), surface_name)
