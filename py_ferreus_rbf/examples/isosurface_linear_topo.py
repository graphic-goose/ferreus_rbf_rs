import time
import numpy as np
import pandas as pd
from pathlib import Path
from ferreus_rbf import RBFInterpolator
from ferreus_rbf.isosurfacing import build_isosurface, BoundaryClosure
from ferreus_rbf.interpolant_config import (
    RBFKernelType,
    InterpolantSettings,
    FittingAccuracyType,
    FittingAccuracy,
)
from ferreus_rbf.progress import (
    Progress,
    SolverIteration,
    SurfacingProgress,
    DuplicatesRemoved,
    Message,
    ProgressEvent,
)


def on_progress(event: ProgressEvent) -> None:
    if isinstance(event, SolverIteration):
        print(
            f"Iteration: {event.iter:3d}  {event.residual:>.5E}  {(event.progress * 100):.1f}%"
        )
    elif isinstance(event, SurfacingProgress):
        print(
            f"Isovalue: {event.isovalue}  Stage: {event.stage}  {(event.progress * 100):.1f}%"
        )
    elif isinstance(event, DuplicatesRemoved):
        print(f"Removed duplicates: {event.num_duplicates}")
    elif isinstance(event, Message):
        print(event.message)


# Define the input paths for the example signed distance points and topography points
project_root = Path(__file__).resolve().parents[2]
datasets_path = project_root / "datasets"
pointset_path = datasets_path / "albatite_SD_points.csv"
topo_path = datasets_path / "Topo points.csv"

# Import the example points files
point_df = pd.read_csv(pointset_path)
topo_df = pd.read_csv(topo_path)

# Extract the source point coordinates and signed distance values
#
# The dataset contains 3D point coordinates in the first 3 columns, followed by
# signed distance (SD) values in the 4th column. The SD values are 0 on the surface
# boundary, -ve inside the surface and +ve outside the surface.
source_points = point_df[["X", "Y", "Z"]].to_numpy().astype(np.float64)
source_values = point_df["SignedDistance"].to_numpy().astype(np.float64)

# Extract the topography points as a 2D interpolant: XY locations with Z elevation values.
topo_points = topo_df[["X", "Y"]].to_numpy().astype(np.float64)
topo_values = topo_df["Z"].to_numpy().astype(np.float64)

# Define the RBF kernel to use
kernel_type = RBFKernelType.Linear

# Define the desired fitting accuracy
fitting_accuracy = FittingAccuracy(0.01, FittingAccuracyType.Absolute)

# Initialise an InterpolantSettings instance
interpolant_settings = InterpolantSettings(
    kernel_type, fitting_accuracy=fitting_accuracy
)

# Create a callback to receive progress updates from the RBFInterpolator
callback = Progress(callback=on_progress)

# Setup and solve the signed distance and topography RBF systems
rbfi = RBFInterpolator(
    source_points, source_values, interpolant_settings, progress_callback=callback
)
topo_rbfi = RBFInterpolator(
    topo_points, topo_values, interpolant_settings, progress_callback=callback
)

# Define the sampling grid resolution for the surfacer
resolution = 5.0

# Define the signed distance isovalue to surface before topography clipping.
isovalue = 20.0

# Define the isosurfacing extents: [minx, miny, minz, maxx, maxy, maxz].
extents = np.array(
    [
        329105.0,
        7744370.0,
        -320.0,
        329845.0,
        7745275.0,
        435.0,
    ],
    dtype=np.float64,
)

# When setting up the RBF evaluators for isosurfacing we need to add a buffer to
# the evaluator extents so we don't evaluate points outside the evaluator domains.
evaluator_padding = 10.0 * resolution

rbf_source_extents = np.hstack(
    (
        np.min(source_points, axis=0),
        np.max(source_points, axis=0),
    )
)
rbf_evaluator_extents = np.hstack(
    (
        np.minimum(rbf_source_extents[:3], extents[:3]) - evaluator_padding,
        np.maximum(rbf_source_extents[3:], extents[3:]) + evaluator_padding,
    )
)
rbfi.build_evaluator(rbf_evaluator_extents.astype(np.float64))

topo_source_extents = np.hstack(
    (
        np.min(topo_points, axis=0),
        np.max(topo_points, axis=0),
    )
)
topo_surfacing_extents = np.array(
    [extents[0], extents[1], extents[3], extents[4]],
    dtype=np.float64,
)
topo_evaluator_extents = np.hstack(
    (
        np.minimum(topo_source_extents[:2], topo_surfacing_extents[:2])
        - evaluator_padding,
        np.maximum(topo_source_extents[2:], topo_surfacing_extents[2:])
        + evaluator_padding,
    )
)
topo_rbfi.build_evaluator(topo_evaluator_extents.astype(np.float64))


def surface_fn(targets: np.ndarray) -> np.ndarray:
    # The surfacer calls this function with batches of 3D sample locations.
    # It expects one scalar value per sample. The zero contour of those returned
    # values is the surface that will be extracted.
    rbf_values = np.asarray(rbfi.evaluate_targets(targets)).ravel()

    # The topography interpolant is 2D: it maps XY locations to a topographic Z
    # elevation. For every 3D sample point, evaluate the topography directly
    # below or above it using only the sample's X and Y coordinates.
    topo_values = np.asarray(topo_rbfi.evaluate_targets(targets[:, :2])).ravel()

    # Shift the signed-distance RBF so the requested isovalue becomes zero.
    # For example, if isovalue is 20, points where the RBF evaluates to 20 now
    # return 0, values below 20 are negative, and values above 20 are positive.
    rbf_isovalue_values = rbf_values - isovalue

    # Build a second implicit field for the topography clipping plane/surface.
    # This is negative below the topography, zero on it, and positive above it.
    topo_clip_values = targets[:, 2] - topo_values

    # Taking the maximum combines the two implicit fields as an intersection:
    # the result is negative only where both fields are negative. Surfacing the
    # combined field at zero keeps the RBF isosurface only where it is below the
    # topography, and the topography itself closes the clipped portion.
    return np.maximum(rbf_isovalue_values, topo_clip_values)


start_surfacing = time.time()

# surface_fn shifts the RBF by isovalue, so the combined clipped field is surfaced at 0.
mesh = build_isosurface(
    source_points,
    extents,
    resolution,
    0.0,
    surface_fn,
    boundary_closure=BoundaryClosure.ClosePositive,
    progress_callback=callback,
)

end_surfacing = time.time()
print(f"Surfacing took {end_surfacing - start_surfacing} seconds")

# Save the isosurface out to an obj file
surface_name = f"isosurface_linear_topo_{isovalue:g}_{resolution:g}m"
outpath = Path(f"{surface_name}.obj")
mesh.save_obj(str(outpath), surface_name)
