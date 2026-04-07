import numpy as np
import pandas as pd
from pathlib import Path
from ferreus_rbf import RBFInterpolator
from ferreus_rbf.isosurfacing import save_obj, surface_nets
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


# Define the input path for the example signed distance points
project_root = Path(__file__).resolve().parent.parent
pointset_path = project_root / "datasets" / "albatite_SD_points.csv"

# Import the example points file
df = pd.read_csv(pointset_path)

# Extract the source point coordinates and signed distance values
#
# The dataset contains 3D point coordinates in the first 3 columns, followed by
# signed distance (SD) values in the 4th column. The SD values are 0 on the surface
# boundary, -ve inside the surface and +ve outside the surface.
source_points = df[["X", "Y", "Z"]].to_numpy().astype(np.float64)
source_values = df["SignedDistance"].to_numpy().reshape((-1, 1)).astype(np.float64)

# Get the axis aligned bounding box extents of the source points
# to use for the isosurface extraction
extents = np.concatenate(
    (
        np.floor(np.min(source_points, axis=0)),
        np.ceil(np.max(source_points, axis=0)),
    )
)

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

# Setup and solve the RBF system
rbfi = RBFInterpolator(
    source_points, source_values, interpolant_settings, progress_callback=callback
)

# Define the sampling grid resolution for the surfacer
resolution = 5

# When setting up the rbf evaluator for isosurfacing we need to add a buffer to the
# extents of the evaluator so we don't end up trying to evaluate points outside the
# extents of the evaluator.
bbox_padding = 2
evaluator_extents = np.hstack(
    (
        extents[:3] - resolution * (bbox_padding + 1),
        extents[3:] + resolution * (bbox_padding + 1),
    )
)

# Build the RBF evaluator
rbfi.build_evaluator(evaluator_extents)

# Define the isosurfacing evaluation function to use the rbf evaluator
isosurface_fn = lambda targets: rbfi.evaluate_targets(targets)

# Define the isovalue at which to surface
isovalue = 0.0

# Extract the isosurface
verts, faces = surface_nets(
    extents,
    resolution,
    isovalue,
    isosurface_fn,
    source_points,
    source_values,
)

# Save the isosurface out to an obj file
surface_name = f"isosurface_linear_{resolution}m"
outpath = Path(f"{surface_name}.obj")
save_obj(str(outpath), surface_name, verts, faces)
