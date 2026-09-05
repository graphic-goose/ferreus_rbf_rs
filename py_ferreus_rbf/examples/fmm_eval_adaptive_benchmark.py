# Temporary example to test dense rbf evaluation performance

import time

import numpy as np

from ferreus_rbf import RBFInterpolator
from ferreus_rbf.config import FmmCompressionType, FmmParams, Params
from ferreus_rbf.interpolant_config import InterpolantSettings, RBFKernelType

RADIUS = 5.0
N_PER_SPHERE = 10_000
BASE_RANGE = 40.0
INTERPOLATION_ORDER = 7

rng = np.random.default_rng(0)


def sphere_sdf_points(center: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points filling the sphere region, denser near the centre, with signed distances."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.0, 1.5 * RADIUS, size=(n, 1))
    points = center + directions * radii
    values = np.linalg.norm(points - center, axis=1) - RADIUS
    return points, values


points_a, values_a = sphere_sdf_points(np.array([5.0, 5.0, 5.0]), N_PER_SPHERE)
points_b, values_b = sphere_sdf_points(np.array([30.0, 30.0, 5.0]), N_PER_SPHERE)

source_points = np.vstack((points_a, points_b))
source_values = np.concatenate((values_a, values_b))

axis = np.arange(0.0, 40.0 + 0.25, 0.25)
targets = np.stack(
    np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
).reshape(-1, 3)

interpolant_settings = InterpolantSettings(
    RBFKernelType.Spheroidal,
    base_range=BASE_RANGE,
)


def make_params(eval_adaptive: bool) -> Params:
    fmm_params = FmmParams(
        interpolation_order=INTERPOLATION_ORDER,
        max_points_per_cell=256,
        compression_type=FmmCompressionType.ACA,
        epsilon=10.0**-INTERPOLATION_ORDER,
        eval_chunk_size=1024,
        eval_adaptive=eval_adaptive,
    )
    return Params(RBFKernelType.Spheroidal, fmm_params=fmm_params)


print(f"Sources: {source_points.shape[0]}  Targets: {targets.shape[0]}")

for eval_adaptive in (True, False):
    rbfi = RBFInterpolator(
        source_points,
        source_values,
        interpolant_settings,
        params=make_params(eval_adaptive),
    )

    start = time.perf_counter()
    values = rbfi.evaluate(targets)
    elapsed = time.perf_counter() - start

    print(
        f"eval_adaptive={str(eval_adaptive):<5} "
        f"{elapsed:7.3f} s  "
        f"min={values.min():+.4f} max={values.max():+.4f}"
    )
