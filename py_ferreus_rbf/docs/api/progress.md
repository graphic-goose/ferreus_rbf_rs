# Progress & Events

::: ferreus_rbf.Progress

## Event payloads

::: ferreus_rbf.progress.DuplicatesRemoved

::: ferreus_rbf.progress.SolverIteration

::: ferreus_rbf.progress.Message

::: ferreus_rbf.progress.SurfacingProgress

### Type aliases

::: ferreus_rbf.progress.ProgressEvent
::: ferreus_rbf.progress.ProgressCallback

Example use:
-----------
```python
from ferreus_rbf.progress import (
    Progress,
    SolverIteration,
    SurfacingProgress,
    DuplicatesRemoved,
    Message,
    ProgressEvent
)

# Create a function to handle each ProgressEvent type
def on_progress(event: ProgressEvent) -> None:
    if isinstance(event, DuplicatesRemoved):
        print(f"Removed duplicates: {event.num_duplicates}")
    elif isinstance(event, SolverIteration):
        print(f"Iteration: {event.iter:3d}  {event.residual:>.5E}  {(event.progress * 100):.1f}%")
    elif isinstance(event, Message):
        print(event.message)
    elif isinstance(event, SurfacingProgress):
        print(f"Isovalue: {event.isovalue}  Stage: {event.stage}  {(event.progress * 100):.1f}%")

# Create a Progress instance using the on_progress function
prog = Progress(callback=on_progress)
```

Example DuplicatesRemove output:
-------------------------------
```
Removed duplicates: 15
```

Example SolverIteration output:
------------------------------
```
Iteration:   1  1.80801E+02  16.2%
Iteration:   2  1.53657E+01  37.3%
Iteration:   3  2.42940E+00  53.1%
Iteration:   4  5.06244E-01  66.5%
Iteration:   5  1.35629E-01  77.7%
Iteration:   6  4.87250E-02  86.5%
Iteration:   7  1.17505E-02  98.6%
Iteration:   8  2.45231E-03  100.0%
```

Example Message output:
----------------------
```
Took 2.870149s to solve RBF for 26988 points using the following settings:
Kernel: Spheroidal, Polynomial degree: -1
Fitting accuracy: 0.01, Tolerance type: Absolute
```

Example SurfacingProgress output:
--------------------------------
```
Isovalue: 0.0  Stage: Calculating surface intersections  0.0%
Isovalue: 0.0  Stage: Calculating surface intersections  38.9%
Isovalue: 0.0  Stage: Calculating normals  70.0%
Isovalue: 0.0  Stage: Building quads  80.0%
Isovalue: 0.0  Stage: Building faces  90.0%
Isovalue: 0.0  Stage: Finished  100.0%
```