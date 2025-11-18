The name of the polynomial order to add to the RBF system.

To ensure a unique solution to the RBF system of equations,
some kernels have a minimum required polynomial that must
be added, as shown below.

|    Kernel       |  Minimum drift  |  Default drift  |
|---------------- |-----------------|-----------------|
| Linear          | Constant        | Constant        |
| ThinPlateSpline | Linear          | Linear          |
| Cubic           | Linear          | Linear          |
| Spheroidal      | None            | None            |