/////////////////////////////////////////////////////////////////////////////////////////////
//
// Writes extracted isosurfaces to OBJ files from RBF-generated vertex and face data.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use std::fs::File;
use std::io::{BufWriter, Write};
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use faer::MatRef;

/// Write an isosurface to an OBJ file.
///
/// - `name`: object name to write as `o <name>`
/// - `verts`: (V × 3) positions (f64)
/// - `faces`: (F × 3) triangle indices (usize, **0-based**)
///
/// # Errors
/// - `InvalidInput` if `verts.ncols() != 3` or `faces.ncols() != 3` or mesh is empty.
/// - `InvalidData` if any face index is out of range (`>= V`).
pub fn save_obj<P: AsRef<Path>>(
    path: P,
    name: &str,
    verts: MatRef<f64>,
    faces: MatRef<usize>,
) -> Result<()> {
    let (nv, dv) = (verts.nrows(), verts.ncols());
    let (nf, k)  = (faces.nrows(), faces.ncols());

    if dv != 3 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("vertices must be (V x 3), got (V x {dv})"),
        ));
    }
    if k != 3 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("faces must be triangles (F x 3), got (F x {k})"),
        ));
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# isosurface (triangles)")?;
    writeln!(w, "# object: {name}")?;

    // If empty, return error
    if nv == 0 || nf == 0 {
        return Err(Error::new(
            ErrorKind::InvalidInput, 
            "mesh is empty (no verts or faces)"
        ));
    }

    // Object header
    writeln!(w, "\no {}", name)?;

    // Vertices
    for r in 0..nv {
        let x = verts.get(r, 0);
        let y = verts.get(r, 1);
        let z = verts.get(r, 2);
        writeln!(w, "v {} {} {}", x, y, z)?;
    }

    // Faces (OBJ is 1-based indexing)
    for r in 0..nf {
        let a = faces.get(r, 0);
        let b = faces.get(r, 1);
        let c = faces.get(r, 2);
        if *a >= nv || *b >= nv || *c >= nv {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("face {r}: index out of bounds (nv = {nv})"),
            ));
        }
        writeln!(w, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }

    w.flush()
}
