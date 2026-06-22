/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines the public triangle mesh representation and OBJ file helpers.
//
// Created on: 17 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Public triangle mesh representation.
//!
//! Meshes are stored as dense `faer` matrices. `vertices` has one row per vertex and three
//! columns for the XYZ coordinates. `facets` has one row per triangle and three columns for
//! zero-based vertex indices.

use faer::Mat;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

/// Triangle mesh returned by RMT extraction.
pub struct Mesh {
    /// Vertex coordinates as an `N x 3` matrix.
    pub vertices: Mat<f64>,

    /// Triangle vertex indices as an `M x 3` matrix.
    pub facets: Mat<usize>,
}

impl Mesh {
    /// Saves this triangle mesh as a Wavefront OBJ file.
    ///
    /// The mesh must contain three coordinate columns and triangular facets. Facet indices are
    /// written using OBJ's 1-based indexing.
    pub fn save_obj<P: AsRef<Path>>(&self, path: P, name: &str) -> Result<()> {
        let (nv, dv) = self.vertices.shape();
        let (nf, k) = self.facets.shape();

        println!("nv: {:?}, mf: {:?}", nv, nf);

        if dv != 3 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("vertices must be (V x 3), got (V x {dv})"),
            ));
        }
        if k != 3 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("facets must be triangles (F x 3), got (F x {k})"),
            ));
        }

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "# isosurface (triangles)")?;
        writeln!(w, "# object: {name}")?;

        // Empty meshes are invalid OBJ outputs for the extraction workflow.
        if nv == 0 || nf == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "mesh is empty (no vertices or facets)",
            ));
        }

        // Object header.
        writeln!(w, "\no {}", name)?;
        writeln!(w, "s off")?;

        // Vertices.
        for r in 0..nv {
            let x = self.vertices.get(r, 0);
            let y = self.vertices.get(r, 1);
            let z = self.vertices.get(r, 2);
            writeln!(w, "v {} {} {}", x, y, z)?;
        }

        // Facets. OBJ uses 1-based indexing.
        for r in 0..nf {
            let a = self.facets.get(r, 0);
            let b = self.facets.get(r, 1);
            let c = self.facets.get(r, 2);
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
}
