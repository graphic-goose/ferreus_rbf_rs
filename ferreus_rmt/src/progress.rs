/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines progress reporting messages and sinks for regularised marching tetrahedra.
//
// Created on: 13 Jun 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Progress reporting primitives for regularised marching tetrahedra extraction.

use std::fmt::{self, Debug};
use std::sync::{Arc, mpsc};
use std::thread;

/// Coarse stage of an isosurface extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsosurfaceStage {
    ProjectingSeeds,
    ExpandingWavefront,
    ClusteringVertices,
    BuildingFacets,
    CleaningMesh,
    BoundaryClosure,
    Finished,
}

impl IsosurfaceStage {
    /// Human-readable stage name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProjectingSeeds => "Projecting seeds",
            Self::ExpandingWavefront => "Expanding wavefront",
            Self::ClusteringVertices => "Clustering vertices",
            Self::BuildingFacets => "Building facets",
            Self::CleaningMesh => "Cleaning mesh",
            Self::BoundaryClosure => "Boundary closure",
            Self::Finished => "Finished",
        }
    }
}

impl fmt::Display for IsosurfaceStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Progress events emitted during isosurface extraction.
#[derive(Debug, Clone)]
pub enum ProgressMsg {
    /// Event indicating progress for isosurface extraction.
    IsosurfaceProgress {
        isovalue: f64,
        stage: IsosurfaceStage,
        progress: f64,
    },

    /// Arbitrary informational message.
    Message { message: String },
}

/// Sink that consumes progress messages.
pub trait ProgressSink: Send + Sync + Debug {
    fn emit(&self, msg: ProgressMsg);
}

/// Progress sink that forwards messages over a channel.
#[derive(Debug)]
pub struct ClosureSink {
    tx: mpsc::SyncSender<ProgressMsg>,
}

impl ProgressSink for ClosureSink {
    #[inline]
    fn emit(&self, msg: ProgressMsg) {
        let _ = self.tx.try_send(msg);
    }
}

/// Spawns a listener thread that runs a handler closure for each progress message.
pub fn closure_sink<F>(
    buffer: usize,
    mut handler: F,
) -> (Arc<dyn ProgressSink>, thread::JoinHandle<()>)
where
    F: FnMut(ProgressMsg) + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel::<ProgressMsg>(buffer.max(1));
    let sink: Arc<dyn ProgressSink> = Arc::new(ClosureSink { tx });

    let handle = thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            handler(msg);
        }
    });

    (sink, handle)
}
