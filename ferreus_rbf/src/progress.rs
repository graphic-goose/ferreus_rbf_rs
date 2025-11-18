/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines progress reporting messages, sinks, and helper functions for long-running processes.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! Progress reporting primitives for long-running computations.

use std::sync::{Arc, mpsc};
use std::thread;
use std::fmt::Debug;

/// Progress events emitted during long-running computations.
#[derive(Debug, Clone)]
pub enum ProgressMsg {
    /// Event indicating that duplicate source points were removed.
    DuplicatesRemoved { num_duplicates: usize },

    /// Event indicating iteration status for an iterative solver.
    SolverIteration { iter: usize, residual: f64, progress: f64},

    /// Event indicating progress for isosurface extraction.
    SurfacingProgress { isovalue: f64, stage: String, progress: f64},

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

/// Calculates the percentage progress of the solver based on the
/// current residual and the requested accuracy tolerance. Returns
/// the percentage as a value between [0, 1].
#[inline]
pub (crate) fn progress_from_rel(current_res: f64, start_res: f64, target_res: f64) -> f64 {
    if current_res <= target_res { 1.0 } else { 
        (start_res.log10() - current_res.log10()) / (start_res.log10() - target_res.log10())
     }
}
