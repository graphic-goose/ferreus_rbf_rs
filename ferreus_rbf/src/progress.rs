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

use std::fmt::Debug;
use std::sync::{Arc, mpsc};
use std::thread;

use ferreus_rmt::progress as rmt_progress;

/// Progress events emitted during long-running computations.
#[derive(Debug, Clone)]
pub enum ProgressMsg {
    /// Event indicating that duplicate source points were removed.
    DuplicatesRemoved { num_duplicates: usize },

    /// Event indicating iteration status for an iterative solver.
    SolverIteration {
        iter: usize,
        residual: f64,
        progress: f64,
    },

    /// Event indicating progress for isosurface extraction.
    SurfacingProgress {
        isovalue: f64,
        stage: String,
        progress: f64,
    },

    /// Arbitrary informational message.
    Message { message: String },
}

/// Sink that consumes progress messages.
pub trait ProgressSink: Send + Sync + Debug {
    fn emit(&self, msg: ProgressMsg);
}

/// Extension helpers for adapting RBF progress sinks to related crates.
pub trait ProgressSinkExt {
    /// Wraps this RBF progress sink so it can receive RMT progress events.
    fn into_rmt_progress(self) -> Arc<dyn rmt_progress::ProgressSink>;
}

impl ProgressSinkExt for Arc<dyn ProgressSink> {
    #[inline]
    fn into_rmt_progress(self) -> Arc<dyn rmt_progress::ProgressSink> {
        Arc::new(RmtProgressSinkAdapter { sink: self })
    }
}

/// Adapter that maps RMT progress messages into RBF progress messages.
#[derive(Debug)]
pub struct RmtProgressSinkAdapter {
    sink: Arc<dyn ProgressSink>,
}

impl rmt_progress::ProgressSink for RmtProgressSinkAdapter {
    #[inline]
    fn emit(&self, msg: rmt_progress::ProgressMsg) {
        let msg = match msg {
            rmt_progress::ProgressMsg::IsosurfaceProgress {
                isovalue,
                stage,
                progress,
            } => ProgressMsg::SurfacingProgress {
                isovalue,
                stage: stage.to_string(),
                progress,
            },
            rmt_progress::ProgressMsg::Message { message } => ProgressMsg::Message { message },
        };

        self.sink.emit(msg);
    }
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
pub(crate) fn progress_from_rel(current_res: f64, start_res: f64, target_res: f64) -> f64 {
    if current_res <= target_res {
        1.0
    } else {
        (start_res.log10() - current_res.log10()) / (start_res.log10() - target_res.log10())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    struct CollectSink {
        messages: Mutex<Vec<ProgressMsg>>,
    }

    impl ProgressSink for CollectSink {
        fn emit(&self, msg: ProgressMsg) {
            self.messages.lock().unwrap().push(msg);
        }
    }

    #[test]
    fn maps_rmt_progress_messages_to_rbf_progress_messages() {
        let sink = Arc::new(CollectSink::default());
        let rbf_sink: Arc<dyn ProgressSink> = sink.clone();
        let rmt_sink = rbf_sink.into_rmt_progress();

        rmt_sink.emit(rmt_progress::ProgressMsg::IsosurfaceProgress {
            isovalue: 1.5,
            stage: rmt_progress::IsosurfaceStage::BuildingFacets,
            progress: 0.75,
        });
        rmt_sink.emit(rmt_progress::ProgressMsg::Message {
            message: "done".to_string(),
        });

        let messages = sink.messages.lock().unwrap();
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            ProgressMsg::SurfacingProgress {
                isovalue,
                stage,
                progress,
            } => {
                assert_eq!(*isovalue, 1.5);
                assert_eq!(stage, "Building facets");
                assert_eq!(*progress, 0.75);
            }
            msg => panic!("expected surfacing progress message, got {msg:?}"),
        }

        match &messages[1] {
            ProgressMsg::Message { message } => assert_eq!(message, "done"),
            msg => panic!("expected text progress message, got {msg:?}"),
        }
    }
}
