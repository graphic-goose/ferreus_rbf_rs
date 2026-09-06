/////////////////////////////////////////////////////////////////////////////////////////////
//
// Builds a threadpool in Windows that only uses performance-cores instead of economy-cores.
//
// Created on: 6 Sep 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # worker_pool
//!
//! Module that builds and reuses a Rayon thread pool for large RBF solves.
//!
//! On Windows, the worker threads are assigned the set of logical processors
//! associated with the CPU's performance cores. This prevents work from being
//! unnecessarily scheduled on the slower efficiency cores of a hybrid CPU,
//! while allowing Windows to move work between the available performance cores.
//!
//! On other operating systems, or when the CPU topology cannot be detected,
//! operations use Rayon's existing execution context.

use rayon::ThreadPool;
use std::sync::OnceLock;

/// The performance-core worker pool, constructed once when it is first required.
static WORKER_POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();

/// Runs an operation inside the performance-core worker pool when one is available.
/// Otherwise, runs the operation in the calling thread's existing Rayon context.
pub(crate) fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    match WORKER_POOL.get_or_init(build_worker_pool) {
        Some(pool) => pool.install(operation),
        None => operation(),
    }
}

/// Builds a Rayon thread pool whose workers prefer the logical processors belonging
/// to the CPU's performance cores.
#[cfg(target_os = "windows")]
fn build_worker_pool() -> Option<ThreadPool> {
    use gdt_cpus::{CpuInfo, set_thread_soft_affinity};

    // Detect the processor topology and select every logical processor associated
    // with a performance core. This includes simultaneous multithreading siblings.
    let cpu_info = CpuInfo::detect().ok()?;
    let performance_cpus = cpu_info.performance_core_mask();
    let worker_count = performance_cpus.count();

    // Fall back to the existing Rayon context if no suitable processors were found.
    if worker_count == 0 {
        return None;
    }

    // Each worker receives the same performance-core CPU set. Rayon remains free to
    // move work between workers, while Windows can place each worker on any available
    // performance core in the set.
    rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .thread_name(|index| format!("ferreus-pcore-{index}"))
        .start_handler(move |_| {
            if let Err(error) = set_thread_soft_affinity(&performance_cpus) {
                eprintln!("Failed to select the P-core CPU set: {error}");
            }
        })
        .build()
        .ok()
}

/// Uses the existing Rayon execution context on platforms where the Windows
/// performance-core worker pool is not applicable.
#[cfg(not(target_os = "windows"))]
fn build_worker_pool() -> Option<ThreadPool> {
    None
}
