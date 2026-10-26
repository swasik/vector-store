/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! VECTOR-951 diagnostics: where a CDC row waits on its way into the index.
//!
//! Label-free on purpose (one index per experiment), registered into the
//! service registry by `Metrics::new`.

use prometheus::GaugeVec;
use prometheus::HistogramVec;
use prometheus::IntCounterVec;
use prometheus::Registry;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

pub(crate) struct Diag {
    pub(crate) wait: HistogramVec,
    pub(crate) events: IntCounterVec,
    pub(crate) queue: GaugeVec,
}

pub(crate) static DIAG: LazyLock<Diag> = LazyLock::new(|| {
    let buckets = vec![
        0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0,
    ];
    Diag {
        wait: HistogramVec::new(
            prometheus::HistogramOpts::new(
                "vsdiag_wait_seconds",
                "VECTOR-951: time spent in one stage of the ingest or search path",
            )
            .buckets(buckets),
            &["stage"],
        )
        .unwrap(),
        events: IntCounterVec::new(
            prometheus::Opts::new("vsdiag_events_total", "VECTOR-951: pipeline events"),
            &["event"],
        )
        .unwrap(),
        queue: GaugeVec::new(
            prometheus::Opts::new("vsdiag_queue_len", "VECTOR-951: last seen queue length"),
            &["queue"],
        )
        .unwrap(),
    }
});

pub(crate) fn register(registry: &Registry) {
    registry.register(Box::new(DIAG.wait.clone())).unwrap();
    registry.register(Box::new(DIAG.events.clone())).unwrap();
    registry.register(Box::new(DIAG.queue.clone())).unwrap();
}

pub(crate) fn observe(stage: &str, start: Instant) {
    DIAG.wait
        .with_label_values(&[stage])
        .observe(start.elapsed().as_secs_f64());
}

pub(crate) fn event(event: &str) {
    DIAG.events.with_label_values(&[event]).inc();
}

pub(crate) fn queue(queue: &str, len: usize) {
    DIAG.queue.with_label_values(&[queue]).set(len as f64);
}

/// Measures how late the runtime runs a task that should wake after 10 ms:
/// a busy runtime shows up here before anything else does.
pub(crate) fn spawn_runtime_probe() {
    const SLEEP: Duration = Duration::from_millis(10);
    tokio::spawn(async {
        loop {
            let start = Instant::now();
            tokio::time::sleep(SLEEP).await;
            DIAG.wait
                .with_label_values(&["runtime_wakeup_delay"])
                .observe(start.elapsed().saturating_sub(SLEEP).as_secs_f64());
            tokio::time::sleep(Duration::from_millis(90)).await;
        }
    });
}
