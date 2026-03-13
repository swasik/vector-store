/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::IndexKey;
use crate::Progress;
use crate::node_state::NodeState;
use crate::node_state::NodeStateExt;
use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::Instrument;
use tracing::debug;
use tracing::debug_span;

const POLL_INTERVAL: Duration = Duration::from_millis(500);
const PROGRESS_BAR_LENGTH: u64 = 1000;

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold} [{wide_bar:.green/dim}] {msg}")
        .expect("valid progress bar template")
        .progress_chars("━╸─")
}

fn finished_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold} [{wide_bar:.green}] {msg}")
        .expect("valid progress bar template")
        .progress_chars("━╸─")
}

/// Spawns a background task that displays progress bars for indexes being built.
///
/// The task periodically polls the node state for building indexes and updates
/// progress bars accordingly. It runs until all building indexes are finished
/// and the node reaches Serving state, or until the node state channel is closed.
///
/// Progress bars are only shown when running in a terminal (stdout is a TTY).
pub fn spawn(
    node_state: mpsc::Sender<NodeState>,
    multi: MultiProgress,
) -> tokio::task::JoinHandle<()> {
    // Use cursor movement instead of line clearing to avoid visible flicker
    // when progress bars are redrawn.
    multi.set_move_cursor(true);

    tokio::spawn(
        async move {
            debug!("starting");
            let mut bars: HashMap<IndexKey, ProgressBar> = HashMap::new();
            let mut seen_building = false;

            loop {
                let building_indexes = node_state.get_building_indexes().await;

                if !building_indexes.is_empty() {
                    seen_building = true;
                }

                // Remove bars for indexes that are no longer building
                bars.retain(|key, bar| {
                    if !building_indexes.iter().any(|b| &b.index_key == key) {
                        bar.set_position(PROGRESS_BAR_LENGTH);
                        bar.set_style(finished_style());
                        bar.set_message("done");
                        bar.finish();
                        false
                    } else {
                        true
                    }
                });

                // Update or create bars for building indexes
                for build_progress in &building_indexes {
                    let bar = bars
                        .entry(build_progress.index_key.clone())
                        .or_insert_with(|| {
                            let pb = multi.add(ProgressBar::new(PROGRESS_BAR_LENGTH));
                            pb.set_style(bar_style());
                            pb.set_prefix(format!("Building {}", build_progress.index_key));
                            pb
                        });

                    match &build_progress.progress {
                        Progress::InProgress(percentage) => {
                            let position =
                                (percentage.get() / 100.0 * PROGRESS_BAR_LENGTH as f64) as u64;
                            bar.set_position(position);
                            bar.set_message(format!("{:.1}%", percentage.get()));
                        }
                        Progress::Done => {
                            bar.set_position(PROGRESS_BAR_LENGTH);
                            bar.set_style(finished_style());
                            bar.set_message("done");
                            bar.finish();
                        }
                    }
                }

                // Stop when we've seen building indexes and they're all done
                if seen_building && building_indexes.is_empty() {
                    break;
                }

                tokio::time::sleep(POLL_INTERVAL).await;
            }

            // Clean up remaining bars
            for (_, bar) in bars {
                bar.finish_and_clear();
            }

            debug!("finished");
        }
        .instrument(debug_span!("progress_display")),
    )
}

/// Writer adapter that routes output through [`MultiProgress`] so that log
/// lines do not overwrite progress bars.
#[derive(Debug)]
pub struct ProgressWriter {
    multi: MultiProgress,
}

impl ProgressWriter {
    pub fn new(multi: MultiProgress) -> Self {
        Self { multi }
    }
}

impl std::io::Write for ProgressWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // indicatif's println expects a string without a trailing newline
        let s = String::from_utf8_lossy(buf);
        let s = s.trim_end_matches('\n');
        if !s.is_empty() {
            self.multi.println(s).map_err(std::io::Error::other)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Clone for ProgressWriter {
    fn clone(&self) -> Self {
        Self {
            multi: self.multi.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_style_is_valid() {
        // Ensure the template compiles without panic
        let _ = bar_style();
    }

    #[test]
    fn finished_style_is_valid() {
        let _ = finished_style();
    }
}
