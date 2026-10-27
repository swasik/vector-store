/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

mod actor;
mod diskann;
mod factory;
mod opensearch;
mod usearch;
mod validator;

use crate::Config;
use crate::ModifyAdmission;
use crate::memory::Memory;
use crate::worker::Worker;
use actor::AnnR;
pub(crate) use actor::CountR;
use actor::Message;
pub(crate) use actor::VsIndexModify;
pub(crate) use actor::VsIndexModifyExt;
pub(crate) use actor::VsIndexSearch;
pub(crate) use actor::VsIndexSearchExt;
pub(crate) use factory::VsIndexConfiguration;
pub(crate) use factory::VsIndexFactory;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::watch;
pub(crate) use validator::Error;

async fn recv(
    rx_search: &mut mpsc::Receiver<VsIndexSearch>,
    rx_modify: &mut mpsc::Receiver<VsIndexModify>,
) -> Option<Message> {
    tokio::select! {
        // The order of the select branches is important. We want to prioritize search messages
        // over modify messages. We shouldn't starve modify messages since tokio runtime uses a
        // fair scheduler. From observations, it is visible that providing new search requests is
        // working in waves: first buffered then fully consumed.
        biased;

        Some(msg) = rx_search.recv() => Some(Message::Search(msg)),
        Some(msg) = rx_modify.recv() => Some(Message::Modify(msg)),
        else => None,
    }
}

/// The messages of an index actor, searches first, with modifications admitted as
/// `Config::modify_admission` says; it follows that setting as the configuration reloads.
struct Admission {
    config: watch::Receiver<Arc<Config>>,
    policy: ModifyAdmission,
    batch_left: usize,
    batch_started: Option<Instant>,
    last_modify: Instant,
}

impl Admission {
    fn new(mut config: watch::Receiver<Arc<Config>>) -> Self {
        let policy = config.borrow_and_update().modify_admission;
        Self {
            config,
            policy,
            batch_left: 0,
            batch_started: None,
            last_modify: Instant::now(),
        }
    }

    async fn recv(
        &mut self,
        rx_search: &mut mpsc::Receiver<VsIndexSearch>,
        rx_modify: &mut mpsc::Receiver<VsIndexModify>,
    ) -> Option<Message> {
        if self.config.has_changed().unwrap_or(false) {
            self.policy = self.config.borrow_and_update().modify_admission;
        }
        if let Some(msg) = self.next_in_batch(rx_modify) {
            return Some(Message::Modify(msg));
        }
        if let Some(msg) = self.overdue(rx_modify) {
            self.start_batch();
            return Some(Message::Modify(msg));
        }
        let msg = recv(rx_search, rx_modify).await?;
        if matches!(msg, Message::Modify(_)) {
            self.start_batch();
        }
        Some(msg)
    }

    fn start_batch(&mut self) {
        self.batch_left = match (self.policy.batch_size, self.policy.batch_time) {
            (0, None) => 0,
            (0, Some(_)) => usize::MAX,
            (size, _) => size,
        };
        self.batch_started = None;
        self.last_modify = Instant::now();
    }

    /// The next queued modification of the current batch. The batch's time counts from the moment
    /// its first modification was dispatched, after the searches in flight finished.
    fn next_in_batch(
        &mut self,
        rx_modify: &mut mpsc::Receiver<VsIndexModify>,
    ) -> Option<VsIndexModify> {
        if self.batch_left == 0 {
            return None;
        }
        let started = *self.batch_started.get_or_insert_with(Instant::now);
        if self
            .policy
            .batch_time
            .is_some_and(|time| started.elapsed() >= time)
        {
            crate::diag::event("actor_batch_end_time");
            self.batch_left = 0;
            return None;
        }
        let Ok(msg) = rx_modify.try_recv() else {
            crate::diag::event("actor_batch_end_empty");
            self.batch_left = 0;
            return None;
        };
        self.batch_left -= 1;
        if self.batch_left == 0 {
            crate::diag::event("actor_batch_end_size");
        }
        self.last_modify = Instant::now();
        Some(msg)
    }

    /// A queued modification taken ahead of waiting searches: none was taken for `max_wait`.
    fn overdue(&mut self, rx_modify: &mut mpsc::Receiver<VsIndexModify>) -> Option<VsIndexModify> {
        let max_wait = self.policy.max_wait?;
        if self.last_modify.elapsed() < max_wait {
            return None;
        }
        let msg = rx_modify.try_recv().ok()?;
        crate::diag::event("actor_modify_overdue");
        Some(msg)
    }
}

pub(crate) fn new_index_factory_usearch(
    config_tx: watch::Receiver<Arc<Config>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
) -> anyhow::Result<Box<dyn VsIndexFactory + Send + Sync>> {
    Ok(Box::new(usearch::new_usearch(config_tx, worker, memory)?))
}

pub(crate) fn new_index_factory_opensearch(
    addr: String,
    config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<Box<dyn VsIndexFactory + Send + Sync>> {
    Ok(Box::new(opensearch::new_opensearch(&addr, config_rx)?))
}

pub(crate) fn new_index_factory_diskann(
    config_rx: watch::Receiver<Arc<Config>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
) -> anyhow::Result<Box<dyn VsIndexFactory + Send + Sync>> {
    Ok(Box::new(diskann::new_diskann(config_rx, worker, memory)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexKey;
    use crate::table::PartitionId;
    use std::time::Duration;
    use tokio::sync::oneshot;

    fn add() -> VsIndexModify {
        VsIndexModify::RemovePartition {
            partition_id: PartitionId::from(0),
        }
    }

    fn search() -> VsIndexSearch {
        let (tx, _rx) = oneshot::channel();
        VsIndexSearch::Count {
            index_key: IndexKey::new(&"ks".into(), &"idx".into()),
            tx,
        }
    }

    struct Actor {
        admission: Admission,
        config: watch::Sender<Arc<Config>>,
        rx_search: mpsc::Receiver<VsIndexSearch>,
        rx_modify: mpsc::Receiver<VsIndexModify>,
        tx_search: mpsc::Sender<VsIndexSearch>,
        tx_modify: mpsc::Sender<VsIndexModify>,
    }

    impl Actor {
        fn new(policy: ModifyAdmission) -> Self {
            let (config, config_rx) = watch::channel(Arc::new(Config {
                modify_admission: policy,
                ..Config::default()
            }));
            let (tx_search, rx_search) = mpsc::channel(100);
            let (tx_modify, rx_modify) = mpsc::channel(100);
            Self {
                admission: Admission::new(config_rx),
                config,
                rx_search,
                rx_modify,
                tx_search,
                tx_modify,
            }
        }

        async fn queue(&self, searches: usize, modifies: usize) {
            for _ in 0..searches {
                self.tx_search.send(search()).await.unwrap();
            }
            for _ in 0..modifies {
                self.tx_modify.send(add()).await.unwrap();
            }
        }

        /// The kinds of the next `count` messages: 'S' for a search, 'M' for a modification.
        async fn take(&mut self, count: usize) -> String {
            let mut kinds = String::new();
            for _ in 0..count {
                let msg = self
                    .admission
                    .recv(&mut self.rx_search, &mut self.rx_modify)
                    .await
                    .unwrap();
                kinds.push(if matches!(msg, Message::Search(_)) {
                    'S'
                } else {
                    'M'
                });
            }
            kinds
        }
    }

    #[tokio::test]
    async fn searches_go_first_and_one_modify_fills_a_gap_by_default() {
        let mut actor = Actor::new(ModifyAdmission::default());
        actor.queue(2, 3).await;
        assert_eq!(actor.take(3).await, "SSM");
        actor.queue(2, 0).await;
        assert_eq!(actor.take(3).await, "SSM");
    }

    #[tokio::test]
    async fn a_batch_takes_queued_modifies_ahead_of_searches() {
        let mut actor = Actor::new(ModifyAdmission {
            batch_size: 2,
            ..ModifyAdmission::default()
        });
        actor.queue(1, 5).await;
        assert_eq!(actor.take(1).await, "S");
        actor.queue(2, 0).await;
        assert_eq!(actor.take(4).await, "SSMM"); // the gap after the searches starts a batch
        actor.queue(1, 0).await;
        assert_eq!(actor.take(3).await, "MSM"); // the batch of three ends at its size
    }

    #[tokio::test]
    async fn a_batch_ends_after_its_time() {
        let mut actor = Actor::new(ModifyAdmission {
            batch_time: Some(Duration::from_millis(50)),
            ..ModifyAdmission::default()
        });
        actor.queue(0, 5).await;
        assert_eq!(actor.take(2).await, "MM");
        tokio::time::sleep(Duration::from_millis(60)).await;
        actor.queue(1, 0).await;
        assert_eq!(actor.take(2).await, "SM");
    }

    #[tokio::test]
    async fn an_overdue_modify_goes_ahead_of_queued_searches() {
        let mut actor = Actor::new(ModifyAdmission {
            max_wait: Some(Duration::from_millis(50)),
            ..ModifyAdmission::default()
        });
        actor.queue(0, 1).await;
        assert_eq!(actor.take(1).await, "M");
        actor.queue(3, 1).await;
        assert_eq!(actor.take(1).await, "S");
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(actor.take(1).await, "M");
    }

    #[tokio::test]
    async fn the_policy_follows_a_config_reload() {
        let mut actor = Actor::new(ModifyAdmission::default());
        actor.config.send_modify(|config| {
            *config = Arc::new(Config {
                modify_admission: ModifyAdmission {
                    batch_size: 3,
                    ..ModifyAdmission::default()
                },
                ..Config::clone(config)
            })
        });
        actor.queue(0, 4).await;
        actor.queue(1, 0).await;
        assert_eq!(actor.take(5).await, "SMMMM");
    }
}
