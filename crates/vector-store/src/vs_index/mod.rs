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
use tokio::sync::mpsc;
use tokio::sync::watch;
pub(crate) use validator::Error;

/// Receives the next message for an index actor, from the search channel or the
/// modify channel.
///
/// While only one of the channels has a message pending, that message is taken,
/// so searches are still served ahead of an empty modify queue. When both have a
/// message pending, `select!` picks a branch at random, which bounds how long a
/// modification can wait behind searches.
///
/// The branches must not be `biased` towards searches. A `biased` select polls
/// branches in declaration order and takes the first ready one, so a query
/// stream that keeps the search channel non-empty starves modifications
/// indefinitely. That backpressures the whole ingest pipeline: the modify
/// channel fills, `monitor_items` blocks on it, the embeddings channel fills,
/// and the CDC consumer tasks block while holding their semaphore permits,
/// which freezes both CDC readers. The index then diverges from the base table
/// silently, because the readers are alive and `cdc_reader_up` stays 1
/// (VECTOR-951).
async fn recv(
    rx_search: &mut mpsc::Receiver<VsIndexSearch>,
    rx_modify: &mut mpsc::Receiver<VsIndexModify>,
) -> Option<Message> {
    tokio::select! {
        Some(msg) = rx_search.recv() => Some(Message::Search(msg)),
        Some(msg) = rx_modify.recv() => Some(Message::Modify(msg)),
        else => None,
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
    use crate::IndexName;
    use crate::KeyspaceName;
    use crate::table::PartitionId;
    use tokio::sync::oneshot;

    const CHANNEL_SIZE: usize = 4;

    fn search() -> VsIndexSearch {
        let (tx, _rx) = oneshot::channel();
        VsIndexSearch::Count {
            index_key: IndexKey::new(
                &KeyspaceName::from("ks".to_string()),
                &IndexName::from("idx".to_string()),
            ),
            tx,
        }
    }

    fn modify() -> VsIndexModify {
        VsIndexModify::RemovePartition {
            partition_id: PartitionId::from(0u64),
        }
    }

    /// Reproduces VECTOR-951: a query stream that keeps the search channel
    /// non-empty must not starve a pending modification.
    ///
    /// Sustained load is modelled by refilling the search channel every time a
    /// search is taken, so the search branch is ready on every iteration, as it
    /// is under a saturated ANN workload.
    #[tokio::test]
    async fn recv_does_not_starve_modify_under_sustained_search_load() {
        const ITERATIONS: usize = 1000;

        let (tx_search, mut rx_search) = mpsc::channel(CHANNEL_SIZE);
        let (tx_modify, mut rx_modify) = mpsc::channel(CHANNEL_SIZE);

        tx_modify.send(modify()).await.unwrap();
        tx_search.send(search()).await.unwrap();

        for _ in 0..ITERATIONS {
            match recv(&mut rx_search, &mut rx_modify).await {
                Some(Message::Modify(_)) => return,
                Some(Message::Search(_)) => {
                    // The next query is already waiting, as under sustained load.
                    _ = tx_search.try_send(search());
                }
                None => panic!("both channels closed while senders are alive"),
            }
        }

        panic!("modify starved: {ITERATIONS} searches served, modify never delivered");
    }

    /// Searches must still win while nothing is pending on the modify channel,
    /// so the fairness above does not cost query latency when there is no
    /// ingest backlog.
    #[tokio::test]
    async fn recv_takes_search_when_no_modify_is_pending() {
        let (tx_search, mut rx_search) = mpsc::channel(CHANNEL_SIZE);
        let (_tx_modify, mut rx_modify) = mpsc::channel::<VsIndexModify>(CHANNEL_SIZE);

        tx_search.send(search()).await.unwrap();

        assert!(matches!(
            recv(&mut rx_search, &mut rx_modify).await,
            Some(Message::Search(_))
        ));
    }

    /// A closed search channel must not disable receiving of modifications.
    #[tokio::test]
    async fn recv_takes_modify_after_search_channel_is_closed() {
        let (tx_search, mut rx_search) = mpsc::channel::<VsIndexSearch>(CHANNEL_SIZE);
        let (tx_modify, mut rx_modify) = mpsc::channel(CHANNEL_SIZE);

        drop(tx_search);
        tx_modify.send(modify()).await.unwrap();

        assert!(matches!(
            recv(&mut rx_search, &mut rx_modify).await,
            Some(Message::Modify(_))
        ));
    }

    /// Both channels closed terminates the actor loop.
    #[tokio::test]
    async fn recv_returns_none_when_both_channels_are_closed() {
        let (tx_search, mut rx_search) = mpsc::channel::<VsIndexSearch>(CHANNEL_SIZE);
        let (tx_modify, mut rx_modify) = mpsc::channel::<VsIndexModify>(CHANNEL_SIZE);

        drop(tx_search);
        drop(tx_modify);

        assert!(recv(&mut rx_search, &mut rx_modify).await.is_none());
    }
}
