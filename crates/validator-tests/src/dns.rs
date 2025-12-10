/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use async_backtrace::framed;
use std::net::Ipv4Addr;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

pub enum Dns {
    Version { tx: oneshot::Sender<String> },
    Domain { tx: oneshot::Sender<String> },
    Remove { name: String },
    Upsert { name: String, ip: Ipv4Addr },
}

pub trait DnsExt {
    /// Returns the version of the DNS server.
    fn version(&self) -> impl Future<Output = String>;

    /// Returns the domain name of the DNS server.
    fn domain(&self) -> impl Future<Output = String>;

    /// Remove an A DNS record with the given name.
    fn remove(&self, name: String) -> impl Future<Output = ()>;

    /// Upserts an A DNS record with the given name and IP address.
    fn upsert(&self, name: String, ip: Ipv4Addr) -> impl Future<Output = ()>;
}

impl DnsExt for mpsc::Sender<Dns> {
    #[framed]
    async fn version(&self) -> String {
        let (tx, rx) = oneshot::channel();
        self.send(Dns::Version { tx })
            .await
            .expect("DnsExt::version: internal actor should receive request");
        rx.await
            .expect("DnsExt::version: internal actor should send response")
    }

    #[framed]
    async fn domain(&self) -> String {
        let (tx, rx) = oneshot::channel();
        self.send(Dns::Domain { tx })
            .await
            .expect("DnsExt::domain: internal actor should receive request");
        rx.await
            .expect("DnsExt::domain: internal actor should send response")
    }

    #[framed]
    async fn remove(&self, name: String) {
        self.send(Dns::Remove { name })
            .await
            .expect("DnsExt::remove: internal actor should receive request");
    }

    #[framed]
    async fn upsert(&self, name: String, ip: Ipv4Addr) {
        self.send(Dns::Upsert { name, ip })
            .await
            .expect("DnsExt::upsert: internal actor should receive request");
    }
}
