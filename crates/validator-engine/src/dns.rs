/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use async_backtrace::frame;
use async_backtrace::framed;
use hickory_server::authority::Catalog;
use hickory_server::authority::ZoneType;
use hickory_server::proto::rr::DNSClass;
use hickory_server::proto::rr::LowerName;
use hickory_server::proto::rr::Name;
use hickory_server::proto::rr::RrKey;
use hickory_server::proto::rr::rdata::a::A;
use hickory_server::proto::rr::rdata::soa::SOA;
use hickory_server::proto::rr::record_data::RData;
use hickory_server::proto::rr::record_type::RecordType;
use hickory_server::proto::rr::resource::Record;
use hickory_server::server::ServerFuture;
use hickory_server::store::in_memory::InMemoryAuthority;
use std::net::Ipv4Addr;
use std::str::FromStr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::Instrument;
use tracing::debug;
use tracing::debug_span;
use vector_search_validator_tests::Dns;

#[framed]
/// Starts the DNS server on the given IP address.
pub(crate) async fn new(ip: Ipv4Addr) -> mpsc::Sender<Dns> {
    assert!(ip.is_loopback(), "DNS server should listen on a localhost");

    let (tx, mut rx) = mpsc::channel(10);

    let mut state = State::new().await;

    let socket = UdpSocket::bind((ip, 53))
        .await
        .expect("dns: failed to bind UDP socket");

    let mut catalog = Catalog::new();
    catalog.upsert(
        LowerName::from_str(ZONE).unwrap(),
        vec![state.authority.clone()],
    );
    let mut server = ServerFuture::new(catalog);
    server.register_socket(socket);

    tokio::spawn(
        frame!(async move {
            debug!("starting");

            while let Some(msg) = rx.recv().await {
                process(msg, &mut state).await;
            }

            server
                .shutdown_gracefully()
                .await
                .expect("stop: failed to shutdown server gracefully");

            debug!("stopped");
        })
        .instrument(debug_span!("dns")),
    );

    tx
}

struct State {
    version: String,
    authority: Arc<InMemoryAuthority>,
    serial: u32,
}

const ZONE: &str = "validator.test.";
const TTL: u32 = 60;

impl State {
    #[framed]
    async fn new() -> Self {
        let version = format!("hicory-server-{}", hickory_server::version());

        let authority = Arc::new(InMemoryAuthority::empty(
            Name::from_str(ZONE).unwrap(),
            ZoneType::Primary,
            false,
        ));
        let mut soa = Record::from_rdata(
            Name::from_str(ZONE).unwrap(),
            TTL,
            RData::SOA(SOA::new(
                Name::from_str(ZONE).unwrap(),
                Name::new(),
                0,
                0,
                0,
                0,
                0,
            )),
        );
        soa.set_dns_class(DNSClass::IN);
        authority.upsert(soa, 0).await;

        Self {
            version,
            authority,
            serial: 0,
        }
    }
}

#[framed]
async fn process(msg: Dns, state: &mut State) {
    match msg {
        Dns::Version { tx } => {
            tx.send(state.version.clone())
                .expect("process Dns::Version: failed to send a response");
        }

        Dns::Domain { tx } => {
            tx.send(ZONE[..ZONE.len() - 1].to_string())
                .expect("process Dns::Domain: failed to send a response");
        }

        Dns::Remove { name } => {
            remove(name, state).await;
        }

        Dns::Upsert { name, ip } => {
            upsert(name, ip, state).await;
        }
    }
}

#[framed]
async fn remove(name: String, state: &mut State) {
    state.authority.records_mut().await.remove(&RrKey::new(
        LowerName::from_str(&format!("{name}.{ZONE}")).expect("remove: failed to parse name"),
        RecordType::A,
    ));
}

#[framed]
async fn upsert(name: String, ip: Ipv4Addr, state: &mut State) {
    let name = Name::from_str(&format!("{name}.{ZONE}")).expect("upsert: failed to parse name");

    let serial = state.serial;
    state.serial += 1;

    let octets = ip.octets();
    let mut record = Record::from_rdata(
        name,
        TTL,
        RData::A(A::new(octets[0], octets[1], octets[2], octets[3])),
    );
    record.set_dns_class(DNSClass::IN);

    state.authority.upsert(record, serial).await;
}
