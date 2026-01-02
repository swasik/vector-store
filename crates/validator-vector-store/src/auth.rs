/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use async_backtrace::framed;
use std::time::Duration;
use tokio::time::sleep;
use tracing::info;
use vector_search_validator_tests::common::*;
use vector_search_validator_tests::*;

const WAITING_FOR_INDEX_DISCOVERY: Duration = Duration::from_secs(10);

#[framed]
pub(crate) async fn new() -> TestCase {
    let timeout = DEFAULT_TEST_TIMEOUT;
    TestCase::empty()
        .with_test(
            "vs_doesnt_work_without_permission",
            timeout,
            vs_doesnt_work_without_permission,
        )
        .with_test(
            "vs_works_when_permission_granted",
            timeout,
            vs_works_when_permission_granted,
        )
}

#[framed]
async fn vs_doesnt_work_without_permission(actors: TestActors) {
    info!("started");

    let mut scylla_configs = get_default_scylla_node_configs(&actors).await;
    let mut vs_configs = get_default_vs_node_configs(&actors);

    for config in scylla_configs.iter_mut() {
        config.config = Some(
            "authenticator: PasswordAuthenticator\nauthorizer: CassandraAuthorizer"
                .as_bytes()
                .to_vec(),
        );
    }

    for config in vs_configs.iter_mut() {
        config.user = Some("alice".to_string());
        config.password = Some("alice_password".to_string());
    }

    init_with_config(actors.clone(), scylla_configs, vs_configs).await;

    let (session, clients) = prepare_connection_with_auth(&actors, "cassandra", "cassandra").await;
    session
        .query_unpaged(
            "CREATE ROLE alice WITH PASSWORD = 'alice_password' AND LOGIN = true",
            (),
        )
        .await
        .expect("failed to create role alice");

    let _ = create_keyspace(&session).await;
    let table = create_table(&session, "pk INT PRIMARY KEY, v1 VECTOR<FLOAT, 3>", None).await;

    // Create index
    session
        .query_unpaged(
            format!("CREATE INDEX idx_v1 ON {table}(v1) USING 'vector_index'"),
            (),
        )
        .await
        .expect("failed to create an index");

    sleep(WAITING_FOR_INDEX_DISCOVERY).await;

    let mut failed_to_create = true;

    for client in clients.iter() {
        if client
            .indexes()
            .await
            .iter()
            .any(|idx| idx.index.to_string() == "idx_v1")
        {
            failed_to_create = false;
            break;
        }
    }
    assert!(
        failed_to_create,
        "Index creation should fail without proper permissions"
    );

    session
        .query_unpaged("DROP ROLE alice", ())
        .await
        .expect("failed to drop role alice");

    cleanup(actors).await;

    info!("finished");
}

#[framed]
async fn vs_works_when_permission_granted(actors: TestActors) {
    info!("started");

    let mut scylla_configs = get_default_scylla_node_configs(&actors).await;
    let mut vs_configs = get_default_vs_node_configs(&actors);

    for config in scylla_configs.iter_mut() {
        config.config = Some(
            "authenticator: PasswordAuthenticator\nauthorizer: CassandraAuthorizer"
                .as_bytes()
                .to_vec(),
        );
    }

    for config in vs_configs.iter_mut() {
        config.user = Some("alice".to_string());
        config.password = Some("alice_password".to_string());
    }

    init_with_config(actors.clone(), scylla_configs, vs_configs).await;

    let (session, clients) = prepare_connection_with_auth(&actors, "cassandra", "cassandra").await;

    session
        .query_unpaged(
            "CREATE ROLE alice WITH PASSWORD = 'alice_password' AND LOGIN = true",
            (),
        )
        .await
        .expect("failed to create role alice");
    session
        .query_unpaged("GRANT VECTOR_SEARCH_INDEXING ON ALL KEYSPACES TO alice", ())
        .await
        .expect("failed to grant permissions to alice");

    let _ = create_keyspace(&session).await;
    let table = create_table(&session, "pk INT PRIMARY KEY, v1 VECTOR<FLOAT, 3>", None).await;

    // Create index on column v1
    let index = create_index(&session, &clients, &table, "v1").await;

    for client in clients.iter() {
        wait_for_index(client, &index).await;
    }

    cleanup(actors).await;

    info!("finished");
}
