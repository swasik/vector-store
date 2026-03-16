import json
from unittest.mock import MagicMock, patch

import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    with app.test_client() as c:
        yield c


def _login(client, email="user@example.com", token="tok123"):
    """Simulate login by setting session values directly."""
    with client.session_transaction() as sess:
        sess["email"] = email
        sess["api_token"] = token


# ---------- index ----------

def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"AI-Assisted Assistant" in resp.data


# ---------- auth/login ----------

@patch("app._jira_get")
def test_login_success_redirects(mock_get, client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    resp = client.post(
        "/auth/login",
        data={"email": "user@example.com", "api_token": "tok123"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/"


@patch("app._jira_get")
def test_login_invalid_credentials(mock_get, client):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    resp = client.post(
        "/auth/login",
        data={"email": "user@example.com", "api_token": "bad"},
    )
    assert resp.status_code == 401
    assert b"Invalid" in resp.data


def test_login_missing_fields(client):
    resp = client.post("/auth/login", data={"email": "", "api_token": ""})
    assert resp.status_code == 400
    assert b"required" in resp.data


@patch("app._jira_get")
def test_login_jira_unreachable(mock_get, client):
    import requests as http_requests

    mock_get.side_effect = http_requests.ConnectionError("timeout")

    resp = client.post(
        "/auth/login",
        data={"email": "user@example.com", "api_token": "tok"},
    )
    assert resp.status_code == 502
    assert b"Could not reach JIRA" in resp.data


# ---------- auth/logout ----------

def test_logout_clears_session(client):
    _login(client)
    resp = client.post("/auth/logout", follow_redirects=False)
    assert resp.status_code == 302

    # Session should be cleared — issues returns 401
    resp = client.get("/api/issues")
    assert resp.status_code == 401


# ---------- api/issues ----------

def test_issues_unauthenticated(client):
    resp = client.get("/api/issues")
    assert resp.status_code == 401


@patch("app._jira_get")
def test_issues_success(mock_get, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "issues": [
            {
                "key": "PM-1",
                "fields": {
                    "summary": "Test issue",
                    "status": {"name": "In Progress"},
                    "priority": {"name": "Medium"},
                    "issuetype": {"name": "Task"},
                    "updated": "2026-03-15T10:00:00.000+0000",
                    "labels": ["backend"],
                },
            }
        ]
    }
    mock_get.return_value = mock_resp

    resp = client.get("/api/issues")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data["issues"]) == 1
    assert data["issues"][0]["key"] == "PM-1"
    assert data["issues"][0]["status"] == "In Progress"
    assert data["issues"][0]["url"] == "https://scylladb.atlassian.net/browse/PM-1"


@patch("app._jira_get")
def test_issues_jira_401_clears_session(mock_get, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp

    resp = client.get("/api/issues")
    assert resp.status_code == 401

    # Session should be cleared
    resp2 = client.get("/api/issues")
    assert resp2.status_code == 401


@patch("app._jira_get")
def test_issues_jira_429(mock_get, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    resp = client.get("/api/issues")
    assert resp.status_code == 429
    assert b"rate limit" in resp.data


@patch("app._jira_get")
def test_issues_jira_500(mock_get, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_get.return_value = mock_resp

    resp = client.get("/api/issues")
    assert resp.status_code == 502


# ---------- api/label ----------

def test_label_unauthenticated(client):
    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1"]}),
        content_type="application/json",
    )
    assert resp.status_code == 401


def test_label_missing_body(client):
    _login(client)
    resp = client.post("/api/label", content_type="application/json")
    assert resp.status_code == 400


def test_label_empty_keys(client):
    _login(client)
    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_label_invalid_key_format(client):
    _login(client)
    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["invalid-key"]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert b"Invalid issue key" in resp.data


def test_label_rejects_injection_attempt(client):
    _login(client)
    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1; DROP TABLE"]}),
        content_type="application/json",
    )
    assert resp.status_code == 400


@patch("app._jira_put")
def test_label_success(mock_put, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_put.return_value = mock_resp

    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1", "PM-2"]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data["results"]) == 2
    assert all(r["success"] for r in data["results"])
    assert mock_put.call_count == 2


@patch("app._jira_put")
def test_label_partial_failure(mock_put, client):
    _login(client)

    resp_ok = MagicMock()
    resp_ok.status_code = 204

    resp_fail = MagicMock()
    resp_fail.status_code = 404

    mock_put.side_effect = [resp_ok, resp_fail]

    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1", "PM-2"]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["results"][0]["success"] is True
    assert data["results"][1]["success"] is False


@patch("app._jira_put")
def test_label_jira_401_clears_session(mock_put, client):
    _login(client)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_put.return_value = mock_resp

    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1"]}),
        content_type="application/json",
    )
    assert resp.status_code == 401


@patch("app._jira_put")
def test_label_network_error(mock_put, client):
    _login(client)

    import requests as http_requests

    mock_put.side_effect = http_requests.ConnectionError("fail")

    resp = client.post(
        "/api/label",
        data=json.dumps({"issueKeys": ["PM-1"]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["results"][0]["success"] is False
