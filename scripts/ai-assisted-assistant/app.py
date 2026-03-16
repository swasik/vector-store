import logging
import os
import re

import requests as http_requests
from flask import Flask, jsonify, redirect, render_template, request, session

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

JIRA_BASE_URL = "https://scylladb.atlassian.net"
ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


def _jira_auth():
    """Return (email, token) from the session or None."""
    email = session.get("email")
    token = session.get("api_token")
    if email and token:
        return (email, token)
    return None


def _jira_get(path, auth, params=None):
    """GET a JIRA REST API endpoint."""
    return http_requests.get(
        f"{JIRA_BASE_URL}{path}",
        auth=auth,
        params=params,
        headers={"Accept": "application/json"},
        timeout=15,
    )


def _jira_put(path, auth, json_body):
    """PUT to a JIRA REST API endpoint."""
    return http_requests.put(
        f"{JIRA_BASE_URL}{path}",
        auth=auth,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=json_body,
        timeout=15,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/auth/login", methods=["POST"])
def auth_login():
    email = request.form.get("email", "").strip()
    api_token = request.form.get("api_token", "").strip()

    if not email or not api_token:
        return jsonify({"error": "Email and API token are required."}), 400

    try:
        resp = _jira_get("/rest/api/3/myself", auth=(email, api_token))
    except http_requests.RequestException as exc:
        log.error("Login: could not reach JIRA: %s", exc)
        return jsonify({"error": "Could not reach JIRA. Please try again."}), 502

    if resp.status_code != 200:
        log.warning("Login: JIRA returned %s: %s", resp.status_code, resp.text[:500])
        return jsonify({"error": "Invalid email or API token."}), 401

    session["email"] = email
    session["api_token"] = api_token
    return redirect("/")


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return redirect("/")


@app.route("/api/issues")
def api_issues():
    auth = _jira_auth()
    if not auth:
        return jsonify({"error": "Not authenticated."}), 401

    jql = (
        'assignee = currentUser() AND updated >= -2w AND status != "New" '
        "ORDER BY updated DESC"
    )
    try:
        resp = _jira_get(
            "/rest/api/3/search/jql",
            auth=auth,
            params={
                "jql": jql,
                "fields": "summary,status,priority,updated,labels,issuetype",
                "maxResults": 100,
            },
        )
    except http_requests.RequestException as exc:
        log.error("Issues: could not reach JIRA: %s", exc)
        return jsonify({"error": "Could not reach JIRA."}), 502

    if resp.status_code == 401:
        log.warning("Issues: JIRA returned 401: %s", resp.text[:500])
        session.clear()
        return jsonify({"error": "JIRA credentials are invalid or revoked."}), 401

    if resp.status_code == 429:
        log.warning("Issues: JIRA rate limit (429)")
        return jsonify({"error": "JIRA rate limit exceeded. Try again later."}), 429

    if resp.status_code != 200:
        log.error("Issues: JIRA returned %s: %s", resp.status_code, resp.text[:500])
        return jsonify({"error": f"JIRA returned status {resp.status_code}."}), 502

    data = resp.json()
    issues = []
    for raw in data.get("issues", []):
        fields = raw.get("fields", {})
        issues.append(
            {
                "key": raw["key"],
                "summary": fields.get("summary", ""),
                "status": (fields.get("status") or {}).get("name", ""),
                "priority": (fields.get("priority") or {}).get("name", ""),
                "issueType": (fields.get("issuetype") or {}).get("name", ""),
                "updated": fields.get("updated", ""),
                "labels": fields.get("labels", []),
                "url": f"{JIRA_BASE_URL}/browse/{raw['key']}",
            }
        )

    return jsonify({"issues": issues})


@app.route("/api/label", methods=["POST"])
def api_label():
    auth = _jira_auth()
    if not auth:
        return jsonify({"error": "Not authenticated."}), 401

    body = request.get_json(silent=True)
    if not body or not isinstance(body.get("issueKeys"), list):
        return jsonify({"error": "Request must include an issueKeys array."}), 400

    issue_keys = body["issueKeys"]
    if not issue_keys:
        return jsonify({"error": "issueKeys must not be empty."}), 400

    for key in issue_keys:
        if not isinstance(key, str) or not ISSUE_KEY_RE.match(key):
            return jsonify({"error": f"Invalid issue key: {key}"}), 400

    results = []
    for key in issue_keys:
        try:
            resp = _jira_put(
                f"/rest/api/3/issue/{key}",
                auth=auth,
                json_body={"update": {"labels": [{"add": "ai-assisted"}]}},
            )
            if resp.status_code in (200, 204):
                results.append({"key": key, "success": True})
            elif resp.status_code == 401:
                session.clear()
                return jsonify({"error": "JIRA credentials are invalid or revoked."}), 401
            else:
                results.append(
                    {"key": key, "success": False, "error": f"JIRA returned {resp.status_code}"}
                )
        except http_requests.RequestException as exc:
            results.append({"key": key, "success": False, "error": str(exc)})

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
