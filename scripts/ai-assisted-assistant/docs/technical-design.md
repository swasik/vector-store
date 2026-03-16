# AI-Assisted Assistant for AI-Assisted PRs — Technical Design

## Overview

A single-page web application that authenticates against the ScyllaDB JIRA
instance, displays recently modified issues assigned to the current user,
and allows bulk-labelling selected issues with `ai-assisted`.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│               Browser (SPA)                 │
│  ┌────────┐  ┌───────────┐  ┌────────────┐ │
│  │  Login  │  │ Issue List│  │ Label API  │ │
│  │  Form   │  │   View    │  │  Caller    │ │
│  └────┬───┘  └─────┬─────┘  └─────┬──────┘ │
│       │            │               │        │
└───────┼────────────┼───────────────┼────────┘
        │            │               │
        ▼            ▼               ▼
┌──────────────────────────────────────────────┐
│         Backend (Python / Flask)             │
│  /auth/login   /api/issues   /api/label      │
└──────────────────┬───────────────────────────┘
                   │  (HTTP Basic Auth)
                   ▼
┌──────────────────────────────────────────────┐
│  JIRA Cloud REST API v3                      │
│  https://scylladb.atlassian.net              │
└──────────────────────────────────────────────┘
```

The app uses a thin Python backend to keep JIRA credentials server-side and
proxy API calls. The frontend is a single HTML page with vanilla JavaScript
(no build step).

---

## Authentication

The app uses **JIRA API tokens** with HTTP Basic Auth — no OAuth app
registration required.

### Prerequisites

Each user creates a personal API token at
<https://id.atlassian.com/manage-profile/security/api-tokens>. No
Atlassian developer console setup or admin approval is needed.

### Flow

1. User opens the app and sees a login form (email + API token).
2. User submits the form via `POST /auth/login`.
3. Backend validates the credentials by calling
   `GET https://scylladb.atlassian.net/rest/api/3/myself` with HTTP
   Basic Auth (`email:api_token`).
4. If valid, the email and token are stored in an encrypted, `HttpOnly`,
   `SameSite=Lax` session cookie.
5. The page reloads and shows the issue list.

All subsequent JIRA API calls use the stored credentials via Basic Auth.
No OAuth scopes, callback URLs, or token refresh logic is needed.

---

## Backend

### Technology

- **Python 3.12+** with **Flask**
- `requests` for JIRA API calls
- `flask` built-in session with `SECRET_KEY` for cookie encryption

### Endpoints

#### `POST /auth/login`

Accepts `email` and `api_token` from the login form. Validates them by
calling the JIRA `/rest/api/3/myself` endpoint with Basic Auth. On
success, stores the credentials in the session and redirects to `/`.
On failure, re-renders the login form with an error message.

#### `POST /auth/logout`

Clears the session and redirects to `/`.

#### `GET /api/issues`

Proxies a JQL search to JIRA. Requires an authenticated session.

**JQL query:**
```
assignee = currentUser()
AND updated >= -2w
AND status != "New"
ORDER BY updated DESC
```

**JIRA endpoint:**
```
GET https://scylladb.atlassian.net/rest/api/3/search/jql
  ?jql=<JQL>
  &fields=summary,status,priority,updated,labels,issuetype,key
  &maxResults=100
```

Authenticated via HTTP Basic Auth using the credentials from the session.

**Response (JSON):**
```json
{
  "issues": [
    {
      "key": "PM-268",
      "summary": "Sync workflow improvements",
      "status": "In Progress",
      "priority": "Medium",
      "issueType": "Task",
      "updated": "2026-03-15T10:30:00Z",
      "labels": ["backend"],
      "url": "https://scylladb.atlassian.net/browse/PM-268"
    }
  ]
}
```

#### `POST /api/label`

Adds the `ai-assisted` label to the specified issues.

**Request body:**
```json
{
  "issueKeys": ["PM-268", "PM-271"]
}
```

**For each issue key**, the backend calls:
```
PUT https://scylladb.atlassian.net/rest/api/3/issue/{key}
```
with body:
```json
{
  "update": {
    "labels": [{ "add": "ai-assisted" }]
  }
}
```

**Response:**
```json
{
  "results": [
    { "key": "PM-268", "success": true },
    { "key": "PM-271", "success": true }
  ]
}
```

Failed updates are reported per-issue with `"success": false` and an
`"error"` message.

#### `GET /`

Serves the static `index.html` page.

### Configuration

All configuration is via environment variables:

| Variable | Description |
|---|---|
| `FLASK_SECRET_KEY` | Session encryption key (used to encrypt the cookie that holds JIRA credentials) |

---

## Frontend

A single `index.html` served by Flask. No framework, no build step —
vanilla HTML + CSS + JavaScript.

### Layout

```
┌─────────────────────────────────────────────┐
│  AI-Assisted Assistant for AI-Assisted PRs  [Sign in / Out]  │
├─────────────────────────────────────────────┤
│                                             │
│  ☐  PM-268  Task  In Progress  Medium       │
│     Sync workflow improvements              │
│     Updated: 2026-03-15  Labels: backend    │
│                                             │
│  ☐  PM-271  Bug   Code Review  High         │
│     Fix auth token refresh                  │
│     Updated: 2026-03-14  Labels: —          │
│                                             │
│  ☐  PM-275  Task  In Progress  Low          │
│     Update sizing docs                      │
│     Updated: 2026-03-13  Labels: docs       │
│                                             │
│  ...                                        │
│                                             │
├─────────────────────────────────────────────┤
│  [Select All]     [✨ I'm feeling lucky]    │
└─────────────────────────────────────────────┘
```

Each issue row is a compact card showing:
- Checkbox for selection
- Issue key (linked to JIRA)
- Issue type icon/badge
- Status
- Priority
- Summary (truncated if needed)
- Last updated (relative, e.g., "2 days ago")
- Current labels

### Behaviour

1. On page load, check if authenticated by calling `GET /api/issues`.
   - If 401 → show login form (email + API token fields) with a
     step-by-step help section explaining how to create a token:
     1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
     2. Click **Create API token**
     3. Give it a label (e.g., "AI-Assisted Assistant for AI-Assisted PRs")
     4. Copy the generated token and paste it into the form
   - If 200 → render the issue list.
2. Checkboxes track selected issue keys in a local `Set`.
3. **"Select All"** toggles all checkboxes.
4. **"I'm feeling lucky"** button:
   - Disabled when no issues are selected.
   - On click, sends `POST /api/label` with the selected keys.
   - Shows a spinner during the request.
   - On success, updates the label badges in-place and unchecks the rows.
   - On partial failure, highlights failed rows with an error message.

### Styling

Minimal CSS targeting a dense, scannable layout:
- Compact rows (~48px height).
- Monospace font for issue keys.
- Status and priority rendered as coloured badges.
- Responsive down to 768px viewport width.

---

## File Structure

```
scripts/ai-assisted-assistant/
├── requirements.md          # Product requirements
├── technical-design.md      # This document
├── app.py                   # Flask backend
├── render.yaml              # Render.com service definition
├── templates/
│   └── index.html           # SPA frontend
└── requirements.txt         # Python dependencies
```

**requirements.txt:**
```
flask>=3.0
gunicorn>=22.0
requests>=2.31
```

---

## Deployment (Render.com)

The app is hosted on [Render](https://render.com) as a **Web Service**.

### render.yaml

```yaml
services:
  - type: web
    name: ai-assisted-assistant
    runtime: python
    rootDir: scripts/ai-assisted-assistant
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: PYTHON_VERSION
        value: "3.12"
```

### Setup Steps

1. Create a new **Web Service** on Render, pointing to this repository.
2. Set the **Root Directory** to `scripts/ai-assisted-assistant`.
3. `FLASK_SECRET_KEY` is auto-generated by Render. No other secrets
   need to be configured — users provide their own JIRA API tokens
   at login time.

### Notes

- Render's free tier spins down after inactivity; the first request after
  idle may take ~30 s. Use a paid instance if this is unacceptable.
- Flask's default file-system sessions work fine on a single Render
  instance. If scaling to multiple instances, switch to a shared session
  store (e.g., Redis via Render's managed Redis).
- Render provides automatic HTTPS — no TLS configuration needed.

---

## Security Considerations

- **No client-side credentials.** JIRA email and API token are stored
  server-side in an encrypted session cookie (`HttpOnly`, `SameSite=Lax`).
  They are never exposed to frontend JavaScript.
- **CSRF protection.** The `/api/label` and `/auth/login` endpoints only
  accept `POST` with a valid session.
- **Per-user tokens.** Each user authenticates with their own API token,
  so actions are attributed to the correct JIRA user. No shared service
  account.
- **Secrets via env vars.** `FLASK_SECRET_KEY` is the only server-side
  secret and is never committed to the repository.
- **Input validation.** Issue keys sent to `/api/label` are validated
  against the pattern `^[A-Z][A-Z0-9]+-\d+$` before being forwarded to
  the JIRA API.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| JIRA API token revoked / invalid | Backend returns 401; frontend shows login form |
| JIRA API rate limit (429) | Backend returns 429 to frontend; frontend shows "try again in X seconds" |
| Label update partially fails | Response includes per-issue success/error; frontend highlights failed rows |
| Network error | Frontend shows a dismissible error banner |
