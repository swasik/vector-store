# AI-Assisted Assistant for AI-Assisted PRs

A web app that lists your recent JIRA issues and lets you bulk-label them
with `ai-assisted`.

## Prerequisites

- Python 3.12+
- A JIRA API token (create one at
  [id.atlassian.com → API tokens](https://id.atlassian.com/manage-profile/security/api-tokens))

## Running locally

```bash
cd scripts/ai-assisted-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000, enter your Atlassian email and API token, and
you're in.

## Running tests

```bash
source .venv/bin/activate
pytest test_app.py -v
```

## Deploying to Render

See the [Deployment section](docs/technical-design.md#deployment-rendercom) in the
technical design.
