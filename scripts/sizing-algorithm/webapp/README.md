# VSS Sizing Calculator — Web Application

A browser-based UI for the ScyllaDB Vector Search sizing algorithm
(`vss_sizing.py`).

## Prerequisites

* Python 3.10+

## Quick Start

```bash
# From the repository root:
cd scripts/sizing-algorithm/webapp

# Create a virtual environment (optional but recommended):
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Run the server:
gunicorn -b 127.0.0.1:5050 app:app
```

Open **http://127.0.0.1:5050** in your browser.

> **Tip:** For development with auto-reload, use:
> ```bash
> gunicorn -b 127.0.0.1:5050 --reload app:app
> ```
> Or fall back to the built-in Flask server:
> ```bash
> python app.py
> ```

## Project Structure

```
webapp/
├── app.py              # Flask server + /api/compute endpoint
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/
│   └── index.html      # Single-page HTML UI
└── static/
    ├── style.css       # Styles
    └── app.js          # Frontend logic (vanilla JS)
```

## How It Works

1. The HTML page presents sliders and controls for every sizing parameter
   (number of vectors, dimensions, QPS, recall, K, quantization, metadata
   size, filtering columns).
2. Results update automatically as you adjust any parameter (debounced at
   150 ms).
3. The frontend `POST`s the current values to `/api/compute`.
3. The Flask backend instantiates a `SizingInput`, calls
   `compute_sizing()`, and returns the full recommendation as JSON.
4. The frontend renders the results: instance selection, search-node RAM /
   vCPU requirements, HNSW parameters, data-node sizing, and estimated
   monthly cost.
