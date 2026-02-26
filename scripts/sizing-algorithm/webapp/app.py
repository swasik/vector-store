"""
Flask web application for the ScyllaDB VSS Sizing Calculator.

Serves a single-page UI and exposes a ``/api/compute`` endpoint that
delegates to ``vss_sizing.compute_sizing``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing vss_sizing from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

import vss_sizing as vs

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the single-page sizing calculator."""
    return render_template("index.html")


@app.route("/api/compute", methods=["POST"])
def compute():
    """Run the sizing algorithm and return the result as JSON."""
    data = request.get_json(force=True)

    try:
        inp = vs.SizingInput(
            num_vectors=int(data.get("num_vectors", vs.DEFAULT_NUM_VECTORS)),
            dimensions=int(data.get("dimensions", vs.DEFAULT_DIMENSIONS)),
            target_qps=int(data.get("target_qps", vs.DEFAULT_TARGET_QPS)),
            recall=int(data.get("recall", vs.DEFAULT_RECALL)),
            k=int(data.get("k", vs.DEFAULT_K)),
            quantization=vs.Quantization(
                data.get("quantization", vs.Quantization.NONE.value)
            ),
            metadata_bytes_per_vector=int(
                data.get("metadata_bytes_per_vector", vs.DEFAULT_METADATA_BYTES)
            ),
            filtering_columns=int(
                data.get("filtering_columns", vs.MIN_FILTERING_COLUMNS)
            ),
        )
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = vs.compute_sizing(inp)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "hnsw": {
            "m": result.hnsw_params.m,
        },
        "compression_ratio": result.compression_ratio,
        "search_node": {
            "index_ram_gb": round(result.search_node.index_ram_bytes / (1024**3), 2),
            "filtering_ram_gb": round(
                result.search_node.filtering_ram_bytes / (1024**3), 2
            ),
            "total_ram_gb": result.search_node.total_ram_gb,
            "required_vcpus": result.search_node.required_vcpus,
            "base_qps_per_vcpu": round(result.search_node.base_qps_per_vcpu, 1),
            "effective_qps_per_vcpu": result.search_node.effective_qps_per_vcpu,
            "throughput_bucket": result.search_node.throughput_bucket,
        },
        "instance_selection": {
            "instance_type": result.instance_selection.instance_type.name,
            "instance_vcpus": result.instance_selection.instance_type.vcpus,
            "instance_ram_gb": result.instance_selection.instance_type.ram_gb,
            "num_instances": result.instance_selection.num_instances,
            "total_vcpus": result.instance_selection.total_vcpus,
            "total_ram_gb": result.instance_selection.total_ram_gb,
            "total_cost_per_hour": result.instance_selection.total_cost_per_hour,
            "total_cost_per_month": round(
                result.instance_selection.total_cost_per_hour * 730, 2
            ),
        },
        "data_node": {
            "num_nodes": result.data_node.num_nodes,
            "vcpus_per_node": result.data_node.vcpus_per_node,
            "total_vcpus": result.data_node.total_vcpus,
            "total_storage_gb": result.data_node.total_storage_gb,
            "embedding_storage_gb": round(
                result.data_node.embedding_storage_bytes / (1024**3), 2
            ),
            "metadata_storage_gb": round(
                result.data_node.metadata_storage_bytes / (1024**3), 2
            ),
        },
        "search_replicas": result.search_replicas,
        "summary": result.summary,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
