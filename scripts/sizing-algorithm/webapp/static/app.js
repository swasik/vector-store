/* ───────────────────────────────────────────────────────────
   ScyllaDB VSS Sizing Calculator — Frontend Logic
   ─────────────────────────────────────────────────────────── */

(function () {
  "use strict";

  // ── Helpers ──────────────────────────────────────────────

  /** Format a number with thousands separators. */
  function fmt(n) {
    return Number(n).toLocaleString("en-US");
  }

  /** Format bytes to a human-readable string. */
  function fmtBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KiB";
    if (bytes < 1024 * 1024 * 1024)
      return (bytes / (1024 * 1024)).toFixed(1) + " MiB";
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GiB";
  }

  /** Convert a log10 slider value to the actual integer. */
  function logToValue(logVal) {
    return Math.round(Math.pow(10, parseFloat(logVal)));
  }

  /** Convert an actual integer value to log10 for the slider. */
  function valueToLog(val) {
    return Math.log10(Math.max(val, 1));
  }

  /** Return a debounced version of *fn* that waits *ms* after the last call. */
  function debounce(fn, ms) {
    var timer;
    return function () {
      clearTimeout(timer);
      timer = setTimeout(fn, ms);
    };
  }

  // ── DOM References ──────────────────────────────────────

  const elNumVectors       = document.getElementById("num_vectors");
  const elNumVectorsVal    = document.getElementById("num_vectors_val");
  const elDimensions       = document.getElementById("dimensions");
  const elDimensionsVal    = document.getElementById("dimensions_val");
  const elTargetQps        = document.getElementById("target_qps");
  const elTargetQpsVal     = document.getElementById("target_qps_val");
  const elRecall           = document.getElementById("recall");
  const elRecallVal        = document.getElementById("recall_val");
  const elK               = document.getElementById("k");
  const elKVal             = document.getElementById("k_val");
  const elMetadataBytes    = document.getElementById("metadata_bytes");
  const elMetadataBytesVal = document.getElementById("metadata_bytes_val");
  const elFilteringCols    = document.getElementById("filtering_columns");
  const elFilteringColsVal = document.getElementById("filtering_columns_val");
  const elResultsContent   = document.getElementById("results-content");
  const elErrorBox         = document.getElementById("error-box");

  // ── Value display updaters ──────────────────────────────

  function updateDisplays() {
    elNumVectorsVal.textContent    = fmt(logToValue(elNumVectors.value));
    elDimensionsVal.textContent    = fmt(elDimensions.value);
    elTargetQpsVal.textContent     = fmt(elTargetQps.value);
    elRecallVal.textContent        = elRecall.value + " %";
    elKVal.textContent             = fmt(elK.value);
    elMetadataBytesVal.textContent = fmtBytes(logToValue(elMetadataBytes.value));
    elFilteringColsVal.textContent = elFilteringCols.value;

    // Highlight active dimension chip.
    document.querySelectorAll('.chip[data-target="dimensions"]').forEach(function (chip) {
      chip.classList.toggle("active", chip.dataset.value === elDimensions.value);
    });
  }

  // Debounced compute triggered on any input change.
  var scheduleCompute = debounce(compute, 150);

  function onInputChange() {
    updateDisplays();
    scheduleCompute();
  }

  // Wire up real-time display updates + auto-compute.
  [elNumVectors, elDimensions, elTargetQps, elRecall, elK, elMetadataBytes, elFilteringCols]
    .forEach(function (el) {
      el.addEventListener("input", onInputChange);
    });

  // ── Preset chips ────────────────────────────────────────

  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var target = document.getElementById(chip.dataset.target);
      if (target) {
        target.value = chip.dataset.value;
        target.dispatchEvent(new Event("input"));
      }
    });
  });

  // ── Quantization radio cards ────────────────────────────

  document.querySelectorAll(".radio-card").forEach(function (card) {
    card.addEventListener("click", function () {
      document.querySelectorAll(".radio-card").forEach(function (c) {
        c.classList.remove("selected");
      });
      card.classList.add("selected");
      card.querySelector("input").checked = true;
      scheduleCompute();
    });
  });

  // ── Collect input values ────────────────────────────────

  function collectInput() {
    var quantEl = document.querySelector('input[name="quantization"]:checked');
    return {
      num_vectors:            logToValue(elNumVectors.value),
      dimensions:             parseInt(elDimensions.value, 10),
      target_qps:             parseInt(elTargetQps.value, 10),
      recall:                 parseInt(elRecall.value, 10),
      k:                      parseInt(elK.value, 10),
      quantization:           quantEl ? quantEl.value : "none",
      metadata_bytes_per_vector: logToValue(elMetadataBytes.value),
      filtering_columns:      parseInt(elFilteringCols.value, 10),
    };
  }

  // ── Render results ──────────────────────────────────────

  function renderResults(data) {
    var s  = data.search_node;
    var i  = data.instance_selection;
    var d  = data.data_node;
    var h  = data.hnsw;

    var html = "";

    // Cost banner
    html += '<div class="cost-banner">';
    html += '  <div class="small-label">Estimated Search-Node Cost</div>';
    html += '  <div class="big-number">$' + fmt(i.total_cost_per_month) + " / mo</div>";
    html += '  <div class="small-label">$' + i.total_cost_per_hour.toFixed(3) + " / hr</div>";
    html += "</div>";

    // Instance Selection
    html += '<div class="result-card">';
    html += '  <div class="result-section-title">Search Node Instances</div>';
    html += resultRow("Instance type",       i.instance_type);
    html += resultRow("Instances",           i.num_instances);
    html += resultRow("vCPUs per instance",  i.instance_vcpus);
    html += resultRow("RAM per instance",    i.instance_ram_gb + " GB");
    html += resultRow("Total vCPUs",         fmt(i.total_vcpus));
    html += resultRow("Total RAM",           fmt(i.total_ram_gb) + " GB");
    html += "</div>";

    // Search Node Sizing
    html += '<div class="result-card">';
    html += '  <div class="result-section-title">Search Node Sizing (per replica)</div>';
    html += resultRow("Index RAM",           s.index_ram_gb.toFixed(2) + " GB");
    html += resultRow("Filtering RAM",       s.filtering_ram_gb.toFixed(2) + " GB");
    html += resultRow("Total RAM required",  s.total_ram_gb.toFixed(2) + " GB", true);
    html += resultRow("Required vCPUs",      fmt(s.required_vcpus), true);
    html += resultRow("Throughput bucket",   s.throughput_bucket);
    html += resultRow("Base QPS/vCPU",       s.base_qps_per_vcpu.toFixed(1));
    html += resultRow("Effective QPS/vCPU",  s.effective_qps_per_vcpu.toFixed(1));
    html += "</div>";

    // HNSW Parameters
    html += '<div class="result-card">';
    html += '  <div class="result-section-title">HNSW Parameters</div>';
    html += resultRow("M (connectivity)",    h.m);
    html += resultRow("Compression ratio",   data.compression_ratio + "×");
    html += "</div>";

    // Data Nodes
    html += '<div class="result-card">';
    html += '  <div class="result-section-title">ScyllaDB Data Nodes</div>';
    html += resultRow("Nodes (RF=3)",        d.num_nodes);
    html += resultRow("vCPUs per node",      d.vcpus_per_node);
    html += resultRow("Total vCPUs",         fmt(d.total_vcpus));
    html += resultRow("Embedding storage",   d.embedding_storage_gb.toFixed(2) + " GB");
    html += resultRow("Metadata storage",    d.metadata_storage_gb.toFixed(2) + " GB");
    html += resultRow("Total storage",       d.total_storage_gb.toFixed(2) + " GB", true);
    html += "</div>";

    elResultsContent.innerHTML = html;
  }

  function resultRow(label, value, highlight) {
    var cls = highlight ? ' highlight' : '';
    return '<div class="result-row">' +
      '<span class="result-label">' + label + '</span>' +
      '<span class="result-value' + cls + '">' + value + '</span>' +
      '</div>';
  }

  // ── API call ────────────────────────────────────────────

  function compute() {

    elErrorBox.classList.add("hidden");

    fetch("/api/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectInput()),
    })
      .then(function (resp) { return resp.json().then(function (d) { return { ok: resp.ok, data: d }; }); })
      .then(function (result) {
        if (!result.ok) {
          elErrorBox.textContent = result.data.error || "Unknown error";
          elErrorBox.classList.remove("hidden");
          return;
        }
        renderResults(result.data);
      })
      .catch(function (err) {
        elErrorBox.textContent = "Request failed: " + err.message;
        elErrorBox.classList.remove("hidden");
      });
  }

  // ── Initialise ──────────────────────────────────────────
  updateDisplays();

  // Auto-compute on load with defaults.
  compute();
})();
