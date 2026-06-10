# medical-triage-agent-ai-poc/frontend/streamlit_app/pages/4_Metrics.py

from __future__ import annotations

import streamlit as st

from streamlit_app.services.metrics_api import get_metrics
from streamlit_app.components.metrics_cards import render_metrics_cards

st.set_page_config(
    page_title="Metrics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Monitoring & Observability Dashboard")

try:
    metrics = get_metrics()
except Exception as exc:
    st.error(f"Unable to retrieve monitoring metrics: {exc}")
    st.stop()

# ==========================================================
# Global Metrics Cards
# ==========================================================
render_metrics_cards(metrics)

# ==========================================================
# Latency Metrics
# ==========================================================
latency = metrics.get("latency", {})

st.subheader("⏱ API Latency")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Average", f"{latency.get('avg_ms', 0):.2f} ms")
with col2:
    st.metric("P95", f"{latency.get('p95_ms', 0):.2f} ms")
with col3:
    st.metric("P99", f"{latency.get('p99_ms', 0):.2f} ms")
with col4:
    st.metric("Min", f"{latency.get('min_ms', 0):.2f} ms")
with col5:
    st.metric("Max", f"{latency.get('max_ms', 0):.2f} ms")

# ==========================================================
# Request Metrics
# ==========================================================
requests = metrics.get("requests", {})

st.subheader("🌐 API Traffic")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Requests", requests.get("total_requests", 0))
with col2:
    st.metric("Successful", requests.get("success_requests", 0))
with col3:
    st.metric("Errors", requests.get("failed_requests", 0))

# ==========================================================
# GPU Metrics
# ==========================================================
gpu = metrics.get("gpu", {})

st.subheader("🖥 GPU Monitoring")

gpu_available = gpu.get("cuda_available", False)

if gpu_available:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VRAM Used", gpu.get("vram_used_mb", 0))
    with col2:
        st.metric("VRAM %", gpu.get("vram_usage_percent", 0))
    with col3:
        st.metric("Throughput", gpu.get("throughput_rps", 0))
    st.json(gpu)
else:
    st.info("No GPU detected.")

# ==========================================================
# Alerts
# ==========================================================
alerts = metrics.get("alerts", [])

st.subheader("🚨 Active Alerts")

if not alerts:
    st.success("No active alerts.")
else:
    for alert in alerts:
        st.warning(alert)

# ==========================================================
# Raw Monitoring Data
# ==========================================================
with st.expander("🔍 Raw Monitoring Payload"):
    st.json(metrics)
