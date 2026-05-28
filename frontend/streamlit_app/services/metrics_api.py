# medical-triage-agent-ai-poc/frontend/streamlit_app/services/metrics_api.py

def get_metrics() -> dict:

    return {
        "latency": "420 ms",
        "requests": 1250,
        "errors": 3,
    }
