# medical-triage-agent-ai-poc/frontend/streamlit_app/services/history_api.py

def get_triage_history() -> list:

    return [
        {
            "patient_id": "P-001",
            "priority": "HIGH",
            "status": "completed",
        },
        {
            "patient_id": "P-002",
            "priority": "LOW",
            "status": "completed",
        },
    ]
