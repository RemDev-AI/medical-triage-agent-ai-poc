# medical-triage-agent-ai-poc/frontend/streamlit_app/utils/formatters.py


def format_priority(priority: str) -> str:

    mapping = {
        "HIGH": "🔴 Haute",
        "MEDIUM": "🟠 Moyenne",
        "LOW": "🟢 Faible",
    }

    return mapping.get(priority, priority)
