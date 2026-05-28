# medical-triage-agent-ai-poc/frontend/streamlit_app/config/theme.py

GLOBAL_CSS = """
<style>

.main {
    background-color: #f7f9fc;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    color: white;
}

.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
}

.medical-card {
    padding: 1rem;
    border-radius: 10px;
    background-color: white;
    border: 1px solid #dbe4ee;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 12px;
    margin-top: 2rem;
}

</style>
"""
