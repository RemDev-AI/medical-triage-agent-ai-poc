# medical-triage-agent-ai-poc/frontend/streamlit_app/components/footer.py

import streamlit as st


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            POC Agent IA de Triage Médical · Streamlit Frontend · FastAPI Backend
        </div>
        """,
        unsafe_allow_html=True,
    )
