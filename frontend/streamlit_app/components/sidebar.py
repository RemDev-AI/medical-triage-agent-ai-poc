# medical-triage-agent-ai-poc/frontend/streamlit_app/components/sidebar.py

import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">
                🏥 Medical AI
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### Navigation")

        st.page_link(
            "app.py",
            label="Accueil",
            icon="🏠",
        )

        st.markdown("---")

        st.markdown("### Informations")

        st.caption("Frontend Streamlit")
        st.caption("Backend FastAPI")
        st.caption("Inference Engine Qwen3")
        st.caption("Pipeline MLOps")
