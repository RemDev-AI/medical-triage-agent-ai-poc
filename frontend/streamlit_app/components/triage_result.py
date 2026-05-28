# medical-triage-agent-ai-poc/frontend/streamlit_app/components/triage_result.py

import streamlit as st


def render_triage_result(result: dict) -> None:

    st.divider()

    st.subheader("📄 Résultat du Triage")

    priority = result.get("priority", "UNKNOWN")

    if priority == "HIGH":
        st.error("🔴 Priorité élevée")

    elif priority == "MEDIUM":
        st.warning("🟠 Priorité moyenne")

    else:
        st.success("🟢 Priorité faible")

    st.markdown(
        f"""
        ### Justification clinique

        {result.get("justification")}
        """
    )

    st.markdown(
        f"""
        ### Recommandations

        {result.get("recommendations")}
        """
    )
