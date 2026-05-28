# medical-triage-agent-ai-poc/frontend/streamlit_app/components/history_table.py

import pandas as pd
import streamlit as st


def render_history_table(history_data: list) -> None:

    dataframe = pd.DataFrame(history_data)

    st.dataframe(
        dataframe,
        use_container_width=True,
    )
