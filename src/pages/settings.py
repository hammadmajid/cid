import streamlit as st
from ..components.layout import header
from ..config import SPACY_MODEL


def render():
    """Render the settings page."""
    header("Settings", "Configuration and system information")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("System Status")

        st.markdown("#### NLP Models")
        st.info(f"spaCy: {SPACY_MODEL} (loaded)")
        st.info("TextBlob: active")
        st.info("scikit-learn: active")

        st.markdown("---")

        st.markdown("#### About")
        st.markdown("""
        Research prototype for NLP-based competitive intelligence.

        **Stack:**
        - UI: Streamlit
        - Sentiment: TextBlob
        - NER: spaCy
        - Topics: scikit-learn (LDA)
        """)

    with col2:
        st.subheader("Setup")
        st.markdown("""
        **Commands:**
        ```bash
        # Install dependencies
        uv sync

        # Download spaCy model
        uv run python -m spacy download en_core_web_lg

        # Run app
        uv run streamlit run app.py
        ```
        """)

        st.markdown("---")

        st.subheader("Documentation")
        st.markdown("""
        - [TextBlob](https://textblob.readthedocs.io/)
        - [spaCy](https://spacy.io/)
        - [scikit-learn](https://scikit-learn.org/)
        """)
