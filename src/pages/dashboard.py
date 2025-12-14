import streamlit as st
from ..components.layout import header, metric_card


def render():
    """Render the dashboard page."""
    header(
        "Competitive Intelligence Dashboard",
        "NLP-powered competitive intelligence analysis",
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Project Overview")
        st.markdown("""
        This dashboard applies Natural Language Processing techniques to automate
        competitive intelligence gathering.

        **The Problem:**
        - Data overload from unstructured text sources
        - Manual analysis is slow and error-prone
        - Fragmented tools lack comprehensive insights

        **The Solution:**
        A unified framework processing text through three NLP techniques:
        sentiment analysis, entity extraction, and topic modeling.
        """)

    with col2:
        st.subheader("Capabilities")
        metric_card("NLP Techniques", "3", "Sentiment, NER, Topic Modeling")
        st.markdown("")
        metric_card("Processing", "Real-time", "Instant analysis of uploaded data")

    st.markdown("---")
    st.subheader("Technical Components")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Sentiment Analysis")
        st.markdown("""
        Detects emotional tone using TextBlob.

        - Polarity: positive/negative
        - Subjectivity: opinion vs fact
        - Use: track reactions to competitor products
        """)

    with col2:
        st.markdown("#### Named Entity Recognition")
        st.markdown("""
        Extracts entities using spaCy.

        - Organizations
        - Products
        - Locations
        - Use: monitor competitor expansion
        """)

    with col3:
        st.markdown("#### Topic Modeling")
        st.markdown("""
        Discovers themes using LDA.

        - Key themes
        - Keyword extraction
        - Use: identify trending topics
        """)

    st.markdown("---")
    st.info("Navigate to Analysis to process text data.")
