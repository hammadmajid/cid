import streamlit as st
import pandas as pd

from ..components.layout import header, metric_card, entity_badges
from ..components.charts import sentiment_chart, keyword_chart
from ..utils.nlp import (
    analyze_sentiment,
    extract_entities,
    run_topic_modeling,
    load_spacy_model,
)


def render(uploaded_text: str = ""):
    """Render the analysis page."""
    header("NLP Analysis", "Text analysis using sentiment, NER, and topic modeling")

    st.markdown("---")

    nlp = load_spacy_model()

    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Entity Extraction", "Topic Modeling"])

    with tab1:
        _render_sentiment_tab(uploaded_text)

    with tab2:
        _render_entity_tab(uploaded_text, nlp)

    with tab3:
        _render_topic_tab(uploaded_text)


def _render_sentiment_tab(uploaded_text: str):
    """Render sentiment analysis tab."""
    st.subheader("Sentiment Analysis")
    st.markdown("Analyze emotional tone and subjectivity of text.")

    default_text = (
        uploaded_text
        or "Our competitor launched a new product. Customers seem excited about it."
    )

    text = st.text_area(
        "Enter text:",
        value=default_text,
        height=150,
        key="sentiment_input",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter text to analyze.")
            return

        result = analyze_sentiment(text)

        col1, col2 = st.columns(2)

        with col1:
            metric_card(
                "Polarity",
                f"{result.polarity:.3f}",
                "Range: -1 (negative) to +1 (positive)",
            )
            if result.sentiment_label == "positive":
                st.success("Positive sentiment")
            elif result.sentiment_label == "negative":
                st.error("Negative sentiment")
            else:
                st.info("Neutral sentiment")

        with col2:
            metric_card(
                "Subjectivity",
                f"{result.subjectivity:.3f}",
                "Range: 0 (objective) to 1 (subjective)",
            )
            if result.subjectivity_label == "subjective":
                st.info("Opinion-based content")
            else:
                st.info("Fact-based content")

        st.markdown("---")
        st.subheader("Visualization")
        fig = sentiment_chart(result.polarity, result.subjectivity)
        st.pyplot(fig)


def _render_entity_tab(uploaded_text: str, nlp):
    """Render entity extraction tab."""
    st.subheader("Named Entity Recognition")
    st.markdown("Extract organizations, products, and locations from text.")

    default_text = (
        uploaded_text
        or "Apple launched the new iPhone in California. Microsoft and Google compete in AI. Tesla plans expansion in China."
    )

    text = st.text_area(
        "Enter text:",
        value=default_text,
        height=150,
        key="entity_input",
    )

    if st.button("Extract Entities", type="primary"):
        if not text.strip():
            st.warning("Enter text to analyze.")
            return

        result = extract_entities(text, nlp)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Organizations")
            entity_badges(result.organizations, "No organizations detected")

        with col2:
            st.markdown("#### Products")
            entity_badges(result.products, "No products detected")

        with col3:
            st.markdown("#### Locations")
            entity_badges(result.locations, "No locations detected")

        st.markdown("---")
        st.subheader("All Entities")

        if result.all_entities:
            df = pd.DataFrame(result.all_entities)
            df.columns = ["Entity", "Type", "Context"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No entities detected.")


def _render_topic_tab(uploaded_text: str):
    """Render topic modeling tab."""
    st.subheader("Topic Modeling")
    st.markdown("Discover themes and extract keywords from text.")

    default_text = uploaded_text or """
    Our product features advanced AI capabilities and machine learning algorithms.
    The software update includes better security and enhanced user interface.
    Customer feedback indicates high satisfaction with product quality and performance.
    Market research shows growing demand for cloud-based solutions and mobile applications.
    """

    text = st.text_area(
        "Enter text:",
        value=default_text,
        height=150,
        key="topic_input",
    )

    num_topics = st.slider("Number of topics:", 1, 5, 3)

    if st.button("Extract Topics", type="primary"):
        if not text.strip():
            st.warning("Enter text to analyze.")
            return

        try:
            result = run_topic_modeling(text, num_topics)

            st.markdown("#### Keywords")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                for i, (word, freq) in enumerate(result.keywords[:5], 1):
                    st.markdown(f"**{i}.** {word} ({int(freq)})")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                fig = keyword_chart(result.keywords)
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("#### Discovered Topics (LDA)")

            if result.topics:
                for idx, topic_words in enumerate(result.topics, 1):
                    st.markdown(f"**Topic {idx}:** {', '.join(topic_words)}")
            else:
                st.info("Text too short for topic modeling. Add more content.")

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
