import streamlit as st
import pandas as pd
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from collections import Counter
import io

# Page configuration
st.set_page_config(
    page_title="Competitive Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Warm Editorial SaaS design
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1A1A1A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4A4A4A;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F5F1E8;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #D4A574;
    }
    .entity-badge {
        display: inline-block;
        background-color: #D4A574;
        color: #1A1A1A;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error(
            "⚠️ spaCy model not found. Please run: `uv run python -m spacy download en_core_web_sm`"
        )
        st.stop()


nlp = load_spacy_model()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    ["Dashboard", "Analysis", "Settings"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your data (CSV/TXT)",
    type=["csv", "txt"],
    help="Upload a file containing text data for analysis",
)

# Handle file upload
uploaded_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.sidebar.success(f"✓ Loaded {len(uploaded_text)} characters")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✓ Loaded {len(df)} rows")
        # Assume first text column
        text_column = df.select_dtypes(include=["object"]).columns[0]
        uploaded_text = " ".join(df[text_column].astype(str).tolist())

st.sidebar.markdown("---")
st.sidebar.caption("NLP for Competitive Intelligence | Research MVP")

# ============================================================
# DASHBOARD PAGE
# ============================================================
if page == "Dashboard":
    st.markdown(
        '<div class="main-header">Competitive Intelligence Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Leveraging NLP to Enhance Competitive Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Bento Grid Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📖 Project Overview")
        st.markdown("""
        This dashboard demonstrates the application of Natural Language Processing (NLP) 
        techniques to automate and enhance competitive intelligence gathering for businesses.
        
        **The Challenge:**
        - **Data Overload**: Companies are inundated with unstructured text from reviews, social media, 
          competitor announcements, and news articles
        - **Manual Inefficiency**: Traditional methods are slow, error-prone, and miss critical market signals
        - **Fragmented Tools**: Existing solutions often focus on single techniques rather than providing 
          comprehensive insights
        
        **Our Solution:**
        A unified analytical framework that processes unstructured text data through three core NLP pillars:
        """)

    with col2:
        st.subheader("📊 Quick Stats")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("NLP Techniques", "3", help="Sentiment Analysis, NER, Topic Modeling")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="metric-card" style="margin-top: 1rem;">',
            unsafe_allow_html=True,
        )
        st.metric(
            "Analysis Types", "Real-time", help="Instant processing of uploaded data"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Technical Pillars
    st.markdown("---")
    st.subheader("🔬 Technical Pillars")

    pillar_col1, pillar_col2, pillar_col3 = st.columns(3)

    with pillar_col1:
        st.markdown("#### 💭 Sentiment Analysis")
        st.markdown("""
        Detects emotional tone in text using **TextBlob**.
        
        - **Polarity**: Positive/Negative sentiment
        - **Subjectivity**: Opinion vs. Fact
        - **Use Case**: Track customer reactions to competitor products
        """)

    with pillar_col2:
        st.markdown("#### 🏷️ Named Entity Recognition")
        st.markdown("""
        Extracts key entities using **spaCy**.
        
        - **Organizations**: Competitor names
        - **Products**: Product mentions
        - **Locations**: Geographic markets
        - **Use Case**: Monitor competitor market expansion
        """)

    with pillar_col3:
        st.markdown("#### 📚 Topic Modeling")
        st.markdown("""
        Discovers hidden themes using **LDA**.
        
        - **Key Themes**: Emerging discussion topics
        - **Keyword Extraction**: Most frequent terms
        - **Use Case**: Identify trending features/concerns
        """)

    # Call to Action
    st.markdown("---")
    st.info("👈 Navigate to the **Analysis** page to start processing your text data.")

# ============================================================
# ANALYSIS PAGE
# ============================================================
elif page == "Analysis":
    st.markdown(
        '<div class="main-header">NLP Analysis Suite</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Real-time text analysis using advanced NLP techniques</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(
        ["💭 Sentiment Analysis", "🏷️ Entity Extraction", "📚 Topic Modeling"]
    )

    # ============================================================
    # SENTIMENT TAB
    # ============================================================
    with tab1:
        st.subheader("Sentiment Analysis with TextBlob")
        st.markdown("Analyze the emotional tone and subjectivity of your text.")

        sentiment_text = st.text_area(
            "Enter text for sentiment analysis:",
            value=uploaded_text
            if uploaded_text
            else "Our competitor launched an amazing new product. Customers seem really excited about it!",
            height=150,
            key="sentiment_input",
        )

        if st.button("🔍 Analyze Sentiment", type="primary"):
            if sentiment_text.strip():
                blob = TextBlob(sentiment_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Display results in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Polarity Score",
                        value=f"{polarity:.3f}",
                        help="Range: -1 (negative) to +1 (positive)",
                    )
                    if polarity > 0.1:
                        st.success("✓ Positive sentiment detected")
                    elif polarity < -0.1:
                        st.error("✗ Negative sentiment detected")
                    else:
                        st.info("≈ Neutral sentiment")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Subjectivity Score",
                        value=f"{subjectivity:.3f}",
                        help="Range: 0 (objective) to 1 (subjective)",
                    )
                    if subjectivity > 0.5:
                        st.info("Highly subjective (opinion-based)")
                    else:
                        st.info("Relatively objective (fact-based)")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Visualization
                st.markdown("---")
                st.subheader("Visual Representation")

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor("#FDFBF7")

                # Polarity bar
                ax[0].barh(["Polarity"], [polarity], color="#D4A574")
                ax[0].set_xlim(-1, 1)
                ax[0].axvline(x=0, color="#1A1A1A", linestyle="--", linewidth=0.8)
                ax[0].set_xlabel("Score", fontsize=11)
                ax[0].set_title(
                    "Polarity (Negative ← → Positive)", fontsize=12, fontweight="bold"
                )
                ax[0].set_facecolor("#FDFBF7")

                # Subjectivity bar
                ax[1].barh(["Subjectivity"], [subjectivity], color="#D4A574")
                ax[1].set_xlim(0, 1)
                ax[1].set_xlabel("Score", fontsize=11)
                ax[1].set_title(
                    "Subjectivity (Objective ← → Subjective)",
                    fontsize=12,
                    fontweight="bold",
                )
                ax[1].set_facecolor("#FDFBF7")

                for a in ax:
                    a.spines["top"].set_visible(False)
                    a.spines["right"].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.warning("⚠️ Please enter some text to analyze.")

    # ============================================================
    # ENTITIES TAB
    # ============================================================
    with tab2:
        st.subheader("Named Entity Recognition with spaCy")
        st.markdown("Extract organizations, products, and locations from your text.")

        entity_text = st.text_area(
            "Enter text for entity extraction:",
            value=uploaded_text
            if uploaded_text
            else "Apple launched the new iPhone in California. Microsoft and Google are competing in the AI market. Tesla announced expansion plans in China.",
            height=150,
            key="entity_input",
        )

        if st.button("🔍 Extract Entities", type="primary"):
            if entity_text.strip():
                doc = nlp(entity_text)

                # Filter entities by type
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
                locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

                # Display in bento grid
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🏢 Organizations")
                    if orgs:
                        for org in set(orgs):
                            st.markdown(
                                f'<span class="entity-badge">{org}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No organizations detected")

                with col2:
                    st.markdown("#### 📦 Products")
                    if products:
                        for prod in set(products):
                            st.markdown(
                                f'<span class="entity-badge">{prod}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No products detected")

                with col3:
                    st.markdown("#### 📍 Locations")
                    if locations:
                        for loc in set(locations):
                            st.markdown(
                                f'<span class="entity-badge">{loc}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No locations detected")

                # All entities table
                st.markdown("---")
                st.subheader("All Detected Entities")

                if doc.ents:
                    entity_data = []
                    for ent in doc.ents:
                        entity_data.append(
                            {
                                "Entity": ent.text,
                                "Type": ent.label_,
                                "Context": entity_text[
                                    max(0, ent.start_char - 20) : min(
                                        len(entity_text), ent.end_char + 20
                                    )
                                ],
                            }
                        )

                    df_entities = pd.DataFrame(entity_data)
                    st.dataframe(df_entities, use_container_width=True, hide_index=True)
                else:
                    st.info("No entities detected in the text.")

            else:
                st.warning("⚠️ Please enter some text to analyze.")

    # ============================================================
    # TOPICS TAB
    # ============================================================
    with tab3:
        st.subheader("Topic Modeling with LDA")
        st.markdown(
            "Discover key themes and extract important keywords from your text."
        )

        topic_text = st.text_area(
            "Enter text for topic modeling:",
            value=uploaded_text
            if uploaded_text
            else """
            Our product features advanced AI capabilities and machine learning algorithms. 
            The new software update includes better security and enhanced user interface. 
            Customer feedback indicates high satisfaction with product quality and performance. 
            Market research shows growing demand for cloud-based solutions and mobile applications.
            """,
            height=150,
            key="topic_input",
        )

        num_topics = st.slider("Number of topics to extract:", 1, 5, 3)

        if st.button("🔍 Extract Topics", type="primary"):
            if topic_text.strip():
                # Use CountVectorizer for keyword extraction
                vectorizer = CountVectorizer(
                    max_features=100, stop_words="english", ngram_range=(1, 2)
                )

                try:
                    # Create document-term matrix
                    doc_term_matrix = vectorizer.fit_transform([topic_text])

                    # Get feature names (words)
                    feature_names = vectorizer.get_feature_names_out()

                    # Get word frequencies
                    word_freq = doc_term_matrix.toarray()[0]
                    word_freq_dict = dict(zip(feature_names, word_freq))

                    # Get top keywords
                    top_keywords = sorted(
                        word_freq_dict.items(), key=lambda x: x[1], reverse=True
                    )[:10]

                    # Display top keywords
                    st.markdown("#### 🔑 Top Keywords")
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        for i, (word, freq) in enumerate(top_keywords[:5], 1):
                            st.markdown(f"**{i}.** {word} ({int(freq)})")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        # Keyword frequency chart
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor("#FDFBF7")
                        ax.set_facecolor("#FDFBF7")

                        words = [w[0] for w in top_keywords[:10]]
                        freqs = [w[1] for w in top_keywords[:10]]

                        ax.barh(words, freqs, color="#D4A574")
                        ax.set_xlabel("Frequency", fontsize=11)
                        ax.set_title(
                            "Top 10 Keywords by Frequency",
                            fontsize=12,
                            fontweight="bold",
                        )
                        ax.invert_yaxis()
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)

                        plt.tight_layout()
                        st.pyplot(fig)

                    # LDA Topic Modeling
                    st.markdown("---")
                    st.markdown("#### 📊 Discovered Topics (LDA)")

                    if len(topic_text.split()) > 10:
                        lda = LatentDirichletAllocation(
                            n_components=num_topics, random_state=42, max_iter=10
                        )
                        lda.fit(doc_term_matrix)

                        # Display topics
                        for idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[-5:][::-1]
                            top_words = [feature_names[i] for i in top_words_idx]

                            st.markdown(f"**Topic {idx + 1}:** {', '.join(top_words)}")
                    else:
                        st.info(
                            "Text is too short for meaningful topic modeling. Try adding more content."
                        )

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try adding more text or varying your vocabulary.")

            else:
                st.warning("⚠️ Please enter some text to analyze.")

# ============================================================
# SETTINGS PAGE
# ============================================================
elif page == "Settings":
    st.markdown('<div class="main-header">Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Configure your analysis preferences</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("⚙️ Analysis Configuration")

        st.markdown("#### NLP Model Information")
        st.info(f"**spaCy Model:** en_core_web_sm (Loaded ✓)")
        st.info(f"**TextBlob:** Active")
        st.info(f"**scikit-learn:** Active")

        st.markdown("---")

        st.markdown("#### About This MVP")
        st.markdown("""
        This is a research prototype demonstrating NLP techniques for competitive intelligence.
        
        **Research Focus:** Leveraging NLP to Enhance Competitive Intelligence
        
        **Technologies:**
        - **UI Framework:** Streamlit
        - **Sentiment Analysis:** TextBlob
        - **Named Entity Recognition:** spaCy (en_core_web_sm)
        - **Topic Modeling:** scikit-learn (LDA)
        
        **Design System:** Warm Editorial SaaS
        - Cream/off-white background (#FDFBF7)
        - Dark gray text (#1A1A1A)
        - Serif fonts for elegant typography
        - Bento-grid layout for clean organization
        """)

    with col2:
        st.subheader("📚 Resources")
        st.markdown("""
        **Documentation:**
        - [TextBlob Docs](https://textblob.readthedocs.io/)
        - [spaCy Docs](https://spacy.io/)
        - [scikit-learn Docs](https://scikit-learn.org/)
        
        **Setup Commands:**
        ```bash
        # Install dependencies
        uv sync
        
        # Download spaCy model
        uv run python -m spacy download en_core_web_sm
        
        # Run the app
        uv run streamlit run app.py
        ```
        """)

# Footer
st.markdown("---")
st.caption(
    "Competitive Intelligence Dashboard | NLP Research MVP | Built with Streamlit"
)
