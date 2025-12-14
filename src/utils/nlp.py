import streamlit as st
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dataclasses import dataclass

from ..config import (
    SPACY_MODEL,
    LDA_MAX_ITER,
    LDA_RANDOM_STATE,
    VECTORIZER_MAX_FEATURES,
)


@dataclass
class SentimentResult:
    polarity: float
    subjectivity: float

    @property
    def sentiment_label(self) -> str:
        if self.polarity > 0.1:
            return "positive"
        elif self.polarity < -0.1:
            return "negative"
        return "neutral"

    @property
    def subjectivity_label(self) -> str:
        return "subjective" if self.subjectivity > 0.5 else "objective"


@dataclass
class EntityResult:
    organizations: list[str]
    products: list[str]
    locations: list[str]
    all_entities: list[dict]


@dataclass
class TopicResult:
    keywords: list[tuple[str, int]]
    topics: list[list[str]]


@st.cache_resource
def load_spacy_model():
    """Load and cache the spaCy model."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        st.error(
            f"spaCy model not found. Run: uv run python -m spacy download {SPACY_MODEL}"
        )
        st.stop()


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    return SentimentResult(
        polarity=blob.sentiment.polarity,
        subjectivity=blob.sentiment.subjectivity,
    )


def extract_entities(text: str, nlp) -> EntityResult:
    """Extract named entities using spaCy."""
    doc = nlp(text)

    orgs = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    products = list(set(ent.text for ent in doc.ents if ent.label_ == "PRODUCT"))
    locations = list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))

    all_entities = [
        {
            "entity": ent.text,
            "type": ent.label_,
            "context": text[max(0, ent.start_char - 20) : min(len(text), ent.end_char + 20)],
        }
        for ent in doc.ents
    ]

    return EntityResult(
        organizations=orgs,
        products=products,
        locations=locations,
        all_entities=all_entities,
    )


def extract_keywords(text: str, max_keywords: int = 10) -> list[tuple[str, int]]:
    """Extract keywords using CountVectorizer."""
    vectorizer = CountVectorizer(
        max_features=VECTORIZER_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    doc_term_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    word_freq = doc_term_matrix.toarray()[0]
    word_freq_dict = dict(zip(feature_names, word_freq))

    return sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:max_keywords]


def run_topic_modeling(text: str, num_topics: int) -> TopicResult:
    """Run LDA topic modeling."""
    vectorizer = CountVectorizer(
        max_features=VECTORIZER_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    doc_term_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    keywords = extract_keywords(text)
    topics = []

    if len(text.split()) > 10:
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=LDA_RANDOM_STATE,
            max_iter=LDA_MAX_ITER,
        )
        lda.fit(doc_term_matrix)

        for topic in lda.components_:
            top_words_idx = topic.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)

    return TopicResult(keywords=keywords, topics=topics)
