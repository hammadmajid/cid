import streamlit as st
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
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


def _get_comprehensive_product_patterns():
    """Get comprehensive product patterns across all categories."""
    return [
        # ==================== TECHNOLOGY ====================
        # Apple products
        {"label": "PRODUCT", "pattern": "iPhone"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "iphone"}]},
        {"label": "PRODUCT", "pattern": [{"TEXT": "iPhone"}, {"IS_DIGIT": True}]},
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "iPhone"},
                {"IS_DIGIT": True},
                {"TEXT": {"REGEX": "Pro|Plus|Max|Mini"}},
            ],
        },
        {"label": "PRODUCT", "pattern": "iPad"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "ipad"}]},
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": "iPad"}, {"TEXT": {"REGEX": "Pro|Air|Mini"}}],
        },
        {"label": "PRODUCT", "pattern": "MacBook"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "macbook"}]},
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": "MacBook"}, {"TEXT": {"REGEX": "Pro|Air"}}],
        },
        {"label": "PRODUCT", "pattern": "AirPods"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "airpods"}]},
        {"label": "PRODUCT", "pattern": "Apple Watch"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "apple"}, {"LOWER": "watch"}]},
        {"label": "PRODUCT", "pattern": "iMac"},
        {"label": "PRODUCT", "pattern": [{"LOWER": "imac"}]},
        {"label": "PRODUCT", "pattern": "Mac Mini"},
        {"label": "PRODUCT", "pattern": "Mac Pro"},
        {"label": "PRODUCT", "pattern": "AirTag"},
        {"label": "PRODUCT", "pattern": "HomePod"},
        # Microsoft products
        {"label": "PRODUCT", "pattern": "Windows"},
        {"label": "PRODUCT", "pattern": [{"TEXT": "Windows"}, {"IS_DIGIT": True}]},
        {"label": "PRODUCT", "pattern": "Surface"},
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "Surface"},
                {"TEXT": {"REGEX": "Pro|Laptop|Book|Go|Studio"}},
            ],
        },
        {"label": "PRODUCT", "pattern": "Xbox"},
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": "Xbox"}, {"TEXT": {"REGEX": "Series|One"}}],
        },
        {"label": "PRODUCT", "pattern": "Microsoft Office"},
        {"label": "PRODUCT", "pattern": "Office 365"},
        {"label": "PRODUCT", "pattern": "Microsoft Teams"},
        {"label": "PRODUCT", "pattern": "Azure"},
        # Google products
        {"label": "PRODUCT", "pattern": "Pixel"},
        {"label": "PRODUCT", "pattern": [{"TEXT": "Pixel"}, {"IS_DIGIT": True}]},
        {"label": "PRODUCT", "pattern": [{"TEXT": "Google"}, {"TEXT": "Pixel"}]},
        {"label": "PRODUCT", "pattern": "Chromebook"},
        {"label": "PRODUCT", "pattern": "Nest"},
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "Nest"},
                {"TEXT": {"REGEX": "Hub|Mini|Cam|Thermostat"}},
            ],
        },
        {"label": "PRODUCT", "pattern": "Chromecast"},
        # Samsung products
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "Galaxy"},
                {"TEXT": {"REGEX": "S|Note|Z|A|Fold|Flip"}},
            ],
        },
        {"label": "PRODUCT", "pattern": [{"TEXT": "Samsung"}, {"TEXT": "Galaxy"}]},
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "Galaxy"},
                {"TEXT": {"REGEX": "S|Note|Z|A|Fold|Flip"}},
                {"IS_DIGIT": True},
            ],
        },
        # Gaming consoles
        {"label": "PRODUCT", "pattern": "PlayStation"},
        {"label": "PRODUCT", "pattern": [{"TEXT": "PlayStation"}, {"IS_DIGIT": True}]},
        {"label": "PRODUCT", "pattern": "PS5"},
        {"label": "PRODUCT", "pattern": "PS4"},
        {"label": "PRODUCT", "pattern": "Nintendo Switch"},
        {"label": "PRODUCT", "pattern": "Steam Deck"},
        # Other tech
        {"label": "PRODUCT", "pattern": "Kindle"},
        {
            "label": "PRODUCT",
            "pattern": [
                {"TEXT": "Kindle"},
                {"TEXT": {"REGEX": "Paperwhite|Oasis|Scribe"}},
            ],
        },
        {"label": "PRODUCT", "pattern": "Fire TV"},
        {"label": "PRODUCT", "pattern": "Echo"},
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": "Echo"}, {"TEXT": {"REGEX": "Dot|Show|Studio"}}],
        },
        {"label": "PRODUCT", "pattern": "Alexa"},
        # ==================== AUTOMOTIVE ====================
        # Tesla
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": "Model"}, {"TEXT": {"REGEX": "S|3|X|Y"}}],
        },
        {"label": "PRODUCT", "pattern": [{"TEXT": "Tesla"}, {"TEXT": "Model"}]},
        {"label": "PRODUCT", "pattern": "Cybertruck"},
        {"label": "PRODUCT", "pattern": "Tesla Roadster"},
        # Other automotive
        {"label": "PRODUCT", "pattern": "Mustang"},
        {"label": "PRODUCT", "pattern": "F-150"},
        {"label": "PRODUCT", "pattern": "Civic"},
        {"label": "PRODUCT", "pattern": "Accord"},
        {"label": "PRODUCT", "pattern": "Camry"},
        {"label": "PRODUCT", "pattern": "Corolla"},
        {"label": "PRODUCT", "pattern": "Prius"},
        {"label": "PRODUCT", "pattern": "911"},
        {"label": "PRODUCT", "pattern": "Porsche 911"},
        {"label": "PRODUCT", "pattern": "Range Rover"},
        {"label": "PRODUCT", "pattern": "BMW"},
        {
            "label": "PRODUCT",
            "pattern": [{"TEXT": {"REGEX": "BMW|Mercedes|Audi"}}, {"IS_DIGIT": True}],
        },
        # ==================== BEVERAGES ====================
        {"label": "PRODUCT", "pattern": "Coca-Cola"},
        {"label": "PRODUCT", "pattern": "Coke"},
        {"label": "PRODUCT", "pattern": "Pepsi"},
        {"label": "PRODUCT", "pattern": "Sprite"},
        {"label": "PRODUCT", "pattern": "Fanta"},
        {"label": "PRODUCT", "pattern": "Mountain Dew"},
        {"label": "PRODUCT", "pattern": "Red Bull"},
        {"label": "PRODUCT", "pattern": "Monster Energy"},
        {"label": "PRODUCT", "pattern": "Gatorade"},
        {"label": "PRODUCT", "pattern": "Starbucks"},
        # ==================== FOOD ====================
        {"label": "PRODUCT", "pattern": "Big Mac"},
        {"label": "PRODUCT", "pattern": "Whopper"},
        {"label": "PRODUCT", "pattern": "Happy Meal"},
        {"label": "PRODUCT", "pattern": "Doritos"},
        {"label": "PRODUCT", "pattern": "Cheetos"},
        {"label": "PRODUCT", "pattern": "Lay's"},
        {"label": "PRODUCT", "pattern": "Pringles"},
        {"label": "PRODUCT", "pattern": "Oreo"},
        {"label": "PRODUCT", "pattern": "KitKat"},
        {"label": "PRODUCT", "pattern": "Snickers"},
        {"label": "PRODUCT", "pattern": "M&M's"},
        # ==================== CONSUMER GOODS ====================
        {"label": "PRODUCT", "pattern": "Tide"},
        {"label": "PRODUCT", "pattern": "Pampers"},
        {"label": "PRODUCT", "pattern": "Gillette"},
        {"label": "PRODUCT", "pattern": "Oral-B"},
        {"label": "PRODUCT", "pattern": "Crest"},
        {"label": "PRODUCT", "pattern": "Colgate"},
        {"label": "PRODUCT", "pattern": "Kleenex"},
        {"label": "PRODUCT", "pattern": "Lysol"},
        # ==================== FASHION & APPAREL ====================
        {"label": "PRODUCT", "pattern": "Air Jordan"},
        {"label": "PRODUCT", "pattern": [{"TEXT": "Air"}, {"TEXT": "Jordan"}]},
        {"label": "PRODUCT", "pattern": "Nike Air"},
        {"label": "PRODUCT", "pattern": "Adidas Ultraboost"},
        {"label": "PRODUCT", "pattern": "Ray-Ban"},
        {"label": "PRODUCT", "pattern": "Levi's"},
        {"label": "PRODUCT", "pattern": [{"TEXT": "Levi's"}, {"IS_DIGIT": True}]},
        # ==================== PHARMACEUTICALS ====================
        {"label": "PRODUCT", "pattern": "Tylenol"},
        {"label": "PRODUCT", "pattern": "Advil"},
        {"label": "PRODUCT", "pattern": "Aspirin"},
        {"label": "PRODUCT", "pattern": "Lipitor"},
        {"label": "PRODUCT", "pattern": "Viagra"},
        {"label": "PRODUCT", "pattern": "Prozac"},
        # ==================== SOFTWARE ====================
        {"label": "PRODUCT", "pattern": "ChatGPT"},
        {"label": "PRODUCT", "pattern": "Google Chrome"},
        {"label": "PRODUCT", "pattern": "Firefox"},
        {"label": "PRODUCT", "pattern": "Safari"},
        {"label": "PRODUCT", "pattern": "Photoshop"},
        {"label": "PRODUCT", "pattern": "Adobe Photoshop"},
        {"label": "PRODUCT", "pattern": "Microsoft Word"},
        {"label": "PRODUCT", "pattern": "Excel"},
        {"label": "PRODUCT", "pattern": "PowerPoint"},
        {"label": "PRODUCT", "pattern": "Slack"},
        {"label": "PRODUCT", "pattern": "Zoom"},
    ]


def _setup_contextual_product_matcher(nlp):
    """Setup matcher for contextual product detection."""
    matcher = Matcher(nlp.vocab)

    # Pattern 1: "product is XXX" / "product was XXX" / "product will be XXX"
    matcher.add(
        "PRODUCT_IS",
        [
            [
                {"LOWER": "product"},
                {"LEMMA": {"IN": ["be", "is", "was", "were", "will"]}},
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 2: "launched XXX" / "released XXX" / "unveiled XXX"
    matcher.add(
        "PRODUCT_LAUNCH",
        [
            [
                {
                    "LEMMA": {
                        "IN": [
                            "launch",
                            "release",
                            "unveil",
                            "introduce",
                            "announce",
                            "debut",
                        ]
                    }
                },
                {"POS": {"IN": ["DET", "ADJ"]}, "OP": "*"},  # Optional: "the new"
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 3: "they launched XXX" / "company released XXX"
    matcher.add(
        "SUBJECT_LAUNCHED",
        [
            [
                {"POS": {"IN": ["PRON", "PROPN", "NOUN"]}},
                {
                    "LEMMA": {
                        "IN": [
                            "launch",
                            "release",
                            "unveil",
                            "introduce",
                            "announce",
                            "debut",
                        ]
                    }
                },
                {"POS": {"IN": ["DET", "ADJ"]}, "OP": "*"},
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 4: "selling XXX" / "bought XXX" / "purchased XXX"
    matcher.add(
        "PRODUCT_TRANSACTION",
        [
            [
                {"LEMMA": {"IN": ["sell", "buy", "purchase", "order", "ship"]}},
                {"POS": {"IN": ["DET", "ADJ"]}, "OP": "*"},
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 5: "using XXX" / "with XXX"
    matcher.add(
        "PRODUCT_USAGE",
        [
            [
                {"LEMMA": {"IN": ["use", "using", "with"]}},
                {"POS": {"IN": ["DET", "ADJ"]}, "OP": "*"},
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 6: "new XXX" / "latest XXX"
    matcher.add(
        "NEW_PRODUCT",
        [
            [
                {"LOWER": {"IN": ["new", "latest", "upcoming", "next"]}},
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
            ]
        ],
    )

    # Pattern 7: "XXX features" / "XXX specs" / "XXX price"
    matcher.add(
        "PRODUCT_ATTRIBUTES",
        [
            [
                {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "+"},
                {
                    "LOWER": {
                        "IN": [
                            "feature",
                            "features",
                            "spec",
                            "specs",
                            "price",
                            "pricing",
                            "cost",
                            "review",
                            "reviews",
                        ]
                    }
                },
            ]
        ],
    )

    return matcher


def _add_contextual_products(doc, matcher):
    """Add products detected by contextual patterns."""
    matches = matcher(doc)
    new_ents = []

    for match_id, start, end in matches:
        # Extract the product name (skip verb/determiner/adjective)
        span = doc[start:end]

        # Find the actual product name (proper nouns and nouns)
        product_tokens = []
        for token in span:
            if token.pos_ in ["PROPN", "NOUN"] and not token.is_stop:
                product_tokens.append(token)
            elif product_tokens and token.pos_ in ["PROPN", "NOUN"]:
                # Continue if we're in a noun phrase
                product_tokens.append(token)

        if product_tokens:
            # Create entity from the product tokens
            product_start = product_tokens[0].i
            product_end = product_tokens[-1].i + 1

            # Check if this overlaps with existing entities
            overlaps = False
            for ent in doc.ents:
                if product_start < ent.end and product_end > ent.start:
                    # Only override if it's not already a PRODUCT
                    if ent.label_ == "PRODUCT":
                        overlaps = True
                        break

            if not overlaps:
                new_span = Span(doc, product_start, product_end, label="PRODUCT")
                new_ents.append(new_span)

    # Merge new entities with existing ones
    if new_ents:
        try:
            # Filter out overlapping entities, prefer PRODUCT label
            original_ents = list(doc.ents)
            all_ents = original_ents + new_ents

            # Remove duplicates and overlaps, preferring PRODUCT entities
            filtered_ents = []
            for ent in sorted(all_ents, key=lambda e: (e.start, -len(e.text))):
                # Check if this entity overlaps with any already added
                overlaps = False
                for existing in filtered_ents:
                    if ent.start < existing.end and ent.end > existing.start:
                        overlaps = True
                        break
                if not overlaps:
                    filtered_ents.append(ent)

            doc.ents = tuple(filtered_ents)
        except Exception:
            # If merging fails, keep original entities
            pass

    return doc


@st.cache_resource
def load_spacy_model():
    """Load and cache the spaCy model with comprehensive product extraction pipeline."""
    try:
        nlp = spacy.load(SPACY_MODEL)

        # Step 1: Add Entity Ruler with known product patterns
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(_get_comprehensive_product_patterns())

        # Step 2: Setup contextual matcher (applied during extraction)
        nlp._product_matcher = _setup_contextual_product_matcher(nlp)

        return nlp
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
    """Extract named entities using spaCy with contextual product detection."""
    doc = nlp(text)

    # Apply contextual product matcher
    if hasattr(nlp, "_product_matcher"):
        doc = _add_contextual_products(doc, nlp._product_matcher)

    orgs = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    products = list(set(ent.text for ent in doc.ents if ent.label_ == "PRODUCT"))
    locations = list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))

    all_entities = [
        {
            "entity": ent.text,
            "type": ent.label_,
            "context": text[
                max(0, ent.start_char - 20) : min(len(text), ent.end_char + 20)
            ],
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

    return sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[
        :max_keywords
    ]


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
