PAGE_CONFIG = {
    "page_title": "Competitive Intelligence Dashboard",
    "page_icon": "chart_with_upwards_trend",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

STYLES = """
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
"""

# Chart colors
CHART_BG_COLOR = "#FDFBF7"
ACCENT_COLOR = "#D4A574"
TEXT_COLOR = "#1A1A1A"

# NLP settings
SPACY_MODEL = "en_core_web_sm"
LDA_MAX_ITER = 10
LDA_RANDOM_STATE = 42
VECTORIZER_MAX_FEATURES = 100
