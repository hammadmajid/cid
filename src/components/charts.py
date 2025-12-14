import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..config import CHART_BG_COLOR, ACCENT_COLOR


def sentiment_chart(polarity: float, subjectivity: float) -> Figure:
    """Create sentiment visualization chart."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(CHART_BG_COLOR)

    # Polarity bar
    ax[0].barh(["Polarity"], [polarity], color=ACCENT_COLOR)
    ax[0].set_xlim(-1, 1)
    ax[0].axvline(x=0, color="#1A1A1A", linestyle="--", linewidth=0.8)
    ax[0].set_xlabel("Score", fontsize=11)
    ax[0].set_title("Polarity (Negative to Positive)", fontsize=12, fontweight="bold")
    ax[0].set_facecolor(CHART_BG_COLOR)

    # Subjectivity bar
    ax[1].barh(["Subjectivity"], [subjectivity], color=ACCENT_COLOR)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Score", fontsize=11)
    ax[1].set_title("Subjectivity (Objective to Subjective)", fontsize=12, fontweight="bold")
    ax[1].set_facecolor(CHART_BG_COLOR)

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def keyword_chart(keywords: list[tuple[str, int]], max_display: int = 10) -> Figure:
    """Create keyword frequency chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(CHART_BG_COLOR)
    ax.set_facecolor(CHART_BG_COLOR)

    display_keywords = keywords[:max_display]
    words = [w[0] for w in display_keywords]
    freqs = [w[1] for w in display_keywords]

    ax.barh(words, freqs, color=ACCENT_COLOR)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title("Top Keywords by Frequency", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig
