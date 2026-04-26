# ============================================================
# modules/visualizer.py  –  Matplotlib charts for stored results
# ============================================================

import matplotlib
matplotlib.use("TkAgg")          # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import Counter

from modules.database_manager import fetch_all


# ── Colour palette ────────────────────────────────────────────
COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#3498db",
}


def _sentiment_counts(records: list) -> Counter:
    return Counter(r["sentiment"] for r in records)


# ── Individual charts ─────────────────────────────────────────

def plot_sentiment_distribution(records: list, ax=None):
    """Bar chart of sentiment class counts."""
    counts = _sentiment_counts(records)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS.get(l, "#95a5a6") for l in labels]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.bar_label(bars, padding=3, fontsize=11, fontweight="bold")
    ax.set_title("Sentiment Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values, default=1) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_sentiment_pie(records: list, ax=None):
    """Pie chart showing percentage breakdown."""
    counts = _sentiment_counts(records)
    if not counts:
        return

    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = [COLORS.get(l, "#95a5a6") for l in labels]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Sentiment Proportions", fontsize=13, fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_top_words(records: list, ax=None):
    """
    Bar chart comparing the top 5 positive vs. top 5 negative
    keyword occurrences across all stored results.
    """
    from modules.model_loader import load_keywords
    from modules.feature_extractor import get_matched_keywords

    keywords = load_keywords()
    pos_counter: Counter = Counter()
    neg_counter: Counter = Counter()

    for r in records:
        text = r.get("clean_text") or r.get("raw_text") or ""
        if not text:
            continue
        matched = get_matched_keywords(text, keywords)
        pos_counter.update(matched["positive"])
        neg_counter.update(matched["negative"])

    top_pos = pos_counter.most_common(5)
    top_neg = neg_counter.most_common(5)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    pos_words = [w for w, _ in top_pos]
    pos_vals  = [c for _, c in top_pos]
    neg_words = [w for w, _ in top_neg]
    neg_vals  = [c for _, c in top_neg]

    x_pos = np.arange(len(pos_words))
    x_neg = np.arange(len(neg_words))

    ax.barh(pos_words, pos_vals, color=COLORS["positive"],
            label="Positive Keywords", alpha=0.85)
    ax.barh(
        [f"-{w}" for w in neg_words],
        [-v for v in neg_vals],
        color=COLORS["negative"],
        label="Negative Keywords", alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Keyword Occurrences", fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency  (positive → | ← negative)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_sentiment_trend(records: list, ax=None):
    """
    Line chart showing cumulative sentiment over time
    (order of insertion as a proxy for time).
    """
    if not records:
        return

    # Map to numeric values for trend
    num_map = {"positive": 1, "neutral": 0, "negative": -1}
    values = [num_map.get(r["sentiment"], 0) for r in reversed(records)]
    x = list(range(1, len(values) + 1))

    # Running average
    window = max(1, len(values) // 10)
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, values, alpha=0.3, color="#7f8c8d", linewidth=1, label="Raw")
    ax.plot(x, smoothed, color="#2980b9", linewidth=2.5, label=f"Trend (avg {window})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(x, smoothed, 0,
                    where=[v > 0 for v in smoothed],
                    alpha=0.15, color=COLORS["positive"])
    ax.fill_between(x, smoothed, 0,
                    where=[v < 0 for v in smoothed],
                    alpha=0.15, color=COLORS["negative"])
    ax.set_title("Sentiment Trend Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Analysis #")
    ax.set_ylabel("Sentiment  (1=pos, 0=neu, -1=neg)")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Negative", "Neutral", "Positive"])
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Combined dashboard ────────────────────────────────────────

def show_dashboard():
    """Open a single window with all four charts arranged in a 2×2 grid."""
    plt.close('all')  # Close any existing figures to allow repetitive calls
    records = fetch_all()
    if not records:
        import tkinter.messagebox as mb
        mb.showinfo("No Data", "No analysis results found in the database yet.")
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Financial Sentiment Analysis Dashboard", fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_sentiment_distribution(records, ax=ax1)
    plot_sentiment_pie(records,          ax=ax2)
    plot_top_words(records,              ax=ax3)
    plot_sentiment_trend(records,        ax=ax4)

    plt.show()
