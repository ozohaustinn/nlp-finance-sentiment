"""
visualise.py
Generates sentiment charts for any ticker analysed
by run_analysis.py. Called automatically at the end
of each run and saves charts to outputs/.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("outputs")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

CATEGORY_LABELS = {
    "credit_risk":    "Credit risk",
    "market_risk":    "Market risk",
    "macro_signals":  "Macro signals",
    "liquidity_risk": "Liquidity risk",
}

SENTIMENT_COLORS = {
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}


def pick_color(score: float) -> str:
    return "#1a1a2e" if score >= 0 else "#c0392b"


def chart_overall(ticker: str, summary: dict):
    score = summary["overall_sentiment_score"]
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.barh([ticker], [score], color=pick_color(score), height=0.35)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    label_x = score + 0.001 if score >= 0 else score - 0.001
    ha = "left" if score >= 0 else "right"
    ax.text(label_x, 0, f"{score:+.4f}",
            va="center", ha=ha, fontsize=13, fontweight="bold")

    ax.set_xlabel("Sentiment score  (negative ← 0 → positive)")
    ax.set_title(f"{ticker} — overall MD&A sentiment score",
                 fontweight="bold", pad=12)
    limit = max(abs(score) * 3, 0.05)
    ax.set_xlim(-limit, limit)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{ticker}_chart_01_overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out.name}")


def chart_distribution(ticker: str, summary: dict):
    dist = summary["distribution"]
    total = summary["total_scored"]
    labels = ["positive", "negative", "neutral"]
    values = [dist[l] / total * 100 for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("% of risk-flagged sentences")
    ax.set_ylim(0, 105)
    ax.set_title(f"{ticker} — sentiment distribution",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{ticker}_chart_02_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out.name}")


def chart_by_category(ticker: str, summary: dict):
    categories = list(CATEGORY_LABELS.keys())
    labels = list(CATEGORY_LABELS.values())
    scores = [summary["by_category"].get(c, 0) for c in categories]
    colors = [pick_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, scores, color=colors, width=0.5, alpha=0.9)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.001 if val >= 0 else val - 0.002,
                f"{val:+.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Sentiment score  (negative ← 0 → positive)")
    ax.set_title(f"{ticker} — sentiment score by risk category",
                 fontweight="bold", pad=12)

    limit = max(max(abs(s) for s in scores) * 1.6, 0.05)
    ax.set_ylim(-limit, limit)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{ticker}_chart_03_by_category.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out.name}")


def chart_top_sentences(ticker: str, scored_sentences: list, n: int = 5):
    import pandas as pd
    df = pd.DataFrame(scored_sentences)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    for ax, sentiment in zip(axes, ["negative", "positive"]):
        subset = df[df["sentiment"] == sentiment].nlargest(n, "confidence")
        if subset.empty:
            ax.text(0.5, 0.5, f"No {sentiment} sentences found",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        truncated = [s[:90] + "..." if len(s) > 90
                     else s for s in subset["sentence"]]
        confidences = subset["confidence"].tolist()
        color = SENTIMENT_COLORS[sentiment]

        y = range(len(truncated))
        ax.barh(list(y), confidences, color=color, alpha=0.85, height=0.5)
        ax.set_yticks(list(y))
        ax.set_yticklabels(truncated, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Confidence")
        ax.set_title(f"Top {n} {sentiment} sentences — {ticker}",
                     fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / f"{ticker}_chart_04_top_sentences.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out.name}")


def run(ticker: str, sentiment_output: dict):
    print(f"\n[Visualise] {ticker}")
    summary = sentiment_output["summary"]
    scored = sentiment_output["scored_sentences"]

    chart_overall(ticker, summary)
    chart_distribution(ticker, summary)
    chart_by_category(ticker, summary)
    chart_top_sentences(ticker, scored)

    print(f"  All charts saved to outputs/")