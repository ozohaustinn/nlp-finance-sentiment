"""
sentiment.py
Scores risk-flagged sentences using FinBERT —
a transformer model pre-trained on financial text.

Produces sentence-level sentiment labels (positive /
negative / neutral) with confidence scores, plus a
structured summary with overall and category-level
sentiment scores.
"""

import json
import pathlib
from datetime import datetime

import torch
import pandas as pd
from transformers import pipeline


OUTPUT_DIR = pathlib.Path("outputs")

RISK_CATEGORIES = [
    "credit_risk",
    "market_risk",
    "macro_signals",
    "liquidity_risk"
]


def load_finbert():
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Device: {'GPU' if device == 0 else 'CPU'}")
    finbert = pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=device
    )
    print(f"  FinBERT loaded.")
    return finbert


def score_sentences(finbert, sentences: list, batch_size: int = 16) -> list:
    from tqdm.auto import tqdm
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Scoring"):
        batch = sentences[i:i + batch_size]
        scored = finbert(batch, truncation=True, max_length=512,
                         batch_size=batch_size)
        results.extend(scored)
    return results


def sentiment_score(label: str, confidence: float) -> float:
    if label == "positive":
        return confidence
    elif label == "negative":
        return -confidence
    return 0.0


def build_dataframe(flagged_sentences: list, results: list) -> pd.DataFrame:
    rows = []
    for item, result in zip(flagged_sentences, results):
        rows.append({
            "sentence": item["sentence"],
            "categories": item["categories"],
            "phrases": item["phrases"],
            "sentiment": result["label"],
            "confidence": round(result["score"], 4),
            "score": sentiment_score(result["label"], result["score"])
        })
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict:
    overall = round(df["score"].mean(), 4)
    distribution = {
        label: int((df["sentiment"] == label).sum())
        for label in ["positive", "negative", "neutral"]
    }
    by_category = {}
    for cat in RISK_CATEGORIES:
        cat_df = df[df["categories"].apply(lambda x: cat in x)]
        by_category[cat] = round(cat_df["score"].mean(), 4) if len(cat_df) else 0.0

    return {
        "total_scored": len(df),
        "overall_sentiment_score": overall,
        "mean_confidence": round(df["confidence"].mean(), 4),
        "low_confidence_count": int((df["confidence"] < 0.6).sum()),
        "distribution": distribution,
        "by_category": by_category
    }


def run(ticker: str, flagged_sentences: list) -> dict:
    print(f"\n[Sentiment] {ticker}")

    finbert = load_finbert()

    sentences_text = [item["sentence"] for item in flagged_sentences]
    print(f"  Sentences to score: {len(sentences_text):,}")

    results = score_sentences(finbert, sentences_text)
    df = build_dataframe(flagged_sentences, results)
    summary = compute_summary(df)

    print(f"\n  Overall sentiment score : {summary['overall_sentiment_score']:+.4f}")
    print(f"  Distribution — "
          f"pos: {summary['distribution']['positive']}  "
          f"neg: {summary['distribution']['negative']}  "
          f"neu: {summary['distribution']['neutral']}")
    print(f"  By category:")
    for cat, score in summary["by_category"].items():
        direction = "+" if score >= 0 else ""
        print(f"    {cat:<20} {direction}{score:.4f}")

    output = {
        "ticker": ticker,
        "scored_at": datetime.now().isoformat(),
        "model": "ProsusAI/finbert",
        "summary": summary,
        "scored_sentences": df.to_dict(orient="records")
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{ticker}_sentiment_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Saved → {out_path.name}")
    return output