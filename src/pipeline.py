"""
pipeline.py
Handles sentence tokenisation, noise removal,
TF-IDF vectorisation and risk phrase flagging.

Noise removal is institution-agnostic — no hardcoded
bank-specific phrases. Uses heuristic signals (number
density, uppercase ratio, special character density)
combined with TF-IDF bottom-percentile removal to catch
both structural noise (tables, headers) and semantic
boilerplate (legal disclaimers, standard disclosures).
"""

import re
import json
import pathlib
from datetime import datetime

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

OUTPUT_DIR = pathlib.Path("outputs")

CUSTOM_STOP_WORDS = [
    "30", "31", "2024", "2025", "june", "december", "january",
    "february", "march", "april", "july", "august", "september",
    "october", "november", "billion", "million", "trillion",
    "certain", "related", "including", "following", "based",
    "used", "using", "also", "may", "one", "two", "three",
    "increase", "decrease", "increased", "decreased", "compared",
    "prior", "period", "quarter", "year", "ended", "page",
    "table", "note", "refer", "information", "form", "total",
    "net", "per", "non", "approximately", "respectively"
]

RISK_CATEGORIES = {
    "credit_risk": [
        "credit loss", "allowance", "charge-off", "nonperforming",
        "net charge", "loan loss", "default", "delinquency",
        "credit quality", "impairment", "provisioning", "provision",
        "past due", "criticized", "classified", "substandard"
    ],
    "market_risk": [
        "value at risk", "var", "market risk", "interest rate risk",
        "volatility", "fair value", "hedging", "derivatives",
        "net interest", "spread", "mark to market", "basis risk",
        "duration", "convexity", "yield curve"
    ],
    "macro_signals": [
        "economic outlook", "recession", "inflation", "federal reserve",
        "rate hike", "unemployment", "gdp", "macroeconomic",
        "geopolitical", "uncertainty", "tariff", "trade",
        "central bank", "monetary policy", "fiscal", "slowdown",
        "contraction", "expansion", "cycle"
    ],
    "liquidity_risk": [
        "liquidity", "funding", "deposit", "borrowing",
        "cash flow", "stress test", "hqla", "lcr",
        "runoff", "outflow", "inflow", "contingency",
        "wholesale funding", "repo", "commercial paper"
    ]
}


def sentence_features(sentence: str) -> dict:
    words = sentence.split()
    if not words:
        return {}

    num_words = len(words)
    num_numbers = sum(1 for w in words if re.search(r'\d', w))
    num_upper = sum(1 for w in words if w.isupper() and len(w) > 1)
    num_special = sum(1 for c in sentence if c in "$%•–—|/\\")
    num_chars = len(sentence)

    return {
        "num_words": num_words,
        "number_ratio": num_numbers / num_words,
        "upper_ratio": num_upper / num_words,
        "special_density": num_special / num_chars,
        "avg_word_len": sum(len(w) for w in words) / num_words,
    }


def is_noise(sentence: str) -> bool:
    f = sentence_features(sentence)
    if not f:
        return True

    # Too short to carry analytical signal
    if f["num_words"] < 12:
        return True

    # Table row — dominated by numbers
    if f["number_ratio"] > 0.35:
        return True

    # Header — dominated by uppercase words
    if f["upper_ratio"] > 0.4:
        return True

    # Dense special characters — bullet lists, financial tables
    if f["special_density"] > 0.08:
        return True

    # Very short average word length — likely abbreviations or codes
    if f["avg_word_len"] < 3.5:
        return True

    return False


def remove_low_tfidf(sentences: list, threshold_pct: float = 0.05) -> list:
    """
    Remove sentences in the bottom threshold_pct of mean TF-IDF scores.
    These are semantically generic — boilerplate legal disclaimers and
    standard disclosures that appear similarly across all filings.
    """
    all_stop_words = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS
    vectorizer = TfidfVectorizer(
        stop_words=all_stop_words,
        ngram_range=(1, 2),
        min_df=2
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return sentences

    sentence_scores = np.asarray(matrix.mean(axis=1)).flatten()
    threshold = np.percentile(sentence_scores, threshold_pct * 100)

    kept = [s for s, score in zip(sentences, sentence_scores)
            if score >= threshold]
    removed = len(sentences) - len(kept)
    print(f"  TF-IDF boilerplate removal: {removed} sentences removed")
    return kept


def flag_risk_sentences(sentences: list) -> list:
    flagged = []
    for sentence in sentences:
        s_lower = sentence.lower()
        matched_categories = []
        matched_phrases = []
        for category, phrases in RISK_CATEGORIES.items():
            hits = [p for p in phrases if p in s_lower]
            if hits:
                matched_categories.append(category)
                matched_phrases.extend(hits)
        if matched_categories:
            flagged.append({
                "sentence": sentence,
                "categories": matched_categories,
                "phrases": matched_phrases
            })
    return flagged


def get_tfidf_terms(sentences: list, top_n: int = 30) -> list:
    all_stop_words = list(ENGLISH_STOP_WORDS) + CUSTOM_STOP_WORDS
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=all_stop_words,
        ngram_range=(1, 2),
        min_df=3,
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    return [
        {"term": feature_names[idx], "score": round(float(scores[idx]), 3)}
        for idx in top_indices
    ]


def run(ticker: str, mda_text: str) -> dict:
    print(f"\n[Pipeline] {ticker}")

    # Step 1: tokenise
    sentences = sent_tokenize(mda_text)
    print(f"  Total sentences: {len(sentences):,}")

    # Step 2: heuristic noise removal
    clean = [s for s in sentences if not is_noise(s)]
    print(f"  After heuristic noise removal: {len(clean):,}")

    # Step 3: semantic boilerplate removal
    clean = remove_low_tfidf(clean, threshold_pct=0.05)
    print(f"  After TF-IDF boilerplate removal: {len(clean):,}")

    # Step 4: risk flagging
    flagged = flag_risk_sentences(clean)
    print(f"  Risk-flagged sentences: {len(flagged):,}")

    # Step 5: TF-IDF top terms on flagged sentences
    tfidf_terms = get_tfidf_terms(
        [s["sentence"] for s in flagged]
    )

    result = {
        "ticker": ticker,
        "processed_at": datetime.now().isoformat(),
        "stats": {
            "total_sentences": len(sentences),
            "clean_sentences": len(clean),
            "risk_flagged": len(flagged),
            "coverage_pct": round(len(flagged) / len(clean) * 100, 1)
                            if clean else 0
        },
        "top_tfidf_terms": tfidf_terms,
        "flagged_sentences": flagged
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{ticker}_nlp_pipeline_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved → {out_path.name}")
    return result