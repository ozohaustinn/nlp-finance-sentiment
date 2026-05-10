"""
run_analysis.py
Single entry point for the NLP Finance Sentiment tool.

Usage:
    python run_analysis.py --ticker JPM
    python run_analysis.py --ticker C --limit 3
    python run_analysis.py --ticker BAC --filing 10-K --limit 1
"""

import argparse
import json
import pathlib
from datetime import datetime

import src.acquisition as acquisition
import src.pipeline as pipeline
import src.sentiment as sentiment
import src.visualise as visualise

OUTPUT_DIR = pathlib.Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLP sentiment analysis on SEC earnings filings."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. JPM, C, BAC)"
    )
    parser.add_argument(
        "--filing",
        type=str,
        default="10-Q",
        help="SEC filing type (default: 10-Q)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of filings to download (default: 3)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="user@example.com",
        help="Email for SEC EDGAR rate limiting"
    )
    return parser.parse_args()


def print_banner(ticker: str, filing: str):
    print("\n" + "=" * 55)
    print(f"  NLP Finance Sentiment Analyser")
    print(f"  Ticker  : {ticker}")
    print(f"  Filing  : {filing}")
    print(f"  Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)


def save_combined_output(ticker: str, acq: dict, pipe: dict, sent: dict):
    combined = {
        "ticker": ticker,
        "period": acq.get("extracted_at", ""),
        "run_at": datetime.now().isoformat(),
        "acquisition": {
            "mda_char_count": acq["mda_char_count"],
        },
        "pipeline": {
            "stats": pipe["stats"],
            "top_tfidf_terms": pipe["top_tfidf_terms"][:15],
        },
        "sentiment": {
            "model": sent["model"],
            "summary": sent["summary"]
        }
    }

    out_path = OUTPUT_DIR / f"{ticker}_full_report_latest.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n  Full report saved → {out_path.name}")
    return combined


def print_summary(ticker: str, combined: dict):
    s = combined["sentiment"]["summary"]
    p = combined["pipeline"]["stats"]

    print(f"  MD&A sentences analysed : {p['clean_sentences']:,}")
    print(f"  Risk-flagged            : {p['risk_flagged']:,} ({p['coverage_pct']}%)")
    print(f"  Overall sentiment score : {s['overall_sentiment_score']:+.4f}")
    print(f"  Mean confidence         : {s['mean_confidence']:.4f}")
    print(f"\n  Sentiment distribution:")
    for label, count in s["distribution"].items():
        pct = count / s["total_scored"] * 100
        bar = "█" * int(pct / 3)
        print(f"    {label.upper():<10} {count:>4}  ({pct:.1f}%)  {bar}")
    print(f"\n  Score by risk category:")
    for cat, score in s["by_category"].items():
        direction = "+" if score >= 0 else ""
        bar = "▓" * int(abs(score) * 20)
        print(f"    {cat:<20} {direction}{score:.4f}  {bar}")
    print("=" * 55)


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    print_banner(ticker, args.filing)

    # Stage 1 — Acquisition
    acq_result = acquisition.run(
        ticker=ticker,
        filing_type=args.filing,
        limit=args.limit,
        email=args.email
    )

    # Stage 2 — NLP Pipeline
    pipe_result = pipeline.run(
        ticker=ticker,
        mda_text=acq_result["mda_text"]
    )

    # Stage 3 — Sentiment
    sent_result = sentiment.run(
        ticker=ticker,
        flagged_sentences=pipe_result["flagged_sentences"]
    )

    # Save combined report
    combined = save_combined_output(ticker, acq_result, pipe_result, sent_result)
    print_summary(ticker, combined)
    visualise.run(ticker, sent_result)


if __name__ == "__main__":
    main()