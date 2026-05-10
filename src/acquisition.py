"""
acquisition.py
Handles SEC EDGAR filing download and MD&A text extraction.
"""

import re
import json
import pathlib
from datetime import datetime
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup


DATA_DIR = pathlib.Path("data/filings")
OUTPUT_DIR = pathlib.Path("outputs")


def download_filings(ticker: str, filing_type: str = "10-Q", limit: int = 3, email: str = "user@example.com"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader("NLPFinance", email, str(DATA_DIR))
    dl.get(filing_type, ticker, limit=limit)
    print(f"Downloaded {limit} {filing_type} filings for {ticker}")


def get_latest_filing_path(ticker: str, filing_type: str = "10-Q") -> pathlib.Path:
    base = DATA_DIR / "sec-edgar-filings" / ticker / filing_type
    folders = sorted(base.iterdir())
    if not folders:
        raise FileNotFoundError(f"No filings found for {ticker}")
    return folders[0] / "full-submission.txt"


def extract_primary_document(filing_path: pathlib.Path) -> str:
    with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    doc_blocks = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', raw, re.DOTALL)

    for doc in doc_blocks:
        type_match = re.search(r'<TYPE>(.*?)\n', doc)
        if type_match and type_match.group(1).strip() == filing_type_from_doc(doc, filing_path):
            html_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            if html_match:
                return html_match.group(1)

    # Fallback: return largest document block
    largest = max(doc_blocks, key=len)
    html_match = re.search(r'<TEXT>(.*?)</TEXT>', largest, re.DOTALL)
    return html_match.group(1) if html_match else largest


def filing_type_from_doc(doc: str, path: pathlib.Path) -> str:
    # Infer filing type from path
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "sec-edgar-filings" and i + 2 < len(parts):
            return parts[i + 2]
    return "10-Q"


def extract_mda(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()

    full_text = soup.get_text(separator=" ", strip=True)
    full_text = re.sub(r'\s+', ' ', full_text)

    # MD&A start patterns — ordered by specificity
    START_PATTERNS = [
        r"management.{0,10}s discussion and analysis of financial condition",
        r"management.{0,10}s discussion and analysis",
        r"MANAGEMENT.{0,10}S DISCUSSION AND ANALYSIS",
    ]

    # End boundary patterns — tried in order, first match wins
    END_PATTERNS = [
        r"quantitative and qualitative disclosures about market risk",
        r"disclosure controls and procedures",
        r"controls and procedures",
        r"legal proceedings",
        r"item\s+3[\.\s]",
        r"item\s+4[\.\s]",
    ]

    # Find start position
    start_pos = None
    for pattern in START_PATTERNS:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        # Take the last match of the start pattern — skips TOC entries
        if matches:
            start_pos = matches[-1].end()
            break

    if start_pos is None:
        raise ValueError("MD&A start heading not found in filing")

    # Find end position — try each boundary, take earliest one after start
    end_pos = None
    for pattern in END_PATTERNS:
        matches = list(re.finditer(pattern, full_text[start_pos:], re.IGNORECASE))
        if matches:
            # Take first match after start, but only if it gives us >10k chars
            candidate = matches[0].start()
            if candidate > 10000:
                end_pos = start_pos + candidate
                print(f"  MD&A end boundary: '{pattern}'")
                break

    if end_pos is None:
        # Fallback: take 600k chars from start — enough for any MD&A
        end_pos = start_pos + 600000
        print(f"  MD&A end boundary: fallback (600k chars)")

    mda = full_text[start_pos:end_pos].strip()

    if len(mda) < 10000:
        raise ValueError(f"MD&A too short ({len(mda):,} chars) — extraction may have failed")

    return mda

    if not matches:
        raise ValueError("MD&A section not found in filing")

    return max(matches, key=lambda m: len(m.group(1))).group(1).strip()


def run(ticker: str, filing_type: str = "10-Q", limit: int = 3, email: str = "user@example.com") -> dict:
    print(f"\n[Acquisition] {ticker} — {filing_type}")

    download_filings(ticker, filing_type, limit, email)
    filing_path = get_latest_filing_path(ticker, filing_type)

    print(f"  Reading: {filing_path.name}")
    html_content = extract_primary_document(filing_path)

    print(f"  Extracting MD&A...")
    mda_text = extract_mda(html_content)

    result = {
        "ticker": ticker,
        "filing_type": filing_type,
        "extracted_at": datetime.now().isoformat(),
        "mda_char_count": len(mda_text),
        "mda_text": mda_text
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{ticker}_mda_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  MD&A extracted: {len(mda_text):,} chars → {out_path.name}")
    return result