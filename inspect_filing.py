"""
inspect_filing.py
Diagnostic — inspect document structure of a SEC filing.
Usage: python inspect_filing.py --ticker C
"""

import re
import argparse
import pathlib
from bs4 import BeautifulSoup

DATA_DIR = pathlib.Path("data/filings")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    return parser.parse_args()


def get_latest_filing(ticker):
    base = DATA_DIR / "sec-edgar-filings" / ticker / "10-Q"
    folders = sorted(base.iterdir())
    return folders[0] / "full-submission.txt"


def main():
    args = parse_args()
    ticker = args.ticker.upper()

    print(f"Inspecting {ticker} filing...\n")
    path = get_latest_filing(ticker)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # List all document blocks
    doc_blocks = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL)
    print(f"Total document blocks: {len(doc_blocks)}\n")

    print("First 25 documents:\n")
    for i, doc in enumerate(doc_blocks[:25]):
        type_match = re.search(r'<TYPE>(.*?)\n', doc)
        filename_match = re.search(r'<FILENAME>(.*?)\n', doc)
        doc_type = type_match.group(1).strip() if type_match else "UNKNOWN"
        filename = filename_match.group(1).strip() if filename_match else "?"
        print(f"  [{i+1:>2}] {doc_type:<30} {filename:<45} {len(doc):>12,} chars")

    # Extract the 10-Q document specifically
    print("\nSearching for 10-Q document block...")
    target = None
    for doc in doc_blocks:
        type_match = re.search(r'<TYPE>(.*?)\n', doc)
        if type_match and type_match.group(1).strip() == "10-Q":
            target = doc
            break

    if not target:
        print("No 10-Q block found — check document types above")
        return

    html_match = re.search(r'<TEXT>(.*?)</TEXT>', target, re.DOTALL)
    htm_content = html_match.group(1) if html_match else target

    soup = BeautifulSoup(htm_content, "html.parser")
    for table in soup.find_all("table"):
        table.decompose()

    full_text = soup.get_text(separator=" ", strip=True)
    full_text = re.sub(r'\s+', ' ', full_text)
    print(f"10-Q narrative text: {len(full_text):,} chars\n")

    print("Management references (first 20):\n")
    for i, m in enumerate(re.finditer(r'management.{0,120}', full_text, re.IGNORECASE)):
        print(f"  {m.group().strip()[:120]}")
        if i >= 19:
            break
    print("\nSearching for section boundaries:\n")
    boundaries = [
        "quantitative and qualitative",
        "controls and procedures",
        "legal proceedings",
        "risk factors",
        "unregistered sales",
        "item 3",
        "item 4"
    ]
    for boundary in boundaries:
        match = re.search(boundary, full_text, re.IGNORECASE)
        if match:
            print(f"  FOUND '{boundary}' at position {match.start():,}")
            print(f"    Context: ...{full_text[match.start()-50:match.start()+80]}...")
        else:
            print(f"  NOT FOUND: '{boundary}'")


if __name__ == "__main__":
    main()