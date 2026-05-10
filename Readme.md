# NLP Finance Sentiment Analyser

A production-grade NLP tool that reads SEC earnings filings and outputs 
structured sentiment scores across four risk categories — automating what 
sell-side analysts do manually.

Built with `sec-edgar-downloader`, `scikit-learn`, and `ProsusAI/FinBERT`.

---

## What it does

1. **Pulls real SEC filings** — downloads 10-Q or 10-K filings directly from 
   EDGAR for any US-listed company
2. **Extracts MD&A narrative** — isolates the Management Discussion & Analysis 
   section using institution-agnostic boundary detection
3. **Cleans and filters** — removes table noise, headers, and boilerplate using 
   heuristic signals and TF-IDF bottom-percentile removal
4. **Flags risk sentences** — identifies sentences carrying credit, market, 
   macro, and liquidity risk language
5. **Scores with FinBERT** — runs each flagged sentence through a transformer 
   model pre-trained on financial text
6. **Outputs structured results** — JSON report + four charts per company, 
   saved automatically to `outputs/`

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/ozohaustinn/nlp-finance-sentiment.git
cd nlp-finance-sentiment

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# Run on any SEC-filing company
python run_analysis.py --ticker JPM --email "you@example.com"
python run_analysis.py --ticker BAC --email "you@example.com"
python run_analysis.py --ticker GS  --filing 10-K --limit 1 --email "you@example.com"
```

---

## Output

For each ticker, the tool generates:

| File | Description |
|------|-------------|
| `{TICKER}_mda_latest.json` | Raw extracted MD&A text |
| `{TICKER}_nlp_pipeline_latest.json` | Clean sentences + TF-IDF terms |
| `{TICKER}_sentiment_latest.json` | Sentence-level FinBERT scores |
| `{TICKER}_full_report_latest.json` | Combined summary report |
| `{TICKER}_chart_01_overall.png` | Overall sentiment score |
| `{TICKER}_chart_02_distribution.png` | Positive / negative / neutral split |
| `{TICKER}_chart_03_by_category.png` | Score by risk category |
| `{TICKER}_chart_04_top_sentences.png` | Most confident flagged sentences |

---

## Sample results — JPM vs Citi Q2 2025

| Metric | JPMorgan Chase | Citigroup |
|--------|---------------|-----------|
| Overall sentiment score | +0.0143 | -0.0254 |
| Credit risk | -0.0115 | -0.0250 |
| Market risk | +0.0280 | -0.0095 |
| Macro signals | -0.0643 | +0.0016 |
| Liquidity risk | +0.0358 | -0.0422 |
| Sentences scored | 886 | 485 |
| Mean confidence | 0.884 | 0.874 |

**Key findings:**
- JPM net positive overall, driven by strong CIB markets revenue
- Citi net negative across market, credit, and liquidity risk categories
- Both banks flag credit risk negatively — consistent with Q2 2025 provisioning environment
- JPM macro language (-0.064) more explicitly cautious than Citi (+0.002)
- Citi liquidity risk (-0.042) most negative reading — reflects ongoing transformation costs

---

## Project structure

```
nlp-finance-sentiment/
├── src/
│   ├── acquisition.py      # SEC EDGAR download + MD&A extraction
│   ├── pipeline.py         # Noise removal, TF-IDF, risk flagging
│   ├── sentiment.py        # FinBERT scoring + summary computation
│   └── visualise.py        # Chart generation (auto-runs per ticker)
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_nlp_pipeline.ipynb
│   ├── 03_sentiment_engine.ipynb
│   └── 04_report.ipynb
├── outputs/
├── run_analysis.py         # Single entry point CLI
├── inspect_filing.py       # Diagnostic tool for new filing structures
└── requirements.txt
```

## Methodology

**Noise removal** is institution-agnostic — no hardcoded bank-specific phrases. 
The pipeline uses two complementary approaches:

- **Heuristic filters** — number density > 35% (tables), uppercase ratio > 40% 
  (headers), special character density > 8% (bullet/table structure)
- **TF-IDF bottom-percentile removal** — sentences scoring in the bottom 5% of 
  mean TF-IDF across the corpus are semantically generic boilerplate

**Risk categorisation** uses domain-specific phrase matching across four 
categories: credit risk, market risk, macro signals, and liquidity risk.

**Sentiment scoring** uses [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert), 
a BERT model fine-tuned on financial phraseology. Scores are computed as:
- Positive sentence → `+confidence`
- Negative sentence → `-confidence`  
- Neutral sentence → `0`

Overall and category scores are the mean across all scored sentences in that group.

---

## Requirements

```
sec-edgar-downloader
transformers
torch
pandas
scikit-learn
nltk
beautifulsoup4
matplotlib
jupyter
ipykernel
tqdm
```

## Author

**Augustine Ozoemena**  
MSc Finance — Grenoble École de Management

[GitHub](https://github.com/ozohaustinn) 
[LinkedIn](https://linkedin.com/in/augustine-ozoemena)

---

*Built as part of a structured NLP for Finance learning series. 