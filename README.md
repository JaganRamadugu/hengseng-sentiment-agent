# Heng Seng Sentiment-Based Stock Price Prediction Agent

An AI-powered quantitative sentiment and next-trading-day stock forecasting system for five Hong Kong Exchange (HKEX) blue-chip equities:
- **0700.HK** — Tencent Holdings Ltd.
- **9988.HK** — Alibaba Group Holding Ltd.
- **0005.HK** — HSBC Holdings plc
- **3690.HK** — Meituan
- **1810.HK** — Xiaomi Corporation

---

## 📌 Project Overview

This project was developed as an end-to-end quantitative financial engineering MVP. It ingests alternative unstructured data (financial news via Exa, retail social posts via Apify REST API from Telegram and LIHKG), applies a bilingual financial sentiment engine (VADER + custom English/Chinese financial lexicon), extracts financial trading signals (support/resistance levels, target prices, trading positions), and benchmarks XGBoost predictive models against a Zero-Return benchmark under strict chronological walk-forward validation.

---

## 🏛️ System Architecture

```
hengseng-sentiment-agent/
├── ingestion/
│   ├── exa_news.py              # Financial news ingestion via Exa API
│   ├── apify_scraper.py         # Apify REST API social scraper (Telegram & LIHKG)
│   ├── enrich_lihkg.py          # Apify LIHKG dataset integration
│   └── social_normalizer.py     # Ticker disambiguation, ranking exclusion, UTC parsing
├── sentiment/
│   └── sentiment_engine.py      # VADER + Custom Financial Lexicon & calibrated probabilities
├── analysis/
│   ├── financial_signals.py     # Rule-based entity & signal extractor (support/resistance/targets)
│   └── audit_social_data.py     # Ticker attribution validation & auditing script
├── models/
│   ├── prediction_pipeline.py   # Technical feature engineering & multi-model evaluation
│   └── *.joblib                 # Serialized XGBoost model artifacts
├── dashboard/
│   └── app.py                   # Interactive visual Streamlit & Plotly dashboard
├── reports/
│   └── final_report.md          # Comprehensive quantitative research & assessment report
├── data/
│   ├── raw/                     # Original preserved raw datasets
│   └── processed/               # Normalized datasets, sentiments, signals, & forecasts
├── requirements.txt             # Validated Python dependencies
├── .env.example                 # Safe environment template
└── README.md                    # System documentation
```

---

## 🛠️ Technologies & Stack

- **Language:** Python 3.13 (Windows 11 Enterprise Environment)
- **Data Ingestion:** `requests`, `exa-py` (REST APIs avoid blocked native DLLs)
- **Data Engineering:** `pandas`, `numpy`, `yfinance`
- **Natural Language Processing:** `vaderSentiment` with custom English & Traditional/Simplified Chinese financial lexicon
- **Machine Learning:** `scikit-learn`, `xgboost` (Time-series chronological splits, walk-forward validation)
- **Model Serialization:** `joblib`
- **Interactive Dashboard:** `streamlit`, `plotly`

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.11+ installed. The environment runs inside a local Python virtual environment (`.venv`).

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```powershell
Copy-Item .env.example .env
```
Fill in `.env`:
```ini
APIFY_API_TOKEN=your_apify_api_token
EXA_API_KEY=your_exa_api_key
```
*(If external APIs are unavailable or keys expire, the pipeline automatically falls back to the preserved local datasets in `data/raw/` without interrupting execution.)*

### 3. Install Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 🚀 Execution Guide (PowerShell)

Execute the end-to-end pipeline in order:

### Step 1: Normalize & Validate Social Data
Corrects ticker attribution, excludes automated ranking tables, deduplicates, and filters to the 10-day window:
```powershell
.\.venv\Scripts\python.exe ingestion\social_normalizer.py
```
*Outputs: `data/processed/social_posts.csv` (123 clean records)*

### Step 2: Run Sentiment Analysis Engine
Computes sentiment scores, intensities, calibrated bullish/bearish probabilities, and engagement-weighted scores:
```powershell
.\.venv\Scripts\python.exe sentiment\sentiment_engine.py
```
*Outputs: `data/processed/news_sentiment.csv`, `data/processed/social_sentiment.csv`, `data/processed/sentiment_summary.csv`*

### Step 3: Extract Financial Information & Signals
Extracts explicit price references, buy/sell/hold opinions, support/resistance levels, and trading positions:
```powershell
.\.venv\Scripts\python.exe analysis\financial_signals.py
```
*Outputs: `data/processed/financial_signals.csv` (170 analyzed records)*

### Step 4: Train Models & Generate Predictions
Trains Market-Only, Market+News, Market+Social, and Market+Combined XGBoost models and benchmarks against Zero-Return:
```powershell
.\.venv\Scripts\python.exe models\prediction_pipeline.py
```
*Outputs: `data/processed/model_comparison.csv`, `data/processed/final_predictions.csv`, `models/*.joblib`*

### Step 5: Launch the Interactive Visual Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Summary of Measured Results

### Sentiment Overview (10-Day Window: Nov 2 – Nov 12, 2026)
| Ticker | Company | Mentions | Avg Sentiment | Weighted Sentiment | Sentiment Stance |
|---|---|---|---|---|---|
| **0005.HK** | HSBC | 26 | +0.3988 | +0.2614 | Bullish |
| **0700.HK** | Tencent | 25 | +0.3124 | +0.2545 | Bullish |
| **1810.HK** | Xiaomi | 22 | +0.5688 | +0.4246 | Strong Bullish |
| **3690.HK** | Meituan | 26 | +0.0841 | +0.0924 | Mild Bullish / Neutral |
| **9988.HK** | Alibaba | 71 | +0.1315 | +0.1125 | Polarized |

### Next-Trading-Day Forecasts (Combined Model)
| Ticker | Company | Latest Close | Pred. Next Close | Pred. Return (%) | Forecast Direction |
|---|---|---|---|---|---|
| **0700.HK** | Tencent | HK$ 409.60 | HK$ 411.48 | +0.460% | BULLISH |
| **9988.HK** | Alibaba | HK$ 100.60 | HK$ 100.48 | -0.118% | BEARISH |
| **0005.HK** | HSBC | HK$ 141.60 | HK$ 143.07 | +1.039% | BULLISH |
| **3690.HK** | Meituan | HK$ 72.85 | HK$ 73.29 | +0.597% | BULLISH |
| **1810.HK** | Xiaomi | HK$ 30.10 | HK$ 30.11 | +0.036% | BULLISH |

---

## 🔬 Empirical Model Evaluation & Honest Benchmarking

| Ticker | Model Name | Test MAE | Test RMSE | R² Score | Directional Accuracy | Beats Zero-Return? |
|---|---|---|---|---|---|---|
| **0700.HK** | Zero-Return Benchmark | **0.017004** | **0.022359** | 0.0000 | 0.00% | — |
| 0700.HK | Market + Combined | 0.018448 | 0.024250 | -0.1762 | 43.16% | False |
| **9988.HK** | Zero-Return Benchmark | **0.020751** | **0.028771** | 0.0000 | 0.00% | — |
| 9988.HK | Market + Combined | 0.021253 | 0.029202 | -0.0302 | 51.58% | False |
| **0005.HK** | Zero-Return Benchmark | **0.011601** | **0.016084** | 0.0000 | 0.00% | — |
| 0005.HK | Market + Combined | 0.013035 | 0.018195 | -0.2796 | 48.42% | False |
| **3690.HK** | Zero-Return Benchmark | **0.019981** | **0.026362** | 0.0000 | 0.00% | — |
| 3690.HK | Market + Combined | 0.021305 | 0.027962 | -0.1251 | 47.37% | False |
| **1810.HK** | Zero-Return Benchmark | **0.025110** | **0.033068** | 0.0000 | 0.00% | — |
| 1810.HK | Market + Combined | 0.025783 | 0.033878 | -0.0496 | 47.37% | False |

**Key Findings:**
1. **Zero-Return Superiority:** In all five equities, the Zero-Return Benchmark achieved lower MAE and RMSE than the XGBoost regression models. This aligns directly with the Efficient Market Hypothesis (EMH).
2. **Marginal Sentiment Value:** Adding sentiment slightly reduced MAE for Alibaba (0.0217 → 0.0214), Meituan (0.0213 → 0.0211), and Xiaomi (0.0260 → 0.0255), and increased directional accuracy for Xiaomi from 40.0% to 47.4%.
3. **No Fabricated Performance:** We do not claim sentiment beats the market; rather, we provide honest empirical evaluation and discuss the sample size limitation.

---

## ⚠️ Known Limitations

1. **10-Day Alternative Data Window:** The available social and news dataset spans 10 days. While highly informative for current qualitative intelligence, it cannot support statistically robust longitudinal ML training.
2. **Lexicon-Based Chinese Sentiment:** VADER is an English-oriented rule-based model. Although extended with financial Chinese tokens, it lacks the deep contextual understanding of native Chinese LLMs.
3. **Environment Security Constraints:** Windows Application Control restricts native DLLs (blocking PyTorch, Transformers, and Apify SDK). The system uses requests and VADER to guarantee 100% stable execution.

---

## 📄 License & Attribution

Internship Quantitative Assessment MVP. Built for academic and quantitative evaluation.
