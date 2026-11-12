"""
Pipeline Integrity Verification Script
Checks all output files, record counts, and data schemas.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

files_to_check = {
    "social_posts.csv": (100, ["source", "ticker", "text", "timestamp_utc"]),
    "news_sentiment.csv": (40, ["mentioned_ticker", "sentiment_score", "sentiment_label"]),
    "social_sentiment.csv": (100, ["mentioned_ticker", "sentiment_score", "sentiment_label"]),
    "sentiment_summary.csv": (10, ["ticker", "source", "average_sentiment"]),
    "financial_signals.csv": (150, ["ticker", "company_name", "sentiment_direction", "current_price_reference"]),
    "model_comparison.csv": (20, ["ticker", "model_name", "mae", "rmse"]),
    "final_predictions.csv": (10, ["ticker", "company", "model_used", "predicted_next_close"]),
}

print("=" * 60)
print("VERIFYING PROCESSED DATASETS")
print("=" * 60)

all_ok = True
for fname, (min_rows, req_cols) in files_to_check.items():
    fpath = PROCESSED_DIR / fname
    if not fpath.exists():
        print(f"❌ MISSING: {fname}")
        all_ok = False
        continue
    df = pd.read_csv(fpath)
    rows = len(df)
    missing_cols = [c for c in req_cols if c not in df.columns]
    if rows < min_rows or missing_cols:
        print(f"⚠️ ISSUE in {fname}: rows={rows} (expected >={min_rows}), missing cols={missing_cols}")
        all_ok = False
    else:
        print(f"✅ {fname}: {rows} rows, all {len(req_cols)} required columns present.")

print("\n=" * 60)
print("VERIFYING MODEL ARTIFACTS")
print("=" * 60)
joblib_files = list(MODELS_DIR.glob("*.joblib"))
print(f"Found {len(joblib_files)} model artifacts in {MODELS_DIR}")
if len(joblib_files) < 10:
    print(f"⚠️ Expected at least 10 artifacts, found {len(joblib_files)}")
    all_ok = False
else:
    print("✅ Model artifacts verified.")

print("\n=" * 60)
print("VERIFYING FINAL REPORT & DOCUMENTATION")
print("=" * 60)
report_path = REPORTS_DIR / "final_report.md"
readme_path = BASE_DIR / "README.md"
dashboard_path = BASE_DIR / "dashboard" / "app.py"

for name, p in [("Final Report", report_path), ("README", readme_path), ("Dashboard App", dashboard_path)]:
    if p.exists() and p.stat().st_size > 500:
        print(f"✅ {name}: exists ({p.stat().st_size} bytes)")
    else:
        print(f"❌ {name}: missing or too small")
        all_ok = False

if all_ok:
    print("\n🎉 ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")
else:
    print("\n❌ SOME VERIFICATIONS FAILED.")

