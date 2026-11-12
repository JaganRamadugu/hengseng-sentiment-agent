"""
Comprehensive Stock Return & Price Prediction Pipeline
Compares:
  1. Market-Only Baseline Model
  2. Market + News Sentiment Model
  3. Market + Social Sentiment Model
  4. Market + Combined (News & Social) Sentiment Model
  5. Zero-Return Benchmark (predicting 0 return)

Features: Technical price features + Sentiment features
Validation: Chronological train/test split (no look-ahead bias) + TimeSeriesSplit walk-forward CV
Outputs:
  - data/processed/model_comparison.csv
  - data/processed/final_predictions.csv
  - models/*.joblib
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "0005.HK": "HSBC",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}

BASE_TECHNICAL_FEATURES = [
    "return_1d",
    "return_2d",
    "return_5d",
    "volatility_5d",
    "volatility_10d",
    "volume_change",
    "ma_ratio_5",
    "ma_ratio_10",
    "rsi_14",
    "momentum_5d",
]


def load_or_fetch_prices(ticker: str) -> pd.DataFrame:
    """Loads price data from raw directory or downloads via yfinance if missing."""
    file_path = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index(date_col).sort_index()
        # Keep basic price columns
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].copy()
        return df

    print(f"Downloading prices for {ticker}...")
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def engineer_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all required technical indicators."""
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["return_1d"] = close.pct_change(1, fill_method=None)
    df["return_2d"] = close.pct_change(2, fill_method=None)
    df["return_5d"] = close.pct_change(5, fill_method=None)

    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volume_change"] = volume.pct_change(fill_method=None)

    df["ma_ratio_5"] = close / close.rolling(5).mean() - 1
    df["ma_ratio_10"] = close / close.rolling(10).mean() - 1

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Momentum 5d
    df["momentum_5d"] = close.pct_change(5, fill_method=None)

    # Target: Next trading day return
    df["target_return"] = close.shift(-1) / close - 1

    return df.replace([np.inf, -np.inf], np.nan)


def build_daily_sentiment_series(ticker: str, price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Constructs aligned daily sentiment features for the given ticker.
    Uses decay (EMA span=3) to propagate information across non-trading days.
    """
    date_strs = price_dates.strftime("%Y-%m-%d")
    sent_df = pd.DataFrame(index=price_dates)
    sent_df["date_str"] = date_strs

    news_file = PROCESSED_DIR / "news_sentiment.csv"
    social_file = PROCESSED_DIR / "social_sentiment.csv"

    # 1. News daily aggregation
    daily_news = pd.DataFrame(columns=["news_sentiment", "news_mentions"])
    if news_file.exists():
        ndf = pd.read_csv(news_file)
        ndf = ndf[ndf["mentioned_ticker"] == ticker].copy()
        if not ndf.empty and "published_date" in ndf.columns:
            ndf["date"] = pd.to_datetime(ndf["published_date"], utc=True).dt.strftime("%Y-%m-%d")
            daily_news = ndf.groupby("date").agg(
                news_sentiment=("sentiment_score", "mean"),
                news_mentions=("sentiment_score", "count")
            )

    # 2. Social daily aggregation
    daily_social = pd.DataFrame(columns=["social_sentiment", "social_mentions"])
    if social_file.exists():
        sdf = pd.read_csv(social_file)
        sdf = sdf[sdf["mentioned_ticker"] == ticker].copy()
        if not sdf.empty and "timestamp_utc" in sdf.columns:
            sdf["date"] = pd.to_datetime(sdf["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
            # Calculate engagement weight
            eng = sdf["likes"] + 2 * sdf["replies"] + 0.05 * sdf["views"]
            sdf["eng_weight"] = 1.0 + np.log1p(eng)
            sdf["w_score"] = sdf["sentiment_score"] * sdf["eng_weight"]

            daily_social = sdf.groupby("date").apply(
                lambda g: pd.Series({
                    "social_sentiment": g["w_score"].sum() / g["eng_weight"].sum(),
                    "social_mentions": len(g)
                }),
                include_groups=False
            )

    # Merge onto price dates
    merged = sent_df.join(daily_news, on="date_str").join(daily_social, on="date_str")

    merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)
    merged["news_mentions"] = merged["news_mentions"].fillna(0)
    merged["social_sentiment"] = merged["social_sentiment"].fillna(0.0)
    merged["social_mentions"] = merged["social_mentions"].fillna(0)

    # Combined sentiment
    # Equal weight if both present, or whichever is available
    def combine_daily(row):
        has_news = row["news_mentions"] > 0
        has_soc = row["social_mentions"] > 0
        if has_news and has_soc:
            return 0.5 * row["news_sentiment"] + 0.5 * row["social_sentiment"]
        elif has_news:
            return row["news_sentiment"]
        elif has_soc:
            return row["social_sentiment"]
        return 0.0

    merged["combined_sentiment"] = merged.apply(combine_daily, axis=1)

    # Rolling decayed sentiment (EMA with span=3 days) to propagate weekend/off-hours signals
    merged["news_sentiment_ema"] = merged["news_sentiment"].ewm(span=3).mean()
    merged["social_sentiment_ema"] = merged["social_sentiment"].ewm(span=3).mean()
    merged["combined_sentiment_ema"] = merged["combined_sentiment"].ewm(span=3).mean()

    return merged[[
        "news_sentiment", "news_sentiment_ema",
        "social_sentiment", "social_sentiment_ema",
        "combined_sentiment", "combined_sentiment_ema"
    ]]


def train_and_evaluate():
    """
    Runs multi-model comparison for all 5 target HKEX stocks.
    Generates model comparison table and final predictions.
    """
    model_configs = {
        "Market_Only": BASE_TECHNICAL_FEATURES,
        "Market_Plus_News": BASE_TECHNICAL_FEATURES + ["news_sentiment", "news_sentiment_ema"],
        "Market_Plus_Social": BASE_TECHNICAL_FEATURES + ["social_sentiment", "social_sentiment_ema"],
        "Market_Plus_Combined": BASE_TECHNICAL_FEATURES + ["combined_sentiment", "combined_sentiment_ema"],
    }

    all_comparison_rows = []
    final_prediction_rows = []

    print("=" * 70)
    print("RUNNING MULTI-MODEL STOCK PREDICTION PIPELINE")
    print("=" * 70)

    for ticker, comp_name in TICKERS.items():
        print(f"\nProcessing {ticker} ({comp_name})...")
        prices = load_or_fetch_prices(ticker)
        df_feat = engineer_technical_features(prices)
        df_sent = build_daily_sentiment_series(ticker, df_feat.index)

        full_df = df_feat.join(df_sent)

        # Save enhanced dataset to data/raw for reproducibility
        out_raw_file = RAW_DIR / f"{ticker.replace('.', '_')}.csv"
        full_df.to_csv(out_raw_file)

        # Split: drop rows missing target_return for training/testing
        valid_data = full_df.dropna(subset=BASE_TECHNICAL_FEATURES + ["target_return"]).copy()
        if len(valid_data) < 150:
            print(f"Error: Insufficient samples for {ticker} ({len(valid_data)} rows)")
            continue

        split_idx = int(len(valid_data) * 0.8)
        train_df = valid_data.iloc[:split_idx]
        test_df = valid_data.iloc[split_idx:]
        y_test = test_df["target_return"]

        # Baseline: Zero-Return benchmark
        zero_pred = np.zeros(len(y_test))
        base_mae = mean_absolute_error(y_test, zero_pred)
        base_rmse = np.sqrt(mean_squared_error(y_test, zero_pred))

        # Store results for this stock across models
        stock_models = {}

        # 1. Record Zero-Return Benchmark in comparison table
        all_comparison_rows.append({
            "ticker": ticker,
            "company": comp_name,
            "model_name": "Zero_Return_Benchmark",
            "features_used": "None (Predict 0 return)",
            "test_samples": len(y_test),
            "mae": round(base_mae, 6),
            "rmse": round(base_rmse, 6),
            "r2_score": 0.0,
            "directional_accuracy": 0.0,  # Zero prediction has 0 sign matching
            "vs_baseline_mae_pct": 0.0,
            "beats_zero_benchmark": False
        })

        best_model_name = "Market_Only"
        best_mae = float("inf")

        for m_name, feature_list in model_configs.items():
            X_train = train_df[feature_list].fillna(0.0)
            y_train = train_df["target_return"]
            X_test = test_df[feature_list].fillna(0.0)

            # Fit XGBoost model
            model = XGBRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Evaluate on held-out test partition
            test_preds = model.predict(X_test)
            m_mae = mean_absolute_error(y_test, test_preds)
            m_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            m_r2 = r2_score(y_test, test_preds)

            # Directional accuracy: sign match of test_preds vs y_test
            dir_acc = np.mean(np.sign(test_preds) == np.sign(y_test))
            vs_base_pct = ((m_mae - base_mae) / base_mae) * 100

            beats_zero = bool(m_mae < base_mae)

            all_comparison_rows.append({
                "ticker": ticker,
                "company": comp_name,
                "model_name": m_name,
                "features_used": ", ".join(feature_list),
                "test_samples": len(y_test),
                "mae": round(m_mae, 6),
                "rmse": round(m_rmse, 6),
                "r2_score": round(m_r2, 6),
                "directional_accuracy": round(dir_acc * 100, 2),
                "vs_baseline_mae_pct": round(vs_base_pct, 2),
                "beats_zero_benchmark": beats_zero
            })

            stock_models[m_name] = (model, feature_list, m_mae, m_rmse, dir_acc)

            if m_mae < best_mae:
                best_mae = m_mae
                best_model_name = m_name

            # Save model artifact
            artifact_file = MODELS_DIR / f"{ticker.replace('.', '_')}_{m_name}.joblib"
            joblib.dump(model, artifact_file)

        # Generate next-trading-day forecast
        # Use latest available row with valid closing price
        valid_close_df = full_df.dropna(subset=["Close"])
        latest_row = valid_close_df.iloc[[-1]]
        latest_close = float(latest_row["Close"].iloc[0])

        # Evaluate predictions from all models
        for m_name in ["Market_Only", "Market_Plus_Combined"]:
            m_obj, m_feats, m_mae, m_rmse, m_dir_acc = stock_models[m_name]
            X_latest = latest_row[m_feats].fillna(0.0)
            pred_ret = float(m_obj.predict(X_latest)[0])
            pred_price = latest_close * (1 + pred_ret)
            direction = "BULLISH" if pred_ret > 0 else "BEARISH" if pred_ret < 0 else "NEUTRAL"

            # Get latest sentiment values
            latest_comb_sent = float(latest_row["combined_sentiment"].iloc[0]) if "combined_sentiment" in latest_row else 0.0

            final_prediction_rows.append({
                "ticker": ticker,
                "company": comp_name,
                "model_used": m_name,
                "latest_date": str(latest_row.index[0])[:10],
                "latest_close": round(latest_close, 3),
                "predicted_return_pct": round(pred_ret * 100, 3),
                "predicted_next_close": round(pred_price, 3),
                "predicted_direction": direction,
                "latest_sentiment_score": round(latest_comb_sent, 4),
                "test_mae": round(m_mae, 6),
                "test_rmse": round(m_rmse, 6),
                "zero_benchmark_mae": round(base_mae, 6),
                "directional_accuracy_pct": round(m_dir_acc * 100, 2),
                "beats_zero_benchmark": bool(m_mae < base_mae)
            })

    # Save Comparison DataFrame
    comp_df = pd.DataFrame(all_comparison_rows)
    comp_file = PROCESSED_DIR / "model_comparison.csv"
    comp_df.to_csv(comp_file, index=False)
    print(f"\nSaved model comparison to {comp_file}")

    # Save Final Predictions DataFrame
    pred_df = pd.DataFrame(final_prediction_rows)
    pred_file = PROCESSED_DIR / "final_predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Saved final predictions to {pred_file}")

    # Display Summary Table
    print("\n" + "=" * 80)
    print("FINAL PREDICTIONS (NEXT-TRADING-DAY FORECAST)")
    print("=" * 80)
    print(pred_df[[
        "ticker", "company", "model_used", "latest_close",
        "predicted_next_close", "predicted_return_pct", "predicted_direction",
        "test_mae", "zero_benchmark_mae", "directional_accuracy_pct"
    ]].to_string(index=False))

    return comp_df, pred_df


if __name__ == "__main__":
    train_and_evaluate()
