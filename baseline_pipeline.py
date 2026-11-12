
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# ==========================================
# CONFIGURATION
# ==========================================

TICKERS = {
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "0005.HK": "HSBC",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "return_1d",
    "return_2d",
    "return_5d",
    "volatility_5d",
    "volatility_10d",
    "volume_change",
    "ma_ratio_5",
    "ma_ratio_10",
    "rsi_14",
]


# ==========================================
# DATA INGESTION
# ==========================================

def download_prices(ticker):

    print(f"\nDownloading {ticker}...")

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"No data available for {ticker}")
        return None

    # Handle possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    df = df[required].copy()

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    df["return_1d"] = close.pct_change(1)
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)

    df["volatility_5d"] = (
        df["return_1d"].rolling(5).std()
    )

    df["volatility_10d"] = (
        df["return_1d"].rolling(10).std()
    )

    df["volume_change"] = volume.pct_change()

    df["ma_ratio_5"] = (
        close / close.rolling(5).mean() - 1
    )

    df["ma_ratio_10"] = (
        close / close.rolling(10).mean() - 1
    )

    # RSI indicator
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Target: next trading day's return
    df["target_return"] = (
        close.shift(-1) / close - 1
    )

    return df.replace([np.inf, -np.inf], np.nan)


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(df, ticker):

    data = df.dropna(
        subset=FEATURES + ["target_return"]
    ).copy()

    if len(data) < 150:
        print(f"Insufficient data for {ticker}")
        return None

    X = data[FEATURES]
    y = data["target_return"]

    # Final 20% held out from model selection
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    # Walk-forward validation on training portion
    cv = TimeSeriesSplit(n_splits=5)

    cv_scores = []

    for train_idx, val_idx in cv.split(X_train):

        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )

        predictions = model.predict(
            X_train.iloc[val_idx]
        )

        score = mean_absolute_error(
            y_train.iloc[val_idx],
            predictions
        )

        cv_scores.append(score)

    # Fit on complete training partition
    final_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train, y_train)

    test_predictions = final_model.predict(X_test)

    model_mae = mean_absolute_error(
        y_test, test_predictions
    )

    model_rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    baseline_predictions = np.zeros(len(y_test))

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    directional_accuracy = np.mean(
        np.sign(test_predictions) == np.sign(y_test)
    )

    # Next-session forecast using latest available features
    latest_features = df[FEATURES].dropna().iloc[[-1]]

    predicted_return = float(
        final_model.predict(latest_features)[0]
    )

    latest_close = float(df["Close"].dropna().iloc[-1])

    predicted_price = latest_close * (
        1 + predicted_return
    )

    results = {
        "ticker": ticker,
        "latest_close": latest_close,
        "predicted_return_pct": predicted_return * 100,
        "predicted_next_close": predicted_price,
        "direction": (
            "BULLISH" if predicted_return > 0
            else "BEARISH" if predicted_return < 0
            else "NEUTRAL"
        ),
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "baseline_mae": baseline_mae,
        "directional_accuracy": directional_accuracy,
        "cv_mean_mae": np.mean(cv_scores),
        "test_samples": len(y_test),
    }

    print(f"\n{'='*45}")
    print(f"{ticker} - {TICKERS[ticker]}")
    print(f"{'='*45}")
    print(f"Latest close: {latest_close:.3f}")
    print(f"Predicted close: {predicted_price:.3f}")
    print(f"Predicted return: {predicted_return*100:.3f}%")
    print(f"Direction: {results['direction']}")
    print(f"Model MAE: {model_mae:.6f}")
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Model RMSE: {model_rmse:.6f}")
    print(f"Directional accuracy: {directional_accuracy:.2%}")
    print(f"CV mean MAE: {np.mean(cv_scores):.6f}")

    return results


# ==========================================
# MAIN
# ==========================================

def main():

    all_results = []

    for ticker, name in TICKERS.items():

        try:
            prices = download_prices(ticker)

            if prices is None:
                continue

            prices = create_features(prices)

            prices.to_csv(
                RAW_DIR / f"{ticker.replace('.', '_')}.csv"
            )

            result = evaluate_model(prices, ticker)

            if result:
                result["asset_name"] = name
                all_results.append(result)

        except Exception as error:
            print(f"Error processing {ticker}: {error}")

    if not all_results:
        print("No successful predictions. Check data and dependencies.")
        return

    report = pd.DataFrame(all_results)

    report = report.sort_values(
        "predicted_return_pct",
        ascending=False
    )

    report.to_csv(
        PROCESSED_DIR / "baseline_predictions.csv",
        index=False
    )

    print("\nFINAL MARKET RANKING")
    print(report[
        [
            "ticker",
            "asset_name",
            "predicted_next_close",
            "predicted_return_pct",
            "direction",
            "directional_accuracy"
        ]
    ].to_string(index=False))

    print("\nSaved results to data/processed/baseline_predictions.csv")


if __name__ == "__main__":
    main()
