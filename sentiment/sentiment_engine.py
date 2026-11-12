"""
Unified Sentiment Analysis Engine for Financial News and Social Media Posts
Combines VADER SentimentIntensityAnalyzer with an extended English & Chinese
financial domain lexicon and defensible probability modeling.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NEWS_INPUT_FILE = RAW_DIR / "news_data.csv"
SOCIAL_INPUT_FILE = PROCESSED_DIR / "social_posts.csv"

NEWS_OUTPUT_FILE = PROCESSED_DIR / "news_sentiment.csv"
SOCIAL_OUTPUT_FILE = PROCESSED_DIR / "social_sentiment.csv"
SUMMARY_OUTPUT_FILE = PROCESSED_DIR / "sentiment_summary.csv"

# ==========================================
# FINANCIAL LEXICON
# ==========================================

# 1. English financial sentiment words
ENGLISH_FINANCIAL_LEXICON = {
    "bullish": 2.5,
    "outperform": 2.5,
    "outperformed": 2.5,
    "upgrade": 2.0,
    "upgraded": 2.5,
    "profitability": 1.8,
    "profitable": 2.0,
    "growth": 1.5,
    "surge": 2.5,
    "surged": 2.5,
    "rally": 2.0,
    "record-high": 2.5,
    "beat": 2.0,
    "beats": 2.0,
    "stronger": 1.8,
    "dividend": 1.0,
    "buyback": 1.4,
    "buybacks": 1.4,
    "recovery": 1.8,
    "rebound": 1.8,
    "rebounded": 1.8,
    "jump": 1.8,
    "jumped": 1.8,
    "overweight": 1.8,
    "gain": 1.5,
    "gains": 1.5,
    "optimistic": 1.8,
    "expansion": 1.5,
    "revenue-growth": 2.0,

    "bearish": -2.5,
    "underperform": -2.5,
    "downgrade": -2.5,
    "downgraded": -2.5,
    "losses": -2.0,
    "loss": -2.0,
    "crash": -3.0,
    "crashed": -3.0,
    "plunge": -2.5,
    "plunged": -2.5,
    "selloff": -2.5,
    "sell-off": -2.5,
    "miss": -1.8,
    "missed": -1.8,
    "weakness": -1.8,
    "weak": -1.5,
    "debt": -1.0,
    "default": -2.5,
    "bankruptcy": -3.0,
    "recession": -2.0,
    "layoffs": -2.0,
    "fraud": -2.8,
    "investigation": -1.5,
    "slump": -2.2,
    "slumped": -2.2,
    "tumble": -2.2,
    "tumbled": -2.2,
    "decline": -1.5,
    "declined": -1.5,
    "underweight": -1.8,
    "pessimistic": -1.8,
    "warning": -1.5,
}

# 2. Chinese financial sentiment words (Traditional & Simplified)
CHINESE_FINANCIAL_LEXICON = {
    # Positive
    "大漲": 2.8,
    "大涨": 2.8,
    "急升": 2.5,
    "暴漲": 3.0,
    "暴涨": 3.0,
    "飆升": 2.8,
    "飙升": 2.8,
    "升穿": 2.0,
    "反彈": 2.0,
    "反弹": 2.0,
    "增長": 1.8,
    "增长": 1.8,
    "盈利": 2.0,
    "淨流入": 2.0,
    "净流入": 2.0,
    "淨買入": 2.0,
    "净买入": 2.0,
    "創新高": 2.8,
    "创新高": 2.8,
    "買入": 2.0,
    "买入": 2.0,
    "增持": 2.0,
    "跑贏": 2.2,
    "跑赢": 2.2,
    "向好": 1.8,
    "看好": 2.2,
    "突破": 2.0,
    "超預期": 2.5,
    "超预期": 2.5,
    "強勁": 2.2,
    "强劲": 2.2,
    "回暖": 1.8,
    "利好": 2.2,
    "走高": 1.8,
    "大動作": 1.5,
    "大动作": 1.5,
    "合作": 1.2,

    # Negative
    "暴跌": -3.0,
    "暴瀉": -3.0,
    "暴泻": -3.0,
    "急挫": -2.8,
    "重挫": -2.8,
    "下瀉": -2.5,
    "下泻": -2.5,
    "瀉": -2.5,
    "泻": -2.5,
    "大跌": -2.5,
    "下挫": -2.2,
    "跌破": -2.2,
    "失守": -2.2,
    "沽空": -2.2,
    "沽出": -2.0,
    "減持": -2.0,
    "减持": -2.0,
    "拋售": -2.5,
    "抛售": -2.5,
    "捱沽": -2.2,
    "死咗": -2.5,
    "虧損": -2.5,
    "亏损": -2.5,
    "看淡": -2.2,
    "走弱": -2.0,
    "承壓": -1.8,
    "承压": -1.8,
    "利空": -2.2,
    "疲軟": -2.0,
    "疲软": -2.0,
    "退步": -1.8,
    "涉嫌": -2.0,
    "違規": -2.2,
    "违规": -2.2,
    "出軌": -2.0,
    "出轨": -2.0,
    "醜聞": -2.5,
    "丑闻": -2.5,
}

# Initialize VADER Analyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(ENGLISH_FINANCIAL_LEXICON)
analyzer.lexicon.update(CHINESE_FINANCIAL_LEXICON)

# Precompile Chinese lexicon terms for spacing preprocessing
SORTED_CHINESE_TERMS = sorted(CHINESE_FINANCIAL_LEXICON.keys(), key=len, reverse=True)


def preprocess_text_for_vader(text: str) -> str:
    """
    Pad recognized Chinese financial sentiment tokens with whitespace
    so that VADER's whitespace tokenizer can isolate and score them.
    """
    if not isinstance(text, str):
        return ""
    for term in SORTED_CHINESE_TERMS:
        if term in text:
            text = text.replace(term, f" {term} ")
    return " ".join(text.split())


def compute_sentiment(text: str, ticker: str, source: str) -> dict:
    """
    Computes sentiment metrics for a single text record.
    Returns dictionary with all required fields.
    """
    clean_txt = str(text).strip()
    if not clean_txt:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "sentiment_intensity": 0.0,
            "bullish_probability": 0.5,
            "bearish_probability": 0.5,
            "mentioned_ticker": ticker,
            "sentiment_source": source,
            "raw_text": ""
        }

    prep_txt = preprocess_text_for_vader(clean_txt)
    scores = analyzer.polarity_scores(prep_txt)
    compound = float(scores["compound"])

    # Classification using compound thresholds
    if compound > 0.05:
        label = "Positive"
    elif compound < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    # Defensible probabilistic calibration via logistic transformation
    # P(bullish) = 1 / (1 + exp(-3.5 * compound))
    # Maps compound 0 -> 0.50, +1 -> 0.97, -1 -> 0.03
    scale_factor = 3.5
    bullish_prob = float(1.0 / (1.0 + np.exp(-scale_factor * compound)))
    bearish_prob = float(1.0 - bullish_prob)

    intensity = float(abs(compound))

    return {
        "sentiment_score": round(compound, 4),
        "sentiment_label": label,
        "sentiment_intensity": round(intensity, 4),
        "bullish_probability": round(bullish_prob, 4),
        "bearish_probability": round(bearish_prob, 4),
        "mentioned_ticker": ticker,
        "sentiment_source": source,
    }


def analyze_news():
    """Processes news articles and produces news_sentiment.csv."""
    if not NEWS_INPUT_FILE.exists():
        print(f"Error: {NEWS_INPUT_FILE} not found.")
        return pd.DataFrame()

    df = pd.read_csv(NEWS_INPUT_FILE)
    print(f"\nProcessing {len(df)} news articles for sentiment...")

    records = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        body = str(row.get("text", "")).strip()
        analysis_text = f"{title}. {body}".strip()
        ticker = row.get("ticker", "")

        res = compute_sentiment(analysis_text, ticker, "News")
        res["title"] = title
        res["url"] = row.get("url", "")
        res["published_date"] = row.get("published_date")
        res["company"] = row.get("company", "")
        res["analysis_text"] = analysis_text
        records.append(res)

    news_sentiment_df = pd.DataFrame(records)

    # Reorder columns
    cols = [
        "mentioned_ticker", "company", "sentiment_source", "sentiment_score",
        "sentiment_label", "sentiment_intensity", "bullish_probability",
        "bearish_probability", "published_date", "title", "url", "analysis_text"
    ]
    news_sentiment_df = news_sentiment_df[cols]
    news_sentiment_df.to_csv(NEWS_OUTPUT_FILE, index=False)
    print(f"Saved news sentiment to {NEWS_OUTPUT_FILE} ({len(news_sentiment_df)} rows).")

    return news_sentiment_df


def analyze_social():
    """Processes normalized social posts and produces social_sentiment.csv."""
    if not SOCIAL_INPUT_FILE.exists():
        print(f"Error: {SOCIAL_INPUT_FILE} not found. Run social_normalizer first.")
        return pd.DataFrame()

    df = pd.read_csv(SOCIAL_INPUT_FILE)
    print(f"\nProcessing {len(df)} social posts for sentiment...")

    records = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        ticker = row.get("ticker", "")
        source = row.get("source", "Social")

        res = compute_sentiment(text, ticker, source)
        res["timestamp_utc"] = row.get("timestamp_utc")
        res["author"] = row.get("author")
        res["likes"] = row.get("likes", 0)
        res["replies"] = row.get("replies", 0)
        res["views"] = row.get("views", 0)
        res["reposts"] = row.get("reposts", 0)
        res["url"] = row.get("url")
        res["text"] = text
        records.append(res)

    social_sentiment_df = pd.DataFrame(records)

    cols = [
        "mentioned_ticker", "sentiment_source", "sentiment_score",
        "sentiment_label", "sentiment_intensity", "bullish_probability",
        "bearish_probability", "timestamp_utc", "author", "likes",
        "replies", "views", "reposts", "url", "text"
    ]
    social_sentiment_df = social_sentiment_df[cols]
    social_sentiment_df.to_csv(SOCIAL_OUTPUT_FILE, index=False)
    print(f"Saved social sentiment to {SOCIAL_OUTPUT_FILE} ({len(social_sentiment_df)} rows).")

    return social_sentiment_df


def generate_sentiment_summaries(news_df: pd.DataFrame, social_df: pd.DataFrame):
    """
    Generates multi-dimensional sentiment summaries:
    - By Stock
    - By Source
    - By Date
    - Distribution of Positive / Negative / Neutral
    - Average sentiment and engagement-weighted sentiment
    """
    combined_rows = []

    if not news_df.empty:
        for _, r in news_df.iterrows():
            dt_str = str(r["published_date"])[:10] if pd.notna(r["published_date"]) else "Unknown"
            combined_rows.append({
                "ticker": r["mentioned_ticker"],
                "source": "News",
                "date": dt_str,
                "sentiment_score": r["sentiment_score"],
                "sentiment_label": r["sentiment_label"],
                "sentiment_intensity": r["sentiment_intensity"],
                "likes": 0,
                "replies": 0,
                "views": 0,
                "reposts": 0,
            })

    if not social_df.empty:
        for _, r in social_df.iterrows():
            dt_str = str(r["timestamp_utc"])[:10] if pd.notna(r["timestamp_utc"]) else "Unknown"
            combined_rows.append({
                "ticker": r["mentioned_ticker"],
                "source": r["sentiment_source"],
                "date": dt_str,
                "sentiment_score": r["sentiment_score"],
                "sentiment_label": r["sentiment_label"],
                "sentiment_intensity": r["sentiment_intensity"],
                "likes": int(r.get("likes", 0)) if pd.notna(r.get("likes")) else 0,
                "replies": int(r.get("replies", 0)) if pd.notna(r.get("replies")) else 0,
                "views": int(r.get("views", 0)) if pd.notna(r.get("views")) else 0,
                "reposts": int(r.get("reposts", 0)) if pd.notna(r.get("reposts")) else 0,
            })

    comb_df = pd.DataFrame(combined_rows)

    # Compute engagement weight for social posts:
    # E_i = 1.0 + ln(1 + likes + 2*replies + 0.05*views + 2*reposts)
    def calc_weight(row):
        tot_eng = row["likes"] + 2 * row["replies"] + 0.05 * row["views"] + 2 * row["reposts"]
        return 1.0 + np.log1p(tot_eng)

    comb_df["eng_weight"] = comb_df.apply(calc_weight, axis=1)
    comb_df["weighted_score"] = comb_df["sentiment_score"] * comb_df["eng_weight"]

    # Summary by Ticker and Source
    grouped = comb_df.groupby(["ticker", "source"], as_index=False).apply(
        lambda g: pd.Series({
            "total_mentions": len(g),
            "average_sentiment": round(g["sentiment_score"].mean(), 4),
            "median_sentiment": round(g["sentiment_score"].median(), 4),
            "engagement_weighted_sentiment": round(g["weighted_score"].sum() / g["eng_weight"].sum(), 4),
            "average_intensity": round(g["sentiment_intensity"].mean(), 4),
            "positive_count": (g["sentiment_label"] == "Positive").sum(),
            "negative_count": (g["sentiment_label"] == "Negative").sum(),
            "neutral_count": (g["sentiment_label"] == "Neutral").sum(),
            "positive_ratio": round((g["sentiment_label"] == "Positive").mean(), 4),
            "negative_ratio": round((g["sentiment_label"] == "Negative").mean(), 4),
            "neutral_ratio": round((g["sentiment_label"] == "Neutral").mean(), 4),
            "total_likes": g["likes"].sum(),
            "total_replies": g["replies"].sum(),
            "total_views": g["views"].sum(),
        }),
        include_groups=False
    ).reset_index(drop=True)

    # Also compute overall per-ticker summary (combined across sources)
    overall_ticker = comb_df.groupby(["ticker"], as_index=False).apply(
        lambda g: pd.Series({
            "source": "ALL_COMBINED",
            "total_mentions": len(g),
            "average_sentiment": round(g["sentiment_score"].mean(), 4),
            "median_sentiment": round(g["sentiment_score"].median(), 4),
            "engagement_weighted_sentiment": round(g["weighted_score"].sum() / g["eng_weight"].sum(), 4),
            "average_intensity": round(g["sentiment_intensity"].mean(), 4),
            "positive_count": (g["sentiment_label"] == "Positive").sum(),
            "negative_count": (g["sentiment_label"] == "Negative").sum(),
            "neutral_count": (g["sentiment_label"] == "Neutral").sum(),
            "positive_ratio": round((g["sentiment_label"] == "Positive").mean(), 4),
            "negative_ratio": round((g["sentiment_label"] == "Negative").mean(), 4),
            "neutral_ratio": round((g["sentiment_label"] == "Neutral").mean(), 4),
            "total_likes": g["likes"].sum(),
            "total_replies": g["replies"].sum(),
            "total_views": g["views"].sum(),
        }),
        include_groups=False
    ).reset_index(drop=True)

    summary_df = pd.concat([grouped, overall_ticker], ignore_index=True)
    summary_df = summary_df.sort_values(["ticker", "source"])
    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False)
    print(f"\nSaved sentiment summary to {SUMMARY_OUTPUT_FILE}")

    print("\nOVERALL SENTIMENT PER TICKER:")
    print(overall_ticker[["ticker", "total_mentions", "average_sentiment", "engagement_weighted_sentiment", "positive_count", "negative_count", "neutral_count"]].to_string(index=False))

    return summary_df


def main():
    print("=" * 60)
    print("STARTING UNIFIED SENTIMENT ANALYSIS ENGINE")
    print("=" * 60)

    news_df = analyze_news()
    social_df = analyze_social()
    generate_sentiment_summaries(news_df, social_df)

    print("\nSentiment analysis completed successfully.")


if __name__ == "__main__":
    main()
