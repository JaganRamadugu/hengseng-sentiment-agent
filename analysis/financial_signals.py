"""
Financial Information & Trading Signal Extraction Engine
Extracts stock mentions, price references, sentiment direction,
trading opinions (Buy/Sell/Hold), support/resistance levels, price targets,
and trading positions using rule-based and regex extraction.
Strictly adheres to: do not invent unmentioned levels/positions.
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "financial_signals.csv"

TICKER_NAMES = {
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "0005.HK": "HSBC",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}

# Regex patterns for signal extraction
PRICE_PATTERNS = [
    re.compile(r"(?:即時價|現報|報|收報|開報|現價|收市價)[:：\s]*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"股價[^\d]{0,10}([0-9]+(?:\.[0-9]+)?)\s*(?:元|港元)", re.I),
    re.compile(r"(?:slipped|fell|rose|jumped|trades?|closed?)\s*(?:[a-z\s]{0,15})?(?:to|at)\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?:HK\$|\$)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per share)?", re.I),
    re.compile(r"([0-9]{2,4}(?:\.[0-9]+)?)\s*(?:元|港元)(?=水平|關|低見|高位|低位)", re.I),
]

TARGET_PRICE_PATTERNS = [
    re.compile(r"(?:目標價|target price|PT|TP)[:：\s]*(?:為|至|約|around)?\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?:上看|目標看至|目標定為)\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:元|港元)?", re.I),
]

SUPPORT_PATTERNS = [
    re.compile(r"(?:支持位|支撐位|支持線|支持關口|support(?: level)?|floor)[:：\s]*(?:約|around)?\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?:守穩|守住|下試|退守)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:元|港元|關|點)?(?:支持)?", re.I),
    re.compile(r"(?:失守|跌穿)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:元|港元|關)", re.I),
]

RESISTANCE_PATTERNS = [
    re.compile(r"(?:阻力位|阻力線|阻力關口|頂部|resistance(?: level)?|ceiling)[:：\s]*(?:約|around)?\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"(?:升穿|突破|挑戰|上試)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:元|港元|關|點)?(?:阻力)?", re.I),
    re.compile(r"阻力處於\s*([0-9]+(?:\.[0-9]+)?)", re.I),
]

TRADING_OPINION_PATTERNS = {
    "BUY": re.compile(r"\b(buy|outperform|overweight)\b|買入|增持|跑贏大市|加倉|超配|看好|大動作", re.I),
    "SELL": re.compile(r"\b(sell|underperform|underweight)\b|沽出|賣出|減持|沽空|看淡|拋售|重挫|瀉", re.I),
    "HOLD": re.compile(r"\b(hold|neutral)\b|持有|中性|觀望|橫行|整固", re.I),
}

TRADING_POSITION_PATTERNS = [
    re.compile(r"\b(Long Put|Long Call|Short Call|Short Put)\b", re.I),
    re.compile(r"(做多|做空|空倉|多倉|長倉|短倉|持倉|淨買入|淨賣出|沽空|平倉|鎖倉)"),
]


def extract_regex_first(text: str, patterns: list) -> str:
    """Returns first matched pattern or 'Not mentioned'."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            val = m.group(1) if m.groups() else m.group(0)
            if val and val.strip():
                return val.strip()
    return "Not mentioned"


def extract_trading_opinion(text: str, sentiment_score: float) -> str:
    """Extracts explicit trading opinion or falls back to 'Not mentioned'."""
    for opinion, pat in TRADING_OPINION_PATTERNS.items():
        if pat.search(text):
            return opinion
    return "Not mentioned"


def extract_trading_position(text: str) -> str:
    """Extracts explicit trading position keywords or 'Not mentioned'."""
    for pat in TRADING_POSITION_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1) if m.groups() else m.group(0)
            if val:
                return val.strip()
    return "Not mentioned"


def extract_signals():
    """
    Parses processed news and social sentiment datasets and extracts
    structured financial signals.
    """
    news_file = PROCESSED_DIR / "news_sentiment.csv"
    social_file = PROCESSED_DIR / "social_sentiment.csv"

    records = []

    # 1. Process News
    if news_file.exists():
        news_df = pd.read_csv(news_file)
        print(f"Extracting signals from {len(news_df)} news articles...")
        for _, row in news_df.iterrows():
            text = str(row.get("analysis_text", "")).strip()
            ticker = str(row.get("mentioned_ticker", "")).strip()
            company = TICKER_NAMES.get(ticker, row.get("company", "Unknown"))
            score = float(row.get("sentiment_score", 0.0))
            direction = "BULLISH" if score > 0.05 else "BEARISH" if score < -0.05 else "NEUTRAL"

            price_ref = extract_regex_first(text, PRICE_PATTERNS)
            target_price = extract_regex_first(text, TARGET_PRICE_PATTERNS)
            support_lvl = extract_regex_first(text, SUPPORT_PATTERNS)
            res_lvl = extract_regex_first(text, RESISTANCE_PATTERNS)
            opinion = extract_trading_opinion(text, score)
            position = extract_trading_position(text)

            records.append({
                "source": "News",
                "ticker": ticker,
                "company_name": company,
                "timestamp": row.get("published_date"),
                "sentiment_score": score,
                "sentiment_direction": direction,
                "current_price_reference": price_ref,
                "trading_opinion": opinion,
                "support_level": support_lvl,
                "resistance_level": res_lvl,
                "price_target": target_price,
                "trading_position": position,
                "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                "url": row.get("url", ""),
            })

    # 2. Process Social
    if social_file.exists():
        social_df = pd.read_csv(social_file)
        print(f"Extracting signals from {len(social_df)} social posts...")
        for _, row in social_df.iterrows():
            text = str(row.get("text", "")).strip()
            ticker = str(row.get("mentioned_ticker", "")).strip()
            company = TICKER_NAMES.get(ticker, "Unknown")
            score = float(row.get("sentiment_score", 0.0))
            direction = "BULLISH" if score > 0.05 else "BEARISH" if score < -0.05 else "NEUTRAL"

            price_ref = extract_regex_first(text, PRICE_PATTERNS)
            target_price = extract_regex_first(text, TARGET_PRICE_PATTERNS)
            support_lvl = extract_regex_first(text, SUPPORT_PATTERNS)
            res_lvl = extract_regex_first(text, RESISTANCE_PATTERNS)
            opinion = extract_trading_opinion(text, score)
            position = extract_trading_position(text)

            records.append({
                "source": row.get("sentiment_source", "Social"),
                "ticker": ticker,
                "company_name": company,
                "timestamp": row.get("timestamp_utc"),
                "sentiment_score": score,
                "sentiment_direction": direction,
                "current_price_reference": price_ref,
                "trading_opinion": opinion,
                "support_level": support_lvl,
                "resistance_level": res_lvl,
                "price_target": target_price,
                "trading_position": position,
                "text_snippet": text[:200] + "..." if len(text) > 200 else text,
                "url": row.get("url", ""),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved financial signals to {OUTPUT_FILE} ({len(df)} records).")

    # Signal coverage summary
    print("\nFINANCIAL SIGNAL EXTRACTION COVERAGE:")
    for col in ["current_price_reference", "trading_opinion", "support_level", "resistance_level", "price_target", "trading_position"]:
        found_count = (df[col] != "Not mentioned").sum()
        print(f"- {col}: {found_count}/{len(df)} ({found_count/len(df)*100:.1f}%) mentions identified")

    return df


if __name__ == "__main__":
    extract_signals()

