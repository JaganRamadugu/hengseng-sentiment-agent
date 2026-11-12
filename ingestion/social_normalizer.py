"""
Social Data Normalizer and Ticker Identification Engine
Provides robust ticker attribution, generic market ranking filtering,
timestamp parsing, and deduplication for Telegram and LIHKG datasets.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Reference date for the 10-day target window (Nov 2 - Nov 12, 2026)
CUTOFF_NOW = datetime(2026, 11, 12, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_START = CUTOFF_NOW - timedelta(days=10)

TICKER_CONFIG = {
    "0700.HK": {
        "name": "Tencent",
        "title_patterns": [r"0700", r"00700", r"騰訊", r"腾讯", r"\bTencent\b"],
        "body_patterns": [
            r"0700\.HK", r"00700\.HK", r"(?<!\d)00700(?!\d)", r"(?<!\d)0700(?!\d)",
            r"騰訊控股", r"騰訊", r"腾讯控股", r"腾讯", r"\bTencent\b"
        ]
    },
    "9988.HK": {
        "name": "Alibaba",
        "title_patterns": [r"9988", r"09988", r"阿里巴巴", r"阿里(?=[－\-\s\(\（]|股價|挫|跌|升|反彈|業績|巴巴)", r"\bAlibaba\b"],
        "body_patterns": [
            r"9988\.HK", r"09988\.HK", r"(?<!\d)09988(?!\d)", r"(?<!\d)9988(?!\d)",
            r"阿里巴巴", r"阿里(?=[－\-\s\(\（]|股價|挫|跌|升|反彈|業績|巴巴)",
            r"\bAlibaba\b"
        ]
    },
    "0005.HK": {
        "name": "HSBC",
        "title_patterns": [r"0005", r"00005", r"滙豐", r"汇丰", r"\bHSBC\b"],
        "body_patterns": [
            r"0005\.HK", r"00005\.HK", r"(?<!\d)00005(?!\d)", r"(?<!\d)0005(?!\d)",
            r"滙豐控股", r"滙豐", r"汇丰控股", r"汇丰", r"\bHSBC\b"
        ]
    },
    "3690.HK": {
        "name": "Meituan",
        "title_patterns": [r"3690", r"03690", r"美團", r"美团", r"\bMeituan\b"],
        "body_patterns": [
            r"3690\.HK", r"03690\.HK", r"(?<!\d)03690(?!\d)", r"(?<!\d)3690(?!\d)",
            r"美團", r"美团", r"美團點評", r"美团点评", r"\bMeituan\b"
        ]
    },
    "1810.HK": {
        "name": "Xiaomi",
        "title_patterns": [r"1810", r"01810", r"小米", r"\bXiaomi\b"],
        "body_patterns": [
            r"1810\.HK", r"01810\.HK", r"(?<!\d)01810(?!\d)", r"(?<!\d)1810(?!\d)",
            r"小米集團", r"小米集团", r"小米", r"\bXiaomi\b"
        ]
    }
}

GENERIC_RANKING_PATTERNS = [
    r"【#即時排行】",
    r"二十大成交額",
    r"二十大成交量",
    r"二十大升幅",
    r"二十大跌幅",
    r"十大成交額",
    r"5大沽空金額股票",
    r"港股半日沽空金額",
    r"南向資金今日淨買入",
    r"活躍重磅股表現",
    r"《半日速報》",
    r"《全日速報》",
    r"恒指走勢丨",
    r"《港股》恒指半日跌",
    r"《港股》恒指低開",
    r"【港股市況】恒指",
    r"國際早班車",
    r"耀才陳偉明：\s*港股",
]


def is_generic_market_ranking(text: str) -> bool:
    """Checks if the post is an automated ranking table or macro roundup."""
    for pattern in GENERIC_RANKING_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def identify_ticker(text: str, title: str = "") -> str | None:
    """
    Identifies the primary target stock discussed in the post.
    Returns ticker (e.g., '0700.HK') or None if generic/multi-stock.
    """
    if not text:
        return None

    full_text = f"{title} {text}".strip()
    first_line = text.split("\n")[0] if "\n" in text else title

    # Exclude automated ranking tables and broad market wrap-ups
    if is_generic_market_ranking(full_text):
        return None

    title_matches = {}
    body_matches = {}

    for ticker, conf in TICKER_CONFIG.items():
        t_cnt = sum(len(re.findall(p, first_line, re.IGNORECASE)) for p in conf["title_patterns"])
        if t_cnt > 0:
            title_matches[ticker] = t_cnt

        b_cnt = sum(len(re.findall(p, full_text, re.IGNORECASE)) for p in conf["body_patterns"])
        if b_cnt > 0:
            body_matches[ticker] = b_cnt

    if not body_matches:
        return None

    # Exactly 1 target stock mentioned in text
    if len(body_matches) == 1:
        return list(body_matches.keys())[0]

    # Multiple stocks mentioned: check if title or dominant mentions give clear evidence
    if len(title_matches) == 1:
        t_ticker = list(title_matches.keys())[0]
        other_body_counts = [cnt for t, cnt in body_matches.items() if t != t_ticker]
        if not other_body_counts or body_matches.get(t_ticker, 0) >= 2 * max(other_body_counts):
            return t_ticker

    # Prominent body count (at least 3 mentions and at least 2x second highest)
    sorted_body = sorted(body_matches.items(), key=lambda x: x[1], reverse=True)
    if sorted_body[0][1] >= 3 and sorted_body[0][1] >= 2 * sorted_body[1][1]:
        return sorted_body[0][0]

    # Otherwise post discusses multiple target stocks without single focus -> exclude from single-stock attribution
    return None


def clean_text(text: str) -> str:
    """Normalize whitespace and formatting."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = text.split("\n")
    if len(lines) >= 2 and lines[0].strip() == lines[1].strip():
        text = "\n".join(lines[1:])
    return text.strip()


def parse_timestamp_utc(val) -> str | None:
    """Safely parse timestamps into ISO 8601 UTC."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    try:
        dt = pd.to_datetime(val, utc=True)
        return dt.isoformat()
    except Exception:
        return None


def normalize_social_datasets():
    """
    Loads raw Telegram and LIHKG datasets, applies robust ticker identification,
    filters to the 10-day window, deduplicates, and saves to data/processed/social_posts.csv.
    """
    records = []

    # 1. Telegram
    tg_path = RAW_DIR / "telegram_posts.csv"
    if tg_path.exists():
        tg_df = pd.read_csv(tg_path)
        print(f"Loaded {len(tg_df)} raw Telegram posts.")
        for _, row in tg_df.iterrows():
            raw_text = clean_text(str(row.get("text", "")))
            ticker = identify_ticker(raw_text)
            if not ticker:
                continue

            ts_utc = parse_timestamp_utc(row.get("timestamp_utc"))
            # 10-day filter
            if ts_utc:
                dt = pd.to_datetime(ts_utc)
                if dt < WINDOW_START or dt > CUTOFF_NOW:
                    continue

            records.append({
                "source": "Telegram",
                "ticker": ticker,
                "text": raw_text,
                "timestamp_utc": ts_utc,
                "author": row.get("author") if pd.notna(row.get("author")) else None,
                "author_followers": np.nan,  # Unavailable in Telegram scraper; keep null per instructions
                "likes": int(row.get("likes", 0)) if pd.notna(row.get("likes")) else 0,
                "replies": int(row.get("replies", 0)) if pd.notna(row.get("replies")) else 0,
                "views": int(row.get("views", 0)) if pd.notna(row.get("views")) else 0,
                "reposts": int(row.get("reposts", 0)) if pd.notna(row.get("reposts")) else 0,
                "url": row.get("url") if pd.notna(row.get("url")) else None,
            })

    # 2. LIHKG
    li_path = RAW_DIR / "lihkg_posts.csv"
    if li_path.exists():
        li_df = pd.read_csv(li_path)
        print(f"Loaded {len(li_df)} raw LIHKG posts.")
        for _, row in li_df.iterrows():
            raw_text = clean_text(str(row.get("text", "")))
            ticker = identify_ticker(raw_text)
            if not ticker:
                continue

            ts_utc = parse_timestamp_utc(row.get("timestamp_utc"))
            if ts_utc:
                dt = pd.to_datetime(ts_utc)
                if dt < WINDOW_START or dt > CUTOFF_NOW:
                    continue

            records.append({
                "source": "LIHKG",
                "ticker": ticker,
                "text": raw_text,
                "timestamp_utc": ts_utc,
                "author": row.get("author") if pd.notna(row.get("author")) else None,
                "author_followers": np.nan,
                "likes": int(row.get("likes", 0)) if pd.notna(row.get("likes")) else 0,
                "replies": int(row.get("replies", 0)) if pd.notna(row.get("replies")) else 0,
                "views": int(row.get("views", 0)) if pd.notna(row.get("views")) else 0,
                "reposts": int(row.get("reposts", 0)) if pd.notna(row.get("reposts")) else 0,
                "url": row.get("url") if pd.notna(row.get("url")) else None,
            })

    if not records:
        print("Warning: No social posts matched criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Required columns order
    cols = [
        "source", "ticker", "text", "timestamp_utc", "author",
        "author_followers", "likes", "replies", "views", "reposts", "url"
    ]
    df = df[cols]

    # Deduplicate:
    # 1. By URL when URL is present
    has_url = df[df["url"].notna() & (df["url"] != "")]
    no_url = df[df["url"].isna() | (df["url"] == "")]
    has_url = has_url.drop_duplicates(subset=["url"])
    df = pd.concat([has_url, no_url], ignore_index=True)

    # 2. By normalized text key and ticker
    df["norm_text_key"] = df["text"].str.replace(r"\s+", "", regex=True)
    df = df.drop_duplicates(subset=["norm_text_key", "ticker"])
    df = df.drop(columns=["norm_text_key"])

    out_path = PROCESSED_DIR / "social_posts.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved normalized social posts to {out_path}")
    print(f"Total valid social records: {len(df)}")
    print("\nSocial Posts per Ticker:")
    print(df["ticker"].value_counts())
    print("\nSocial Posts per Source:")
    print(df["source"].value_counts())

    return df


if __name__ == "__main__":
    normalize_social_datasets()
