"""
Apify REST API Scraper for Social Data (Telegram & LIHKG)
Uses requests to interact with Apify REST endpoints (avoids Windows DLL blocking).
Preserves existing raw data in case of network or API rate limits.
"""

import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd
from dotenv import load_dotenv

from ingestion.social_normalizer import (
    identify_ticker,
    clean_text,
    parse_timestamp_utc,
    TICKER_CONFIG
)

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv("APIFY_API_TOKEN")

BASE_URL = "https://api.apify.com/v2"

TELEGRAM_CHANNELS = [
    "StockIntelligenceNews",
    "StockIntelligence",
    "mingtak"
]

# Generate keyword search terms across all 5 assets
SEARCH_KEYWORDS = []
for conf in TICKER_CONFIG.values():
    SEARCH_KEYWORDS.extend([conf["name"]] + [p.replace(r"\b", "").replace(r"\.", ".") for p in conf["title_patterns"]])
SEARCH_KEYWORDS = list(set([k for k in SEARCH_KEYWORDS if not k.startswith("(?")]))

ACTORS = {
    "LIHKG": {
        "id": "claude_code_reviewer/lihkg-scraper-en",
        "input": {
            "mode": "search",
            "keywords": ["Tencent", "騰訊", "Alibaba", "阿里巴巴", "HSBC", "滙豐", "Meituan", "美團", "Xiaomi", "小米"],
            "dateFrom": "10d",
            "order": "desc_reply_time",
            "maxItems": 30,
            "aiLabels": False
        },
        "output": "lihkg_posts.csv"
    },

    "Telegram": {
        "id": "innolabs/telegram-scraper",
        "input": {
            "channels": TELEGRAM_CHANNELS,
            "maxMessagesPerChannel": 200,
            "newerThan": "10 days",
            "keywords": ["Tencent", "Alibaba", "HSBC", "Meituan", "Xiaomi", "騰訊", "阿里巴巴", "滙豐", "美團", "小米"],
            "includeChannelInfo": True,
            "maxItems": 600
        },
        "output": "telegram_posts.csv"
    }
}


# ==========================================
# APIFY REST API
# ==========================================

def run_actor(actor_id, run_input):
    if not TOKEN:
        raise ValueError("APIFY_API_TOKEN missing in .env")

    encoded_actor = quote(actor_id, safe="")
    print(f"\nStarting actor: {actor_id}")

    response = requests.post(
        f"{BASE_URL}/acts/{encoded_actor}/runs",
        params={"token": TOKEN},
        json=run_input,
        timeout=60
    )

    if not response.ok:
        raise RuntimeError(
            f"Actor start failed ({response.status_code}): {response.text[:500]}"
        )

    run_data = response.json()["data"]
    run_id = run_data["id"]
    print(f"Run ID: {run_id}")

    # Poll actor status (max 10 minutes)
    for attempt in range(120):
        status_response = requests.get(
            f"{BASE_URL}/actor-runs/{run_id}",
            params={"token": TOKEN},
            timeout=30
        )
        status_response.raise_for_status()
        run_status = status_response.json()["data"]
        status = run_status["status"]

        print(f"Actor status: {status} ({attempt + 1}/120)")

        if status == "SUCCEEDED":
            dataset_id = run_status["defaultDatasetId"]
            break

        if status in ["FAILED", "ABORTED", "TIMED-OUT"]:
            raise RuntimeError(f"Actor ended with status: {status}")

        time.sleep(5)
    else:
        raise TimeoutError("Actor did not finish within 10 minutes")

    # Retrieve dataset
    dataset_response = requests.get(
        f"{BASE_URL}/datasets/{dataset_id}/items",
        params={"token": TOKEN, "format": "json", "clean": "true"},
        timeout=120
    )
    dataset_response.raise_for_status()
    records = dataset_response.json()
    print(f"Downloaded {len(records)} records")

    return records


# ==========================================
# NORMALIZE SOCIAL POSTS
# ==========================================

def get_value(item, keys, default=None):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_records(records, source):
    normalized = []
    for item in records:
        text = str(get_value(item, ["text", "content", "body", "message"], ""))
        title = str(get_value(item, ["title", "threadTitle"], ""))

        combined_text = clean_text(f"{title} {text}".strip())
        ticker = identify_ticker(combined_text, title=title)

        if not ticker:
            continue

        normalized.append({
            "source": source,
            "ticker": ticker,
            "text": combined_text,
            "timestamp_utc": parse_timestamp_utc(get_value(
                item,
                ["date", "createdAt", "created_at", "timestamp", "publishedAt", "published_date", "time"]
            )),
            "author": get_value(item, ["author", "username", "user", "senderName"]),
            "author_followers": None,
            "likes": int(get_value(item, ["likes", "likeCount", "likesCount", "like_count"], 0)),
            "replies": int(get_value(item, ["replies", "replyCount", "commentCount", "reply_count"], 0)),
            "views": int(get_value(item, ["views", "viewCount"], 0)),
            "reposts": int(get_value(item, ["reposts", "repostCount", "shares"], 0)),
            "url": get_value(item, ["url", "link", "threadUrl", "thread_url"])
        })

    return normalized


# ==========================================
# MAIN
# ==========================================

def main():
    if not TOKEN:
        print("Warning: APIFY_API_TOKEN is not configured. Retaining existing raw files.")
        return

    for source, config in ACTORS.items():
        try:
            records = run_actor(config["id"], config["input"])
            normalized = normalize_records(records, source)
            df = pd.DataFrame(normalized)
            output = RAW_DIR / config["output"]

            if not df.empty:
                df.to_csv(output, index=False)
                print(f"\n{source} results saved to {output}: {len(df)} records")
            else:
                print(f"\n{source}: No relevant records found. Preserving existing data.")

        except Exception as error:
            print(f"\n{source} run encountered error: {error}")
            print(f"Preserving existing {config['output']} raw file.")

    print("\nSocial ingestion check completed.")


if __name__ == "__main__":
    main()
