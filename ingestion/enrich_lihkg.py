"""
Enrich and validate raw LIHKG dataset from Apify dataset LxmsGtiBoceQG9KP2.
Populates correct thread_url, like_count, and reply_count.
"""
import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("APIFY_API_TOKEN")

raw_li_path = Path("data/raw/lihkg_posts.csv")

if token:
    try:
        r = requests.get(
            "https://api.apify.com/v2/datasets/LxmsGtiBoceQG9KP2/items",
            params={"token": token, "clean": "true"},
            timeout=30
        )
        if r.ok:
            items = r.json()
            rows = []
            for item in items:
                title = item.get("title", "")
                content = item.get("content") or ""
                combined = f"{title} {content}".strip()
                rows.append({
                    "source": "LIHKG",
                    "ticker": None,  # Will be attributed accurately by social_normalizer
                    "text": combined,
                    "timestamp_utc": item.get("created_at"),
                    "author": item.get("author"),
                    "author_followers": None,
                    "likes": item.get("like_count", 0),
                    "replies": item.get("reply_count", 0),
                    "views": 0,
                    "reposts": 0,
                    "url": item.get("thread_url")
                })
            df = pd.DataFrame(rows)
            df.to_csv(raw_li_path, index=False)
            print(f"Successfully enriched {len(df)} records in {raw_li_path} from Apify dataset.")
    except Exception as e:
        print(f"Failed to fetch from Apify: {e}")

