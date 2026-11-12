
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "0700.HK": "Tencent",
    "9988.HK": "Alibaba",
    "0005.HK": "HSBC",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
}


def fetch_news():

    api_key = os.getenv("EXA_API_KEY")

    if not api_key:
        raise ValueError(
            "EXA_API_KEY missing. Add it to your .env file."
        )

    exa = Exa(api_key=api_key)

    start_date = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).date().isoformat()

    all_news = []

    for ticker, company in ASSETS.items():

        print(f"Fetching news for {company}...")

        query = (
            f"{company} {ticker} Hong Kong stock "
            "financial market news"
        )

        try:

            response = exa.search(
                query,
                category="news",
                num_results=10,
                start_published_date=start_date,
                contents={
                    "highlights": True
                }
            )

            for article in response.results:

                highlights = getattr(
                    article, "highlights", None
                ) or []

                text = " ".join(highlights)

                if not text:
                    text = getattr(article, "text", "") or ""

                all_news.append({
                    "ticker": ticker,
                    "company": company,
                    "title": getattr(article, "title", ""),
                    "url": getattr(article, "url", ""),
                    "published_date": getattr(
                        article, "published_date", None
                    ),
                    "author": getattr(article, "author", ""),
                    "text": text,
                    "source": "exa"
                })

            print(f"Retrieved {len(response.results)} articles")

        except Exception as error:

            print(f"News retrieval failed for {ticker}: {error}")

    if not all_news:
        print("No news retrieved.")
        return

    df = pd.DataFrame(all_news)

    df = df.drop_duplicates(subset=["url"])

    output_file = OUTPUT_DIR / "news_data.csv"

    df.to_csv(output_file, index=False)

    print("\nNews ingestion completed.")
    print(f"Total articles: {len(df)}")
    print(f"Saved to: {output_file}")

    return df


if __name__ == "__main__":
    fetch_news()
