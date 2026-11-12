import pandas as pd

news = pd.read_csv("data/raw/news_data.csv")
print("Sample news articles:")
for i, r in news.head(5).iterrows():
    print(f"[{r['ticker']}] {r['title']}")
    print(f"Text: {r['text'][:250]}")
    print("=" * 50)

