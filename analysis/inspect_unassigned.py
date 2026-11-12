import sys
sys.path.append(".")
import re
import pandas as pd
from analysis.audit_social_data import identify_ticker, TICKERS_CONFIG

tg = pd.read_csv("data/raw/telegram_posts.csv")

unassigned = tg[tg["text"].apply(identify_ticker).isna()]
print(f"Total unassigned: {len(unassigned)}")

for i, row in unassigned.head(15).iterrows():
    print(f"--- Unassigned Post {i} (Original ticker: {row['ticker']}) ---")
    print(str(row["text"])[:300].replace("\n", " "))
    print()
