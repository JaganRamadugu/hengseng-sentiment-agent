
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FILE = BASE_DIR / "data" / "raw" / "telegram_posts.csv"

df = pd.read_csv(FILE)

print("\nDATASET SHAPE:", df.shape)

print("\nPOSTS PER STOCK")
print(df["ticker"].value_counts())

print("\nPOSTS PER SOURCE")
print(df["source"].value_counts())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nSAMPLE POSTS")

for _, row in df.head(10).iterrows():

    print("\nTicker:", row["ticker"])
    print("Text:", str(row["text"])[:300])
    print("Timestamp:", row["timestamp_utc"])
    print("Views:", row["views"])
    print("Likes:", row["likes"])

print("\nUNIQUE POSTS:", df["text"].nunique())
