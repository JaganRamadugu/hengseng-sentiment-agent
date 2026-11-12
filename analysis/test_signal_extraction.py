import re
import pandas as pd

news = pd.read_csv("data/raw/news_data.csv")
tg = pd.read_csv("data/raw/telegram_posts.csv")

pt_pat = re.compile(r"(?:目標價|target price|PT|TP)[:：\s]*(?:為|至|約|around)?\s*(?:HK\$|\$)?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
supp_pat = re.compile(r"(?:支持位|支撐位|支持線|support)[:：\s]*(?:約|around)?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
res_pat = re.compile(r"(?:阻力位|阻力線|resistance)[:：\s]*(?:約|around)?\s*([0-9]+(?:\.[0-9]+)?)", re.I)

print("--- Searching News for Target Prices, Support, Resistance ---")
matches_news = 0
for idx, r in news.iterrows():
    txt = str(r["title"]) + " " + str(r["text"])
    m_pt = pt_pat.findall(txt)
    m_supp = supp_pat.findall(txt)
    m_res = res_pat.findall(txt)
    if m_pt or m_supp or m_res:
        matches_news += 1
        print(f"{r['ticker']} -> PT: {m_pt}, Supp: {m_supp}, Res: {m_res}")
        print(txt[:180])
        print("-" * 40)
print(f"Total news articles with signals: {matches_news}/{len(news)}")

