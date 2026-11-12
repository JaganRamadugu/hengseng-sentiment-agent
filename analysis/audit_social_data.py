import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

TICKERS_CONFIG = {
    "0700.HK": {
        "name": "Tencent",
        "title_patterns": [r"0700", r"騰訊", r"腾讯", r"\bTencent\b"],
        "body_patterns": [
            r"0700\.HK", r"00700\.HK", r"(?<!\d)00700(?!\d)", r"(?<!\d)0700(?!\d)",
            r"騰訊控股", r"騰訊", r"腾讯控股", r"腾讯", r"\bTencent\b"
        ]
    },
    "9988.HK": {
        "name": "Alibaba",
        "title_patterns": [r"9988", r"阿里巴巴", r"阿里(?!巴)", r"\bAlibaba\b"],
        "body_patterns": [
            r"9988\.HK", r"09988\.HK", r"(?<!\d)09988(?!\d)", r"(?<!\d)9988(?!\d)",
            r"阿里巴巴", r"阿里(?=[－\-\s\(\（]|股價|挫|跌|升|反彈|業績|巴巴)",
            r"\bAlibaba\b"
        ]
    },
    "0005.HK": {
        "name": "HSBC",
        "title_patterns": [r"0005", r"滙豐", r"汇丰", r"\bHSBC\b"],
        "body_patterns": [
            r"0005\.HK", r"00005\.HK", r"(?<!\d)00005(?!\d)", r"(?<!\d)0005(?!\d)",
            r"滙豐控股", r"滙豐", r"汇丰控股", r"汇丰", r"\bHSBC\b"
        ]
    },
    "3690.HK": {
        "name": "Meituan",
        "title_patterns": [r"3690", r"美團", r"美团", r"\bMeituan\b"],
        "body_patterns": [
            r"3690\.HK", r"03690\.HK", r"(?<!\d)03690(?!\d)", r"(?<!\d)3690(?!\d)",
            r"美團", r"美团", r"\bMeituan\b"
        ]
    },
    "1810.HK": {
        "name": "Xiaomi",
        "title_patterns": [r"1810", r"小米", r"\bXiaomi\b"],
        "body_patterns": [
            r"1810\.HK", r"01810\.HK", r"(?<!\d)01810(?!\d)", r"(?<!\d)1810(?!\d)",
            r"小米集團", r"小米集团", r"小米", r"\bXiaomi\b"
        ]
    }
}

GENERIC_MARKET_PATTERNS = [
    r"【#即時排行】",
    r"二十大成交額",
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

def is_market_roundup(text):
    for pat in GENERIC_MARKET_PATTERNS:
        if re.search(pat, text):
            return True
    return False

def identify_ticker(text):
    # Check if market roundup
    if is_market_roundup(text):
        return None  # Exclude generic market rankings/roundups from individual stock sentiment
    
    first_line = text.split("\n")[0]
    
    title_matches = {}
    body_matches = {}
    
    for ticker, conf in TICKERS_CONFIG.items():
        t_cnt = sum(len(re.findall(p, first_line, re.IGNORECASE)) for p in conf["title_patterns"])
        if t_cnt > 0:
            title_matches[ticker] = t_cnt
        b_cnt = sum(len(re.findall(p, text, re.IGNORECASE)) for p in conf["body_patterns"])
        if b_cnt > 0:
            body_matches[ticker] = b_cnt
            
    # If exactly 1 stock mentioned in title and/or body
    if len(body_matches) == 1:
        return list(body_matches.keys())[0]
        
    # If 1 stock in title and body count is dominant (>= 2x next)
    if len(title_matches) == 1:
        t_ticker = list(title_matches.keys())[0]
        other_counts = [cnt for t, cnt in body_matches.items() if t != t_ticker]
        if not other_counts or body_matches.get(t_ticker, 0) >= 2 * max(other_counts):
            return t_ticker
            
    # If body count has a clear dominant winner (e.g. 5 vs 1)
    if len(body_matches) > 1:
        sorted_counts = sorted(body_matches.items(), key=lambda x: x[1], reverse=True)
        if sorted_counts[0][1] >= 3 and sorted_counts[0][1] >= 2 * sorted_counts[1][1]:
            return sorted_counts[0][0]
            
    return None

tg = pd.read_csv("data/raw/telegram_posts.csv")
tg["assigned_ticker"] = tg["text"].apply(identify_ticker)

print("Telegram reassignment summary:")
print("Original assigned counts:")
print(tg["ticker"].value_counts())
print("\nNew cleaned ticker counts:")
print(tg["assigned_ticker"].value_counts(dropna=False))
