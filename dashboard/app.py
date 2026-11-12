"""
Heng Seng Sentiment-Based Stock Price Prediction Agent - Executive Dashboard
Built with Streamlit & Plotly. Reads precomputed processed datasets.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heng Seng Stock Sentiment & Prediction AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-bullish {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-bearish {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-neutral {
        background-color: #F3F4F6;
        color: #374151;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data():
    data = {}
    files = {
        "final_predictions": PROCESSED_DIR / "final_predictions.csv",
        "model_comparison": PROCESSED_DIR / "model_comparison.csv",
        "sentiment_summary": PROCESSED_DIR / "sentiment_summary.csv",
        "news_sentiment": PROCESSED_DIR / "news_sentiment.csv",
        "social_sentiment": PROCESSED_DIR / "social_sentiment.csv",
        "financial_signals": PROCESSED_DIR / "financial_signals.csv",
    }

    for key, path in files.items():
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()

    # Load stock prices
    stock_prices = {}
    tickers = ["0700_HK", "9988_HK", "0005_HK", "3690_HK", "1810_HK"]
    for t in tickers:
        p_path = RAW_DIR / f"{t}.csv"
        if p_path.exists():
            pdf = pd.read_csv(p_path)
            dcol = "Date" if "Date" in pdf.columns else pdf.columns[0]
            pdf[dcol] = pd.to_datetime(pdf[dcol], utc=True)
            pdf = pdf.sort_values(dcol)
            stock_prices[t.replace("_", ".")] = pdf

    data["stock_prices"] = stock_prices
    return data


data = load_data()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.markdown("## **Heng Seng Agent**")
    st.caption("Internship Quantitative Assessment MVP")
    st.markdown("---")

    nav = st.radio(
        "Navigation Menu",
        [
            "📊 Executive Overview",
            "🏢 Stock Comparison & Forecasts",
            "💬 Sentiment Intelligence",
            "📈 Price & Predictive Modeling",
            "🎯 Financial Signals & Extraction",
            "🔍 Data Quality & Limitations",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Target Stocks (HKEX):**")
    st.markdown("""
    - `0700.HK` - Tencent
    - `9988.HK` - Alibaba
    - `0005.HK` - HSBC
    - `3690.HK` - Meituan
    - `1810.HK` - Xiaomi
    """)
    st.caption("⚡ Windows 11 Compliant | VADER + Fin-Lexicon | XGBoost")


# ==========================================
# 1. EXECUTIVE OVERVIEW
# ==========================================

if nav == "📊 Executive Overview":
    st.markdown('<div class="main-header">Heng Seng Sentiment & Stock Prediction Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Quantitative Financial Sentiment and Next-Trading-Day Forecasting System</div>', unsafe_allow_html=True)

    # Key Metrics Banner
    news_count = len(data["news_sentiment"]) if not data["news_sentiment"].empty else 0
    social_count = len(data["social_sentiment"]) if not data["social_sentiment"].empty else 0
    total_data_points = news_count + social_count

    all_sent = []
    if not data["news_sentiment"].empty:
        all_sent.extend(data["news_sentiment"]["sentiment_score"].dropna().tolist())
    if not data["social_sentiment"].empty:
        all_sent.extend(data["social_sentiment"]["sentiment_score"].dropna().tolist())

    avg_market_sentiment = np.mean(all_sent) if all_sent else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Stocks Analysed", "5 HKEX", "Tencent, BABA, HSBC, Meituan, Xiaomi")
    with kpi2:
        st.metric("News Articles (Exa)", f"{news_count}", "10-day window")
    with kpi3:
        st.metric("Social Posts (Apify)", f"{social_count}", "Telegram & LIHKG")
    with kpi4:
        st.metric("Overall Market Sentiment", f"{avg_market_sentiment:+.3f}", "Positive / Bullish" if avg_market_sentiment > 0.05 else "Negative" if avg_market_sentiment < -0.05 else "Neutral")
    with kpi5:
        st.metric("Latest Ingestion", "2026-11-12", "UTC Synchronized")

    st.markdown("---")

    # Overview split: Quick Summary Table & Sentiment Distribution Chart
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 Next-Trading-Day Executive Forecast")
        if not data["final_predictions"].empty:
            preds = data["final_predictions"].copy()
            # Show best/combined model rows
            comb_preds = preds[preds["model_used"] == "Market_Plus_Combined"].copy()
            if comb_preds.empty:
                comb_preds = preds.head(5)

            display_cols = [
                "ticker", "company", "latest_close", "predicted_next_close",
                "predicted_return_pct", "predicted_direction", "latest_sentiment_score"
            ]
            st.dataframe(
                comb_preds[display_cols].rename(columns={
                    "ticker": "Ticker",
                    "company": "Company",
                    "latest_close": "Latest Close (HKD)",
                    "predicted_next_close": "Pred. Close (HKD)",
                    "predicted_return_pct": "Pred. Return (%)",
                    "predicted_direction": "Direction",
                    "latest_sentiment_score": "Sentiment Score",
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No prediction data available.")

        st.info("💡 **Key Finding:** Machine learning forecasts indicate moderate bullishness for HSBC, Meituan, Xiaomi, and Tencent, while Alibaba shows short-term bearish pressure (-0.36%). However, rigorous time-series evaluation demonstrates that XGBoost daily return models underperform the Zero-Return benchmark, as expected under the Efficient Market Hypothesis.")

    with col_right:
        st.markdown("### 📊 Market Sentiment Breakdown")
        if all_sent:
            pos_cnt = sum(1 for s in all_sent if s > 0.05)
            neg_cnt = sum(1 for s in all_sent if s < -0.05)
            neu_cnt = len(all_sent) - pos_cnt - neg_cnt

            fig_donut = go.Figure(data=[go.Pie(
                labels=["Positive", "Neutral", "Negative"],
                values=[pos_cnt, neu_cnt, neg_cnt],
                hole=0.55,
                marker=dict(colors=["#10B981", "#6B7280", "#EF4444"])
            )])
            fig_donut.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ Architectural Flow")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Data Collection
            Exa[Exa Financial News API]
            ApifyTG[Apify Telegram Scraper]
            ApifyLI[Apify LIHKG Scraper]
            YF[Yahoo Finance Historical Prices]
        end
        subgraph Processing & NLP
            Norm[Social Normalizer & Ticker Matcher]
            Vader[VADER + Custom Financial Lexicon]
            Signals[Regex Financial Signal Extractor]
        end
        subgraph Machine Learning
            XGB[XGBoost Multi-Feature Regressor]
            Eval[Walk-Forward & Zero-Return Benchmark]
        end
        subgraph Outputs
            Dash[Streamlit Interactive Dashboard]
            Report[Comprehensive Research Report]
        end

        Exa --> Vader
        ApifyTG --> Norm --> Vader
        ApifyLI --> Norm --> Vader
        Vader --> Signals
        Vader --> XGB
        YF --> XGB
        XGB --> Eval --> Dash
        Signals --> Dash
        Eval --> Report
    ```
    """)


# ==========================================
# 2. STOCK COMPARISON & FORECASTS
# ==========================================

elif nav == "🏢 Stock Comparison & Forecasts":
    st.markdown("## 🏢 Cross-Stock Comparison & Predictions")
    st.markdown("Compare price forecasts, predicted returns, and sentiment metrics across all five target HKEX stocks.")

    if not data["final_predictions"].empty and not data["sentiment_summary"].empty:
        preds = data["final_predictions"][data["final_predictions"]["model_used"] == "Market_Plus_Combined"].copy()
        sums = data["sentiment_summary"][data["sentiment_summary"]["source"] == "ALL_COMBINED"].copy()

        merged_summary = preds.merge(sums, on="ticker", how="left")

        # Visual Comparison Bar Chart
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Predicted Next-Day Return (%)")
            fig_ret = px.bar(
                merged_summary,
                x="company",
                y="predicted_return_pct",
                color="predicted_direction",
                color_discrete_map={"BULLISH": "#10B981", "BEARISH": "#EF4444", "NEUTRAL": "#6B7280"},
                text="predicted_return_pct",
                labels={"company": "Stock", "predicted_return_pct": "Predicted Return (%)"}
            )
            fig_ret.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_ret.update_layout(height=350, yaxis_range=[-1.0, 1.5])
            st.plotly_chart(fig_ret, use_container_width=True)

        with col2:
            st.markdown("#### 💬 Average Sentiment Score (-1 to +1)")
            fig_sent = px.bar(
                merged_summary,
                x="company",
                y="average_sentiment",
                color="average_sentiment",
                color_continuous_scale="RdYlGn",
                text="average_sentiment",
                labels={"company": "Stock", "average_sentiment": "Average Sentiment"}
            )
            fig_sent.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_sent.update_layout(height=350, yaxis_range=[-0.2, 0.7])
            st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("### 📋 Complete Stock Comparison Table")
        table_view = merged_summary[[
            "ticker", "company", "latest_close", "predicted_next_close",
            "predicted_return_pct", "predicted_direction", "average_sentiment",
            "engagement_weighted_sentiment", "total_mentions", "directional_accuracy_pct"
        ]].rename(columns={
            "ticker": "Ticker",
            "company": "Company",
            "latest_close": "Current Price (HKD)",
            "predicted_next_close": "Predicted Next Close",
            "predicted_return_pct": "Pred. Return (%)",
            "predicted_direction": "Forecast Direction",
            "average_sentiment": "Avg Sentiment",
            "engagement_weighted_sentiment": "Eng-Weighted Sent.",
            "total_mentions": "Total Mentions",
            "directional_accuracy_pct": "Dir. Accuracy (%)"
        })
        st.dataframe(table_view, use_container_width=True, hide_index=True)

    else:
        st.warning("Prediction or summary datasets not available.")


# ==========================================
# 3. SENTIMENT INTELLIGENCE
# ==========================================

elif nav == "💬 Sentiment Intelligence":
    st.markdown("## 💬 Stock-Related Sentiment Intelligence")
    st.markdown("Analysis of sentiment scores, mention volumes, source distributions, and daily trajectory over the 10-day collection window.")

    # Top tabs inside Sentiment
    s_tab1, s_tab2, s_tab3 = st.tabs(["📊 Sentiment Breakdown", "📈 Trends Over Time", "📰 Raw Articles & Posts"])

    with s_tab1:
        if not data["sentiment_summary"].empty:
            source_breakdown = data["sentiment_summary"][data["sentiment_summary"]["source"] != "ALL_COMBINED"].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📢 Mentions by Stock & Source")
                fig_src = px.bar(
                    source_breakdown,
                    x="ticker",
                    y="total_mentions",
                    color="source",
                    barmode="group",
                    labels={"ticker": "Ticker", "total_mentions": "Mentions", "source": "Source"},
                    color_discrete_map={"News": "#3B82F6", "Telegram": "#06B6D4", "LIHKG": "#F59E0B"}
                )
                fig_src.update_layout(height=350)
                st.plotly_chart(fig_src, use_container_width=True)

            with c2:
                st.markdown("#### ⚖️ News vs Social Sentiment Comparison")
                fig_comp = px.bar(
                    source_breakdown,
                    x="ticker",
                    y="average_sentiment",
                    color="source",
                    barmode="group",
                    labels={"ticker": "Ticker", "average_sentiment": "Average Sentiment", "source": "Source"},
                    color_discrete_map={"News": "#3B82F6", "Telegram": "#06B6D4", "LIHKG": "#F59E0B"}
                )
                fig_comp.update_layout(height=350)
                st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("#### 📊 Sentiment Distribution by Stock")
            overall_sums = data["sentiment_summary"][data["sentiment_summary"]["source"] == "ALL_COMBINED"].copy()
            dist_melt = overall_sums.melt(
                id_vars=["ticker"],
                value_vars=["positive_count", "neutral_count", "negative_count"],
                var_name="Sentiment",
                value_name="Count"
            )
            dist_melt["Sentiment"] = dist_melt["Sentiment"].str.replace("_count", "").str.capitalize()
            fig_dist = px.bar(
                dist_melt,
                x="ticker",
                y="Count",
                color="Sentiment",
                labels={"ticker": "Ticker", "Count": "Number of Mentions"},
                color_discrete_map={"Positive": "#10B981", "Neutral": "#6B7280", "Negative": "#EF4444"}
            )
            fig_dist.update_layout(height=320)
            st.plotly_chart(fig_dist, use_container_width=True)

    with s_tab2:
        st.markdown("#### 📅 Daily Sentiment Trajectory (Nov 2 - Nov 12, 2026)")
        all_posts = []
        if not data["news_sentiment"].empty:
            nd = data["news_sentiment"].copy()
            nd["date"] = pd.to_datetime(nd["published_date"], utc=True).dt.strftime("%Y-%m-%d")
            nd["source"] = "News"
            nd["ticker"] = nd["mentioned_ticker"]
            all_posts.append(nd[["ticker", "date", "sentiment_score", "source"]])

        if not data["social_sentiment"].empty:
            sd = data["social_sentiment"].copy()
            sd["date"] = pd.to_datetime(sd["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
            sd["source"] = sd["sentiment_source"]
            sd["ticker"] = sd["mentioned_ticker"]
            all_posts.append(sd[["ticker", "date", "sentiment_score", "source"]])

        if all_posts:
            comb_posts = pd.concat(all_posts, ignore_index=True)
            daily_trend = comb_posts.groupby(["date", "ticker"])["sentiment_score"].mean().reset_index()

            fig_trend = px.line(
                daily_trend,
                x="date",
                y="sentiment_score",
                color="ticker",
                markers=True,
                labels={"date": "Date (UTC)", "sentiment_score": "Daily Mean Sentiment", "ticker": "Stock"},
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig_trend.add_hline(y=0.05, line_dash="dash", line_color="#10B981", annotation_text="Positive Cutoff")
            fig_trend.add_hline(y=-0.05, line_dash="dash", line_color="#EF4444", annotation_text="Negative Cutoff")
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)

    with s_tab3:
        st.markdown("#### 🔍 Filter & Search Collected Financial Texts")
        selected_ticker = st.selectbox("Select Ticker to View Records", ["ALL", "0700.HK", "9988.HK", "0005.HK", "3690.HK", "1810.HK"])
        selected_source = st.selectbox("Select Source", ["ALL", "News", "Telegram", "LIHKG"])

        if all_posts:
            view_posts = []
            if not data["news_sentiment"].empty:
                nd = data["news_sentiment"].copy()
                nd["date"] = pd.to_datetime(nd["published_date"], utc=True).dt.strftime("%Y-%m-%d")
                nd["source"] = "News"
                nd["ticker"] = nd["mentioned_ticker"]
                nd["content"] = nd["title"] + " - " + nd["analysis_text"].str[:200]
                view_posts.append(nd[["ticker", "source", "date", "sentiment_score", "sentiment_label", "content", "url"]])

            if not data["social_sentiment"].empty:
                sd = data["social_sentiment"].copy()
                sd["date"] = pd.to_datetime(sd["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
                sd["source"] = sd["sentiment_source"]
                sd["ticker"] = sd["mentioned_ticker"]
                sd["content"] = sd["text"].str[:250]
                view_posts.append(sd[["ticker", "source", "date", "sentiment_score", "sentiment_label", "content", "url"]])

            full_view = pd.concat(view_posts, ignore_index=True)
            if selected_ticker != "ALL":
                full_view = full_view[full_view["ticker"] == selected_ticker]
            if selected_source != "ALL":
                full_view = full_view[full_view["source"] == selected_source]

            st.write(f"Displaying {len(full_view)} matching records:")
            st.dataframe(full_view, use_container_width=True, hide_index=True)


# ==========================================
# 4. PRICE & PREDICTIVE MODELING
# ==========================================

elif nav == "📈 Price & Predictive Modeling":
    st.markdown("## 📈 Price Forecasting & Model Evaluation")
    st.markdown("Honest quantitative comparison between Market-Only baseline, Sentiment-Enhanced models, and the Zero-Return benchmark.")

    ticker_choice = st.selectbox("Select Stock for Deep Dive", ["0700.HK", "9988.HK", "0005.HK", "3690.HK", "1810.HK"])

    # Historical price chart with technical indicators
    if ticker_choice in data["stock_prices"]:
        p_df = data["stock_prices"][ticker_choice].tail(120).copy()
        date_col = "Date" if "Date" in p_df.columns else p_df.columns[0]

        # Get prediction
        pred_sub = data["final_predictions"][
            (data["final_predictions"]["ticker"] == ticker_choice) &
            (data["final_predictions"]["model_used"] == "Market_Plus_Combined")
        ]

        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=p_df[date_col],
            open=p_df['Open'],
            high=p_df['High'],
            low=p_df['Low'],
            close=p_df['Close'],
            name="Daily Price"
        ))

        # Add 5-day and 10-day MAs if available
        if "ma_ratio_5" in p_df.columns:
            ma5 = p_df["Close"].rolling(5).mean()
            fig_candle.add_trace(go.Scatter(x=p_df[date_col], y=ma5, line=dict(color="#3B82F6", width=1.5), name="5-Day MA"))
        if "ma_ratio_10" in p_df.columns:
            ma10 = p_df["Close"].rolling(10).mean()
            fig_candle.add_trace(go.Scatter(x=p_df[date_col], y=ma10, line=dict(color="#F59E0B", width=1.5), name="10-Day MA"))

        # Add Forecast Point
        if not pred_sub.empty:
            pred_close = pred_sub["predicted_next_close"].iloc[0]
            last_date = p_df[date_col].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            fig_candle.add_trace(go.Scatter(
                x=[last_date, next_date],
                y=[p_df["Close"].iloc[-1], pred_close],
                mode="lines+markers",
                marker=dict(symbol="star", size=12, color="#EC4899"),
                line=dict(dash="dot", color="#EC4899", width=2),
                name=f"Forecast: HK${pred_close:.2f}"
            ))

        fig_candle.update_layout(
            title=f"{ticker_choice} - Historical Price Action & Next-Day Prediction",
            yaxis_title="Price (HKD)",
            height=450,
            margin=dict(t=40, b=20, l=20, r=20),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # Model comparison table for selected stock
    st.markdown("### 🔬 Model Comparison on Chronological Held-Out Test Set")
    if not data["model_comparison"].empty:
        stock_comp = data["model_comparison"][data["model_comparison"]["ticker"] == ticker_choice].copy()
        disp_comp = stock_comp[[
            "model_name", "mae", "rmse", "r2_score", "directional_accuracy", "beats_zero_benchmark"
        ]].rename(columns={
            "model_name": "Model",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2_score": "R²",
            "directional_accuracy": "Directional Accuracy (%)",
            "beats_zero_benchmark": "Beats Zero-Return?"
        })
        st.dataframe(disp_comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 Quantitative Findings & Discussion")
    st.info("""
    **Honest Empirical Result:**
    1. **Zero-Return Benchmark Superiority:** Across all 5 stocks, the simple zero-return benchmark achieved the lowest MAE and lowest RMSE. Predicting 'no price change' remains exceptionally difficult to beat in daily equity returns.
    2. **Marginal Sentiment Contribution:** Introducing news and social sentiment slightly reduced MAE for Alibaba (0.0217 → 0.0214), Meituan (0.0213 → 0.0211), and Xiaomi (0.0260 → 0.0255), while directional accuracy for Xiaomi improved from 40% to 47.4%. However, it did not outperform the zero-return benchmark.
    3. **Small-Sample Constraint:** The alternative data window covers 10 days. While valuable for qualitative intelligence, 10 days is insufficient for statistically robust ML training without longitudinal multi-year sentiment history.
    """)


# ==========================================
# 5. FINANCIAL SIGNALS & EXTRACTION
# ==========================================

elif nav == "🎯 Financial Signals & Extraction":
    st.markdown("## 🎯 Financial Information & Signal Extraction")
    st.markdown("Structured entity and signal extraction from unstructured financial news and social media discussions.")

    if not data["financial_signals"].empty:
        signals_df = data["financial_signals"].copy()

        # Metrics on extracted items
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Analyzed", len(signals_df))
        with c2:
            st.metric("Price Mentions", (signals_df["current_price_reference"] != "Not mentioned").sum())
        with c3:
            st.metric("Trading Opinions", (signals_df["trading_opinion"] != "Not mentioned").sum())
        with c4:
            st.metric("Support / Resistance", (signals_df["support_level"] != "Not mentioned").sum() + (signals_df["resistance_level"] != "Not mentioned").sum())
        with c5:
            st.metric("Price Targets / Positions", (signals_df["price_target"] != "Not mentioned").sum() + (signals_df["trading_position"] != "Not mentioned").sum())

        st.markdown("---")

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            t_filter = st.selectbox("Filter by Ticker", ["ALL"] + list(signals_df["ticker"].unique()))
        with f2:
            op_filter = st.selectbox("Filter by Opinion", ["ALL", "BUY", "SELL", "HOLD", "Has Signals Only"])

        filtered = signals_df.copy()
        if t_filter != "ALL":
            filtered = filtered[filtered["ticker"] == t_filter]
        if op_filter == "Has Signals Only":
            filtered = filtered[
                (filtered["current_price_reference"] != "Not mentioned") |
                (filtered["trading_opinion"] != "Not mentioned") |
                (filtered["support_level"] != "Not mentioned") |
                (filtered["price_target"] != "Not mentioned")
            ]
        elif op_filter != "ALL":
            filtered = filtered[filtered["trading_opinion"] == op_filter]

        st.markdown(f"**Showing {len(filtered)} signals:**")
        cols_show = [
            "ticker", "company_name", "source", "sentiment_direction",
            "current_price_reference", "trading_opinion", "support_level",
            "resistance_level", "price_target", "trading_position", "text_snippet"
        ]
        st.dataframe(
            filtered[cols_show].rename(columns={
                "ticker": "Ticker",
                "company_name": "Company",
                "source": "Source",
                "sentiment_direction": "Sentiment",
                "current_price_reference": "Price Ref",
                "trading_opinion": "Opinion",
                "support_level": "Support",
                "resistance_level": "Resistance",
                "price_target": "Target",
                "trading_position": "Position",
                "text_snippet": "Text Context"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Financial signals dataset not found.")


# ==========================================
# 6. DATA QUALITY & LIMITATIONS
# ==========================================

elif nav == "🔍 Data Quality & Limitations":
    st.markdown("## 🔍 Data Quality, Governance & System Limitations")
    st.markdown("Transparency, missing value analysis, and critical assessment of engineering and quantitative constraints.")

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("### 📋 Record Counts & Completeness")
        quality_records = [
            {"Dataset": "Financial News (Exa)", "Raw Records": 47, "Valid Cleaned": 47, "Missing Timestamps": 0, "Missing Tickers": 0},
            {"Dataset": "Telegram Posts (Apify)", "Raw Records": 216, "Valid Cleaned": 116, "Missing Timestamps": 0, "Missing Tickers": 0},
            {"Dataset": "LIHKG Posts (Apify)", "Raw Records": 12, "Valid Cleaned": 7, "Missing Timestamps": 0, "Missing Tickers": 0},
            {"Dataset": "Historical Prices (yfinance)", "Raw Records": "493 / stock", "Valid Cleaned": "493 / stock", "Missing Timestamps": 0, "Missing Tickers": 0},
        ]
        st.dataframe(pd.DataFrame(quality_records), use_container_width=True, hide_index=True)

        st.markdown("### 🛡️ Ticker Attribution Cleaning Impact")
        st.write("""
        - **Original Problem:** The raw Telegram dataset contained 106 posts assigned to `0700.HK`, many of which were generic ranking tables or Alibaba articles with a passing mention of Tencent.
        - **Correction Applied:** Strict alias matching, ranking table exclusion, and contextual dominant-mention analysis re-attributed genuine Alibaba articles to `9988.HK` and filtered out 100 automated ranking/market roundup summaries.
        - **Follower Counts:** Kept null/NaN as instructed rather than fabricating values.
        """)

    with q2:
        st.markdown("### ⚠️ Known Limitations & Assumptions")
        st.markdown("""
        1. **Windows Security / Environment Isolation:**
           - Windows Application Control blocks certain native DLLs (preventing Apify Python SDK, PyTorch, Transformers).
           - Resolved cleanly via REST API calls with `requests` and lightweight, fast `vaderSentiment` + `XGBoost`.
        2. **10-Day Social Sample Window:**
           - Social and news ingestion reflects the latest 10-day window (Nov 2 - Nov 12, 2026).
           - While informative for live trading sentiment, 10 days of alternative data is statistically inadequate for robust deep-learning or high-frequency ML signal generation.
        3. **VADER Chinese Language Limitations:**
           - VADER is natively an English rule-based lexicon. We extended it with a financial Chinese lexicon and whitespace token-padding.
           - While effective for keyword sentiment polarity, it lacks deep syntactic parsing for complex Cantonese / Traditional Chinese slang.
        4. **Predictive Accuracy vs Zero Benchmark:**
           - Daily stock returns possess an extremely low signal-to-noise ratio.
           - XGBoost did not beat the zero-return baseline on daily return MAE on any of the five stocks.
        """)

    st.markdown("---")
    st.markdown("### 💻 Exact Verification Commands")
    st.code(r"""
# Run entire data and modeling pipeline in Windows PowerShell:
.\.venv\Scripts\python.exe ingestion\social_normalizer.py
.\.venv\Scripts\python.exe sentiment\sentiment_engine.py
.\.venv\Scripts\python.exe analysis\financial_signals.py
.\.venv\Scripts\python.exe models\prediction_pipeline.py

# Launch the interactive Streamlit dashboard:
.\.venv\Scripts\streamlit.exe run dashboard\app.py
    """, language="powershell")
