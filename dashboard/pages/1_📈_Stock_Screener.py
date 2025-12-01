"""
Stock Screener Page - Filter and screen stocks before portfolio analysis
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")

st.title("📈 Stock Screener")
st.markdown("Filter stocks based on fundamental and technical criteria before adding to portfolio analysis.")


@st.cache_data(ttl=3600)
def get_stock_data(tickers):
    """Fetch stock data for screening"""
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            data.append({
                "Ticker": ticker,
                "Name": info.get("shortName", ticker),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": info.get("marketCap", 0),
                "P/E Ratio": info.get("trailingPE", 0),
                "Forward P/E": info.get("forwardPE", 0),
                "PEG Ratio": info.get("pegRatio", 0),
                "Div Yield": info.get("dividendYield", 0) or 0,
                "Beta": info.get("beta", 0),
                "52W High": info.get("fiftyTwoWeekHigh", 0),
                "52W Low": info.get("fiftyTwoWeekLow", 0),
                "Price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "50D MA": info.get("fiftyDayAverage", 0),
                "200D MA": info.get("twoHundredDayAverage", 0),
                "Revenue Growth": info.get("revenueGrowth", 0) or 0,
                "Profit Margin": info.get("profitMargins", 0) or 0,
            })
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")

    return pd.DataFrame(data)


# Default universe
default_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "ACN",
    "CSCO", "ABT", "DHR", "NEE", "LLY", "ADBE", "CRM", "AMD", "INTC",
    "QCOM", "TXN", "NFLX", "PYPL",
]

# Sidebar filters
st.sidebar.markdown("## Screening Filters")

# Market cap filter
st.sidebar.markdown("### Market Cap")
min_cap = st.sidebar.selectbox(
    "Minimum Market Cap",
    options=["Any", "$1B+", "$10B+", "$50B+", "$100B+", "$500B+"],
    index=2,
)

cap_map = {
    "Any": 0,
    "$1B+": 1e9,
    "$10B+": 10e9,
    "$50B+": 50e9,
    "$100B+": 100e9,
    "$500B+": 500e9,
}

# P/E filter
st.sidebar.markdown("### Valuation")
max_pe = st.sidebar.slider("Maximum P/E Ratio", 0, 100, 40)
max_peg = st.sidebar.slider("Maximum PEG Ratio", 0.0, 5.0, 2.0, 0.1)

# Dividend filter
st.sidebar.markdown("### Dividends")
min_div = st.sidebar.slider("Minimum Dividend Yield (%)", 0.0, 10.0, 0.0, 0.1)

# Technical filters
st.sidebar.markdown("### Technical")
above_50ma = st.sidebar.checkbox("Price above 50-day MA", value=False)
above_200ma = st.sidebar.checkbox("Price above 200-day MA", value=False)

# Sector filter
st.sidebar.markdown("### Sectors")
all_sectors = st.sidebar.checkbox("All Sectors", value=True)
if not all_sectors:
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=["Technology", "Healthcare", "Financials", "Consumer Cyclical",
                 "Communication Services", "Industrials", "Consumer Defensive",
                 "Energy", "Utilities", "Real Estate", "Basic Materials"],
    )

# Fetch data
with st.spinner("Fetching stock data..."):
    df = get_stock_data(default_tickers)

# Apply filters
filtered_df = df.copy()

# Market cap filter
filtered_df = filtered_df[filtered_df["Market Cap"] >= cap_map[min_cap]]

# P/E filter
filtered_df = filtered_df[(filtered_df["P/E Ratio"] <= max_pe) | (filtered_df["P/E Ratio"] == 0)]

# PEG filter
filtered_df = filtered_df[(filtered_df["PEG Ratio"] <= max_peg) | (filtered_df["PEG Ratio"] == 0)]

# Dividend filter
filtered_df = filtered_df[filtered_df["Div Yield"] >= min_div / 100]

# Technical filters
if above_50ma:
    filtered_df = filtered_df[filtered_df["Price"] > filtered_df["50D MA"]]
if above_200ma:
    filtered_df = filtered_df[filtered_df["Price"] > filtered_df["200D MA"]]

# Sector filter
if not all_sectors and selected_sectors:
    filtered_df = filtered_df[filtered_df["Sector"].isin(selected_sectors)]

# Display results
st.markdown(f"### Screening Results: {len(filtered_df)} stocks match criteria")

# Format display
display_df = filtered_df.copy()
display_df["Market Cap"] = display_df["Market Cap"].apply(lambda x: f"${x/1e9:.1f}B" if x > 1e9 else f"${x/1e6:.1f}M")
display_df["Div Yield"] = display_df["Div Yield"].apply(lambda x: f"{x:.2%}")
display_df["Revenue Growth"] = display_df["Revenue Growth"].apply(lambda x: f"{x:.1%}")
display_df["Profit Margin"] = display_df["Profit Margin"].apply(lambda x: f"{x:.1%}")
display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")

# Show table
st.dataframe(
    display_df[[
        "Ticker", "Name", "Sector", "Market Cap", "P/E Ratio", "PEG Ratio",
        "Div Yield", "Beta", "Price", "Revenue Growth"
    ]],
    use_container_width=True,
    hide_index=True,
)

# Charts
col1, col2 = st.columns(2)

with col1:
    # Sector distribution
    if len(filtered_df) > 0:
        sector_counts = filtered_df["Sector"].value_counts()
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Sector Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # P/E vs PEG scatter
    if len(filtered_df) > 0:
        fig = px.scatter(
            filtered_df[filtered_df["P/E Ratio"] > 0],
            x="P/E Ratio",
            y="PEG Ratio",
            size="Market Cap",
            color="Sector",
            hover_name="Ticker",
            title="Valuation Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

# Add to portfolio button
st.markdown("---")
st.markdown("### Add to Portfolio Analysis")

selected_for_portfolio = st.multiselect(
    "Select stocks to add to portfolio analysis",
    options=filtered_df["Ticker"].tolist(),
    default=filtered_df["Ticker"].tolist()[:5] if len(filtered_df) >= 5 else filtered_df["Ticker"].tolist(),
)

if st.button("Add to Portfolio", type="primary"):
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = []
    st.session_state.selected_tickers = list(set(st.session_state.get("selected_tickers", []) + selected_for_portfolio))
    st.success(f"Added {len(selected_for_portfolio)} stocks to portfolio analysis!")
    st.info("Go to the main dashboard to run the analysis.")
