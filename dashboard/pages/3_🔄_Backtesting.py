"""
Backtesting Page - Historical performance simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Backtesting", page_icon="🔄", layout="wide")

st.title("🔄 Portfolio Backtesting")
st.markdown("Simulate historical performance of your portfolio strategy.")


class PortfolioBacktest:
    """Simple portfolio backtesting engine"""

    def __init__(self, initial_capital=100000, rebalance_freq="monthly"):
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.results = None

    def run_backtest(self, weights, start_date, end_date):
        """Run backtest simulation"""
        tickers = list(weights.keys())

        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None

        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']

        # Handle single ticker case
        if len(tickers) == 1:
            prices = pd.DataFrame(prices)
            prices.columns = tickers

        prices = prices.dropna()

        if prices.empty:
            return None

        # Initialize portfolio
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        holdings = {t: 0 for t in tickers}
        cash = self.initial_capital
        trades = []

        # Get rebalance dates
        if self.rebalance_freq == "monthly":
            rebal_dates = prices.resample("ME").last().index
        elif self.rebalance_freq == "quarterly":
            rebal_dates = prices.resample("QE").last().index
        elif self.rebalance_freq == "annually":
            rebal_dates = prices.resample("YE").last().index
        else:
            rebal_dates = [prices.index[0]]  # One-time at start

        rebal_set = set(rebal_dates)

        for date in prices.index:
            current_prices = prices.loc[date]

            # Rebalance if needed
            if date in rebal_set or date == prices.index[0]:
                # Calculate current portfolio value
                current_value = cash + sum(
                    holdings[t] * current_prices[t] for t in tickers if t in current_prices.index
                )

                # Sell all
                for t in tickers:
                    if holdings[t] > 0:
                        cash += holdings[t] * current_prices.get(t, 0)
                        holdings[t] = 0

                # Buy according to weights
                for t, w in weights.items():
                    if t in current_prices.index and current_prices[t] > 0:
                        target_value = current_value * w
                        shares = int(target_value / current_prices[t])
                        cost = shares * current_prices[t]
                        if cost <= cash:
                            holdings[t] = shares
                            cash -= cost
                            trades.append({
                                "date": date,
                                "ticker": t,
                                "action": "BUY",
                                "shares": shares,
                                "price": current_prices[t],
                                "value": cost,
                            })

            # Calculate daily portfolio value
            portfolio_value[date] = cash + sum(
                holdings[t] * current_prices.get(t, 0) for t in tickers
            )

        self.results = {
            "portfolio_value": portfolio_value,
            "returns": portfolio_value.pct_change().dropna(),
            "trades": pd.DataFrame(trades),
            "final_holdings": holdings,
            "final_cash": cash,
        }

        return self.results

    def get_metrics(self):
        """Calculate performance metrics"""
        if self.results is None:
            return {}

        returns = self.results["returns"]
        portfolio_value = self.results["portfolio_value"]

        # Basic metrics
        total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.05) / volatility if volatility > 0 else 0

        # Drawdown
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        positive_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = positive_days / total_days if total_days > 0 else 0

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "Total Return": total_return,
            "Annual Return": annual_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Calmar Ratio": calmar,
            "Final Value": portfolio_value.iloc[-1],
            "Number of Trades": len(self.results["trades"]),
        }


# Sidebar configuration
st.sidebar.markdown("## Backtest Configuration")

# Portfolio allocation
st.sidebar.markdown("### Portfolio Allocation")
tickers_input = st.sidebar.text_area(
    "Tickers and Weights (one per line, format: TICKER,WEIGHT)",
    value="AAPL,0.25\nMSFT,0.25\nGOOGL,0.20\nAMZN,0.15\nNVDA,0.15",
    height=150,
)

# Parse weights
weights = {}
for line in tickers_input.strip().split("\n"):
    if "," in line:
        parts = line.strip().split(",")
        if len(parts) == 2:
            ticker = parts[0].strip().upper()
            try:
                weight = float(parts[1].strip())
                weights[ticker] = weight
            except ValueError:
                st.sidebar.warning(f"Invalid weight for {ticker}")

# Normalize weights
total_weight = sum(weights.values())
if total_weight > 0:
    weights = {k: v / total_weight for k, v in weights.items()}

# Show parsed weights
st.sidebar.markdown("**Parsed Weights:**")
for t, w in weights.items():
    st.sidebar.text(f"{t}: {w:.1%}")

# Backtest parameters
st.sidebar.markdown("### Parameters")

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=10000,
)

rebalance_freq = st.sidebar.selectbox(
    "Rebalancing Frequency",
    options=["monthly", "quarterly", "annually", "never"],
    index=0,
)

# Date range
st.sidebar.markdown("### Time Period")
years_back = st.sidebar.slider("Years of History", 1, 10, 3)
end_date = datetime.now()
start_date = end_date - timedelta(days=years_back * 365)

st.sidebar.markdown(f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Run backtest
if st.button("Run Backtest", type="primary"):
    if not weights:
        st.error("Please enter at least one ticker and weight")
    else:
        with st.spinner("Running backtest simulation..."):
            # Run backtest
            backtester = PortfolioBacktest(
                initial_capital=initial_capital,
                rebalance_freq=rebalance_freq,
            )

            results = backtester.run_backtest(
                weights,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            if results is None:
                st.error("Failed to run backtest. Please check ticker symbols.")
            else:
                metrics = backtester.get_metrics()

                # Display key metrics
                st.markdown("### Performance Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Return",
                        f"{metrics['Total Return']:.1%}",
                        delta=f"${metrics['Final Value'] - initial_capital:,.0f}"
                    )

                with col2:
                    st.metric("Annual Return", f"{metrics['Annual Return']:.1%}")

                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

                with col4:
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")

                col5, col6, col7, col8 = st.columns(4)

                with col5:
                    st.metric("Volatility", f"{metrics['Volatility']:.1%}")

                with col6:
                    st.metric("Win Rate", f"{metrics['Win Rate']:.1%}")

                with col7:
                    st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")

                with col8:
                    st.metric("Total Trades", metrics['Number of Trades'])

                # Portfolio value chart
                st.markdown("### Portfolio Value Over Time")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results["portfolio_value"].index,
                    y=results["portfolio_value"].values,
                    name="Portfolio Value",
                    fill='tozeroy',
                    line=dict(color='#1f77b4'),
                ))

                # Add benchmark
                spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
                if not spy_data.empty:
                    spy_prices = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
                    spy_normalized = (spy_prices / spy_prices.iloc[0]) * initial_capital
                    fig.add_trace(go.Scatter(
                        x=spy_normalized.index,
                        y=spy_normalized.values,
                        name="SPY Benchmark",
                        line=dict(color='#ff7f0e', dash='dash'),
                    ))

                fig.update_layout(
                    title="Portfolio Value vs Benchmark",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown chart
                st.markdown("### Drawdown Analysis")

                portfolio_value = results["portfolio_value"]
                rolling_max = portfolio_value.expanding().max()
                drawdown = (portfolio_value - rolling_max) / rolling_max

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    name="Drawdown",
                    fill='tozeroy',
                    line=dict(color='#d62728'),
                ))

                fig.update_layout(
                    title="Drawdown (%)",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Monthly returns heatmap
                st.markdown("### Monthly Returns")

                monthly_returns = results["returns"].resample("ME").apply(
                    lambda x: (1 + x).prod() - 1
                )

                # Create pivot table for heatmap
                monthly_df = pd.DataFrame({
                    "Year": monthly_returns.index.year,
                    "Month": monthly_returns.index.month,
                    "Return": monthly_returns.values,
                })

                pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
                pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

                fig = px.imshow(
                    pivot * 100,
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    labels=dict(x="Month", y="Year", color="Return (%)"),
                    title="Monthly Returns Heatmap",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Trade log
                st.markdown("### Trade Log")
                trades_df = results["trades"]
                if not trades_df.empty:
                    trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
                    trades_df["price"] = trades_df["price"].apply(lambda x: f"${x:.2f}")
                    trades_df["value"] = trades_df["value"].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No trades executed")

                # Return distribution
                st.markdown("### Return Distribution")

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=results["returns"] * 100,
                    nbinsx=50,
                    name="Daily Returns",
                ))

                fig.update_layout(
                    title="Distribution of Daily Returns",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Configure your portfolio in the sidebar and click 'Run Backtest' to simulate historical performance.")

    # Show example
    st.markdown("""
    ### How to Use

    1. **Enter your portfolio allocation** in the sidebar (ticker and weight pairs)
    2. **Set initial capital** for the simulation
    3. **Choose rebalancing frequency** (how often to rebalance to target weights)
    4. **Select time period** for historical analysis
    5. **Click 'Run Backtest'** to see results

    ### What You'll See

    - **Performance metrics**: Total return, Sharpe ratio, max drawdown, etc.
    - **Portfolio value chart** compared to SPY benchmark
    - **Drawdown analysis** showing peak-to-trough declines
    - **Monthly returns heatmap** for seasonal patterns
    - **Trade log** of all rebalancing trades
    - **Return distribution** histogram
    """)
