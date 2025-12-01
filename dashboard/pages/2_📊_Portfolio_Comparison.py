"""
Portfolio Comparison Page - Compare multiple portfolio configurations
Now with AI-Powered Portfolio Seeking across stocks, ETFs, bonds, commodities, REITs, and more!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tradingagents.portfolio.seekers import PortfolioSeeker
from tradingagents.portfolio.seekers.asset_universes import get_ticker_info

st.set_page_config(page_title="Portfolio Comparison", page_icon="📊", layout="wide")

st.title("📊 Portfolio Comparison")
st.markdown("Compare different portfolio strategies and configurations side-by-side.")


def calculate_portfolio_returns(weights, tickers, start_date, end_date):
    """Calculate portfolio returns for a given allocation"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None, None

        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            # Single ticker case
            prices = data[['Close']].rename(columns={'Close': tickers[0]}) if len(tickers) == 1 else data

        # Handle single ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices = prices.dropna()
        if prices.empty:
            return None, None

        returns = prices.pct_change().dropna()

        # Normalize weights
        available_tickers = [t for t in tickers if t in returns.columns]
        if not available_tickers:
            return None, None

        weight_array = np.array([weights.get(t, 0) for t in available_tickers])
        if weight_array.sum() == 0:
            return None, None
        weight_array = weight_array / weight_array.sum()

        portfolio_returns = (returns[available_tickers] * weight_array).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        return cumulative_returns, returns[available_tickers]
    except Exception as e:
        st.error(f"Error calculating returns: {e}")
        return None, None


def calculate_metrics(returns):
    """Calculate portfolio metrics from returns series"""
    if returns is None or len(returns) == 0:
        return {}

    # Ensure we're working with a Series, not DataFrame
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0] if returns.shape[1] == 1 else returns.sum(axis=1)

    annual_return = float(returns.mean() * 252)
    volatility = float(returns.std() * np.sqrt(252))
    sharpe = (annual_return - 0.05) / volatility if volatility > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Sortino
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = float(np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252))
        sortino = (annual_return - 0.05) / downside_std if downside_std > 0 else 0
    else:
        sortino = 0

    return {
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
    }


# Initialize session state for AI-discovered portfolios
if 'ai_portfolios' not in st.session_state:
    st.session_state.ai_portfolios = None
if 'seeking_in_progress' not in st.session_state:
    st.session_state.seeking_in_progress = False


# Main tabs
tab1, tab2 = st.tabs(["📊 Manual Comparison", "🤖 AI Portfolio Seeker"])

with tab1:
    # Original manual portfolio comparison
    st.markdown("### Manual Portfolio Configuration")

    # Sidebar for manual configuration
    st.sidebar.markdown("## Portfolio Configurations")

    # Portfolio 1 - Aggressive Growth
    st.sidebar.markdown("### Portfolio 1: Aggressive Growth")
    p1_name = st.sidebar.text_input("Name", value="Aggressive Growth", key="p1_name")
    p1_tickers = st.sidebar.text_input("Tickers (comma-separated)", value="NVDA,TSLA,AMD,AMZN,META", key="p1_tickers")
    p1_weights = st.sidebar.text_input("Weights (comma-separated)", value="0.25,0.20,0.20,0.20,0.15", key="p1_weights")

    # Portfolio 2 - Balanced
    st.sidebar.markdown("### Portfolio 2: Balanced")
    p2_name = st.sidebar.text_input("Name", value="Balanced", key="p2_name")
    p2_tickers = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL,MSFT,JNJ,PG,VZ", key="p2_tickers")
    p2_weights = st.sidebar.text_input("Weights (comma-separated)", value="0.25,0.25,0.20,0.15,0.15", key="p2_weights")

    # Portfolio 3 - Conservative
    st.sidebar.markdown("### Portfolio 3: Conservative")
    p3_name = st.sidebar.text_input("Name", value="Conservative", key="p3_name")
    p3_tickers = st.sidebar.text_input("Tickers (comma-separated)", value="JNJ,PG,KO,PEP,VZ", key="p3_tickers")
    p3_weights = st.sidebar.text_input("Weights (comma-separated)", value="0.25,0.20,0.20,0.20,0.15", key="p3_weights")

    # Benchmark
    st.sidebar.markdown("### Benchmark")
    benchmark = st.sidebar.selectbox("Benchmark", options=["SPY", "QQQ", "DIA", "IWM"], index=0)

    # Time period
    st.sidebar.markdown("### Time Period")
    period = st.sidebar.selectbox(
        "Backtest Period",
        options=["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        index=3,
    )

    period_map = {
        "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_map[period])

    def parse_portfolio(tickers_str, weights_str):
        """Parse portfolio configuration"""
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        weights = []
        for w in weights_str.split(","):
            try:
                weights.append(float(w.strip()))
            except ValueError:
                weights.append(0)
        return dict(zip(tickers, weights))

    # Parse portfolios
    portfolios = {
        p1_name: parse_portfolio(p1_tickers, p1_weights),
        p2_name: parse_portfolio(p2_tickers, p2_weights),
        p3_name: parse_portfolio(p3_tickers, p3_weights),
    }

    # Run comparison
    if st.button("Compare Portfolios", type="primary", key="manual_compare"):
        with st.spinner("Calculating portfolio performance..."):
            # Calculate returns for each portfolio
            results = {}
            cumulative_data = {}

            for name, weights in portfolios.items():
                tickers = list(weights.keys())
                cum_returns, daily_returns = calculate_portfolio_returns(
                    weights, tickers, start_date, end_date
                )
                if cum_returns is not None and daily_returns is not None:
                    cumulative_data[name] = cum_returns
                    weight_arr = np.array([weights[t] for t in daily_returns.columns])
                    portfolio_rets = (daily_returns * weight_arr).sum(axis=1)
                    results[name] = calculate_metrics(portfolio_rets)

            # Benchmark
            bench_data = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            if not bench_data.empty:
                if 'Adj Close' in bench_data.columns:
                    bench_prices = bench_data['Adj Close']
                else:
                    bench_prices = bench_data['Close']
                bench_returns = bench_prices.pct_change().dropna()
                cumulative_data[benchmark] = (1 + bench_returns).cumprod()
                results[benchmark] = calculate_metrics(bench_returns)

            if cumulative_data:
                # Display cumulative returns chart
                st.markdown("### Cumulative Returns Comparison")

                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

                for i, (name, cum_ret) in enumerate(cumulative_data.items()):
                    fig.add_trace(go.Scatter(
                        x=cum_ret.index,
                        y=cum_ret.values,
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))

                fig.update_layout(
                    title=f"Portfolio Performance ({period})",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics comparison table
                st.markdown("### Performance Metrics Comparison")

                metrics_df = pd.DataFrame(results).T
                metrics_df["Annual Return"] = metrics_df["Annual Return"].apply(lambda x: f"{x:.1%}")
                metrics_df["Volatility"] = metrics_df["Volatility"].apply(lambda x: f"{x:.1%}")
                metrics_df["Sharpe Ratio"] = metrics_df["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
                metrics_df["Sortino Ratio"] = metrics_df["Sortino Ratio"].apply(lambda x: f"{x:.2f}")
                metrics_df["Max Drawdown"] = metrics_df["Max Drawdown"].apply(lambda x: f"{x:.1%}")

                st.dataframe(metrics_df, use_container_width=True)

                # Allocation visualization
                st.markdown("### Portfolio Allocations")

                cols = st.columns(3)
                for i, (col, (name, weights)) in enumerate(zip(cols, portfolios.items())):
                    with col:
                        fig = px.pie(
                            values=list(weights.values()),
                            names=list(weights.keys()),
                            title=name,
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not calculate returns for any portfolio. Please check ticker symbols.")

with tab2:
    st.markdown("""
    ### 🤖 AI-Powered Portfolio Discovery

    Let our multi-agent AI system discover optimal portfolios for you!

    **How it works:**
    1. **Growth Seeker Agent** - Finds high-growth stocks and tech ETFs
    2. **Value Seeker Agent** - Identifies undervalued stocks and dividend aristocrats
    3. **Income Seeker Agent** - Discovers bonds, REITs, and dividend payers
    4. **Defensive Seeker Agent** - Finds low-volatility and safe-haven assets
    5. **Global Seeker Agent** - Identifies international diversification opportunities
    6. **Portfolio Architect** - Combines all recommendations into coherent portfolios

    **Asset Classes Covered:**
    - 📈 **Stocks**: Large-cap, mid-cap, growth, value, dividend
    - 📊 **ETFs**: Market, sector, factor, thematic
    - 💵 **Bonds**: Treasury, corporate, municipal, high-yield, international
    - 🏠 **REITs**: Residential, commercial, data centers, healthcare
    - 🥇 **Commodities**: Gold, silver, oil, agriculture
    - 🌍 **International**: Developed markets, emerging markets, global bonds
    """)

    # Configuration
    st.markdown("### Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_tolerance_selected = st.selectbox(
            "Risk Tolerance",
            options=["Low", "Moderate", "High"],
            index=1,
            key="ai_risk_select",
            help="Your comfort level with investment risk"
        )

    with col2:
        portfolio_style_selected = st.selectbox(
            "Investment Style",
            options=["Diversified", "Growth", "Income", "Balanced"],
            index=0,
            key="ai_style_select",
            help="Your preferred investment approach"
        )

    with col3:
        investment_horizon = st.selectbox(
            "Investment Horizon",
            options=["1-2 Years (Short)", "3-5 Years (Medium)", "5-10 Years (Long)", "10+ Years (Very Long)"],
            index=1,
            key="ai_horizon_select",
            help="How long you plan to hold investments"
        )

    col4, col5 = st.columns(2)

    with col4:
        llm_provider_selected = st.selectbox(
            "AI Provider",
            options=["Anthropic (Claude)", "OpenAI (GPT)"],
            index=0,
            key="ai_provider_select",
        )

    with col5:
        num_holdings = st.slider(
            "Target Holdings per Portfolio",
            min_value=5,
            max_value=20,
            value=10,
            key="ai_holdings_select",
            help="Approximate number of holdings in each portfolio"
        )

    # Map display values to internal values
    risk_map = {"Low": "low", "Moderate": "moderate", "High": "high"}
    style_map = {"Diversified": "diversified", "Growth": "growth", "Income": "income", "Balanced": "balanced"}
    provider_map = {"Anthropic (Claude)": "anthropic", "OpenAI (GPT)": "openai"}
    horizon_map = {
        "1-2 Years (Short)": "short",
        "3-5 Years (Medium)": "medium",
        "5-10 Years (Long)": "long",
        "10+ Years (Very Long)": "very_long"
    }

    risk_tolerance = risk_map[risk_tolerance_selected]
    portfolio_style = style_map[portfolio_style_selected]
    llm_provider = provider_map[llm_provider_selected]
    horizon = horizon_map[investment_horizon]

    # Show current configuration
    st.info(f"**Current Selection:** {risk_tolerance_selected} risk, {portfolio_style_selected} style, {investment_horizon} horizon, ~{num_holdings} holdings")

    # Seek Portfolios Button
    if st.button("🔍 Seek Portfolios", type="primary", use_container_width=True, key="seek_btn"):
        st.session_state.seeking_in_progress = True

        # Store config for display
        st.session_state.last_seek_config = {
            "risk": risk_tolerance_selected,
            "style": portfolio_style_selected,
            "horizon": investment_horizon,
            "holdings": num_holdings,
        }

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(update):
            stage = update.get("stage", "")
            detail = update.get("detail", "")

            progress_map = {
                "growth": 0.15,
                "value": 0.30,
                "income": 0.45,
                "defensive": 0.60,
                "global": 0.75,
                "architect": 0.90,
                "complete": 1.0,
            }

            progress = progress_map.get(stage, 0)
            progress_bar.progress(progress)
            status_text.text(f"🔄 {detail}")

        try:
            seeker = PortfolioSeeker(llm_provider=llm_provider)

            portfolios = seeker.seek_portfolios(
                risk_tolerance=risk_tolerance,
                portfolio_style=portfolio_style,
                investment_horizon=horizon,
                target_holdings=num_holdings,
                progress_callback=progress_callback,
            )

            st.session_state.ai_portfolios = portfolios
            progress_bar.progress(1.0)
            status_text.text("✅ Portfolio discovery complete!")
            st.rerun()

        except Exception as e:
            st.error(f"Error during portfolio seeking: {str(e)}")
            st.session_state.seeking_in_progress = False
            raise e

        st.session_state.seeking_in_progress = False

    # Display AI-discovered portfolios
    if st.session_state.ai_portfolios:
        st.markdown("---")
        st.markdown("## 🎯 Discovered Portfolios")

        portfolios = st.session_state.ai_portfolios

        # Create tabs for each portfolio
        if portfolios:
            portfolio_tabs = st.tabs([p.name for p in portfolios])

            for tab, portfolio in zip(portfolio_tabs, portfolios):
                with tab:
                    # Portfolio header
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Strategy", portfolio.strategy.title())
                    with col2:
                        st.metric("Risk Level", portfolio.risk_level)
                    with col3:
                        st.metric("Expected Return", f"{portfolio.expected_return:.1%}")
                    with col4:
                        st.metric("Expected Volatility", f"{portfolio.expected_volatility:.1%}")

                    st.markdown(f"**Description:** {portfolio.description}")

                    # Allocations
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Holdings")
                        if portfolio.allocations:
                            # Create holdings dataframe
                            holdings_data = []
                            for ticker, weight in sorted(portfolio.allocations.items(), key=lambda x: -x[1]):
                                info = get_ticker_info(ticker)
                                holdings_data.append({
                                    "Ticker": ticker,
                                    "Weight": f"{weight:.1%}",
                                    "Type": info.get("type", "Unknown"),
                                    "Category": info.get("category", "Unknown"),
                                })

                            holdings_df = pd.DataFrame(holdings_data)
                            st.dataframe(holdings_df, use_container_width=True, hide_index=True)

                            # Pie chart
                            fig = px.pie(
                                values=list(portfolio.allocations.values()),
                                names=list(portfolio.allocations.keys()),
                                title="Allocation Breakdown",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("### Asset Type Breakdown")
                        if portfolio.asset_breakdown:
                            breakdown_df = pd.DataFrame([
                                {"Asset Type": k.title(), "Weight": f"{v:.1%}"}
                                for k, v in portfolio.asset_breakdown.items()
                            ])
                            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

                            fig = px.pie(
                                values=list(portfolio.asset_breakdown.values()),
                                names=[k.title() for k in portfolio.asset_breakdown.keys()],
                                title="Asset Class Distribution",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Rationale
                    with st.expander("📝 Investment Rationale"):
                        st.markdown(portfolio.rationale)

            # Compare discovered portfolios
            st.markdown("---")
            st.markdown("## 📊 Compare Discovered Portfolios")

            compare_period = st.selectbox(
                "Backtest Period",
                options=["1M", "3M", "6M", "1Y"],
                index=2,
                key="ai_compare_period"
            )

            if st.button("Compare AI Portfolios", key="compare_ai"):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=period_map.get(compare_period, 180))

                results = {}
                cumulative_data = {}

                for portfolio in portfolios:
                    if portfolio.allocations:
                        tickers = list(portfolio.allocations.keys())
                        cum_returns, daily_returns = calculate_portfolio_returns(
                            portfolio.allocations, tickers, start_date, end_date
                        )
                        if cum_returns is not None and daily_returns is not None:
                            cumulative_data[portfolio.name] = cum_returns
                            weight_arr = np.array([portfolio.allocations.get(t, 0) for t in daily_returns.columns])
                            if weight_arr.sum() > 0:
                                weight_arr = weight_arr / weight_arr.sum()
                                portfolio_rets = (daily_returns * weight_arr).sum(axis=1)
                                results[portfolio.name] = calculate_metrics(portfolio_rets)

                # Add benchmark
                bench_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
                if not bench_data.empty:
                    bench_prices = bench_data['Adj Close'] if 'Adj Close' in bench_data.columns else bench_data['Close']
                    bench_returns = bench_prices.pct_change().dropna()
                    cumulative_data["SPY (Benchmark)"] = (1 + bench_returns).cumprod()
                    results["SPY (Benchmark)"] = calculate_metrics(bench_returns)

                if cumulative_data:
                    # Chart
                    fig = go.Figure()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                    for i, (name, cum_ret) in enumerate(cumulative_data.items()):
                        fig.add_trace(go.Scatter(
                            x=cum_ret.index,
                            y=cum_ret.values,
                            name=name,
                            line=dict(color=colors[i % len(colors)], width=2),
                        ))

                    fig.update_layout(
                        title=f"AI Portfolio Performance ({compare_period})",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        hovermode="x unified",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics table
                    if results:
                        metrics_df = pd.DataFrame(results).T
                        for col in ["Annual Return", "Volatility", "Max Drawdown"]:
                            if col in metrics_df.columns:
                                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.1%}")
                        for col in ["Sharpe Ratio", "Sortino Ratio"]:
                            if col in metrics_df.columns:
                                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}")

                        st.dataframe(metrics_df, use_container_width=True)

            # Use in main dashboard
            st.markdown("---")
            selected_portfolio = st.selectbox(
                "Select a portfolio to use in main dashboard",
                options=[p.name for p in portfolios],
                key="select_ai_portfolio"
            )

            if st.button("📤 Export to Main Dashboard", key="export_ai"):
                for p in portfolios:
                    if p.name == selected_portfolio:
                        st.session_state.exported_portfolio = p
                        st.success(f"Portfolio '{p.name}' exported! Go to the main dashboard to run full analysis.")
                        break

    else:
        st.info("👆 Click 'Seek Portfolios' to discover AI-recommended portfolios across multiple asset classes.")

        # Show asset universe info
        with st.expander("📚 Available Asset Universe"):
            st.markdown("""
            **Stocks (100+)**
            - Mega Cap: AAPL, MSFT, GOOGL, AMZN, NVDA...
            - Large Cap Growth: CRM, ADBE, AMD, NFLX...
            - Large Cap Value: PEP, KO, PFE, IBM...
            - Dividend Aristocrats: JNJ, PG, MMM, ABT...

            **ETFs (50+)**
            - Market: SPY, QQQ, IWM, VTI, VOO...
            - Factor: VTV, VUG, MTUM, QUAL, USMV...
            - Dividend: VYM, SCHD, DVY...

            **Bonds (25+)**
            - Treasury: TLT, IEF, SHY, GOVT, TIP...
            - Corporate: LQD, VCIT, VCSH...
            - High Yield: HYG, JNK, USHY...
            - Municipal: MUB, VTEB, HYD...

            **Sector ETFs (30+)**
            - Technology: XLK, VGT, SOXX, IGV...
            - Healthcare: XLV, VHT, IBB, XBI...
            - Financials: XLF, VFH, KRE...
            - Energy: XLE, VDE, XOP...

            **Commodities (15+)**
            - Precious Metals: GLD, IAU, SLV...
            - Energy: USO, UNG, DBE...
            - Agriculture: DBA, CORN, WEAT...

            **REITs (15+)**
            - Diversified: VNQ, IYR, SCHH...
            - Specialty: AMT, EQIX, DLR, PLD...

            **International (25+)**
            - Developed: EFA, VEA, EWJ, EWG...
            - Emerging: EEM, VWO, FXI, INDA...
            - Global Bonds: BNDX, EMB, VWOB...
            """)
