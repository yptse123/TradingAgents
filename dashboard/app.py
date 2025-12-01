"""
Portfolio Selection Dashboard - Comprehensive Stock Analysis & Portfolio Construction

A full-featured Streamlit dashboard for:
- Multi-stock analysis using TradingAgents
- Portfolio construction with AI-driven allocation
- Risk metrics and performance analytics
- Sector analysis and diversification insights
- Benchmark comparisons
- Rebalancing recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import yfinance as yf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.portfolio import PortfolioGraph, PortfolioAnalytics
from tradingagents.portfolio.states import SignalType, RiskLevel
from tradingagents.default_config import DEFAULT_CONFIG

# Page configuration
st.set_page_config(
    page_title="Portfolio Selection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .signal-buy {
        color: #00c853;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff1744;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffc107;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'portfolio_state' not in st.session_state:
        st.session_state.portfolio_state = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    if 'historical_portfolios' not in st.session_state:
        st.session_state.historical_portfolios = []


def get_sp500_tickers():
    """Get S&P 500 tickers for selection"""
    # Common large-cap stocks for quick selection
    popular_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "ACN", "CSCO", "ABT", "DHR",
        "NEE", "LLY", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "NFLX", "PYPL",
    ]
    return popular_stocks


def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.markdown("## 📊 Portfolio Configuration")

    # Stock selection
    st.sidebar.markdown("### Stock Selection")
    available_tickers = get_sp500_tickers()

    selected_tickers = st.sidebar.multiselect(
        "Select Stocks to Analyze",
        options=available_tickers,
        default=["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
        help="Choose stocks for portfolio analysis"
    )

    # Custom ticker input
    custom_tickers = st.sidebar.text_input(
        "Add Custom Tickers (comma-separated)",
        placeholder="TICKER1, TICKER2",
        help="Add any ticker not in the list"
    )

    if custom_tickers:
        custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        selected_tickers = list(set(selected_tickers + custom_list))

    # Portfolio parameters
    st.sidebar.markdown("### Portfolio Parameters")

    portfolio_size = st.sidebar.number_input(
        "Portfolio Size ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        format="%d",
    )

    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance",
        options=["low", "moderate", "high"],
        value="moderate",
    )

    analysis_date = st.sidebar.date_input(
        "Analysis Date",
        value=datetime.now() - timedelta(days=1),
        max_value=datetime.now(),
    )

    # LLM Configuration
    st.sidebar.markdown("### AI Configuration")

    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=["anthropic", "openai", "google"],
        index=0,
    )

    if llm_provider == "anthropic":
        deep_model = st.sidebar.selectbox(
            "Deep Thinking Model",
            options=["claude-sonnet-4-20250514", "claude-opus-4-5-20251101"],
            index=0,
        )
        quick_model = st.sidebar.selectbox(
            "Quick Thinking Model",
            options=["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"],
            index=0,
        )
    elif llm_provider == "openai":
        deep_model = st.sidebar.selectbox(
            "Deep Thinking Model",
            options=["gpt-4o", "gpt-4-turbo", "o4-mini"],
            index=0,
        )
        quick_model = st.sidebar.selectbox(
            "Quick Thinking Model",
            options=["gpt-4o-mini", "gpt-3.5-turbo"],
            index=0,
        )
    else:
        deep_model = "gemini-pro"
        quick_model = "gemini-pro"

    # Analyst selection
    st.sidebar.markdown("### Analysts")
    analysts = st.sidebar.multiselect(
        "Select Analysts",
        options=["market", "social", "news", "fundamentals"],
        default=["market", "news", "fundamentals"],
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        max_parallel = st.number_input(
            "Max Parallel Analyses",
            min_value=1,
            max_value=5,
            value=2,
        )
        debug_mode = st.checkbox("Debug Mode", value=False)

    return {
        "tickers": selected_tickers,
        "portfolio_size": portfolio_size,
        "risk_tolerance": risk_tolerance,
        "analysis_date": analysis_date.strftime("%Y-%m-%d"),
        "llm_provider": llm_provider,
        "deep_model": deep_model,
        "quick_model": quick_model,
        "analysts": analysts,
        "max_parallel": max_parallel,
        "debug": debug_mode,
    }


def render_portfolio_overview(state):
    """Render portfolio overview section"""
    st.markdown("## 📈 Portfolio Overview")

    metrics = state.get("portfolio_metrics")
    if not metrics:
        st.warning("No portfolio metrics available")
        return

    # Convert to dict if needed
    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Expected Return",
            f"{metrics.get('expected_return', 0):.1%}",
            delta=f"{metrics.get('expected_return', 0) - 0.10:.1%} vs 10% target"
        )

    with col2:
        st.metric(
            "Portfolio Volatility",
            f"{metrics.get('portfolio_volatility', 0):.1%}",
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta="Good" if metrics.get('sharpe_ratio', 0) > 1 else "Moderate"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.1%}",
        )

    with col5:
        st.metric(
            "Positions",
            f"{metrics.get('total_positions', 0)}",
            delta=f"{metrics.get('cash_weight', 0):.0%} cash"
        )


def render_allocation_chart(state):
    """Render portfolio allocation pie chart"""
    allocations = state.get("final_allocations", [])
    if not allocations:
        st.info("No allocations to display")
        return

    # Prepare data
    data = []
    for alloc in allocations:
        if hasattr(alloc, 'ticker'):
            data.append({
                "Ticker": alloc.ticker,
                "Weight": alloc.weight,
                "Amount": alloc.dollar_amount or alloc.weight * state.get("portfolio_size_usd", 100000),
                "Signal": alloc.signal.value if hasattr(alloc.signal, 'value') else alloc.signal,
            })
        else:
            data.append({
                "Ticker": alloc.get("ticker", "Unknown"),
                "Weight": alloc.get("weight", 0),
                "Amount": alloc.get("dollar_amount", 0),
                "Signal": alloc.get("signal", "HOLD"),
            })

    # Add cash position
    metrics = state.get("portfolio_metrics", {})
    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()
    cash_weight = metrics.get("cash_weight", 0)
    if cash_weight > 0:
        data.append({
            "Ticker": "CASH",
            "Weight": cash_weight,
            "Amount": cash_weight * state.get("portfolio_size_usd", 100000),
            "Signal": "HOLD",
        })

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart
        fig = px.pie(
            df,
            values="Weight",
            names="Ticker",
            title="Portfolio Allocation",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Allocation table
        st.markdown("### Allocation Details")
        display_df = df.copy()
        display_df["Weight"] = display_df["Weight"].apply(lambda x: f"{x:.1%}")
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.0f}")

        # Color code signals
        def color_signal(val):
            if val in ["BUY", "STRONG_BUY"]:
                return 'background-color: #c8e6c9'
            elif val in ["SELL", "STRONG_SELL"]:
                return 'background-color: #ffcdd2'
            return ''

        styled_df = display_df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_sector_analysis(state):
    """Render sector allocation analysis"""
    st.markdown("## 🏢 Sector Analysis")

    metrics = state.get("portfolio_metrics", {})
    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()

    sector_weights = metrics.get("sector_weights", {})

    if not sector_weights:
        st.info("No sector data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Sector pie chart
        df = pd.DataFrame([
            {"Sector": k, "Weight": v}
            for k, v in sector_weights.items()
        ])

        fig = px.pie(
            df,
            values="Weight",
            names="Sector",
            title="Sector Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sector metrics
        st.markdown("### Sector Metrics")

        concentration = metrics.get("sector_concentration", 0)
        hhi = metrics.get("herfindahl_index", 0)
        div_ratio = metrics.get("diversification_ratio", 0)

        st.metric("Max Sector Concentration", f"{concentration:.1%}")
        st.metric("Herfindahl Index", f"{hhi:.3f}", help="Lower is more diversified. <0.15 is unconcentrated")
        st.metric("Diversification Ratio", f"{div_ratio:.2f}", help="Higher indicates better diversification")

        # Warning for high concentration
        if concentration > 0.4:
            st.warning("⚠️ High sector concentration detected. Consider diversifying.")


def render_stock_analysis_details(state):
    """Render detailed stock analysis section"""
    st.markdown("## 📑 Stock Analysis Details")

    analyses = state.get("stock_analyses", {})
    if not analyses:
        st.info("No stock analyses available")
        return

    # Create tabs for each stock
    tabs = st.tabs(list(analyses.keys()))

    for tab, (ticker, analysis) in zip(tabs, analyses.items()):
        with tab:
            if hasattr(analysis, 'to_dict'):
                data = analysis.to_dict()
            else:
                data = analysis

            # Signal and confidence
            col1, col2, col3 = st.columns(3)
            with col1:
                signal = data.get("signal", "HOLD")
                signal_color = {"BUY": "🟢", "STRONG_BUY": "🟢🟢", "SELL": "🔴", "STRONG_SELL": "🔴🔴", "HOLD": "🟡"}.get(signal, "⚪")
                st.markdown(f"### {signal_color} {signal}")
            with col2:
                st.metric("Confidence", f"{data.get('confidence', 0):.0%}")
            with col3:
                st.metric("Risk Level", data.get("risk_level", "N/A"))

            # Key metrics
            st.markdown("#### Key Metrics")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("P/E Ratio", data.get("pe_ratio") or "N/A")
            with mcol2:
                st.metric("RSI", data.get("rsi") or "N/A")
            with mcol3:
                st.metric("Beta", data.get("beta") or "N/A")
            with mcol4:
                st.metric("Sector", data.get("sector") or "Unknown")

            # Reports in expanders
            with st.expander("📊 Market Report"):
                st.markdown(data.get("market_report") or "No report available")

            with st.expander("📰 News Report"):
                st.markdown(data.get("news_report") or "No report available")

            with st.expander("💰 Fundamentals Report"):
                st.markdown(data.get("fundamentals_report") or "No report available")

            with st.expander("📢 Sentiment Report"):
                st.markdown(data.get("sentiment_report") or "No report available")

            with st.expander("🎯 Investment Plan"):
                st.markdown(data.get("investment_plan") or "No plan available")


def render_risk_analysis(state):
    """Render risk analysis section"""
    st.markdown("## ⚠️ Risk Analysis")

    metrics = state.get("portfolio_metrics", {})
    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Metrics")

        # Risk metrics table
        risk_data = {
            "Metric": ["Value at Risk (95%)", "Conditional VaR (95%)", "Max Drawdown", "Portfolio Beta"],
            "Value": [
                f"{metrics.get('var_95', 0):.2%}",
                f"{metrics.get('cvar_95', 0):.2%}",
                f"{metrics.get('max_drawdown', 0):.2%}",
                f"{metrics.get('correlation_avg', 0):.2f}",
            ],
            "Status": [
                "🟢" if abs(metrics.get('var_95', 0)) < 0.02 else "🟡" if abs(metrics.get('var_95', 0)) < 0.04 else "🔴",
                "🟢" if abs(metrics.get('cvar_95', 0)) < 0.03 else "🟡" if abs(metrics.get('cvar_95', 0)) < 0.05 else "🔴",
                "🟢" if abs(metrics.get('max_drawdown', 0)) < 0.15 else "🟡" if abs(metrics.get('max_drawdown', 0)) < 0.25 else "🔴",
                "🟢" if metrics.get('correlation_avg', 0) < 0.5 else "🟡" if metrics.get('correlation_avg', 0) < 0.7 else "🔴",
            ],
        }
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Correlation Matrix")

        corr_matrix = state.get("correlation_matrix", {})
        if corr_matrix:
            # Convert to DataFrame
            tickers = list(corr_matrix.keys())
            corr_df = pd.DataFrame(
                [[corr_matrix.get(t1, {}).get(t2, 0) for t2 in tickers] for t1 in tickers],
                index=tickers,
                columns=tickers,
            )

            fig = px.imshow(
                corr_df,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Stock Correlations",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlation matrix not available")


def render_benchmark_comparison(state):
    """Render benchmark comparison section"""
    st.markdown("## 📊 Benchmark Comparison")

    benchmark = state.get("benchmark_comparison", {})
    if not benchmark or "error" in benchmark:
        st.info("Benchmark comparison not available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        portfolio_return = benchmark.get("portfolio_return", 0)
        benchmark_return = benchmark.get("benchmark_return", 0)
        excess = benchmark.get("excess_return", 0)

        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=portfolio_return * 100,
            delta={'reference': benchmark_return * 100, 'relative': False, 'suffix': '%'},
            title={'text': f"Expected Return vs {benchmark.get('benchmark', 'SPY')}"},
            number={'suffix': '%'},
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=benchmark.get("portfolio_sharpe", 0),
            delta={'reference': benchmark.get("benchmark_sharpe", 0), 'relative': False},
            title={'text': "Sharpe Ratio Comparison"},
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=benchmark.get("portfolio_volatility", 0) * 100,
            delta={'reference': benchmark.get("benchmark_volatility", 0) * 100, 'relative': False, 'suffix': '%'},
            title={'text': "Volatility Comparison"},
            number={'suffix': '%'},
        ))
        st.plotly_chart(fig, use_container_width=True)


def render_rebalancing(state):
    """Render rebalancing recommendations"""
    st.markdown("## 🔄 Rebalancing Recommendations")

    if state.get("rebalancing_needed"):
        st.warning("Portfolio rebalancing is recommended!")

        actions = state.get("rebalancing_actions", [])
        if actions:
            st.markdown("### Recommended Actions")
            for action in actions:
                st.markdown(f"- {action.get('trigger', 'Unknown trigger')}")
    else:
        st.success("Portfolio is well-balanced. No rebalancing needed at this time.")


def render_portfolio_summary(state):
    """Render executive summary"""
    st.markdown("## 📋 Executive Summary")

    summary = state.get("portfolio_summary", "")
    rationale = state.get("portfolio_rationale", "")

    if summary:
        st.info(summary)

    with st.expander("View Full Portfolio Rationale"):
        st.markdown(rationale or "No detailed rationale available")


def run_portfolio_analysis(config):
    """Run the portfolio analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(update):
        stage = update.get("stage", "")
        ticker = update.get("ticker", "")
        status = update.get("status", "")
        details = update.get("details", "")

        # Update progress based on stage
        progress_map = {
            "initializing": 0.05,
            "stock_analysis": 0.4,
            "aggregation": 0.6,
            "allocation_debate": 0.75,
            "final_decision": 0.85,
            "metrics": 0.90,
            "benchmark": 0.95,
            "complete": 1.0,
        }

        progress = progress_map.get(stage, 0)
        progress_bar.progress(progress)

        if ticker:
            status_text.text(f"{stage.title()}: {ticker} - {details}")
        else:
            status_text.text(f"{stage.title()}: {details}")

    # Build config
    analysis_config = DEFAULT_CONFIG.copy()
    analysis_config["llm_provider"] = config["llm_provider"]
    analysis_config["deep_think_llm"] = config["deep_model"]
    analysis_config["quick_think_llm"] = config["quick_model"]

    # Create portfolio graph
    pg = PortfolioGraph(
        config=analysis_config,
        selected_analysts=config["analysts"],
        max_parallel_analyses=config["max_parallel"],
        debug=config["debug"],
    )
    pg.set_progress_callback(progress_callback)

    # Run analysis
    state = pg.analyze_portfolio(
        tickers=config["tickers"],
        trade_date=config["analysis_date"],
        portfolio_size_usd=config["portfolio_size"],
        risk_tolerance=config["risk_tolerance"],
    )

    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")

    return state


def main():
    """Main dashboard application"""
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">📊 Portfolio Selection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar configuration
    config = render_sidebar()

    # Main content area
    if not st.session_state.portfolio_state:
        # Show welcome/start screen
        st.markdown("""
        ## Welcome to the AI-Powered Portfolio Selection Tool

        This dashboard uses **TradingAgents** multi-agent AI system to:

        1. **Analyze individual stocks** using Market, News, Sentiment, and Fundamentals analysts
        2. **Conduct investment debates** between Bull and Bear researchers
        3. **Generate portfolio allocations** through Aggressive, Conservative, and Balanced strategists
        4. **Provide risk-adjusted recommendations** with comprehensive metrics

        ### Getting Started

        1. Select stocks from the sidebar (or add custom tickers)
        2. Configure your portfolio parameters
        3. Choose your preferred AI models
        4. Click "Run Portfolio Analysis" below

        """)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Portfolio Analysis", type="primary", use_container_width=True):
                if not config["tickers"]:
                    st.error("Please select at least one stock to analyze")
                else:
                    with st.spinner("Running portfolio analysis..."):
                        try:
                            state = run_portfolio_analysis(config)
                            st.session_state.portfolio_state = state
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            raise e

    else:
        # Show analysis results
        state = st.session_state.portfolio_state

        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("🔄 New Analysis"):
                st.session_state.portfolio_state = None
                st.rerun()
        with col2:
            if st.button("📥 Export Report"):
                # TODO: Implement export
                st.info("Export functionality coming soon!")
        with col3:
            if st.button("💾 Save Portfolio"):
                st.session_state.historical_portfolios.append(state)
                st.success("Portfolio saved!")
        with col4:
            st.download_button(
                "📊 Download JSON",
                data=json.dumps({
                    "portfolio_id": state.get("portfolio_id"),
                    "date": state.get("analysis_date"),
                    "allocations": [
                        a.to_dict() if hasattr(a, 'to_dict') else a
                        for a in state.get("final_allocations", [])
                    ],
                }, default=str, indent=2),
                file_name=f"portfolio_{state.get('portfolio_id', 'export')}.json",
                mime="application/json",
            )

        # Render all sections
        render_portfolio_overview(state)
        st.markdown("---")

        render_allocation_chart(state)
        st.markdown("---")

        render_sector_analysis(state)
        st.markdown("---")

        render_risk_analysis(state)
        st.markdown("---")

        render_benchmark_comparison(state)
        st.markdown("---")

        render_stock_analysis_details(state)
        st.markdown("---")

        render_rebalancing(state)
        st.markdown("---")

        render_portfolio_summary(state)


if __name__ == "__main__":
    main()
