"""
Settings Page - Configure API keys and preferences
"""

import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.markdown("Configure API keys, preferences, and system settings.")

# API Keys section
st.markdown("## API Configuration")

st.warning("⚠️ API keys are stored in environment variables. For security, set them in your shell profile or .env file.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### LLM Providers")

    # OpenAI
    st.markdown("#### OpenAI")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    openai_status = "✅ Configured" if openai_key else "❌ Not Set"
    st.text(f"Status: {openai_status}")
    if st.button("Test OpenAI Connection"):
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-3.5-turbo")
                response = llm.invoke("Say 'OK'")
                st.success("OpenAI connection successful!")
            except Exception as e:
                st.error(f"OpenAI connection failed: {e}")
        else:
            st.error("OpenAI API key not set")

    # Anthropic
    st.markdown("#### Anthropic")
    anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_status = "✅ Configured" if anthropic_key else "❌ Not Set"
    st.text(f"Status: {anthropic_status}")
    if st.button("Test Anthropic Connection"):
        if anthropic_key:
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=anthropic_key)
                response = llm.invoke("Say 'OK'")
                st.success("Anthropic connection successful!")
            except Exception as e:
                st.error(f"Anthropic connection failed: {e}")
        else:
            st.error("Anthropic API key not set")

    # Google
    st.markdown("#### Google AI")
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    google_status = "✅ Configured" if google_key else "❌ Not Set"
    st.text(f"Status: {google_status}")

with col2:
    st.markdown("### Data Providers")

    # Alpha Vantage
    st.markdown("#### Alpha Vantage")
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    av_status = "✅ Configured" if av_key else "❌ Not Set"
    st.text(f"Status: {av_status}")
    st.caption("Used for fundamental data and news")

    # Yahoo Finance (no key needed)
    st.markdown("#### Yahoo Finance")
    st.text("Status: ✅ Available (no key required)")
    st.caption("Used for price data and basic fundamentals")

# Display environment setup instructions
st.markdown("---")
st.markdown("## Environment Setup")

st.markdown("""
### Setting Environment Variables

Add the following to your `~/.bashrc`, `~/.zshrc`, or create a `.env` file:

```bash
# LLM Providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_AUTH_TOKEN="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Data Providers
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```
""")

# Application settings
st.markdown("---")
st.markdown("## Application Settings")

if 'app_settings' not in st.session_state:
    st.session_state.app_settings = {
        "default_portfolio_size": 100000,
        "default_risk_tolerance": "moderate",
        "default_analysts": ["market", "news", "fundamentals"],
        "max_parallel_analyses": 2,
        "cache_data": True,
        "debug_mode": False,
    }

settings = st.session_state.app_settings

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Default Values")

    settings["default_portfolio_size"] = st.number_input(
        "Default Portfolio Size ($)",
        min_value=1000,
        max_value=10000000,
        value=settings["default_portfolio_size"],
        step=10000,
    )

    settings["default_risk_tolerance"] = st.selectbox(
        "Default Risk Tolerance",
        options=["low", "moderate", "high"],
        index=["low", "moderate", "high"].index(settings["default_risk_tolerance"]),
    )

    settings["default_analysts"] = st.multiselect(
        "Default Analysts",
        options=["market", "social", "news", "fundamentals"],
        default=settings["default_analysts"],
    )

with col2:
    st.markdown("### Performance Settings")

    settings["max_parallel_analyses"] = st.slider(
        "Max Parallel Stock Analyses",
        min_value=1,
        max_value=5,
        value=settings["max_parallel_analyses"],
    )

    settings["cache_data"] = st.checkbox(
        "Cache Market Data",
        value=settings["cache_data"],
        help="Cache API responses to reduce API calls"
    )

    settings["debug_mode"] = st.checkbox(
        "Debug Mode",
        value=settings["debug_mode"],
        help="Show detailed logs and intermediate results"
    )

if st.button("Save Settings"):
    st.session_state.app_settings = settings
    st.success("Settings saved!")

# System information
st.markdown("---")
st.markdown("## System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Python Environment")
    import sys
    st.text(f"Python Version: {sys.version.split()[0]}")
    st.text(f"Platform: {sys.platform}")

    st.markdown("### Key Dependencies")
    try:
        import langchain
        st.text(f"LangChain: {langchain.__version__}")
    except:
        st.text("LangChain: Not installed")

    try:
        import streamlit
        st.text(f"Streamlit: {streamlit.__version__}")
    except:
        st.text("Streamlit: Not installed")

    try:
        import yfinance
        st.text(f"yfinance: {yfinance.__version__}")
    except:
        st.text("yfinance: Not installed")

with col2:
    st.markdown("### Data Directories")

    results_dir = Path("./results")
    st.text(f"Results: {results_dir.absolute()}")

    cache_dir = Path("./tradingagents/dataflows/data_cache")
    st.text(f"Cache: {cache_dir.absolute()}")

    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.glob("**/*") if f.is_file())
        st.text(f"Cache Size: {cache_size / 1024 / 1024:.1f} MB")

        if st.button("Clear Cache"):
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            st.success("Cache cleared!")

# Help section
st.markdown("---")
st.markdown("## Help & Documentation")

with st.expander("Getting Started"):
    st.markdown("""
    1. **Set up API keys** in your environment
    2. **Go to the main dashboard** to start analyzing stocks
    3. **Select stocks** from the popular list or enter custom tickers
    4. **Configure portfolio parameters** (size, risk tolerance)
    5. **Run the analysis** and review recommendations
    """)

with st.expander("Understanding the Analysis"):
    st.markdown("""
    The system uses multiple AI agents to analyze stocks:

    - **Market Analyst**: Technical analysis, price patterns, indicators
    - **News Analyst**: News sentiment, company announcements
    - **Social Media Analyst**: Social sentiment, trending discussions
    - **Fundamentals Analyst**: Financial statements, valuations

    These analyses feed into:
    - **Bull/Bear Debate**: Investment thesis discussion
    - **Allocation Debate**: Portfolio weight recommendations
    - **Risk Assessment**: Position sizing and risk management
    """)

with st.expander("Interpreting Results"):
    st.markdown("""
    **Signals:**
    - STRONG_BUY: High conviction buy
    - BUY: Positive outlook
    - HOLD: Neutral/maintain position
    - SELL: Consider reducing
    - STRONG_SELL: High conviction sell

    **Key Metrics:**
    - Sharpe Ratio: Risk-adjusted return (>1 is good)
    - Max Drawdown: Largest peak-to-trough decline
    - Diversification Ratio: Portfolio diversification effectiveness
    """)
