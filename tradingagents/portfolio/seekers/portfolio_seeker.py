"""
Portfolio Seeker - AI-Powered Multi-Agent Portfolio Discovery System

Uses specialized agents to discover and recommend diversified portfolios
across stocks, ETFs, bonds, commodities, REITs, and international assets.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .asset_universes import (
    STOCK_UNIVERSE,
    ETF_UNIVERSE,
    BOND_ETF_UNIVERSE,
    SECTOR_ETF_UNIVERSE,
    COMMODITY_UNIVERSE,
    REIT_UNIVERSE,
    INTERNATIONAL_UNIVERSE,
    get_ticker_info,
)


@dataclass
class PortfolioRecommendation:
    """A portfolio recommendation from the seeker"""
    name: str
    strategy: str
    description: str
    allocations: Dict[str, float]
    asset_breakdown: Dict[str, float]  # By asset type
    expected_return: float
    expected_volatility: float
    risk_level: str
    rationale: str
    holdings_detail: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "description": self.description,
            "allocations": self.allocations,
            "asset_breakdown": self.asset_breakdown,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "risk_level": self.risk_level,
            "rationale": self.rationale,
            "holdings_detail": self.holdings_detail,
        }


def _get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment variables."""
    return os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")


class PortfolioSeeker:
    """
    Multi-agent system for discovering optimal portfolio allocations.

    Uses specialized agents:
    - Growth Seeker: Finds high-growth stocks and growth ETFs
    - Value Seeker: Identifies undervalued stocks and value plays
    - Income Seeker: Discovers dividend stocks, bonds, and REITs
    - Defensive Seeker: Finds low-volatility, safe-haven assets
    - Global Seeker: Identifies international diversification opportunities
    - Portfolio Architect: Combines recommendations into coherent portfolios
    """

    def __init__(self, llm_provider: str = "anthropic", model: str = None):
        """Initialize the portfolio seeker with LLM configuration"""
        self.llm_provider = llm_provider

        if llm_provider == "anthropic":
            api_key = _get_anthropic_api_key()
            self.llm = ChatAnthropic(
                model=model or "claude-sonnet-4-20250514",
                api_key=api_key if api_key else None,
                max_tokens=4096,
            )
        else:
            self.llm = ChatOpenAI(
                model=model or "gpt-4o-mini",
            )

        self.market_data_cache = {}

    def _fetch_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch current market data for tickers"""
        cache_key = ",".join(sorted(tickers))
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]

        data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1y")

                if hist.empty:
                    continue

                returns = hist['Close'].pct_change().dropna()
                annual_return = float(returns.mean() * 252) if len(returns) > 0 else 0
                volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0

                data.append({
                    "ticker": ticker,
                    "name": info.get("shortName", ticker),
                    "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "dividend_yield": info.get("dividendYield", 0) or 0,
                    "beta": info.get("beta", 1),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                    "annual_return": annual_return,
                    "volatility": volatility,
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                })
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue

        df = pd.DataFrame(data)
        self.market_data_cache[cache_key] = df
        return df

    def _run_growth_seeker(self, risk_tolerance: str, num_picks: int = 8) -> Dict:
        """Find high-growth stocks and ETFs"""
        # Get growth-focused tickers
        growth_tickers = (
            STOCK_UNIVERSE["large_cap_growth"][:15] +
            STOCK_UNIVERSE["mid_cap_growth"][:10] +
            [e["ticker"] for e in ETF_UNIVERSE["factor_etfs"] if "Growth" in e["name"]] +
            [e["ticker"] for e in SECTOR_ETF_UNIVERSE["technology"]]
        )

        market_data = self._fetch_market_data(growth_tickers[:25])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Growth Investment Specialist focused on identifying high-growth opportunities.

Your expertise:
- Finding companies with strong revenue and earnings growth
- Identifying emerging technology trends
- Evaluating growth sustainability and competitive moats
- Understanding market momentum and sentiment

Risk Tolerance: {risk_tolerance}
Target: Select {num_picks} growth-focused investments

Market Data Available:
{market_data}

Asset Universe:
- Growth Stocks: Companies with high revenue/earnings growth potential
- Tech ETFs: Broad technology exposure (XLK, VGT, QQQ)
- Sector ETFs: Focused sector plays (SOXX for semiconductors, IGV for software)
- Innovation ETFs: Disruptive technology (ARKK)

Provide your recommendations in JSON format:
```json
{{
    "picks": [
        {{"ticker": "TICKER", "weight": 0.15, "rationale": "Brief reason"}}
    ],
    "theme": "Overall growth theme",
    "risk_assessment": "Risk level and key risks"
}}
```"""),
            ("human", "Analyze the market data and recommend {num_picks} growth investments suitable for {risk_tolerance} risk tolerance.")
        ])

        chain = prompt | self.llm

        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "num_picks": num_picks,
            "market_data": market_data.to_string() if not market_data.empty else "Limited data available",
        })

        return self._parse_agent_response(result.content, "growth")

    def _run_value_seeker(self, risk_tolerance: str, num_picks: int = 6) -> Dict:
        """Find undervalued stocks and value plays"""
        value_tickers = (
            STOCK_UNIVERSE["large_cap_value"][:15] +
            STOCK_UNIVERSE["dividend_aristocrats"][:10] +
            [e["ticker"] for e in ETF_UNIVERSE["factor_etfs"] if "Value" in e["name"]]
        )

        market_data = self._fetch_market_data(value_tickers[:20])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Value Investment Specialist focused on finding undervalued opportunities.

Your expertise:
- Fundamental analysis and intrinsic value calculation
- Identifying margin of safety
- Finding quality companies trading below fair value
- Evaluating balance sheet strength and cash flows

Risk Tolerance: {risk_tolerance}
Target: Select {num_picks} value-focused investments

Market Data Available:
{market_data}

Asset Universe:
- Value Stocks: Companies with low P/E, P/B ratios
- Dividend Aristocrats: Companies with 25+ years of dividend growth
- Value ETFs: VTV, IVE for broad value exposure

Provide your recommendations in JSON format:
```json
{{
    "picks": [
        {{"ticker": "TICKER", "weight": 0.15, "rationale": "Brief reason"}}
    ],
    "theme": "Overall value theme",
    "risk_assessment": "Risk level and key risks"
}}
```"""),
            ("human", "Analyze the market data and recommend {num_picks} value investments suitable for {risk_tolerance} risk tolerance.")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "num_picks": num_picks,
            "market_data": market_data.to_string() if not market_data.empty else "Limited data available",
        })

        return self._parse_agent_response(result.content, "value")

    def _run_income_seeker(self, risk_tolerance: str, num_picks: int = 8) -> Dict:
        """Find income-generating assets: dividends, bonds, REITs"""
        income_tickers = (
            STOCK_UNIVERSE["dividend_aristocrats"][:10] +
            [e["ticker"] for e in ETF_UNIVERSE["factor_etfs"] if "Dividend" in e["name"]] +
            [b["ticker"] for b in BOND_ETF_UNIVERSE["aggregate_bonds"]] +
            [b["ticker"] for b in BOND_ETF_UNIVERSE["corporate_bonds"]] +
            [r["ticker"] for r in REIT_UNIVERSE["diversified"]] +
            [r["ticker"] for r in REIT_UNIVERSE["commercial"][:3]]
        )

        market_data = self._fetch_market_data(income_tickers[:25])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Income Investment Specialist focused on generating reliable income streams.

Your expertise:
- Dividend sustainability analysis
- Bond duration and credit risk evaluation
- REIT fundamentals and FFO analysis
- Yield curve positioning

Risk Tolerance: {risk_tolerance}
Target: Select {num_picks} income-focused investments

Market Data Available:
{market_data}

Asset Universe:
- Dividend Stocks: High-quality dividend payers (JNJ, PG, KO)
- Dividend ETFs: SCHD, VYM, DVY for broad dividend exposure
- Bond ETFs: AGG, BND for aggregate bonds; LQD for corporate; TLT for treasuries
- REITs: VNQ for diversified; specific REITs for focused exposure

Provide your recommendations in JSON format:
```json
{{
    "picks": [
        {{"ticker": "TICKER", "weight": 0.15, "rationale": "Brief reason", "yield": 0.03}}
    ],
    "theme": "Overall income theme",
    "expected_yield": 0.04,
    "risk_assessment": "Risk level and key risks"
}}
```"""),
            ("human", "Analyze the market data and recommend {num_picks} income investments suitable for {risk_tolerance} risk tolerance.")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "num_picks": num_picks,
            "market_data": market_data.to_string() if not market_data.empty else "Limited data available",
        })

        return self._parse_agent_response(result.content, "income")

    def _run_defensive_seeker(self, risk_tolerance: str, num_picks: int = 6) -> Dict:
        """Find defensive, low-volatility assets"""
        defensive_tickers = (
            [e["ticker"] for e in ETF_UNIVERSE["factor_etfs"] if "Vol" in e["name"] or "Quality" in e["name"]] +
            [b["ticker"] for b in BOND_ETF_UNIVERSE["government_bonds"]] +
            [b["ticker"] for b in BOND_ETF_UNIVERSE["aggregate_bonds"]] +
            [c["ticker"] for c in COMMODITY_UNIVERSE["precious_metals"]] +
            [s["ticker"] for s in SECTOR_ETF_UNIVERSE["utilities"]] +
            [s["ticker"] for s in SECTOR_ETF_UNIVERSE["consumer"][:2]]
        )

        market_data = self._fetch_market_data(defensive_tickers[:20])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Defensive Investment Specialist focused on capital preservation and downside protection.

Your expertise:
- Low-volatility investing strategies
- Safe-haven asset allocation
- Defensive sector positioning
- Hedging and risk management

Risk Tolerance: {risk_tolerance}
Target: Select {num_picks} defensive investments

Market Data Available:
{market_data}

Asset Universe:
- Low Volatility ETFs: USMV, SPLV for min vol strategies
- Treasury Bonds: TLT, IEF, SHY for different durations
- Gold: GLD, IAU as safe havens
- Defensive Sectors: XLU (utilities), XLP (consumer staples)
- Quality Factor: QUAL for high-quality companies

Provide your recommendations in JSON format:
```json
{{
    "picks": [
        {{"ticker": "TICKER", "weight": 0.15, "rationale": "Brief reason"}}
    ],
    "theme": "Overall defensive theme",
    "risk_assessment": "Expected volatility reduction"
}}
```"""),
            ("human", "Analyze the market data and recommend {num_picks} defensive investments suitable for {risk_tolerance} risk tolerance.")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "num_picks": num_picks,
            "market_data": market_data.to_string() if not market_data.empty else "Limited data available",
        })

        return self._parse_agent_response(result.content, "defensive")

    def _run_global_seeker(self, risk_tolerance: str, num_picks: int = 6) -> Dict:
        """Find international diversification opportunities"""
        global_tickers = (
            [i["ticker"] for i in INTERNATIONAL_UNIVERSE["developed_markets"]] +
            [i["ticker"] for i in INTERNATIONAL_UNIVERSE["emerging_markets"]] +
            [i["ticker"] for i in INTERNATIONAL_UNIVERSE["global"]] +
            [i["ticker"] for i in INTERNATIONAL_UNIVERSE["international_bonds"]]
        )

        market_data = self._fetch_market_data(global_tickers[:20])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Global Investment Specialist focused on international diversification.

Your expertise:
- Global macro analysis
- Currency considerations
- Emerging vs developed market dynamics
- Geographic risk assessment

Risk Tolerance: {risk_tolerance}
Target: Select {num_picks} international investments

Market Data Available:
{market_data}

Asset Universe:
- Developed Markets: EFA, VEA for broad developed ex-US; country-specific like EWJ (Japan)
- Emerging Markets: EEM, VWO for broad EM; country-specific like FXI (China), INDA (India)
- Global: VT, ACWI for total world exposure
- International Bonds: BNDX, EMB for global fixed income

Provide your recommendations in JSON format:
```json
{{
    "picks": [
        {{"ticker": "TICKER", "weight": 0.15, "rationale": "Brief reason", "region": "Region"}}
    ],
    "theme": "Overall global theme",
    "geographic_breakdown": {{"developed": 0.6, "emerging": 0.4}},
    "risk_assessment": "Currency and geopolitical risks"
}}
```"""),
            ("human", "Analyze the market data and recommend {num_picks} international investments suitable for {risk_tolerance} risk tolerance.")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "num_picks": num_picks,
            "market_data": market_data.to_string() if not market_data.empty else "Limited data available",
        })

        return self._parse_agent_response(result.content, "global")

    def _run_portfolio_architect(
        self,
        growth_picks: Dict,
        value_picks: Dict,
        income_picks: Dict,
        defensive_picks: Dict,
        global_picks: Dict,
        risk_tolerance: str,
        portfolio_style: str,
        investment_horizon: str = "medium",
        target_holdings: int = 10,
    ) -> List[PortfolioRecommendation]:
        """Combine all agent recommendations into coherent portfolios"""

        # Map horizon to description
        horizon_descriptions = {
            "short": "1-2 years (prioritize liquidity, lower volatility)",
            "medium": "3-5 years (balanced approach)",
            "long": "5-10 years (can tolerate more volatility for growth)",
            "very_long": "10+ years (maximize long-term growth potential)",
        }
        horizon_desc = horizon_descriptions.get(investment_horizon, "3-5 years")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Chief Investment Officer responsible for constructing optimal portfolios.

Your task: Create 3 distinct portfolio recommendations by combining inputs from specialist agents.

## Investor Profile:
- **Risk Tolerance**: {risk_tolerance}
- **Investment Style**: {portfolio_style}
- **Investment Horizon**: {investment_horizon}
- **Target Holdings**: {target_holdings} positions per portfolio

## Investment Horizon Considerations:
- **Short (1-2 years)**: Prioritize stability, bonds, dividend stocks. Avoid high-volatility growth.
- **Medium (3-5 years)**: Balanced approach with moderate growth exposure.
- **Long (5-10 years)**: Can include more growth stocks, emerging markets, tolerate volatility.
- **Very Long (10+ years)**: Maximize growth potential, heavy equity allocation, compound growth.

## Specialist Agent Recommendations:

### Growth Seeker:
{growth_picks}

### Value Seeker:
{value_picks}

### Income Seeker:
{income_picks}

### Defensive Seeker:
{defensive_picks}

### Global Seeker:
{global_picks}

## Portfolio Construction Guidelines (adjust based on horizon):

1. **Aggressive Portfolio**:
   - Heavy growth allocation (40-60% for long horizon, 30-40% for short)
   - Technology and innovation focus
   - Include emerging markets for long horizons
   - {target_holdings} holdings

2. **Balanced Portfolio**:
   - Equal growth and value (25% each)
   - Income/bonds allocation (20-30%, higher for short horizons)
   - Global diversification (15-20%)
   - {target_holdings} holdings

3. **Conservative/Income Portfolio**:
   - Income/bonds focus (40-50%, higher for short horizons)
   - Defensive assets and dividend stocks
   - Value stocks with stable earnings
   - {target_holdings} holdings

Provide your recommendations in JSON format:
```json
{{
    "portfolios": [
        {{
            "name": "Portfolio Name",
            "strategy": "aggressive|balanced|conservative",
            "description": "Brief description mentioning horizon suitability",
            "allocations": {{"TICKER": 0.10, "TICKER2": 0.08}},
            "asset_breakdown": {{"stocks": 0.70, "etfs": 0.15, "bonds": 0.10, "commodities": 0.05}},
            "expected_return": 0.12,
            "expected_volatility": 0.20,
            "risk_level": "High|Moderate-High|Moderate|Low-Moderate|Low",
            "rationale": "Why this portfolio fits the investor profile and horizon"
        }}
    ]
}}
```

IMPORTANT:
- Each portfolio MUST have exactly {target_holdings} holdings (tickers)
- Allocations MUST sum to 1.0 for each portfolio
- Consider the investment horizon when selecting assets"""),
            ("human", "Construct 3 optimal portfolios for a {risk_tolerance} risk investor with {investment_horizon} horizon seeking {portfolio_style} strategy. Target {target_holdings} holdings per portfolio.")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "risk_tolerance": risk_tolerance,
            "portfolio_style": portfolio_style,
            "investment_horizon": horizon_desc,
            "target_holdings": target_holdings,
            "growth_picks": json.dumps(growth_picks, indent=2),
            "value_picks": json.dumps(value_picks, indent=2),
            "income_picks": json.dumps(income_picks, indent=2),
            "defensive_picks": json.dumps(defensive_picks, indent=2),
            "global_picks": json.dumps(global_picks, indent=2),
        })

        return self._parse_portfolio_response(result.content)

    def _parse_agent_response(self, response: str, agent_type: str) -> Dict:
        """Parse JSON response from an agent"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

        return {"picks": [], "theme": f"Error parsing {agent_type} response", "error": True}

    def _parse_portfolio_response(self, response: str) -> List[PortfolioRecommendation]:
        """Parse portfolio recommendations from architect response"""
        portfolios = []

        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    data = {}

            for p in data.get("portfolios", []):
                portfolios.append(PortfolioRecommendation(
                    name=p.get("name", "Unknown"),
                    strategy=p.get("strategy", "balanced"),
                    description=p.get("description", ""),
                    allocations=p.get("allocations", {}),
                    asset_breakdown=p.get("asset_breakdown", {}),
                    expected_return=p.get("expected_return", 0),
                    expected_volatility=p.get("expected_volatility", 0),
                    risk_level=p.get("risk_level", "Medium"),
                    rationale=p.get("rationale", ""),
                ))
        except json.JSONDecodeError as e:
            print(f"Error parsing portfolio response: {e}")

        return portfolios

    def seek_portfolios(
        self,
        risk_tolerance: str = "moderate",
        portfolio_style: str = "diversified",
        investment_horizon: str = "medium",
        target_holdings: int = 10,
        progress_callback=None,
    ) -> List[PortfolioRecommendation]:
        """
        Run all specialist agents and generate portfolio recommendations.

        Args:
            risk_tolerance: "low", "moderate", or "high"
            portfolio_style: "growth", "income", "balanced", "diversified"
            investment_horizon: "short", "medium", "long", "very_long"
            target_holdings: Target number of holdings per portfolio (5-20)
            progress_callback: Optional callback for progress updates

        Returns:
            List of PortfolioRecommendation objects
        """
        # Adjust picks per agent based on target holdings
        # Each agent contributes, architect combines
        picks_per_agent = max(4, target_holdings // 2)

        # Store horizon for architect
        self.investment_horizon = investment_horizon
        self.target_holdings = target_holdings

        def update_progress(stage, detail=""):
            if progress_callback:
                progress_callback({"stage": stage, "detail": detail})

        update_progress("growth", f"Running Growth Seeker agent ({picks_per_agent} picks)...")
        growth_picks = self._run_growth_seeker(risk_tolerance, num_picks=picks_per_agent)

        update_progress("value", f"Running Value Seeker agent ({picks_per_agent} picks)...")
        value_picks = self._run_value_seeker(risk_tolerance, num_picks=picks_per_agent)

        update_progress("income", f"Running Income Seeker agent ({picks_per_agent} picks)...")
        income_picks = self._run_income_seeker(risk_tolerance, num_picks=picks_per_agent)

        update_progress("defensive", f"Running Defensive Seeker agent ({picks_per_agent} picks)...")
        defensive_picks = self._run_defensive_seeker(risk_tolerance, num_picks=picks_per_agent)

        update_progress("global", f"Running Global Seeker agent ({picks_per_agent} picks)...")
        global_picks = self._run_global_seeker(risk_tolerance, num_picks=picks_per_agent)

        update_progress("architect", "Portfolio Architect combining recommendations...")
        portfolios = self._run_portfolio_architect(
            growth_picks,
            value_picks,
            income_picks,
            defensive_picks,
            global_picks,
            risk_tolerance,
            portfolio_style,
            investment_horizon,
            target_holdings,
        )

        # Enrich with holdings detail
        for portfolio in portfolios:
            portfolio.holdings_detail = []
            for ticker, weight in portfolio.allocations.items():
                info = get_ticker_info(ticker)
                portfolio.holdings_detail.append({
                    "ticker": ticker,
                    "weight": weight,
                    "type": info.get("type", "Unknown"),
                    "category": info.get("category", "Unknown"),
                })

        update_progress("complete", f"Generated {len(portfolios)} portfolio recommendations")

        return portfolios
