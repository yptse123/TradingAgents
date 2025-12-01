"""Portfolio Graph - Orchestrates multi-stock analysis and portfolio construction"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from .states import (
    PortfolioState,
    StockAnalysisResult,
    PortfolioAllocation,
    PortfolioMetrics,
    SignalType,
    RiskLevel,
    create_initial_portfolio_state,
)
from .agents import (
    create_portfolio_aggregator,
    create_aggressive_allocator,
    create_conservative_allocator,
    create_balanced_allocator,
    create_portfolio_manager,
)
from .metrics import PortfolioAnalytics


def _get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment variables."""
    return os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")


class PortfolioGraph:
    """
    Portfolio Selection and Construction Graph.

    Orchestrates multi-stock analysis using TradingAgents,
    then performs portfolio-level aggregation and allocation.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        selected_analysts: List[str] = None,
        max_parallel_analyses: int = 3,
        debug: bool = False,
    ):
        """
        Initialize the portfolio graph.

        Args:
            config: Configuration dictionary (uses DEFAULT_CONFIG if None)
            selected_analysts: List of analyst types to use
            max_parallel_analyses: Max concurrent stock analyses
            debug: Enable debug mode
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.selected_analysts = selected_analysts or ["market", "social", "news", "fundamentals"]
        self.max_parallel = max_parallel_analyses
        self.debug = debug

        # Initialize LLMs
        self._init_llms()

        # Initialize portfolio-specific agents
        self.aggregator = create_portfolio_aggregator(self.quick_llm)
        self.aggressive_allocator = create_aggressive_allocator(self.quick_llm)
        self.conservative_allocator = create_conservative_allocator(self.quick_llm)
        self.balanced_allocator = create_balanced_allocator(self.quick_llm)
        self.portfolio_manager = create_portfolio_manager(self.deep_llm)

        # Analytics
        self.analytics = PortfolioAnalytics()

        # State tracking
        self.current_state: Optional[PortfolioState] = None
        self.analysis_results: Dict[str, StockAnalysisResult] = {}

        # Progress callback
        self.progress_callback = None

    def _init_llms(self):
        """Initialize LLM instances based on config"""
        provider = self.config.get("llm_provider", "openai").lower()

        if provider in ["openai", "ollama", "openrouter"]:
            self.deep_llm = ChatOpenAI(
                model=self.config["deep_think_llm"],
                base_url=self.config["backend_url"]
            )
            self.quick_llm = ChatOpenAI(
                model=self.config["quick_think_llm"],
                base_url=self.config["backend_url"]
            )
        elif provider == "anthropic":
            api_key = _get_anthropic_api_key()
            base_url = self.config.get("anthropic_base_url")
            max_tokens = self.config.get("anthropic_max_tokens", 8192)

            self.deep_llm = ChatAnthropic(
                model=self.config["deep_think_llm"],
                api_key=api_key if api_key else None,
                base_url=base_url if base_url else None,
                max_tokens=max_tokens,
            )
            self.quick_llm = ChatAnthropic(
                model=self.config["quick_think_llm"],
                api_key=api_key if api_key else None,
                base_url=base_url if base_url else None,
                max_tokens=max_tokens,
            )
        elif provider == "google":
            self.deep_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def set_progress_callback(self, callback):
        """Set a callback function for progress updates"""
        self.progress_callback = callback

    def _report_progress(self, stage: str, ticker: str = None, status: str = "running", details: str = ""):
        """Report progress to callback if set"""
        if self.progress_callback:
            self.progress_callback({
                "stage": stage,
                "ticker": ticker,
                "status": status,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            })

    def _analyze_single_stock(
        self,
        ticker: str,
        trade_date: str,
    ) -> Tuple[str, Optional[StockAnalysisResult]]:
        """
        Analyze a single stock using TradingAgents.

        Returns:
            Tuple of (ticker, analysis_result or None if failed)
        """
        try:
            self._report_progress("analyzing", ticker, "running", f"Starting analysis of {ticker}")

            # Create a fresh TradingAgents instance for this analysis
            ta = TradingAgentsGraph(
                selected_analysts=self.selected_analysts,
                debug=self.debug,
                config=self.config,
            )

            # Run the analysis
            final_state, signal = ta.propagate(ticker, trade_date)

            # Extract analysis result
            result = self._extract_analysis_result(ticker, trade_date, final_state, signal)

            self._report_progress("analyzing", ticker, "completed", f"Completed analysis: {signal}")

            return ticker, result

        except Exception as e:
            self._report_progress("analyzing", ticker, "failed", str(e))
            print(f"Error analyzing {ticker}: {e}")
            return ticker, None

    def _extract_analysis_result(
        self,
        ticker: str,
        trade_date: str,
        state: Dict,
        signal: str,
    ) -> StockAnalysisResult:
        """Extract structured analysis result from TradingAgents state"""

        # Map signal string to SignalType
        signal_upper = signal.upper() if signal else "HOLD"
        signal_map = {
            "STRONG BUY": SignalType.STRONG_BUY,
            "STRONG_BUY": SignalType.STRONG_BUY,
            "BUY": SignalType.BUY,
            "HOLD": SignalType.HOLD,
            "SELL": SignalType.SELL,
            "STRONG SELL": SignalType.STRONG_SELL,
            "STRONG_SELL": SignalType.STRONG_SELL,
        }
        signal_type = signal_map.get(signal_upper, SignalType.HOLD)

        # Calculate confidence from debate state
        confidence = self._calculate_confidence(state)

        # Extract metrics from reports
        metrics = self._extract_metrics_from_reports(state)

        return StockAnalysisResult(
            ticker=ticker,
            trade_date=trade_date,
            signal=signal_type,
            confidence=confidence,
            market_report=state.get("market_report", ""),
            sentiment_report=state.get("sentiment_report", ""),
            news_report=state.get("news_report", ""),
            fundamentals_report=state.get("fundamentals_report", ""),
            investment_plan=state.get("investment_plan", ""),
            trader_plan=state.get("trader_investment_plan", ""),
            final_decision=state.get("final_trade_decision", ""),
            price_target=metrics.get("price_target"),
            stop_loss=metrics.get("stop_loss"),
            current_price=metrics.get("current_price"),
            risk_level=metrics.get("risk_level", RiskLevel.MODERATE),
            volatility=metrics.get("volatility"),
            beta=metrics.get("beta"),
            pe_ratio=metrics.get("pe_ratio"),
            market_cap=metrics.get("market_cap"),
            dividend_yield=metrics.get("dividend_yield"),
            revenue_growth=metrics.get("revenue_growth"),
            profit_margin=metrics.get("profit_margin"),
            rsi=metrics.get("rsi"),
            macd_signal=metrics.get("macd_signal"),
            moving_avg_signal=metrics.get("moving_avg_signal"),
            sector=metrics.get("sector", ""),
            industry=metrics.get("industry", ""),
            raw_state=state,
        )

    def _calculate_confidence(self, state: Dict) -> float:
        """Calculate confidence score from analysis state"""
        # Base confidence
        confidence = 0.5

        # Boost if debate reached consensus
        invest_debate = state.get("investment_debate_state", {})
        if invest_debate.get("judge_decision"):
            confidence += 0.1

        # Boost if risk debate completed
        risk_debate = state.get("risk_debate_state", {})
        if risk_debate.get("judge_decision"):
            confidence += 0.1

        # Boost if all reports are present
        reports = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        present_reports = sum(1 for r in reports if state.get(r))
        confidence += 0.05 * present_reports

        return min(confidence, 1.0)

    def _extract_metrics_from_reports(self, state: Dict) -> Dict:
        """Extract numerical metrics from analysis reports"""
        metrics = {}

        # Try to extract from fundamentals report
        fundamentals = state.get("fundamentals_report", "")
        if fundamentals:
            # Extract P/E ratio
            pe_match = re.search(r'P/E[:\s]*ratio[:\s]*(\d+\.?\d*)', fundamentals, re.IGNORECASE)
            if pe_match:
                metrics["pe_ratio"] = float(pe_match.group(1))

            # Extract market cap
            cap_match = re.search(r'market\s*cap[:\s]*\$?(\d+\.?\d*)\s*(B|M|T)', fundamentals, re.IGNORECASE)
            if cap_match:
                value = float(cap_match.group(1))
                unit = cap_match.group(2).upper()
                multiplier = {"M": 1e6, "B": 1e9, "T": 1e12}.get(unit, 1)
                metrics["market_cap"] = value * multiplier

        # Try to extract from market report
        market = state.get("market_report", "")
        if market:
            # Extract RSI
            rsi_match = re.search(r'RSI[:\s]*(\d+\.?\d*)', market, re.IGNORECASE)
            if rsi_match:
                metrics["rsi"] = float(rsi_match.group(1))

            # Extract beta
            beta_match = re.search(r'beta[:\s]*(\d+\.?\d*)', market, re.IGNORECASE)
            if beta_match:
                metrics["beta"] = float(beta_match.group(1))

        # Extract sector/industry from any report
        for report_key in ["fundamentals_report", "news_report"]:
            report = state.get(report_key, "")
            if report:
                sector_match = re.search(r'sector[:\s]*([A-Za-z\s]+)', report, re.IGNORECASE)
                if sector_match and "sector" not in metrics:
                    metrics["sector"] = sector_match.group(1).strip()[:50]

                industry_match = re.search(r'industry[:\s]*([A-Za-z\s]+)', report, re.IGNORECASE)
                if industry_match and "industry" not in metrics:
                    metrics["industry"] = industry_match.group(1).strip()[:50]

        return metrics

    def analyze_portfolio(
        self,
        tickers: List[str],
        trade_date: str,
        portfolio_size_usd: float = 100000.0,
        risk_tolerance: str = "moderate",
        parallel: bool = True,
    ) -> PortfolioState:
        """
        Analyze a portfolio of stocks and generate allocation recommendations.

        Args:
            tickers: List of stock tickers to analyze
            trade_date: Date for the analysis
            portfolio_size_usd: Total portfolio value
            risk_tolerance: Risk tolerance level (low, moderate, high)
            parallel: Whether to run stock analyses in parallel

        Returns:
            PortfolioState with complete analysis and recommendations
        """
        self._report_progress("initializing", None, "running", f"Starting portfolio analysis for {len(tickers)} stocks")

        # Initialize state
        state = create_initial_portfolio_state(
            tickers=tickers,
            analysis_date=trade_date,
            portfolio_size_usd=portfolio_size_usd,
            risk_tolerance=risk_tolerance,
        )
        self.current_state = state

        # Phase 1: Analyze individual stocks
        self._report_progress("stock_analysis", None, "running", "Beginning individual stock analysis")

        if parallel and len(tickers) > 1:
            state = self._analyze_stocks_parallel(state, trade_date)
        else:
            state = self._analyze_stocks_sequential(state, trade_date)

        # Phase 2: Portfolio aggregation
        self._report_progress("aggregation", None, "running", "Aggregating stock analyses")
        state = self.aggregator(state)

        # Phase 3: Portfolio allocation debate
        self._report_progress("allocation_debate", None, "running", "Running allocation debate")
        state = self._run_allocation_debate(state)

        # Phase 4: Final allocation decision
        self._report_progress("final_decision", None, "running", "Making final allocation decision")
        state = self.portfolio_manager(state)

        # Phase 5: Calculate portfolio metrics
        self._report_progress("metrics", None, "running", "Calculating portfolio metrics")
        state = self._calculate_portfolio_metrics(state, trade_date)

        # Phase 6: Benchmark comparison
        self._report_progress("benchmark", None, "running", "Comparing to benchmarks")
        state["benchmark_comparison"] = self.analytics.get_benchmark_comparison(
            state["final_allocations"],
            "SPY",
            trade_date,
        )

        # Save results
        self._save_results(state)

        self._report_progress("complete", None, "completed", "Portfolio analysis complete")

        self.current_state = state
        return state

    def _analyze_stocks_parallel(self, state: PortfolioState, trade_date: str) -> PortfolioState:
        """Analyze stocks in parallel"""
        tickers = state["pending_tickers"]
        results = {}
        failed = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(self._analyze_single_stock, ticker, trade_date): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result_ticker, result = future.result()
                    if result:
                        results[result_ticker] = result
                    else:
                        failed.append(result_ticker)
                except Exception as e:
                    print(f"Error in parallel analysis of {ticker}: {e}")
                    failed.append(ticker)

        state["stock_analyses"] = results
        state["analyzed_tickers"] = list(results.keys())
        state["failed_tickers"] = failed
        state["pending_tickers"] = []

        return state

    def _analyze_stocks_sequential(self, state: PortfolioState, trade_date: str) -> PortfolioState:
        """Analyze stocks sequentially"""
        tickers = state["pending_tickers"]
        results = {}
        failed = []

        for ticker in tickers:
            result_ticker, result = self._analyze_single_stock(ticker, trade_date)
            if result:
                results[result_ticker] = result
            else:
                failed.append(result_ticker)

        state["stock_analyses"] = results
        state["analyzed_tickers"] = list(results.keys())
        state["failed_tickers"] = failed
        state["pending_tickers"] = []

        return state

    def _run_allocation_debate(self, state: PortfolioState) -> PortfolioState:
        """Run the three-way allocation debate"""
        max_rounds = self.config.get("max_debate_rounds", 1)

        for round_num in range(max_rounds):
            self._report_progress(
                "allocation_debate",
                None,
                "running",
                f"Debate round {round_num + 1}/{max_rounds}"
            )

            # Aggressive allocator proposes
            state = self.aggressive_allocator(state)

            # Conservative allocator responds
            state = self.conservative_allocator(state)

            # Balanced allocator synthesizes
            state = self.balanced_allocator(state)

        return state

    def _calculate_portfolio_metrics(self, state: PortfolioState, trade_date: str) -> PortfolioState:
        """Calculate comprehensive portfolio metrics"""
        allocations = state.get("final_allocations", [])

        if not allocations:
            return state

        # Convert to list if needed
        if not isinstance(allocations, list):
            allocations = list(allocations)

        # Calculate full metrics
        stock_analyses = state.get("stock_analyses", {})
        metrics = self.analytics.calculate_full_metrics(
            allocations,
            stock_analyses,
            trade_date,
        )

        state["portfolio_metrics"] = metrics

        # Calculate correlation matrix
        tickers = [a.ticker for a in allocations if a.weight > 0]
        if tickers:
            state["correlation_matrix"] = self.analytics.calculate_correlation_matrix(
                tickers,
                trade_date,
            )

        return state

    def _save_results(self, state: PortfolioState):
        """Save portfolio analysis results to files"""
        from datetime import datetime as dt

        # Get portfolio_id with fallback
        portfolio_id = state.get("portfolio_id", f"portfolio_{dt.now().strftime('%Y%m%d_%H%M%S')}")

        results_dir = Path(self.config.get("results_dir", "./results"))
        portfolio_dir = results_dir / "portfolios" / portfolio_id
        portfolio_dir.mkdir(parents=True, exist_ok=True)

        # Save state summary
        summary = {
            "portfolio_id": portfolio_id,
            "analysis_date": state.get("analysis_date", ""),
            "portfolio_size_usd": state.get("portfolio_size_usd", 0),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "analyzed_tickers": state.get("analyzed_tickers", []),
            "failed_tickers": state.get("failed_tickers", []),
            "allocations": [a.to_dict() for a in state.get("final_allocations", [])],
            "metrics": state.get("portfolio_metrics", PortfolioMetrics()).to_dict(),
            "benchmark_comparison": state.get("benchmark_comparison", {}),
            "portfolio_summary": state.get("portfolio_summary", ""),
        }

        with open(portfolio_dir / "portfolio_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save individual stock analyses
        analyses_dir = portfolio_dir / "stock_analyses"
        analyses_dir.mkdir(exist_ok=True)

        for ticker, analysis in state.get("stock_analyses", {}).items():
            if hasattr(analysis, 'to_dict'):
                analysis_dict = analysis.to_dict()
            else:
                analysis_dict = analysis

            with open(analyses_dir / f"{ticker}_analysis.json", "w") as f:
                json.dump(analysis_dict, f, indent=2, default=str)

        # Save portfolio rationale
        with open(portfolio_dir / "portfolio_rationale.md", "w") as f:
            f.write(f"# Portfolio Analysis: {portfolio_id}\n\n")
            f.write(f"**Date:** {state.get('analysis_date', 'N/A')}\n")
            f.write(f"**Size:** ${state.get('portfolio_size_usd', 0):,.2f}\n")
            f.write(f"**Risk Tolerance:** {state.get('risk_tolerance', 'moderate')}\n\n")
            f.write("## Aggregation Report\n\n")
            f.write(state.get("aggregation_report", "N/A") + "\n\n")
            f.write("## Portfolio Rationale\n\n")
            f.write(state.get("portfolio_rationale", "N/A") + "\n")

    def get_allocation_summary(self) -> Optional[Dict]:
        """Get a summary of the current portfolio allocation"""
        if not self.current_state:
            return None

        state = self.current_state
        allocations = state.get("final_allocations", [])

        return {
            "portfolio_id": state.get("portfolio_id", "unknown"),
            "date": state.get("analysis_date", ""),
            "portfolio_size": state.get("portfolio_size_usd", 0),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "positions": [
                {
                    "ticker": a.ticker,
                    "weight": f"{a.weight:.1%}",
                    "amount": f"${a.dollar_amount:,.2f}" if a.dollar_amount else "N/A",
                    "signal": a.signal.value if hasattr(a.signal, 'value') else a.signal,
                    "rationale": a.rationale[:100] + "..." if len(a.rationale) > 100 else a.rationale,
                }
                for a in allocations
            ],
            "cash": f"{state.get('portfolio_metrics', PortfolioMetrics()).cash_weight:.1%}",
            "expected_return": f"{state.get('portfolio_metrics', PortfolioMetrics()).expected_return:.1%}",
            "volatility": f"{state.get('portfolio_metrics', PortfolioMetrics()).portfolio_volatility:.1%}",
            "sharpe_ratio": f"{state.get('portfolio_metrics', PortfolioMetrics()).sharpe_ratio:.2f}",
        }
