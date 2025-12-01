#!/usr/bin/env python3
"""
Portfolio Selection - Programmatic Entry Point

This script demonstrates how to use the PortfolioGraph for
multi-stock analysis and portfolio construction.

Example Usage:
    python portfolio_main.py

    # Or with custom parameters:
    from tradingagents.portfolio import PortfolioGraph
    pg = PortfolioGraph(config=my_config)
    state = pg.analyze_portfolio(tickers, date, portfolio_size, risk_tolerance)
"""

import os
import json
from datetime import datetime, timedelta
from pprint import pprint

from tradingagents.portfolio import PortfolioGraph
from tradingagents.default_config import DEFAULT_CONFIG


def progress_callback(update):
    """Simple progress callback to print updates"""
    stage = update.get("stage", "")
    ticker = update.get("ticker", "")
    status = update.get("status", "")
    details = update.get("details", "")

    if ticker:
        print(f"[{stage.upper()}] {ticker}: {details}")
    else:
        print(f"[{stage.upper()}] {details}")


def main():
    """Run portfolio analysis"""
    print("=" * 60)
    print("  Portfolio Selection Tool - TradingAgents")
    print("=" * 60)
    print()

    # Configuration
    config = DEFAULT_CONFIG.copy()

    # Use Anthropic by default (can be changed)
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-20250514"
    config["quick_think_llm"] = "claude-haiku-4-5-20251001"

    # Portfolio parameters
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    portfolio_size = 100000
    risk_tolerance = "moderate"
    analysis_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Analyzing Portfolio:")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Portfolio Size: ${portfolio_size:,}")
    print(f"  Risk Tolerance: {risk_tolerance}")
    print(f"  Analysis Date: {analysis_date}")
    print()

    # Check API keys
    anthropic_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("WARNING: ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not set!")
        print("Please set your API key before running.")
        return

    # Create portfolio graph
    print("Initializing PortfolioGraph...")
    pg = PortfolioGraph(
        config=config,
        selected_analysts=["market", "news", "fundamentals"],
        max_parallel_analyses=2,
        debug=False,
    )
    pg.set_progress_callback(progress_callback)

    print()
    print("Starting portfolio analysis...")
    print("-" * 60)

    # Run analysis
    try:
        state = pg.analyze_portfolio(
            tickers=tickers,
            trade_date=analysis_date,
            portfolio_size_usd=portfolio_size,
            risk_tolerance=risk_tolerance,
            parallel=True,
        )
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

    print("-" * 60)
    print()

    # Print results
    print("=" * 60)
    print("  PORTFOLIO RECOMMENDATIONS")
    print("=" * 60)
    print()

    # Allocations
    print("Recommended Allocations:")
    print("-" * 40)
    allocations = state.get("final_allocations", [])
    for alloc in allocations:
        ticker = alloc.ticker if hasattr(alloc, 'ticker') else alloc.get("ticker", "?")
        weight = alloc.weight if hasattr(alloc, 'weight') else alloc.get("weight", 0)
        signal = alloc.signal if hasattr(alloc, 'signal') else alloc.get("signal", "HOLD")
        if hasattr(signal, 'value'):
            signal = signal.value
        dollar = weight * portfolio_size
        print(f"  {ticker:6s}  {weight:6.1%}  ${dollar:10,.0f}  [{signal}]")

    cash = state.get("portfolio_metrics", {})
    if hasattr(cash, 'cash_weight'):
        cash_weight = cash.cash_weight
    elif isinstance(cash, dict):
        cash_weight = cash.get("cash_weight", 0)
    else:
        cash_weight = 0
    print(f"  {'CASH':6s}  {cash_weight:6.1%}  ${cash_weight * portfolio_size:10,.0f}")
    print()

    # Metrics
    print("Portfolio Metrics:")
    print("-" * 40)
    metrics = state.get("portfolio_metrics", {})
    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()

    print(f"  Expected Return:  {metrics.get('expected_return', 0):6.1%}")
    print(f"  Volatility:       {metrics.get('portfolio_volatility', 0):6.1%}")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):6.2f}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):6.1%}")
    print()

    # Summary
    print("Portfolio Summary:")
    print("-" * 40)
    summary = state.get("portfolio_summary", "")
    if summary:
        print(summary[:500])
        if len(summary) > 500:
            print("...")
    print()

    # Benchmark comparison
    benchmark = state.get("benchmark_comparison", {})
    if benchmark and "error" not in benchmark:
        print("Benchmark Comparison (vs SPY):")
        print("-" * 40)
        print(f"  Portfolio Return:  {benchmark.get('portfolio_return', 0):6.1%}")
        print(f"  Benchmark Return:  {benchmark.get('benchmark_return', 0):6.1%}")
        print(f"  Excess Return:     {benchmark.get('excess_return', 0):6.1%}")
        print()

    # Save results
    summary = pg.get_allocation_summary()
    if summary:
        output_file = f"portfolio_results_{state.get('portfolio_id', 'output')}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")

    print()
    print("=" * 60)
    print("  Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
