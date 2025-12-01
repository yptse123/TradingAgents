"""Portfolio Analytics and Metrics Calculation"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from .states import (
    PortfolioMetrics,
    PortfolioAllocation,
    StockAnalysisResult,
    SignalType,
    RiskLevel,
)


class PortfolioAnalytics:
    """Calculate portfolio-level metrics and analytics"""

    def __init__(self, lookback_days: int = 252):
        """
        Initialize portfolio analytics.

        Args:
            lookback_days: Number of trading days for historical calculations
        """
        self.lookback_days = lookback_days
        self.risk_free_rate = 0.05  # 5% annual risk-free rate

    def get_historical_prices(
        self,
        tickers: List[str],
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical prices for tickers"""
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = datetime.now()

        start = end - timedelta(days=self.lookback_days * 1.5)  # Extra buffer for holidays

        try:
            data = yf.download(
                tickers,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False,
            )
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif len(tickers) == 1:
                prices = data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                prices = data['Close']

            return prices.dropna()
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from prices"""
        return prices.pct_change().dropna()

    def calculate_correlation_matrix(
        self,
        tickers: List[str],
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between stocks"""
        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return {}

        returns = self.calculate_returns(prices)
        corr_matrix = returns.corr()

        # Convert to dict format
        result = {}
        for ticker in corr_matrix.columns:
            result[ticker] = corr_matrix[ticker].to_dict()

        return result

    def calculate_portfolio_volatility(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate portfolio volatility"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        returns = self.calculate_returns(prices)

        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252  # Annualize

        # Portfolio variance
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        return float(portfolio_vol)

    def calculate_expected_return(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate expected portfolio return based on historical data"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        returns = self.calculate_returns(prices)
        mean_returns = returns.mean() * 252  # Annualize

        expected_return = np.dot(weights, mean_returns.values)
        return float(expected_return)

    def calculate_sharpe_ratio(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate portfolio Sharpe ratio"""
        expected_return = self.calculate_expected_return(allocations, end_date)
        volatility = self.calculate_portfolio_volatility(allocations, end_date)

        if volatility == 0:
            return 0.0

        sharpe = (expected_return - self.risk_free_rate) / volatility
        return float(sharpe)

    def calculate_sortino_ratio(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate Sortino ratio (downside risk adjusted)"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        returns = self.calculate_returns(prices)

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Annualized return
        annual_return = portfolio_returns.mean() * 252

        # Downside deviation
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if len(negative_returns) == 0:
            return float('inf')

        downside_std = np.sqrt((negative_returns ** 2).mean()) * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        sortino = (annual_return - self.risk_free_rate) / downside_std
        return float(sortino)

    def calculate_max_drawdown(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate maximum drawdown"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        # Calculate portfolio value
        normalized_prices = prices / prices.iloc[0]
        portfolio_value = (normalized_prices * weights).sum(axis=1)

        # Calculate drawdown
        rolling_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - rolling_max) / rolling_max

        return float(drawdown.min())

    def calculate_var(
        self,
        allocations: List[PortfolioAllocation],
        confidence: float = 0.95,
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate Value at Risk"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        returns = self.calculate_returns(prices)
        portfolio_returns = (returns * weights).sum(axis=1)

        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        return float(var)

    def calculate_sector_weights(
        self,
        allocations: List[PortfolioAllocation],
        stock_analyses: Dict[str, StockAnalysisResult],
    ) -> Dict[str, float]:
        """Calculate sector weights in portfolio"""
        sector_weights = {}

        for alloc in allocations:
            if alloc.ticker in stock_analyses:
                analysis = stock_analyses[alloc.ticker]
                sector = analysis.sector if hasattr(analysis, 'sector') else "Unknown"
                if isinstance(analysis, dict):
                    sector = analysis.get('sector', 'Unknown')
            else:
                sector = "Unknown"

            sector_weights[sector] = sector_weights.get(sector, 0) + alloc.weight

        return sector_weights

    def calculate_herfindahl_index(
        self,
        allocations: List[PortfolioAllocation],
    ) -> float:
        """Calculate Herfindahl-Hirschman Index (concentration measure)"""
        if not allocations:
            return 0.0

        weights = [a.weight for a in allocations if a.weight > 0]
        hhi = sum(w ** 2 for w in weights)
        return float(hhi)

    def calculate_diversification_ratio(
        self,
        allocations: List[PortfolioAllocation],
        end_date: Optional[str] = None,
    ) -> float:
        """Calculate diversification ratio"""
        if not allocations:
            return 0.0

        tickers = [a.ticker for a in allocations if a.weight > 0]
        weights = np.array([a.weight for a in allocations if a.weight > 0])

        if len(tickers) == 0:
            return 0.0

        prices = self.get_historical_prices(tickers, end_date)
        if prices.empty:
            return 0.0

        returns = self.calculate_returns(prices)

        # Individual volatilities
        individual_vols = returns.std() * np.sqrt(252)

        # Weighted average individual vol
        weighted_vol = np.dot(weights, individual_vols.values)

        # Portfolio vol
        portfolio_vol = self.calculate_portfolio_volatility(allocations, end_date)

        if portfolio_vol == 0:
            return 0.0

        return float(weighted_vol / portfolio_vol)

    def calculate_full_metrics(
        self,
        allocations: List[PortfolioAllocation],
        stock_analyses: Dict[str, StockAnalysisResult],
        end_date: Optional[str] = None,
    ) -> PortfolioMetrics:
        """Calculate all portfolio metrics"""

        # Calculate all metrics
        expected_return = self.calculate_expected_return(allocations, end_date)
        volatility = self.calculate_portfolio_volatility(allocations, end_date)
        sharpe = self.calculate_sharpe_ratio(allocations, end_date)
        sortino = self.calculate_sortino_ratio(allocations, end_date)
        max_dd = self.calculate_max_drawdown(allocations, end_date)
        var_95 = self.calculate_var(allocations, 0.95, end_date)
        sector_weights = self.calculate_sector_weights(allocations, stock_analyses)
        hhi = self.calculate_herfindahl_index(allocations)
        div_ratio = self.calculate_diversification_ratio(allocations, end_date)

        # Calculate sector concentration (max sector weight)
        sector_concentration = max(sector_weights.values()) if sector_weights else 0.0

        # Calculate average correlation
        tickers = [a.ticker for a in allocations if a.weight > 0]
        corr_matrix = self.calculate_correlation_matrix(tickers, end_date)
        if corr_matrix:
            all_corrs = []
            for t1 in corr_matrix:
                for t2 in corr_matrix[t1]:
                    if t1 != t2:
                        all_corrs.append(corr_matrix[t1][t2])
            avg_corr = np.mean(all_corrs) if all_corrs else 0.0
        else:
            avg_corr = 0.0

        # Count positions
        total_positions = len([a for a in allocations if a.weight > 0])
        long_positions = len([
            a for a in allocations
            if a.weight > 0 and a.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        ])
        cash_weight = 1.0 - sum(a.weight for a in allocations)

        return PortfolioMetrics(
            expected_return=expected_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            information_ratio=0.0,  # Would need benchmark data
            portfolio_volatility=volatility,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=var_95 * 1.2,  # Approximate CVaR
            sector_concentration=sector_concentration,
            herfindahl_index=hhi,
            correlation_avg=avg_corr,
            diversification_ratio=div_ratio,
            total_positions=total_positions,
            long_positions=long_positions,
            cash_weight=cash_weight,
            sector_weights=sector_weights,
        )

    def get_benchmark_comparison(
        self,
        allocations: List[PortfolioAllocation],
        benchmark_ticker: str = "SPY",
        end_date: Optional[str] = None,
    ) -> Dict:
        """Compare portfolio to benchmark"""
        portfolio_return = self.calculate_expected_return(allocations, end_date)
        portfolio_vol = self.calculate_portfolio_volatility(allocations, end_date)
        portfolio_sharpe = self.calculate_sharpe_ratio(allocations, end_date)

        # Get benchmark data
        benchmark_prices = self.get_historical_prices([benchmark_ticker], end_date)
        if benchmark_prices.empty:
            return {
                "benchmark": benchmark_ticker,
                "error": "Could not fetch benchmark data",
            }

        benchmark_returns = self.calculate_returns(benchmark_prices)
        benchmark_annual_return = float(benchmark_returns.mean().iloc[0] * 252)
        benchmark_vol = float(benchmark_returns.std().iloc[0] * np.sqrt(252))
        benchmark_sharpe = (benchmark_annual_return - self.risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0

        return {
            "benchmark": benchmark_ticker,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_annual_return,
            "excess_return": portfolio_return - benchmark_annual_return,
            "portfolio_volatility": portfolio_vol,
            "benchmark_volatility": benchmark_vol,
            "portfolio_sharpe": portfolio_sharpe,
            "benchmark_sharpe": benchmark_sharpe,
            "information_ratio": (portfolio_return - benchmark_annual_return) / (portfolio_vol - benchmark_vol) if portfolio_vol != benchmark_vol else 0,
        }

    def suggest_rebalancing(
        self,
        current_allocations: List[PortfolioAllocation],
        target_allocations: List[PortfolioAllocation],
        threshold: float = 0.05,
    ) -> List[Dict]:
        """Suggest rebalancing actions based on drift from target"""
        actions = []

        current_dict = {a.ticker: a.weight for a in current_allocations}
        target_dict = {a.ticker: a.weight for a in target_allocations}

        all_tickers = set(current_dict.keys()) | set(target_dict.keys())

        for ticker in all_tickers:
            current_weight = current_dict.get(ticker, 0)
            target_weight = target_dict.get(ticker, 0)
            drift = target_weight - current_weight

            if abs(drift) > threshold:
                action = "BUY" if drift > 0 else "SELL"
                actions.append({
                    "ticker": ticker,
                    "action": action,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift": drift,
                    "priority": "HIGH" if abs(drift) > 0.10 else "MEDIUM",
                })

        # Sort by absolute drift
        actions.sort(key=lambda x: abs(x["drift"]), reverse=True)

        return actions
