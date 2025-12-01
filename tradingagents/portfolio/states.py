"""Portfolio State Definitions for Multi-Stock Analysis and Selection"""

from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalType(str, Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(str, Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class StockAnalysisResult:
    """Result of individual stock analysis from TradingAgents"""
    ticker: str
    trade_date: str
    signal: SignalType
    confidence: float  # 0-1 confidence score

    # Analysis reports from different agents
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""

    # Debate and decision summaries
    investment_plan: str = ""
    trader_plan: str = ""
    final_decision: str = ""

    # Extracted metrics
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    current_price: Optional[float] = None

    # Risk metrics
    risk_level: RiskLevel = RiskLevel.MODERATE
    volatility: Optional[float] = None
    beta: Optional[float] = None

    # Fundamental metrics
    pe_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    dividend_yield: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None

    # Technical indicators
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    moving_avg_signal: Optional[str] = None

    # Sector and industry
    sector: str = ""
    industry: str = ""

    # Raw state (for detailed access)
    raw_state: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "signal": self.signal.value if isinstance(self.signal, SignalType) else self.signal,
            "confidence": self.confidence,
            "market_report": self.market_report,
            "sentiment_report": self.sentiment_report,
            "news_report": self.news_report,
            "fundamentals_report": self.fundamentals_report,
            "investment_plan": self.investment_plan,
            "trader_plan": self.trader_plan,
            "final_decision": self.final_decision,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "current_price": self.current_price,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            "volatility": self.volatility,
            "beta": self.beta,
            "pe_ratio": self.pe_ratio,
            "market_cap": self.market_cap,
            "dividend_yield": self.dividend_yield,
            "revenue_growth": self.revenue_growth,
            "profit_margin": self.profit_margin,
            "rsi": self.rsi,
            "macd_signal": self.macd_signal,
            "moving_avg_signal": self.moving_avg_signal,
            "sector": self.sector,
            "industry": self.industry,
        }


@dataclass
class PortfolioAllocation:
    """Allocation for a single stock in the portfolio"""
    ticker: str
    weight: float  # 0-1 percentage of portfolio
    shares: Optional[int] = None
    dollar_amount: Optional[float] = None
    signal: SignalType = SignalType.HOLD
    rationale: str = ""

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "weight": self.weight,
            "shares": self.shares,
            "dollar_amount": self.dollar_amount,
            "signal": self.signal.value if isinstance(self.signal, SignalType) else self.signal,
            "rationale": self.rationale,
        }


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance and risk metrics"""
    # Performance metrics
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0

    # Risk metrics
    portfolio_volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Diversification metrics
    sector_concentration: float = 0.0
    herfindahl_index: float = 0.0  # Concentration index
    correlation_avg: float = 0.0
    diversification_ratio: float = 0.0

    # Holdings summary
    total_positions: int = 0
    long_positions: int = 0
    cash_weight: float = 0.0

    # Sector allocation
    sector_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "expected_return": self.expected_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "information_ratio": self.information_ratio,
            "portfolio_volatility": self.portfolio_volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sector_concentration": self.sector_concentration,
            "herfindahl_index": self.herfindahl_index,
            "correlation_avg": self.correlation_avg,
            "diversification_ratio": self.diversification_ratio,
            "total_positions": self.total_positions,
            "long_positions": self.long_positions,
            "cash_weight": self.cash_weight,
            "sector_weights": self.sector_weights,
        }


class PortfolioDebateState(TypedDict):
    """State for portfolio-level allocation debate"""
    aggressive_history: Annotated[str, "Aggressive allocator's debate history"]
    conservative_history: Annotated[str, "Conservative allocator's debate history"]
    balanced_history: Annotated[str, "Balanced allocator's debate history"]
    history: Annotated[str, "Combined debate history"]

    current_aggressive_response: Annotated[str, "Latest aggressive allocator response"]
    current_conservative_response: Annotated[str, "Latest conservative allocator response"]
    current_balanced_response: Annotated[str, "Latest balanced allocator response"]

    proposed_allocations: Annotated[Dict[str, Dict], "Proposed allocations from each debater"]
    judge_decision: Annotated[str, "Portfolio manager's final decision"]
    count: Annotated[int, "Number of debate rounds"]


class PortfolioState(TypedDict):
    """Main state for portfolio selection and analysis workflow"""
    # Portfolio configuration
    portfolio_id: Annotated[str, "Unique identifier for this portfolio analysis"]
    analysis_date: Annotated[str, "Date of portfolio analysis"]
    portfolio_size_usd: Annotated[float, "Total portfolio value in USD"]
    risk_tolerance: Annotated[str, "Risk tolerance: low, moderate, high"]

    # Input tickers
    candidate_tickers: Annotated[List[str], "List of candidate stock tickers to analyze"]

    # Individual stock analyses
    stock_analyses: Annotated[
        Dict[str, StockAnalysisResult],
        "Analysis results for each ticker"
    ]

    # Analysis progress tracking
    analyzed_tickers: Annotated[List[str], "Tickers that have been analyzed"]
    pending_tickers: Annotated[List[str], "Tickers pending analysis"]
    failed_tickers: Annotated[List[str], "Tickers that failed analysis"]

    # Portfolio aggregation
    aggregation_report: Annotated[str, "Portfolio aggregator's summary report"]
    correlation_matrix: Annotated[Optional[Dict], "Correlation matrix between stocks"]
    sector_analysis: Annotated[str, "Sector concentration and diversification analysis"]

    # Portfolio debate
    portfolio_debate_state: Annotated[
        PortfolioDebateState,
        "State of portfolio allocation debate"
    ]

    # Final allocation
    final_allocations: Annotated[
        List[PortfolioAllocation],
        "Final portfolio allocation weights"
    ]
    portfolio_metrics: Annotated[PortfolioMetrics, "Portfolio-level metrics"]

    # Portfolio decision
    portfolio_summary: Annotated[str, "Executive summary of portfolio recommendation"]
    portfolio_rationale: Annotated[str, "Detailed rationale for allocation decisions"]

    # Rebalancing recommendations
    rebalancing_needed: Annotated[bool, "Whether portfolio rebalancing is recommended"]
    rebalancing_actions: Annotated[List[Dict], "List of rebalancing actions to take"]

    # Comparison with benchmarks
    benchmark_comparison: Annotated[Dict, "Comparison with market benchmarks"]

    # Historical context
    previous_allocations: Annotated[Optional[List[PortfolioAllocation]], "Previous portfolio allocations"]
    performance_history: Annotated[Optional[List[Dict]], "Historical performance data"]


# Factory function for creating initial portfolio state
def create_initial_portfolio_state(
    tickers: List[str],
    analysis_date: str,
    portfolio_size_usd: float = 100000.0,
    risk_tolerance: str = "moderate",
    portfolio_id: Optional[str] = None,
) -> PortfolioState:
    """Create initial portfolio state for analysis"""

    if portfolio_id is None:
        portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return PortfolioState(
        portfolio_id=portfolio_id,
        analysis_date=analysis_date,
        portfolio_size_usd=portfolio_size_usd,
        risk_tolerance=risk_tolerance,
        candidate_tickers=tickers,
        stock_analyses={},
        analyzed_tickers=[],
        pending_tickers=list(tickers),
        failed_tickers=[],
        aggregation_report="",
        correlation_matrix=None,
        sector_analysis="",
        portfolio_debate_state=PortfolioDebateState(
            aggressive_history="",
            conservative_history="",
            balanced_history="",
            history="",
            current_aggressive_response="",
            current_conservative_response="",
            current_balanced_response="",
            proposed_allocations={},
            judge_decision="",
            count=0,
        ),
        final_allocations=[],
        portfolio_metrics=PortfolioMetrics(),
        portfolio_summary="",
        portfolio_rationale="",
        rebalancing_needed=False,
        rebalancing_actions=[],
        benchmark_comparison={},
        previous_allocations=None,
        performance_history=None,
    )
