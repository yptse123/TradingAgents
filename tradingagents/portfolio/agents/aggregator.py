"""Portfolio Aggregator Agent - Synthesizes individual stock analyses into portfolio insights"""

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

AGGREGATOR_PROMPT = """You are a Senior Portfolio Strategist responsible for synthesizing multiple stock analyses into a cohesive portfolio recommendation.

## Your Role
You analyze individual stock reports and identify:
1. **Cross-Stock Patterns**: Common themes, sector trends, and market signals
2. **Diversification Opportunities**: How stocks complement or overlap each other
3. **Risk Correlations**: Which stocks move together and which provide hedging
4. **Sector Concentration**: Identify over/under-exposure to specific sectors
5. **Quality Ranking**: Rank stocks by conviction level based on analysis quality

## Portfolio Context
- **Analysis Date**: {analysis_date}
- **Portfolio Size**: ${portfolio_size_usd:,.2f}
- **Risk Tolerance**: {risk_tolerance}
- **Number of Candidates**: {num_candidates}

## Individual Stock Analyses
{stock_analyses_summary}

## Sector Distribution
{sector_distribution}

## Your Tasks
1. **Synthesize Findings**: Identify key investment themes across all analyses
2. **Rank Opportunities**: Order stocks by investment attractiveness
3. **Identify Conflicts**: Note any contradictory signals between stocks
4. **Assess Correlations**: Estimate which stocks are likely correlated
5. **Recommend Focus**: Suggest which stocks deserve higher allocation

## Output Format
Provide your aggregation report with:

### 1. Market Theme Summary
[Key themes emerging from the analyses]

### 2. Stock Rankings (by conviction)
| Rank | Ticker | Signal | Confidence | Sector | Key Catalyst |
|------|--------|--------|------------|--------|--------------|
[Ranked list of stocks]

### 3. Sector Analysis
[Sector distribution and concentration risks]

### 4. Correlation Insights
[Expected correlations between stocks]

### 5. Diversification Assessment
[How well this set of stocks diversifies]

### 6. Risk Factors
[Key risks across the portfolio candidates]

### 7. Top Recommendations
[Your top 3-5 stocks with brief rationale]

Be quantitative where possible. Reference specific data from the analyses.
"""


def create_portfolio_aggregator(llm):
    """Create a portfolio aggregator agent that synthesizes stock analyses"""

    def format_stock_analyses(stock_analyses: Dict) -> str:
        """Format individual stock analyses for the prompt"""
        summaries = []
        for ticker, analysis in stock_analyses.items():
            if hasattr(analysis, 'to_dict'):
                data = analysis.to_dict()
            else:
                data = analysis

            summary = f"""
### {ticker}
- **Signal**: {data.get('signal', 'N/A')} (Confidence: {data.get('confidence', 0):.1%})
- **Sector**: {data.get('sector', 'Unknown')} | Industry: {data.get('industry', 'Unknown')}
- **Price Target**: ${data.get('price_target', 'N/A')} | Current: ${data.get('current_price', 'N/A')}
- **Risk Level**: {data.get('risk_level', 'N/A')}
- **Key Metrics**: P/E: {data.get('pe_ratio', 'N/A')}, RSI: {data.get('rsi', 'N/A')}, Beta: {data.get('beta', 'N/A')}
- **Investment Thesis**: {(data.get('investment_plan', '') or data.get('final_decision', ''))[:500]}...
"""
            summaries.append(summary)
        return "\n".join(summaries) if summaries else "No stock analyses available."

    def get_sector_distribution(stock_analyses: Dict) -> str:
        """Calculate sector distribution"""
        sectors = {}
        for ticker, analysis in stock_analyses.items():
            if hasattr(analysis, 'sector'):
                sector = analysis.sector or "Unknown"
            elif isinstance(analysis, dict):
                sector = analysis.get('sector', 'Unknown')
            else:
                sector = "Unknown"
            sectors[sector] = sectors.get(sector, []) + [ticker]

        lines = []
        for sector, tickers in sorted(sectors.items(), key=lambda x: -len(x[1])):
            lines.append(f"- **{sector}**: {', '.join(tickers)} ({len(tickers)} stocks)")
        return "\n".join(lines) if lines else "No sector data available."

    def portfolio_aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio aggregation"""
        stock_analyses = state.get("stock_analyses", {})

        prompt = ChatPromptTemplate.from_messages([
            ("system", AGGREGATOR_PROMPT),
            ("human", "Please provide your portfolio aggregation analysis based on the stock analyses provided.")
        ])

        chain = prompt | llm

        result = chain.invoke({
            "analysis_date": state.get("analysis_date", "N/A"),
            "portfolio_size_usd": state.get("portfolio_size_usd", 100000),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "num_candidates": len(stock_analyses),
            "stock_analyses_summary": format_stock_analyses(stock_analyses),
            "sector_distribution": get_sector_distribution(stock_analyses),
        })

        return {
            "aggregation_report": result.content,
            "sector_analysis": get_sector_distribution(stock_analyses),
        }

    return portfolio_aggregator_node
