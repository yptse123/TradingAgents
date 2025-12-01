"""Stock Screener Agent - Pre-filters stocks based on criteria before deep analysis"""

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import json
import re


SCREENER_PROMPT = """You are a Stock Screening Specialist responsible for filtering candidate stocks before detailed analysis.

## Your Role
Quickly evaluate stocks based on fundamental criteria and market conditions to:
1. Identify stocks worth deep analysis
2. Flag stocks that should be excluded
3. Categorize stocks by investment style (growth, value, income)
4. Prioritize analysis order

## Screening Criteria
Apply these filters based on the risk tolerance:

### For Low Risk Tolerance:
- Market cap > $10B (large cap only)
- P/E ratio < 25
- Dividend yield > 1%
- Beta < 1.2
- Exclude highly volatile sectors

### For Moderate Risk Tolerance:
- Market cap > $2B
- P/E ratio < 35
- No strict dividend requirement
- Beta < 1.5
- All sectors allowed

### For High Risk Tolerance:
- Market cap > $500M
- P/E ratio unrestricted (growth stocks OK)
- No dividend requirement
- Beta unrestricted
- All sectors including speculative

## Input Data
- **Candidate Tickers**: {tickers}
- **Risk Tolerance**: {risk_tolerance}
- **Portfolio Size**: ${portfolio_size:,.2f}
- **Analysis Date**: {analysis_date}

## Stock Data
{stock_data}

## Required Output
Provide your screening results in this JSON format:

```json
{{
    "recommended_for_analysis": [
        {{"ticker": "AAPL", "priority": 1, "category": "growth", "rationale": "..."}},
        {{"ticker": "MSFT", "priority": 2, "category": "value", "rationale": "..."}}
    ],
    "excluded": [
        {{"ticker": "XYZ", "reason": "Market cap too small"}}
    ],
    "screening_summary": "Summary of screening process and findings"
}}
```

Order recommended stocks by analysis priority (1 = highest priority).
"""


def create_stock_screener(llm, data_interface=None):
    """Create a stock screener agent that pre-filters candidates"""

    def get_quick_stock_data(tickers: List[str]) -> str:
        """Get quick fundamental data for screening"""
        data_lines = []

        for ticker in tickers:
            try:
                if data_interface:
                    # Try to get basic data
                    fundamentals = data_interface.get_fundamentals(ticker)
                    data_lines.append(f"**{ticker}**: {fundamentals[:500]}...")
                else:
                    data_lines.append(f"**{ticker}**: Data not available - will analyze anyway")
            except Exception as e:
                data_lines.append(f"**{ticker}**: Error fetching data - {str(e)[:100]}")

        return "\n".join(data_lines) if data_lines else "No data available"

    def stock_screener_node(state: Dict[str, Any]) -> Dict[str, Any]:
        tickers = state.get("candidate_tickers", [])

        if not tickers:
            return {
                "pending_tickers": [],
                "failed_tickers": [],
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system", SCREENER_PROMPT),
            ("human", "Please screen the candidate stocks and provide your recommendations.")
        ])

        chain = prompt | llm

        result = chain.invoke({
            "tickers": ", ".join(tickers),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "portfolio_size": state.get("portfolio_size_usd", 100000),
            "analysis_date": state.get("analysis_date", "N/A"),
            "stock_data": get_quick_stock_data(tickers),
        })

        response_content = result.content

        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        screening_data = {}
        if json_match:
            try:
                screening_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Get ordered tickers for analysis
        recommended = screening_data.get("recommended_for_analysis", [])
        excluded = screening_data.get("excluded", [])

        if recommended:
            ordered_tickers = [item["ticker"] for item in sorted(recommended, key=lambda x: x.get("priority", 999))]
        else:
            # If screening failed, analyze all tickers
            ordered_tickers = list(tickers)

        excluded_tickers = [item["ticker"] for item in excluded] if excluded else []

        return {
            "pending_tickers": ordered_tickers,
            "failed_tickers": excluded_tickers,
        }

    return stock_screener_node
