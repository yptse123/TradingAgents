"""Portfolio Allocator Agents - Different allocation strategies for portfolio debate"""

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import json
import re


AGGRESSIVE_ALLOCATOR_PROMPT = """You are an Aggressive Portfolio Allocator focused on maximizing returns through growth and momentum.

## Your Philosophy
- **Growth First**: Prioritize stocks with highest upside potential
- **Concentrated Bets**: Larger positions in high-conviction ideas
- **Momentum Driven**: Favor stocks with positive technical signals
- **Accept Volatility**: Higher risk tolerance for higher returns
- **Sector Agnostic**: Go where the opportunities are

## Portfolio Context
- **Analysis Date**: {analysis_date}
- **Portfolio Size**: ${portfolio_size_usd:,.2f}
- **Risk Tolerance**: {risk_tolerance}

## Aggregation Report
{aggregation_report}

## Stock Analyses Summary
{stock_analyses_summary}

## Previous Discussion
{debate_history}

## Your Task
Propose an aggressive allocation that:
1. Concentrates on top 3-5 highest-conviction stocks
2. Allocates up to 30% in single positions for best opportunities
3. Minimizes or eliminates defensive positions
4. Keeps cash allocation below 5%
5. Focuses on growth stocks with strong momentum

## Required Output Format
Provide your allocation in this exact JSON format, followed by your rationale:

```json
{{
    "allocations": {{
        "TICKER1": {{"weight": 0.25, "rationale": "reason"}},
        "TICKER2": {{"weight": 0.20, "rationale": "reason"}}
    }},
    "cash_weight": 0.05,
    "expected_return": 0.15,
    "expected_volatility": 0.25,
    "strategy_summary": "Brief strategy description"
}}
```

Then explain your aggressive allocation strategy and respond to any previous debate points.
"""


CONSERVATIVE_ALLOCATOR_PROMPT = """You are a Conservative Portfolio Allocator focused on capital preservation and risk management.

## Your Philosophy
- **Safety First**: Prioritize capital preservation over growth
- **Diversified**: Spread risk across many positions
- **Quality Focus**: Favor established companies with stable earnings
- **Income Oriented**: Value dividend-paying stocks
- **Defensive Sectors**: Overweight utilities, healthcare, consumer staples

## Portfolio Context
- **Analysis Date**: {analysis_date}
- **Portfolio Size**: ${portfolio_size_usd:,.2f}
- **Risk Tolerance**: {risk_tolerance}

## Aggregation Report
{aggregation_report}

## Stock Analyses Summary
{stock_analyses_summary}

## Previous Discussion
{debate_history}

## Your Task
Propose a conservative allocation that:
1. Diversifies across 5-8 stocks minimum
2. Limits single position size to 15% maximum
3. Overweights defensive sectors
4. Keeps cash buffer of 10-20%
5. Focuses on quality and stability over growth

## Required Output Format
Provide your allocation in this exact JSON format, followed by your rationale:

```json
{{
    "allocations": {{
        "TICKER1": {{"weight": 0.12, "rationale": "reason"}},
        "TICKER2": {{"weight": 0.10, "rationale": "reason"}}
    }},
    "cash_weight": 0.15,
    "expected_return": 0.08,
    "expected_volatility": 0.12,
    "strategy_summary": "Brief strategy description"
}}
```

Then explain your conservative allocation strategy and respond to any previous debate points.
"""


BALANCED_ALLOCATOR_PROMPT = """You are a Balanced Portfolio Allocator focused on optimal risk-adjusted returns.

## Your Philosophy
- **Risk-Adjusted Returns**: Maximize Sharpe ratio, not just returns
- **Core-Satellite**: Stable core with tactical satellite positions
- **Sector Balance**: Maintain diversification but allow tilts
- **Moderate Concentration**: Meaningful positions without excess risk
- **Adaptable**: Adjust to market conditions

## Portfolio Context
- **Analysis Date**: {analysis_date}
- **Portfolio Size**: ${portfolio_size_usd:,.2f}
- **Risk Tolerance**: {risk_tolerance}

## Aggregation Report
{aggregation_report}

## Stock Analyses Summary
{stock_analyses_summary}

## Previous Discussion
{debate_history}

## Your Task
Propose a balanced allocation that:
1. Holds 4-6 core positions (5-15% each)
2. Allows 1-2 tactical overweights up to 20%
3. Maintains sector diversification
4. Keeps cash at 5-10%
5. Balances growth and value stocks

## Required Output Format
Provide your allocation in this exact JSON format, followed by your rationale:

```json
{{
    "allocations": {{
        "TICKER1": {{"weight": 0.18, "rationale": "reason"}},
        "TICKER2": {{"weight": 0.15, "rationale": "reason"}}
    }},
    "cash_weight": 0.08,
    "expected_return": 0.12,
    "expected_volatility": 0.18,
    "strategy_summary": "Brief strategy description"
}}
```

Then explain your balanced allocation strategy, weighing the aggressive and conservative viewpoints.
"""


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON allocation from LLM response"""
    # Try to find JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r'\{[^{}]*"allocations"[^{}]*\{.*?\}.*?\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def format_stock_summary(stock_analyses: Dict) -> str:
    """Format stock analyses for allocator prompts"""
    lines = []
    for ticker, analysis in stock_analyses.items():
        if hasattr(analysis, 'to_dict'):
            data = analysis.to_dict()
        else:
            data = analysis

        line = f"- **{ticker}**: {data.get('signal', 'N/A')} ({data.get('confidence', 0):.0%} conf), Sector: {data.get('sector', 'N/A')}, Risk: {data.get('risk_level', 'N/A')}"
        lines.append(line)
    return "\n".join(lines) if lines else "No analyses available."


def create_aggressive_allocator(llm):
    """Create aggressive portfolio allocator agent"""

    def aggressive_allocator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGGRESSIVE_ALLOCATOR_PROMPT),
            ("human", "Propose your aggressive portfolio allocation.")
        ])

        chain = prompt | llm

        debate_state = state.get("portfolio_debate_state", {})

        result = chain.invoke({
            "analysis_date": state.get("analysis_date", "N/A"),
            "portfolio_size_usd": state.get("portfolio_size_usd", 100000),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "aggregation_report": state.get("aggregation_report", ""),
            "stock_analyses_summary": format_stock_summary(state.get("stock_analyses", {})),
            "debate_history": debate_state.get("history", "No previous discussion."),
        })

        response_content = result.content
        allocation_data = extract_json_from_response(response_content)

        # Update debate state
        new_debate_state = dict(debate_state)
        new_debate_state["current_aggressive_response"] = response_content
        new_debate_state["aggressive_history"] = (
            debate_state.get("aggressive_history", "") +
            f"\n\n### Aggressive Allocator (Round {debate_state.get('count', 0) + 1}):\n{response_content}"
        )
        new_debate_state["history"] = (
            debate_state.get("history", "") +
            f"\n\n### Aggressive Allocator:\n{response_content}"
        )

        if allocation_data:
            proposed = new_debate_state.get("proposed_allocations", {})
            proposed["aggressive"] = allocation_data
            new_debate_state["proposed_allocations"] = proposed

        return {
            "portfolio_debate_state": new_debate_state,
        }

    return aggressive_allocator_node


def create_conservative_allocator(llm):
    """Create conservative portfolio allocator agent"""

    def conservative_allocator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CONSERVATIVE_ALLOCATOR_PROMPT),
            ("human", "Propose your conservative portfolio allocation.")
        ])

        chain = prompt | llm

        debate_state = state.get("portfolio_debate_state", {})

        result = chain.invoke({
            "analysis_date": state.get("analysis_date", "N/A"),
            "portfolio_size_usd": state.get("portfolio_size_usd", 100000),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "aggregation_report": state.get("aggregation_report", ""),
            "stock_analyses_summary": format_stock_summary(state.get("stock_analyses", {})),
            "debate_history": debate_state.get("history", "No previous discussion."),
        })

        response_content = result.content
        allocation_data = extract_json_from_response(response_content)

        # Update debate state
        new_debate_state = dict(debate_state)
        new_debate_state["current_conservative_response"] = response_content
        new_debate_state["conservative_history"] = (
            debate_state.get("conservative_history", "") +
            f"\n\n### Conservative Allocator (Round {debate_state.get('count', 0) + 1}):\n{response_content}"
        )
        new_debate_state["history"] = (
            debate_state.get("history", "") +
            f"\n\n### Conservative Allocator:\n{response_content}"
        )

        if allocation_data:
            proposed = new_debate_state.get("proposed_allocations", {})
            proposed["conservative"] = allocation_data
            new_debate_state["proposed_allocations"] = proposed

        return {
            "portfolio_debate_state": new_debate_state,
        }

    return conservative_allocator_node


def create_balanced_allocator(llm):
    """Create balanced portfolio allocator agent"""

    def balanced_allocator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", BALANCED_ALLOCATOR_PROMPT),
            ("human", "Propose your balanced portfolio allocation, considering both aggressive and conservative viewpoints.")
        ])

        chain = prompt | llm

        debate_state = state.get("portfolio_debate_state", {})

        result = chain.invoke({
            "analysis_date": state.get("analysis_date", "N/A"),
            "portfolio_size_usd": state.get("portfolio_size_usd", 100000),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "aggregation_report": state.get("aggregation_report", ""),
            "stock_analyses_summary": format_stock_summary(state.get("stock_analyses", {})),
            "debate_history": debate_state.get("history", "No previous discussion."),
        })

        response_content = result.content
        allocation_data = extract_json_from_response(response_content)

        # Update debate state
        new_debate_state = dict(debate_state)
        new_debate_state["current_balanced_response"] = response_content
        new_debate_state["balanced_history"] = (
            debate_state.get("balanced_history", "") +
            f"\n\n### Balanced Allocator (Round {debate_state.get('count', 0) + 1}):\n{response_content}"
        )
        new_debate_state["history"] = (
            debate_state.get("history", "") +
            f"\n\n### Balanced Allocator:\n{response_content}"
        )
        new_debate_state["count"] = debate_state.get("count", 0) + 1

        if allocation_data:
            proposed = new_debate_state.get("proposed_allocations", {})
            proposed["balanced"] = allocation_data
            new_debate_state["proposed_allocations"] = proposed

        return {
            "portfolio_debate_state": new_debate_state,
        }

    return balanced_allocator_node
