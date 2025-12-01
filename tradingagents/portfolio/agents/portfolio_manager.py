"""Portfolio Manager Agent - Makes final allocation decisions based on debate"""

from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import json
import re
from ..states import PortfolioAllocation, SignalType, PortfolioMetrics


PORTFOLIO_MANAGER_PROMPT = """You are the Chief Investment Officer (CIO) making the final portfolio allocation decision.

## Your Role
You have reviewed proposals from three allocation strategists:
1. **Aggressive Allocator**: Growth and momentum focused
2. **Conservative Allocator**: Capital preservation focused
3. **Balanced Allocator**: Risk-adjusted return focused

You must synthesize these viewpoints and make the final allocation decision that best serves the client's objectives.

## Portfolio Context
- **Analysis Date**: {analysis_date}
- **Portfolio Size**: ${portfolio_size_usd:,.2f}
- **Risk Tolerance**: {risk_tolerance}

## Aggregation Report
{aggregation_report}

## Allocation Proposals

### Aggressive Proposal
{aggressive_proposal}

### Conservative Proposal
{conservative_proposal}

### Balanced Proposal
{balanced_proposal}

## Your Decision Framework
1. **Weight by Risk Tolerance**:
   - Low risk: Lean toward conservative
   - Moderate: Weight balanced heavily
   - High risk: Consider aggressive positions

2. **Conviction Override**: If all three agree on a position, increase confidence

3. **Conflict Resolution**: When proposals conflict, use the balanced view as tiebreaker

4. **Sector Limits**: No sector should exceed 40% of portfolio

5. **Position Limits**:
   - Minimum position: 5%
   - Maximum position: 25%
   - Recommended positions: 4-8 stocks

## Required Output
Provide your final decision in this exact JSON format:

```json
{{
    "final_allocations": [
        {{"ticker": "AAPL", "weight": 0.20, "signal": "BUY", "rationale": "..."}},
        {{"ticker": "MSFT", "weight": 0.15, "signal": "BUY", "rationale": "..."}}
    ],
    "cash_weight": 0.10,
    "portfolio_metrics": {{
        "expected_return": 0.12,
        "expected_volatility": 0.18,
        "sharpe_estimate": 0.67
    }},
    "portfolio_summary": "Executive summary of portfolio strategy",
    "key_risks": ["Risk 1", "Risk 2"],
    "rebalancing_triggers": ["Trigger 1", "Trigger 2"]
}}
```

Then provide your detailed rationale explaining:
1. Why you chose this allocation over alternatives
2. Key positions and their importance
3. How this serves the client's risk tolerance
4. Monitoring recommendations
"""


def create_portfolio_manager(llm):
    """Create the portfolio manager agent that makes final allocation decisions"""

    def extract_final_allocation(response: str) -> Dict:
        """Extract final allocation from response"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    def portfolio_manager_node(state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", PORTFOLIO_MANAGER_PROMPT),
            ("human", "Please make your final portfolio allocation decision.")
        ])

        chain = prompt | llm

        debate_state = state.get("portfolio_debate_state", {})
        proposed = debate_state.get("proposed_allocations", {})

        result = chain.invoke({
            "analysis_date": state.get("analysis_date", "N/A"),
            "portfolio_size_usd": state.get("portfolio_size_usd", 100000),
            "risk_tolerance": state.get("risk_tolerance", "moderate"),
            "aggregation_report": state.get("aggregation_report", ""),
            "aggressive_proposal": debate_state.get("current_aggressive_response", "No proposal."),
            "conservative_proposal": debate_state.get("current_conservative_response", "No proposal."),
            "balanced_proposal": debate_state.get("current_balanced_response", "No proposal."),
        })

        response_content = result.content
        allocation_data = extract_final_allocation(response_content)

        # Build final allocations
        final_allocations = []
        portfolio_size = state.get("portfolio_size_usd", 100000)

        if allocation_data and "final_allocations" in allocation_data:
            for alloc in allocation_data["final_allocations"]:
                weight = alloc.get("weight", 0)
                signal_str = alloc.get("signal", "HOLD").upper()

                # Map signal string to SignalType
                signal_map = {
                    "STRONG_BUY": SignalType.STRONG_BUY,
                    "BUY": SignalType.BUY,
                    "HOLD": SignalType.HOLD,
                    "SELL": SignalType.SELL,
                    "STRONG_SELL": SignalType.STRONG_SELL,
                }
                signal = signal_map.get(signal_str, SignalType.HOLD)

                final_allocations.append(PortfolioAllocation(
                    ticker=alloc.get("ticker", ""),
                    weight=weight,
                    dollar_amount=weight * portfolio_size,
                    signal=signal,
                    rationale=alloc.get("rationale", ""),
                ))

        # Build portfolio metrics
        metrics_data = allocation_data.get("portfolio_metrics", {})
        portfolio_metrics = PortfolioMetrics(
            expected_return=metrics_data.get("expected_return", 0.0),
            portfolio_volatility=metrics_data.get("expected_volatility", 0.0),
            sharpe_ratio=metrics_data.get("sharpe_estimate", 0.0),
            total_positions=len(final_allocations),
            long_positions=len([a for a in final_allocations if a.signal in [SignalType.BUY, SignalType.STRONG_BUY]]),
            cash_weight=allocation_data.get("cash_weight", 0.0),
        )

        # Update debate state with judge decision
        new_debate_state = dict(debate_state)
        new_debate_state["judge_decision"] = response_content

        return {
            "portfolio_debate_state": new_debate_state,
            "final_allocations": final_allocations,
            "portfolio_metrics": portfolio_metrics,
            "portfolio_summary": allocation_data.get("portfolio_summary", ""),
            "portfolio_rationale": response_content,
            "rebalancing_needed": bool(allocation_data.get("rebalancing_triggers")),
            "rebalancing_actions": [
                {"trigger": t} for t in allocation_data.get("rebalancing_triggers", [])
            ],
        }

    return portfolio_manager_node
