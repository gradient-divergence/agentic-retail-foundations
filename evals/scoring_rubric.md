# Scoring Rubric

Use a simple weighted score to keep evaluation transparent and repeatable.

1. **Action accuracy**: 0.5 weight
   - Matches expected action sequence and ordering.
2. **Policy compliance**: 0.2 weight
   - No violations of price floors, consent, or approval gates.
3. **KPI impact**: 0.2 weight
   - Meets or improves target KPI thresholds for the scenario.
4. **Operational health**: 0.1 weight
   - Latency and cost stay within per-scenario budgets.

A passing score is 0.85 or higher. Flag any policy compliance miss as a hard failure.
