"""Reproducible SIMULATION of SKU-123's next Stage 1 -> 2 gate review.

Criteria quoted from ``summary-and-future-directions.qmd``:
    "value over a measured baseline across a representative window, within
    availability, capital, waste, and workload limits; initial policies;
    incident response exercised"
    "Before the review, the owner fixes the baseline, the exposure window,
    value and risk thresholds with their uncertainty, the sign-offs, and the
    rollback triggers."

The chapter supplies no numerical limits. This fixture fixes illustrative
limits BEFORE measuring: positive lower 95% bound on net contribution gain;
nonnegative lower bound on fill-rate change; nonpositive upper bounds on
leftover capital, waste units, and review-workload changes. Controls require
approval before every commitment, no unauthorized execution, exercised
pre-acceptance timeout recovery AND retry exhaustion, and duplicate suppression.
The synthetic window must include ordinary and holiday weekends and delays.
Basic data audit, sandbox, and named owner are local fixture prerequisites;
real owner sign-off and representative field evidence remain outside this demo.
Any failed gate holds the simulated pilot at Stage 1 with buyer approval intact.

Design fixed independently of the results:
* Baseline period: 32 independent SKU-123 store-weekends, seeds 1..32, under
  a fixed 30-unit reorder point and 80-unit order. Evaluation: seeds 1001..1064,
  32 ordinary and 32 holiday episodes, with both policies on EACH same scenario.
  These are independent inventory cycles, each opening with 20 units, not
  sequential weeks with a stock reset hidden in a continuous inventory ledger.
* Both receive the same forecast (30 ordinary / 60 holiday over two days), stock,
  and supplier state. These numbers stand in for ForecastAgent's output; no LLM
  or fitted forecast model runs here. The rule's fixed point represents one
  ordinary weekend of demand and does not change for holidays or delays.
* The agent implements Chapter 13's "Architecture and Agent Interactions":
  target = forecast demand over lead time + review period, plus safety stock;
  order = max(0, target - on_hand - on_order). It reuses ``InventoryAgent.decide``
  (``agents/inventory.py``) with that target and inventory position. A forecast
  is spread evenly across its two days and held constant beyond the weekend.
  Planned lead time is one day for stable suppliers, two for delayed suppliers,
  matching the capstone MDP's one-period delay adjustment; actual receipts still
  arrive on days 1..3. Only the stable/delayed signal is available at proposal,
  not the eventual receipt date or realized demand. Review period is one day;
  safety stock is a fixed illustrative 10 units, not a calibrated service target.
  Both are fixed before measurement, without searching for a passing verdict.
  Round targets up to whole units. Opening on-order is zero; no reservations,
  backorders, case packs, minimums, capacity, or expiry constraints are modeled.
* ``PolicyEvaluator.evaluate`` checks the EXACT proposed quantity at USD 75/unit:
  commitments at or above USD 5,000 require independent buyer approval. This
  Stage 1 pilot additionally retains buyer review even below that limit, as the
  quoted controls gate and Chapter 14's hold require. Every synthetic buyer
  approves the unchanged proposal via ``apply_supervisor_action``; no denials
  or approval delays are simulated. Smaller orders do not gain new authority.
* A local Random(seed) draws weekend demand uniformly within +/-50% of forecast,
  post-weekend demand uniformly 10..40, receipt day from (1,1,1,2,3), pre-acceptance
  failures from (0,0,1,3), and a duplicate receipt with probability 0.4. Weekend
  demand splits over days 1/2; day 3 is the post-holiday clearance window. Lost
  sales are not backlogged. All randomness is generated before either policy.
* USD price 100 and unit cost 75 match ``run_governed_episode``. Post-holiday
  leftovers sell at USD 60 (40% markdown); all remaining units are written off
  at zero salvage. Demand at that markdown is assumed, not an elasticity model.
  Leftover capital includes confirmed incoming units due on day 3. Sell-through
  is post-window sales / that exposure (zero when there is no exposure).
* Contribution = weekend and markdown revenue minus acquisition of ALL opening
  and received units, USD 20 receiving cost per successful order, USD 0.30 per
  unit at each of the two closing snapshots, and USD 1/minute labor. Acquisition
  already charges terminal waste; it is not charged twice. No stockout penalty
  is treated as cash. Buyer review is 5 minutes, each retry 2 minutes; the agent
  adds 1 audit minute plus USD 1/episode and USD 0.25/supplier attempt for tools.
  Both use the same buyer and incident procedures, avoiding a weakened baseline.
* At most 3 supplier attempts; failures occur BEFORE acceptance. Confirmation
  changes only incoming stock; a receipt changes on-hand exactly once. No real
  clocks, network calls, or automatic retry after ambiguous acceptance.
* Metrics are episode means; 95% paired Student-t intervals are computed on
  agent-minus-rule episode differences. They describe Monte Carlo uncertainty
  conditional on these assumptions, not uncertainty about a real retailer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import mean, stdev

from scipy.stats import t

from agents.inventory import InventoryAgent
from capstone.policies import PolicyEvaluator, PolicyInput, apply_supervisor_action
from demos.capstone_scaffold_demo import lookup_inventory_risk, run_governed_episode
from models.governance import SupervisorAction

INITIAL_STOCK, ORDER_UNITS = 20, 80
REORDER_POINT, REVIEW_PERIOD_DAYS, SAFETY_STOCK = 30, 1, 10
PRICE, UNIT_COST, MARKDOWN_PRICE = 100.0, 75.0, 60.0
BASELINE_SEEDS = tuple(range(1, 33))
PAIRED_SEEDS = tuple(range(1001, 1065))
METRICS = (
    "fill_rate",
    "leftover_units",
    "leftover_cost",
    "contribution",
    "post_sell_through",
    "waste_units",
    "workload_minutes",
)


@dataclass(frozen=True)
class Scenario:
    seed: int
    holiday: bool
    demand: int
    post_demand: int
    receipt_day: int
    failures: int
    duplicate: bool


def seeded_scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    holiday = bool(seed % 2)
    forecast = 60 if holiday else 30
    return Scenario(
        seed,
        holiday,
        rng.randint(forecast // 2, forecast * 3 // 2),
        rng.randint(10, 40),
        rng.choice((1, 1, 1, 2, 3)),
        rng.choice((0, 0, 1, 3)),
        rng.random() < 0.4,
    )


def proposed_order_quantity(
    scenario: Scenario, policy: str, *, on_hand: int = INITIAL_STOCK, on_order: int = 0
) -> int:
    position = on_hand + on_order
    if policy == "rule":
        return ORDER_UNITS if position < REORDER_POINT else 0
    if policy == "agent":
        daily_forecast = (60 if scenario.holiday else 30) / 2
        lead_time_days = 1 + int(scenario.receipt_day > 1)
        target = ceil(daily_forecast * (lead_time_days + REVIEW_PERIOD_DAYS) + SAFETY_STOCK)
        agent = InventoryAgent(reorder_threshold=target, max_capacity=target)
        agent.perceive({"stock_level": position})
        return int(agent.decide().get("amount", 0))
    raise ValueError("policy must be agent or rule")


def approved_order(scenario: Scenario, quantity: int, controls: dict[str, int]) -> bool:
    proposal_id = f"proposal-{scenario.seed}"
    proposal = PolicyInput(
        action="supplier_commitment",
        amount=quantity * UNIT_COST,
        currency="USD",
        risk_score=lookup_inventory_risk("SKU-123").risk_score,
        evidence=[proposal_id, "SKU-123", f"quantity={quantity}"],
    )
    decision = PolicyEvaluator().evaluate(proposal)
    controls["policy_stops"] += int(not decision.allowed)
    audit = []
    approved = apply_supervisor_action(
        decision,
        SupervisorAction(action="approve", reviewer="buyer-2"),
        requester="planner-1",
        trace_id=proposal_id,
        audit_log=audit,
        action=proposal.action,
        evidence=proposal.evidence,
    )
    controls["approvals"] += int(approved.allowed and audit[0].actor == "buyer-2")
    return approved.allowed


def execute_order(scenario: Scenario, controls: dict[str, int]) -> tuple[bool, int]:
    for attempt in range(3):
        if attempt:
            controls["retries"] += 1
        try:
            if attempt < scenario.failures:
                raise TimeoutError("synthetic failure before supplier acceptance")
            controls["recoveries"] += int(attempt > 0)
            return True, attempt + 1
        except TimeoutError:
            controls["timeouts"] += 1
    controls["exhausted"] += 1
    return False, 3


def simulate_episode(scenario: Scenario, policy: str) -> dict:
    controls = dict.fromkeys(
        (
            "policy_stops",
            "approvals",
            "retries",
            "duplicates_ignored",
            "timeouts",
            "exhausted",
            "recoveries",
            "unauthorized",
        ),
        0,
    )
    ordered, attempts = False, 0
    quantity = proposed_order_quantity(scenario, policy)
    if quantity and approved_order(scenario, quantity, controls):
        ordered, attempts = execute_order(scenario, controls)
    incoming = quantity if ordered else 0
    on_hand, sales, trace, receipts = INITIAL_STOCK, [], [], set()
    if ordered:
        trace.append(("confirmed", on_hand, incoming))
        controls["unauthorized"] += int(controls["approvals"] == 0)
    leftover = waste = post_sales = 0
    for day, demand in enumerate(
        (scenario.demand // 2, scenario.demand - scenario.demand // 2, scenario.post_demand), start=1
    ):
        if ordered and day == scenario.receipt_day:
            for _ in range(1 + scenario.duplicate):
                if scenario.seed in receipts:
                    controls["duplicates_ignored"] += 1
                    continue
                receipts.add(scenario.seed)
                on_hand += quantity
                incoming -= quantity
                trace.append(("received", on_hand, incoming))
        sold = min(demand, on_hand)
        on_hand -= sold
        sales.append(sold)
        if day == 2:
            leftover = on_hand + incoming
        if day == 3:
            post_sales, waste = sold, on_hand
    assert sum(sales) + waste == INITIAL_STOCK + (quantity if ordered else 0)
    assert incoming == 0 and len(receipts) == int(ordered)
    review_minutes = controls["approvals"] * 5 + controls["retries"] * 2 + int(policy == "agent")
    costs = (INITIAL_STOCK + quantity * ordered) * UNIT_COST
    costs += 20 * ordered + 0.30 * (leftover + waste) + review_minutes
    costs += (1 + 0.25 * attempts) if policy == "agent" else 0
    return dict(
        seed=scenario.seed,
        fill_rate=sum(sales[:2]) / scenario.demand,
        leftover_units=leftover,
        leftover_cost=leftover * UNIT_COST,
        contribution=sum(sales[:2]) * PRICE + post_sales * MARKDOWN_PRICE - costs,
        post_sell_through=post_sales / leftover if leftover else 0.0,
        waste_units=waste,
        workload_minutes=review_minutes,
        controls=controls,
        inventory_trace=trace,
    )


def paired_interval(differences: list[float]) -> tuple[float, float, float]:
    if len(differences) < 2:
        raise ValueError("At least two paired episodes are required")
    center = mean(differences)
    radius = float(t.ppf(0.975, len(differences) - 1)) * stdev(differences) / sqrt(len(differences))
    return center, center - radius, center + radius


def control_totals(runs: list[dict]) -> dict[str, int]:
    return {name: sum(run["controls"][name] for run in runs) for name in runs[0]["controls"]}


def evaluate_gates(paired: dict, controls: dict, scenarios: list[Scenario]) -> dict[str, bool]:
    return dict(
        value=paired["contribution"][1] > 0,
        availability=paired["fill_rate"][1] >= 0,
        capital=paired["leftover_cost"][2] <= 0,
        waste=paired["waste_units"][2] <= 0,
        workload=paired["workload_minutes"][2] <= 0,
        controls=(
            controls["unauthorized"] == 0
            and controls["approvals"] >= controls["policy_stops"] > 0
            and controls["retries"] > 0
            and controls["exhausted"] > 0
            and controls["recoveries"] > 0
            and controls["duplicates_ignored"] > 0
        ),
        window=(
            {s.holiday for s in scenarios} == {False, True} and any(s.receipt_day > 1 for s in scenarios)
        ),
    )


def run_review() -> dict:
    reference = run_governed_episode()
    assert (reference["selling_price"], reference["unit_cost"], reference["currency"]) == (
        PRICE,
        UNIT_COST,
        "USD",
    )
    baseline = [simulate_episode(seeded_scenario(seed), "rule") for seed in BASELINE_SEEDS]
    scenarios = [seeded_scenario(seed) for seed in PAIRED_SEEDS]
    rule = [simulate_episode(scenario, "rule") for scenario in scenarios]
    agent = [simulate_episode(scenario, "agent") for scenario in scenarios]
    paired = {
        name: paired_interval([a[name] - r[name] for a, r in zip(agent, rule, strict=True)])
        for name in METRICS
    }
    gates = evaluate_gates(paired, control_totals(agent), scenarios)
    failed = [name for name, passed in gates.items() if not passed]
    return dict(
        baseline=baseline,
        rule=rule,
        agent=agent,
        paired=paired,
        gates=gates,
        failed_gates=failed,
        verdict="hold" if failed else "advance",
    )


def print_means(label: str, runs: list[dict]) -> None:
    values = {name: mean(run[name] for run in runs) for name in METRICS}
    print(
        f"{label}: fill={100 * values['fill_rate']:.2f}%; "
        f"leftover={values['leftover_units']:.2f} units (USD {values['leftover_cost']:.2f}); "
        f"net=USD {values['contribution']:.2f}; post-sell={100 * values['post_sell_through']:.2f}%; "
        f"waste={values['waste_units']:.2f} units; workload={values['workload_minutes']:.2f} min"
    )


def print_paired(paired: dict) -> None:
    for name, (center, low, high) in paired.items():
        factor = 100 if name in {"fill_rate", "post_sell_through"} else 1
        unit = (
            "pp"
            if factor == 100
            else "USD"
            if name in {"leftover_cost", "contribution"}
            else "min"
            if name == "workload_minutes"
            else "units"
        )
        print(f"{name}: {center * factor:+.2f} {unit} [{low * factor:+.2f}, {high * factor:+.2f}]")


def print_controls_and_gates(review: dict) -> None:
    for key in ("baseline", "rule", "agent"):
        counts = control_totals(review[key])
        print(f"{key} controls: " + "; ".join(f"{name}={count}" for name, count in counts.items()))
    print(
        "Gates: "
        + "; ".join(f"{name}={'pass' if passed else 'fail'}" for name, passed in review["gates"].items())
    )


# region book:capstone-gate-review-run
def main() -> None:
    review = run_review()
    print("SIMULATION ONLY: SKU-123 Stage 1 -> 2 gate review (USD)")
    print("Baseline: 32 cycles, seeds 1..32; paired: 64 cycles, seeds 1001..1064.")
    print("Price=100.00; unit cost=75.00; markdown=60.00; terminal salvage=0.00.")
    print("Rule: 80 units below a 30-unit point; agent: forecast over L+R + 10, net of stock.")
    print("Forecast=30/60 per 2 days; L=1 stable/2 delayed days; R=1 day; opening on-order=0.")
    print("Episode means; leftover includes confirmed incoming stock.")
    for key, label in (("baseline", "baseline-rule"), ("rule", "paired-rule"), ("agent", "paired-agent")):
        print_means(label, review[key])
    print("Paired agent-minus-rule differences [95% Student-t interval]:")
    print_paired(review["paired"])
    print_controls_and_gates(review)
    print(f"VERDICT: {review['verdict']}; failed gates: {', '.join(review['failed_gates']) or 'none'}.")
    print("Synthetic owner: head of replenishment; buyer approval retained.")
    print("Real advancement still requires field evidence and owner sign-off.")


if __name__ == "__main__":
    main()
# endregion book:capstone-gate-review-run
