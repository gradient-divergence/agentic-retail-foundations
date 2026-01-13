from __future__ import annotations

from statistics import mean

from pydantic import BaseModel

from capstone.tracing import TraceContext
from utils.monitoring import AgentMonitor


def simulate_agent_decision(inventory_risk: float) -> str:
    if inventory_risk >= 0.7:
        return "reorder"
    return "hold"


class EvalCase(BaseModel):
    inventory_risk: float
    expected: str


class EvalSuiteMetrics(BaseModel):
    pass_rate: float
    latency_ms: float
    cost_usd: float


def run_eval_suite() -> EvalSuiteMetrics:
    cases = [
        EvalCase(inventory_risk=0.9, expected="reorder"),
        EvalCase(inventory_risk=0.2, expected="hold"),
        EvalCase(inventory_risk=0.75, expected="reorder"),
        EvalCase(inventory_risk=0.4, expected="hold"),
    ]

    latencies = []
    passed = 0
    for case in cases:
        trace = TraceContext(route="capstone/eval", tenant_id="retail-demo")
        decision = simulate_agent_decision(case.inventory_risk)
        if decision == case.expected:
            passed += 1
        latencies.append(trace.finish()["latency_ms"])

    pass_rate = passed / len(cases)
    return EvalSuiteMetrics(
        pass_rate=pass_rate,
        latency_ms=mean(latencies),
        cost_usd=round(len(cases) * 0.02, 2),
    )


def main() -> None:
    metrics = run_eval_suite()
    monitor = AgentMonitor(
        agent_id="capstone-agent",
        metric_thresholds={
            "pass_rate": (0.9, 1.0),
            "latency_ms": (0.0, 250.0),
            "cost_usd": (0.0, 1.0),
        },
        alert_endpoints=[],
    )
    monitor.record_metrics(metrics.model_dump())
    print("AgentOps metrics:", metrics.model_dump())


if __name__ == "__main__":
    main()
