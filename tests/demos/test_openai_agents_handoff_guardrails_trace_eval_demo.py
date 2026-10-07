import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("discount,new_price", [(30, 70), (15, 70)])
def test_guardrail_checks_actual_price_reduction(discount, new_price):
    from demos.openai_agents_handoff_guardrails_trace_eval_demo import PriceProposal, guardrail_check

    result = guardrail_check(
        PriceProposal(product_id="SKU123", current_price=100, discount_pct=discount, new_price=new_price)
    )
    assert not result.passed


def test_planner_output_is_checked_before_handoff(monkeypatch, capsys):
    import demos.openai_agents_handoff_guardrails_trace_eval_demo as demo

    calls = []

    def run_sync(agent, task):
        calls.append(agent.name)
        return SimpleNamespace(
            final_output=json.dumps(
                dict(product_id="SKU123", current_price=49.99, discount_pct=80, new_price=9.99)
            )
        )

    monkeypatch.setattr(
        demo,
        "import_openai_agents_sdk",
        lambda: SimpleNamespace(
            Agent=lambda **kwargs: SimpleNamespace(**kwargs),
            function_tool=lambda function, **kwargs: SimpleNamespace(func=function, **kwargs),
            Runner=SimpleNamespace(run_sync=run_sync),
        ),
    )
    demo.run_demo()
    assert calls == ["PlannerAgent"]
    assert "Guardrail blocked handoff" in capsys.readouterr().out


def test_failed_execution_text_cannot_pass_eval():
    from demos.openai_agents_handoff_guardrails_trace_eval_demo import eval_output

    assert not eval_output("Price change was not executed.", "trace-1").passed
    assert not eval_output('{"status":"failed"}', "trace-1").passed
    assert eval_output('{"status":"executed"}', "trace-1").passed
