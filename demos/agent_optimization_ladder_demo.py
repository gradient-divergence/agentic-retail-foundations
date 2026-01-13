#!/usr/bin/env python3
# region book:optimization-ladder-demo
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class OptimizationStep(str, Enum):
    PROMPT = "prompt_tuning"
    DSPY = "automated_prompt_optimization"
    SFT = "supervised_finetuning"
    PEFT = "peft_lora"
    DPO = "preference_optimization"
    RLHF = "rlhf_or_rlaif"
    RFT = "reinforcement_fine_tuning"
    ONLINE = "guarded_online_learning"


@dataclass(frozen=True)
class OptimizationSignal:
    eval_gain: float
    data_volume: int
    preference_pairs: int
    simulator_ready: bool
    risk_budget: Literal["low", "medium", "high"]


def choose_optimization_step(signal: OptimizationSignal) -> OptimizationStep:
    if signal.eval_gain < 0.02:
        return OptimizationStep.PROMPT
    if signal.data_volume < 500:
        return OptimizationStep.DSPY
    if signal.data_volume < 5000:
        return OptimizationStep.PEFT
    if signal.preference_pairs >= 2000:
        return OptimizationStep.DPO
    if signal.risk_budget == "high" and signal.simulator_ready:
        return OptimizationStep.RFT
    if signal.risk_budget == "high" and signal.data_volume > 20000:
        return OptimizationStep.RLHF
    return OptimizationStep.SFT


def run_example() -> None:
    signal = OptimizationSignal(
        eval_gain=0.05,
        data_volume=8000,
        preference_pairs=0,
        simulator_ready=False,
        risk_budget="medium",
    )
    step = choose_optimization_step(signal)
    print({"next_step": step.value, "signal": signal})


if __name__ == "__main__":
    run_example()
# endregion book:optimization-ladder-demo
