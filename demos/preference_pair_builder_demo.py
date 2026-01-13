#!/usr/bin/env python3
# region book:preference-pair-demo
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    candidate_a: str
    candidate_b: str
    score_a: float
    score_b: float


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    reason: str


def to_preference_pairs(cases: Iterable[EvalCase]) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for case in cases:
        if case.score_a == case.score_b:
            continue
        chosen, rejected = (
            (case.candidate_a, case.candidate_b)
            if case.score_a > case.score_b
            else (case.candidate_b, case.candidate_a)
        )
        pairs.append(
            PreferencePair(
                prompt=case.prompt,
                chosen=chosen,
                rejected=rejected,
                reason="preference_eval",
            )
        )
    return pairs


def run_example() -> None:
    cases = [
        EvalCase(
            prompt="Draft a compliant return response",
            candidate_a="Returns accepted within 30 days with receipt.",
            candidate_b="Sure, just send it back anytime.",
            score_a=0.92,
            score_b=0.31,
        )
    ]
    pairs = to_preference_pairs(cases)
    print([pair.__dict__ for pair in pairs])


if __name__ == "__main__":
    run_example()
# endregion book:preference-pair-demo
