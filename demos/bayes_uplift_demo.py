#!/usr/bin/env python3
# region book:bayes-uplift-demo
from __future__ import annotations

import random

from pydantic import BaseModel, Field, model_validator


class AlphaBeta(BaseModel):
    alpha: float = Field(..., gt=0)
    beta: float = Field(..., gt=0)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class PromoOutcome(BaseModel):
    conversions: int = Field(..., ge=0)
    trials: int = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_conversions(self) -> PromoOutcome:
        if self.conversions > self.trials:
            raise ValueError("conversions cannot exceed trials")
        return self

    @property
    def rate(self) -> float:
        return self.conversions / self.trials


def bayesian_update(prior: AlphaBeta, outcome: PromoOutcome) -> AlphaBeta:
    return AlphaBeta(
        alpha=prior.alpha + outcome.conversions,
        beta=prior.beta + (outcome.trials - outcome.conversions),
    )


def estimate_uplift(control: PromoOutcome, treatment: PromoOutcome) -> float:
    return treatment.rate - control.rate


def simulate_outcome(trials: int, true_rate: float, rng: random.Random) -> PromoOutcome:
    conversions = sum(1 for _ in range(trials) if rng.random() < true_rate)
    return PromoOutcome(conversions=conversions, trials=trials)


if __name__ == "__main__":
    rng = random.Random(42)
    prior = AlphaBeta(alpha=2, beta=8)

    control = simulate_outcome(trials=200, true_rate=0.08, rng=rng)
    treatment = simulate_outcome(trials=200, true_rate=0.11, rng=rng)

    posterior = bayesian_update(prior, treatment)
    uplift = estimate_uplift(control, treatment)

    print("Posterior mean conversion rate:", round(posterior.mean(), 4))
    print("Estimated uplift:", round(uplift, 4))
# endregion book:bayes-uplift-demo
