"""Synthetic rule/planner/frozen Q-policy comparison on paired held-out seeds."""

import json
import random
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np

from demos.capstone_mdp_simulator_demo import Action, InventoryRiskMDP, baseline_action

ACTIONS = list(Action)
EVALUATION_SEEDS = tuple(range(1001, 1031))


def state_key(state):
    return (
        state.week,
        state.inventory_on_hand // 20,
        state.supplier_status.value,
        tuple(state.pipeline),
        state.promo_active,
    )


def plan_action(env, state):
    random_state = random.getstate()
    scores = {}
    try:
        for action in ACTIONS:
            random.seed(991)  # Common planning noise, independent of the evaluation episode.
            imagined = deepcopy(env)
            total = 0.0
            next_action = action
            for step in range(min(4, env.horizon_weeks - state.week)):
                next_state, reward, done, _ = imagined.step(next_action)
                total += 0.95**step * reward
                if done:
                    break
                next_action = baseline_action(imagined, next_state)
            scores[action] = total
    finally:
        random.setstate(random_state)
    return max(scores, key=scores.get)


def train_policy(episodes=500, seed=7):
    random.seed(seed)
    q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    for _ in range(episodes):
        env = InventoryRiskMDP()
        state = env.reset()
        while state.week < env.horizon_weeks:
            key = state_key(state)
            index = random.randrange(len(ACTIONS)) if random.random() < 0.2 else int(np.argmax(q[key]))
            next_state, reward, done, _ = env.step(ACTIONS[index])
            target = reward + (0 if done else 0.95 * float(np.max(q[state_key(next_state)])))
            q[key][index] += 0.1 * (target - q[key][index])
            state = next_state
    return {key: ACTIONS[int(np.argmax(values))] for key, values in q.items()}


def compare_policies(seeds=EVALUATION_SEEDS, episodes=500, max_reorder_units=80):
    if len(seeds) < 2 or episodes <= 0:
        raise ValueError("Comparison requires at least two held-out seeds and positive training episodes")
    policy = train_policy(episodes)
    records = {}
    for name in ("rule", "planner", "learned"):
        runs = []
        started = time.perf_counter()
        for seed in seeds:
            random.seed(seed)
            env = InventoryRiskMDP()
            state = env.reset()
            contribution = demand = sales = stockouts = reward_total = 0.0
            violations, trace = 0, []
            while state.week < env.horizon_weeks:
                action = (
                    baseline_action(env, state)
                    if name == "rule"
                    else plan_action(env, state)
                    if name == "planner"
                    else policy.get(state_key(state), Action.WAIT)
                )
                state, reward, _, info = env.step(action)
                contribution += info.contribution
                reward_total += reward
                demand += info.demand
                sales += info.sales
                stockouts += info.stockouts
                violations += int(action == Action.REORDER and env.reorder_qty > max_reorder_units)
                trace.append(
                    dict(
                        week=state.week,
                        action=action.value,
                        stockouts=info.stockouts,
                        inventory_on_hand=state.inventory_on_hand,
                        pipeline=list(state.pipeline),
                    )
                )
            runs.append(
                dict(
                    contribution=contribution,
                    net_reward=reward_total,
                    demand_units=demand,
                    sales_units=sales,
                    stockout_units=stockouts,
                    policy_violations=violations,
                    trace=trace,
                )
            )
        values = np.array([run["net_reward"] for run in runs])
        radius = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        records[name] = dict(
            mean_net_reward=float(values.mean()),
            confidence_interval=[float(values.mean() - radius), float(values.mean() + radius)],
            mean_contribution=float(np.mean([r["contribution"] for r in runs])),
            fill_rate=sum(r["sales_units"] for r in runs) / sum(r["demand_units"] for r in runs),
            mean_stockout_units=float(np.mean([r["stockout_units"] for r in runs])),
            policy_violations=sum(r["policy_violations"] for r in runs),
            runtime_seconds=time.perf_counter() - started,
            runs=runs,
        )
    paired = {}
    for name in ("planner", "learned"):
        differences = np.array(
            [
                a["net_reward"] - b["net_reward"]
                for a, b in zip(records[name]["runs"], records["rule"]["runs"], strict=True)
            ]
        )
        radius = float(1.96 * differences.std(ddof=1) / np.sqrt(len(differences)))
        paired[name] = dict(
            mean_gain=float(differences.mean()),
            confidence_interval=[float(differences.mean() - radius), float(differences.mean() + radius)],
        )
    lower = paired["learned"]["confidence_interval"][0]
    return dict(
        synthetic=True,
        currency="USD",
        guardrail=f"reorder_quantity<={max_reorder_units}",
        training_seed=7,
        evaluation_seeds=list(seeds),
        policies=records,
        paired_planner_minus_rule=paired["planner"],
        paired_learned_minus_rule=paired["learned"],
        paired_learned_minus_rule_lower=lower,
        decision="promote" if lower > 0 and records["learned"]["policy_violations"] == 0 else "retain_rule",
    )


if __name__ == "__main__":
    print(json.dumps(compare_policies(), indent=2))
