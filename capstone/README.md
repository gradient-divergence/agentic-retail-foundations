# Capstone Runbook

This folder holds the minimal capstone workflow building blocks. Use the demos below for deterministic runs that align with the book.

## Quick start

From the companion repo root:

```bash
python demos/capstone_scaffold_demo.py
python demos/capstone_gateway_demo.py
python demos/capstone_mdp_simulator_demo.py
python demos/agentops_eval_demo.py
```

## What each demo covers

- `capstone_scaffold_demo.py` demonstrates the orchestrator and tool gateway.
- `capstone_gateway_demo.py` runs a simple event flow with audit logging.
- `capstone_mdp_simulator_demo.py` simulates inventory risk as a sequential decision problem.
- `agentops_eval_demo.py` runs a tiny evaluation suite and records metrics.

## Determinism

- The MDP simulator demo sets a random seed.
- Other demos use fixed inputs and should be deterministic.

## Expected outputs

Each demo prints a short trace to stdout. Use these runs as smoke tests for alignment with the manuscript and contracts.
