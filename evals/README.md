# Evaluation Artifacts

This folder holds offline evaluation assets used by the capstone and book demos.

- `schema.json` defines the scenario format.
  Its `$defs/golden_trace` definition validates reference traces separately from scenarios.
- `scenarios.json` provides a small curated set of test cases.
- `scoring_rubric.md` defines how to score outcomes.
- `golden_traces.json` stores reference tool traces for regression checks.
- `scoring.py` computes the rubric's weighted score and policy hard-failure gate from four component scores.

These assets are intentionally small so they run fast in local tests and CI.
