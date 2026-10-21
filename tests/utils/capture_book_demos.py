"""Capture quoted offline demonstrations, including the paired capstone comparison.

Run from the companion root: python -m tests.utils.capture_book_demos /tmp/outputs.json
Wall times, audit UUIDs and live timestamps are runtime metadata, not seeded figures.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

DEMO_MODULES = (
    "bdi_inventory_demo_short",
    "ooda_pricing_demo_short",
    "bayes_uplift_demo",
    "bayesian_reco_book_demo",
    "capstone_mdp_simulator_demo",
    "mdp_dynamic_pricing_book_demo",
    "capstone_policy_comparison_demo",
    "store_fulfillment_book_demo",
    "structured_output_schema_demo",
    "tool_calling_rag_demo",
    "sensor_processor_demo",
    "promotion_causal_book_demo",
    "task_allocation_cnp_demo",
    "procurement_auction_demo_short",
    "inventory_sharing_demo_short",
    "capstone_scaffold_demo",
)


def capture(output: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "PYTHONPATH": str(root)}
    captures = {}
    for name in DEMO_MODULES:
        with TemporaryDirectory(prefix="book-demo-") as workdir:
            result = subprocess.run(
                [sys.executable, "-m", f"demos.{name}"],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        captures[name] = result.stdout
    output.write_text(json.dumps(captures, indent=2))
    return captures


if __name__ == "__main__":
    print(f"Captured {len(capture(Path(sys.argv[1])))} demos")  # noqa: T201 -- CLI result count.
