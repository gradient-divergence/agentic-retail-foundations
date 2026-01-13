#!/usr/bin/env python3
from __future__ import annotations

from environments.inventory_risk_sim import (
    InventoryRiskSimConfig,
    InventoryRiskSimulator,
)


def main() -> None:
    config = InventoryRiskSimConfig()
    simulator = InventoryRiskSimulator(config, seed=21)
    metrics = simulator.run()
    print("Inventory risk simulation metrics:")
    print(metrics.model_dump())


if __name__ == "__main__":
    main()
