"""
Module: connectors.dummy_inventory_system

Provides a dummy in-memory inventory system for testing shelf monitoring agents.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DummyInventorySystem:
    """
    Dummy in-memory inventory system for shelf monitoring.
    """

    _audit_reports: list[dict[str, Any]] = []

    async def report_visual_audit(self, audit_report: dict[str, Any]) -> None:
        """Store or print the visual audit report for testing."""
        await asyncio.sleep(0.01)
        self._audit_reports.append(audit_report)
        logger.info("[DummyInventorySystem] Visual audit report received: %s", audit_report)
