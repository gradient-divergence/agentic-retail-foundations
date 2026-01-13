"""
Module: connectors.dummy_planogram_db

Provides a dummy in-memory planogram database for testing shelf monitoring agents.
"""

import asyncio
from typing import Any


class DummyPlanogramDB:
    """
    Dummy in-memory planogram database for shelf monitoring.
    """

    _cams = {"LOC1-SEC001": "CAM01", "LOC1-SEC002": "CAM02"}
    _pgs = {
        "LOC1-SEC001": {
            "products": [
                {
                    "product_id": "S1",
                    "expected_count": 5,
                    "position": {"x": 0.2, "y": 0.3},
                },
                {
                    "product_id": "S2",
                    "expected_count": 5,
                    "position": {"x": 0.6, "y": 0.3},
                },
                {
                    "product_id": "S3",
                    "expected_count": 3,
                    "position": {"x": 0.4, "y": 0.7},
                },
            ]
        },
        "LOC1-SEC002": {
            "products": [
                {
                    "product_id": "S4",
                    "expected_count": 4,
                    "position": {"x": 0.1, "y": 0.2},
                },
                {
                    "product_id": "S5",
                    "expected_count": 6,
                    "position": {"x": 0.5, "y": 0.5},
                },
            ]
        },
    }

    async def get_section_camera(self, location_id: str, section_id: str) -> str | None:
        """Get the camera ID for a given location and section."""
        await asyncio.sleep(0.01)
        key = f"{location_id}-{section_id}"
        return self._cams.get(key)

    async def get_section_planogram(self, location_id: str, section_id: str) -> dict[str, Any] | None:
        """Get the planogram for a given location and section."""
        await asyncio.sleep(0.01)
        key = f"{location_id}-{section_id}"
        return self._pgs.get(key)
