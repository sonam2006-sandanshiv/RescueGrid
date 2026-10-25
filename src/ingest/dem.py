"""DEM source metadata helpers; download is provider-configured."""
from __future__ import annotations
from typing import Any


def find_dem_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only catalog items explicitly identifying Copernicus GLO-30."""
    return [item for item in items if "copernicus" in str(item.get("collection", "")).lower()
            or "copernicus" in str(item.get("properties", {}).get("title", "")).lower()]
