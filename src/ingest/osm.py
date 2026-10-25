"""Historical OSM feature extraction for pre-event infrastructure snapshots."""
from __future__ import annotations

from datetime import date
from io import BytesIO
import logging
import os
from typing import Any

import pyarrow.parquet as parquet
import requests
from shapely import wkb
from shapely.geometry import mapping

LOG = logging.getLogger(__name__)
OHSOME_V2 = "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"
COMBINED_FILTER = (
    "(building=* and building!=no and geometry:polygon) or "
    "(highway=* and geometry:line) or "
    "((bridge=yes or man_made=bridge) and (geometry:line or geometry:polygon)) or "
    "((amenity=hospital or amenity=clinic or healthcare=*) and (geometry:point or geometry:line or geometry:polygon)) or "
    "(place in (city,town,village,hamlet) and (geometry:point or geometry:polygon))"
)


def validate_osm_date(snapshot_date: date, flood_date: date) -> None:
    if snapshot_date >= flood_date:
        raise ValueError(f"OSM snapshot date {snapshot_date} must be strictly earlier than flood date {flood_date}")


def fetch_osm_layers(bbox: tuple[float, float, float, float], snapshot_date: date,
                     flood_date: date, endpoint: str = OHSOME_V2, timeout: int = 90) -> dict[str, dict[str, Any]]:
    """Retrieve one pre-event GeoParquet snapshot and divide it into rescue layers."""
    validate_osm_date(snapshot_date, flood_date)
    api_key = os.environ.get("OHSOME_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Historical OSM requires OHSOME_API_KEY. Create a free ohsome API key and set it in .env.")
    response = requests.post(endpoint, json={"aoi": list(bbox), "filter": COMBINED_FILTER,
                                              "time": snapshot_date.isoformat()},
                             headers={"Authorization": api_key}, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"ohsome historical snapshot failed ({response.status_code}): {response.text[:400]}")
    table = parquet.read_table(BytesIO(response.content))
    result = {name: {"type": "FeatureCollection", "features": []}
              for name in ("buildings", "roads", "bridges", "health", "settlements")}
    for row in table.to_pylist():
        geometry_value = row.get("geometry")
        if not geometry_value:
            continue
        try:
            geom = wkb.loads(geometry_value)
        except (TypeError, ValueError):
            continue
        raw_tags = row.get("tags") or {}
        tags = dict(raw_tags) if not isinstance(raw_tags, dict) else raw_tags
        feature = {"type": "Feature", "geometry": mapping(geom),
                   "properties": {**tags, "osm_type": row.get("osm_type"), "osm_id": row.get("osm_id")}}
        if "building" in tags and tags.get("building") != "no" and geom.geom_type in ("Polygon", "MultiPolygon"):
            result["buildings"]["features"].append(feature)
        if "highway" in tags and geom.geom_type in ("LineString", "MultiLineString"):
            result["roads"]["features"].append(feature)
        if (tags.get("bridge") == "yes" or tags.get("man_made") == "bridge"):
            result["bridges"]["features"].append(feature)
        if tags.get("amenity") in ("hospital", "clinic") or "healthcare" in tags:
            result["health"]["features"].append(feature)
        if tags.get("place") in ("city", "town", "village", "hamlet"):
            result["settlements"]["features"].append(feature)
    LOG.info("Fetched pre-event OSM at %s: %s", snapshot_date,
             {key: len(value["features"]) for key, value in result.items()})
    return result
