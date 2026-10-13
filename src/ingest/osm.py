"""Pre-event OpenStreetMap layers from the ohsome history API."""
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

logger = logging.getLogger(__name__)
OHSOME_V2_API = "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"
COMBINED_FILTER = (
    "(building=* and building!=no and geometry:polygon) or "
    "(highway=* and geometry:line) or "
    "((bridge=yes or man_made=bridge) and (geometry:line or geometry:polygon)) or "
    "((amenity=hospital or amenity=clinic or healthcare=*) and "
    "(geometry:point or geometry:line or geometry:polygon)) or "
    "(place in (city,town,village,hamlet) and (geometry:point or geometry:polygon))"
)


def _validate_osm_date(osm_date: str, flood_date: str) -> None:
    if date.fromisoformat(osm_date) >= date.fromisoformat(flood_date):
        raise ValueError(f"OSM date ({osm_date}) must be strictly BEFORE flood date ({flood_date})")


def _geometry(row: dict[str, Any]):
    value = row.get("geometry")
    if value is None:
        value = row.get("geom")
    if value is None:
        value = row.get("wkb_geometry")
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, str):
        return wkb.loads(value, hex=True)
    return wkb.loads(value)


def fetch_osm_layers(bbox: tuple[float, float, float, float], snapshot_date: date,
                     flood_date: date, timeout: int = 40) -> dict[str, dict[str, Any]]:
    """Fetch one real pre-event GeoParquet snapshot and split it into map layers."""
    _validate_osm_date(snapshot_date.isoformat(), flood_date.isoformat())
    api_key = os.environ.get("OHSOME_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Historical OSM requires OHSOME_API_KEY; no synthetic features were used.")
    minx, miny, maxx, maxy = bbox
    aoi = {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny],
             [maxx, maxy], [minx, maxy], [minx, miny]]]}
    response = requests.post(OHSOME_V2_API, json={"aoi": aoi,
        "filter": COMBINED_FILTER, "time": snapshot_date.isoformat()},
        headers={"Authorization": api_key, "User-Agent": "FloodLensAI/1.0"}, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"ohsome snapshot failed ({response.status_code}): {response.text[:300]}")
    table = parquet.read_table(BytesIO(response.content))
    result = {name: {"type": "FeatureCollection", "features": []}
              for name in ("buildings", "roads", "bridges", "health", "settlements")}
    for row in table.to_pylist():
        try:
            geom = _geometry(row)
        except (TypeError, ValueError):
            continue
        if geom is None:
            continue
        raw_tags = row.get("tags") or {}
        if not isinstance(raw_tags, dict):
            try:
                raw_tags = dict(raw_tags)
            except (TypeError, ValueError):
                raw_tags = {}
        tags = {str(key): str(value) for key, value in raw_tags.items()}
        feature = {"type": "Feature", "geometry": mapping(geom),
                   "properties": {**tags, "osm_type": row.get("osm_type"), "osm_id": row.get("osm_id")}}
        if tags.get("building") not in (None, "no") and geom.geom_type in ("Polygon", "MultiPolygon"):
            result["buildings"]["features"].append(feature)
        if tags.get("highway") and geom.geom_type in ("LineString", "MultiLineString"):
            result["roads"]["features"].append(feature)
        if tags.get("bridge") == "yes" or tags.get("man_made") == "bridge":
            result["bridges"]["features"].append(feature)
        if tags.get("amenity") in ("hospital", "clinic") or "healthcare" in tags:
            result["health"]["features"].append(feature)
        if tags.get("place") in ("city", "town", "village", "hamlet"):
            result["settlements"]["features"].append(feature)
    logger.info("Fetched pre-event OSM layers at %s: %s", snapshot_date,
                {key: len(value["features"]) for key, value in result.items()})
    return result


def fetch_osm_data(bbox, flood_date, osm_date="2026-07-27"):
    """Legacy pipeline adapter that returns the real layers as one GeoDataFrame."""
    import geopandas as gpd
    layers = fetch_osm_layers(tuple(bbox), date.fromisoformat(osm_date),
                              date.fromisoformat(flood_date))
    features = [feature for collection in layers.values() for feature in collection["features"]]
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
