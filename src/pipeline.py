"""Run satellite discovery, image change analysis, and rescue exposure mapping."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from src.analysis import analyze_event
from src.ingest import s1, s2
from src.ingest.catalog import FALLBACK_STAC, PLANETARY_COMPUTER_STAC
from src.ingest.osm import fetch_osm_layers, validate_osm_date
from src.preprocess.s1_pairing import select_s1_pair, select_s2_pair

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def run(aoi: tuple[float, float, float, float] | dict[str, Any], flood_date: date,
        osm_date: date | None = None, output_root: Path | None = None,
        endpoints: tuple[str, ...] | None = None) -> Path:
    """Find event scenes, process image pixels, and write map-ready run artifacts."""
    if isinstance(aoi, dict):
        coords = aoi["geometry"]["coordinates"][0]
        bbox = (min(p[0] for p in coords), min(p[1] for p in coords),
                max(p[0] for p in coords), max(p[1] for p in coords))
        aoi_value: Any = aoi
    else:
        bbox, aoi_value = aoi, {"bbox": aoi}
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("AOI bbox must be west,south,east,north with positive area")
    osm_snapshot = osm_date or (flood_date - timedelta(days=1))
    validate_osm_date(osm_snapshot, flood_date)
    aoi_hash = hashlib.sha256(json.dumps(aoi_value, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"{flood_date.isoformat()}-{aoi_hash}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = (output_root or ROOT / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "aoi": aoi_value, "aoi_hash": aoi_hash, "flood_date": flood_date.isoformat(),
        "osm_snapshot_date": osm_snapshot.isoformat(), "inputs": [], "warnings": [], "status": "discovery"}
    paired = {}
    for sensor, query, selector in (("Sentinel-1", s1.query, select_s1_pair),
                                    ("Sentinel-2", s2.query, select_s2_pair)):
        try:
            catalog_kwargs = {"endpoints": endpoints} if endpoints else {
                "endpoints": (PLANETARY_COMPUTER_STAC,) if sensor == "Sentinel-1" else (FALLBACK_STAC,)}
            if sensor == "Sentinel-1":
                catalog_kwargs["collection"] = "sentinel-1-rtc"
            items = query(bbox, flood_date, timeout=15, limit=100, **catalog_kwargs)
            pair = selector(items, flood_date)
            paired[sensor] = pair
            for role, item in zip(("before", "after"), pair):
                props = item.get("properties", {})
                manifest["inputs"].append({"sensor": sensor, "role": role, "id": item.get("id"),
                    "collection": item.get("collection"), "platform": props.get("platform"),
                    "datetime": props.get("datetime"), "relative_orbit": props.get("sat:relative_orbit"),
                    "orbit_direction": props.get("sat:orbit_state"),
                    "cloud_cover_percent": props.get("eo:cloud_cover"),
                    "asset_keys": sorted(item.get("assets", {}).keys()),
                    "stac_api": "https://planetarycomputer.microsoft.com/api/stac/v1"})
        except Exception as exc:
            manifest["warnings"].append(f"{sensor} discovery/pairing unavailable: {exc}")
            LOG.warning("%s discovery/pairing unavailable: %s", sensor, exc)
    if not paired:
        manifest["status"] = "discovery_incomplete"
        manifest["note"] = "No usable pre/post pair was found; no image analysis was performed."
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_dir
    try:
        osm_layers = fetch_osm_layers(bbox, osm_snapshot, flood_date, timeout=45)
    except Exception as exc:
        osm_layers = {}
        manifest["warnings"].append(f"Pre-event OSM retrieval unavailable: {exc}")
    try:
        stats = analyze_event(paired, bbox, osm_layers, run_dir)
        manifest["analysis"] = stats
        manifest["warnings"].extend(stats["warnings"])
        manifest["status"] = "analysis_complete"
        manifest["note"] = stats["note"]
    except Exception as exc:
        manifest["status"] = "analysis_incomplete"
        manifest["warnings"].append(f"Pixel analysis unavailable: {exc}")
        LOG.exception("Pixel analysis failed")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir
