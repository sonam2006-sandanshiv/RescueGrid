"""Same relative orbit and pass-direction Sentinel-1 pair selection."""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any
from src.ingest.catalog import scene_datetime

LOG = logging.getLogger(__name__)


def select_s1_pair(items: list[dict[str, Any]], flood_date: date,
                   preferred_repeat_days: int = 12, allowed_repeats: tuple[int, ...] = (6, 12, 24)
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the closest pre/post pair sharing relative orbit and direction.

    The pair spans the event date and its acquisition gap is an allowed repeat
    interval. Missing orbit metadata is rejected rather than guessed.
    """
    groups: dict[tuple[Any, Any], dict[str, list[dict[str, Any]]]] = {}
    for item in items:
        props = item.get("properties", {})
        orbit = props.get("sat:relative_orbit")
        direction = props.get("sat:orbit_state")
        if orbit is None or not direction:
            LOG.warning("Rejecting %s: relative orbit or pass direction missing", item.get("id"))
            continue
        day = scene_datetime(item).date()
        side = "pre" if day <= flood_date else "post"
        groups.setdefault((orbit, str(direction).lower()), {"pre": [], "post": []})[side].append(item)
    candidates: list[tuple[tuple[int, int, int], dict[str, Any], dict[str, Any]]] = []
    for (orbit, direction), sides in groups.items():
        for pre in sides["pre"]:
            for post in sides["post"]:
                gap = (scene_datetime(post).date() - scene_datetime(pre).date()).days
                if gap not in allowed_repeats:
                    continue
                score = (abs(gap - preferred_repeat_days), abs((flood_date - scene_datetime(pre).date()).days - preferred_repeat_days), gap)
                candidates.append((score, pre, post))
    if not candidates:
        LOG.error("No valid same-orbit/pass S1 pair found; mismatched tracks are never differenced")
        raise ValueError("No pre/post Sentinel-1 pair on the same relative orbit and pass direction with an allowed 6/12/24-day gap")
    candidates.sort(key=lambda candidate: (candidate[0], scene_datetime(candidate[1]), candidate[1].get("id", "")))
    _, pre, post = candidates[0]
    LOG.info("Selected same-track S1 pair %s -> %s (orbit=%s, pass=%s, gap=%s days)",
             pre.get("id"), post.get("id"), pre["properties"].get("sat:relative_orbit"),
             pre["properties"].get("sat:orbit_state"), (scene_datetime(post).date()-scene_datetime(pre).date()).days)
    return pre, post


def _s2_tile(item: dict[str, Any]) -> str | None:
    props = item.get("properties", {})
    for key in ("s2:mgrs_tile", "mgrs:tile", "grid:code"):
        if props.get(key): return str(props[key]).upper()
    match = re.search(r"_T(\d{2}[A-Z]{3})_", str(item.get("id", "")))
    return match.group(1) if match else None


def select_s2_pair(items: list[dict[str, Any]], flood_date: date,
                   max_distance_days: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose least-cloudy near-event scenes from the same MGRS tile.

    Refuse distant post-event scenes and mismatched tiles: either can create an
    apparently good comparison from changed seasonal conditions or coverage.
    """
    pre = [x for x in items if 0 <= (flood_date-scene_datetime(x).date()).days <= max_distance_days]
    post = [x for x in items if 0 < (scene_datetime(x).date()-flood_date).days <= max_distance_days]
    if not pre or not post:
        raise ValueError(f"No Sentinel-2 pre/post scene within {max_distance_days} days of flood date")
    cloud = lambda item: float(item.get("properties", {}).get("eo:cloud_cover", 100.0))
    candidates = []
    for before in pre:
        for after in post:
            before_tile, after_tile = _s2_tile(before), _s2_tile(after)
            if before_tile != after_tile:
                continue
            temporal_distance = abs((flood_date-scene_datetime(before).date()).days) + abs((scene_datetime(after).date()-flood_date).days)
            candidates.append(((cloud(before)+cloud(after), temporal_distance), before, after))
    if not candidates:
        raise ValueError("No near-event Sentinel-2 pre/post pair shares an MGRS tile")
    candidates.sort(key=lambda row: (row[0], str(row[1].get("id", "")), str(row[2].get("id", ""))))
    return candidates[0][1], candidates[0][2]
