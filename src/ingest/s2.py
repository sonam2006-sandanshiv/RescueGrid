"""Sentinel-2 L2A scene discovery."""
from datetime import date, timedelta
from src.ingest.catalog import FALLBACK_STAC, search_planetary_computer, search_stac
from src.preprocess.s1_pairing import select_s2_pair

def query(bbox: tuple[float, float, float, float], event: date, *,
          max_days_before: int = 48, max_days_after: int = 48,
          max_cloud_cover: float = 100, limit: int = 200, timeout: int = 60, **_kwargs):
    start, end = event-timedelta(days=max_days_before), event+timedelta(days=max_days_after)
    try:
        items = search_planetary_computer("sentinel-2-l2a", bbox, start, end,
                                          limit=limit, timeout=timeout)
    except Exception:
        # Element84 is an independent public STAC fallback with COG assets.
        items = search_stac("sentinel-2-l2a", bbox, start, end,
                            endpoints=(FALLBACK_STAC,), limit=limit, timeout=timeout)
    return [item for item in items
            if float(item.get("properties", {}).get("eo:cloud_cover", 100)) <= max_cloud_cover]


def search_s2(bbox: tuple[float, float, float, float], event: str | date,
              max_days_before: int = 48, max_days_after: int = 48,
              max_cloud_cover: float = 100, **kwargs):
    """Dashboard-compatible query returning the least-cloudy pre/post pair."""
    event_day = date.fromisoformat(event) if isinstance(event, str) else event
    items = query(bbox, event_day, max_days_before=max_days_before,
                  max_days_after=max_days_after, max_cloud_cover=max_cloud_cover, **kwargs)
    return select_s2_pair(items, event_day)
