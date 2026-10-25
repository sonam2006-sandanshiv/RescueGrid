"""Terrain-corrected Sentinel-1 scene discovery."""
from datetime import date
from src.ingest.catalog import search_planetary_computer
from src.preprocess.s1_pairing import select_s1_pair

def query(bbox: tuple[float, float, float, float], event: date, *,
          max_days_before: int = 48, max_days_after: int = 48,
          limit: int = 200, timeout: int = 60, **_kwargs):
    """Return real Sentinel-1 RTC items; unsupported/un-calibrated assets are not substituted."""
    from datetime import timedelta
    return search_planetary_computer("sentinel-1-rtc", bbox,
        event - timedelta(days=max_days_before), event + timedelta(days=max_days_after),
        limit=limit, timeout=timeout)


def search_s1(bbox: tuple[float, float, float, float], event: str | date,
              max_days_before: int = 48, max_days_after: int = 48, **kwargs):
    """Dashboard-compatible query that returns a validated same-track pair."""
    from datetime import timedelta
    event_day = date.fromisoformat(event) if isinstance(event, str) else event
    items = query(bbox, event_day, max_days_before=max_days_before,
                  max_days_after=max_days_after, **kwargs)
    return select_s1_pair(items, event_day)
