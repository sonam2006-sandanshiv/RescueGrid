import logging
from datetime import datetime, timedelta, timezone

import planetary_computer as pc
from src.ingest.catalog import get_catalog
from src.preprocess.s1_pairing import select_s1_pair

logger = logging.getLogger(__name__)

def search_s1(bbox, flood_date, max_days_before=48, max_days_after=48, before_target=None, after_target=None):
    """Search real RTC scenes around the event and rank against preferred pre/post dates."""
    try:
        event = datetime.fromisoformat(str(flood_date)).replace(tzinfo=timezone.utc)
        before = datetime.fromisoformat(str(before_target or flood_date)).replace(tzinfo=timezone.utc)
        after = datetime.fromisoformat(str(after_target or flood_date)).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        logger.warning("Invalid Sentinel-1 date input")
        return None
    start = event - timedelta(days=max_days_before)
    end = event + timedelta(days=max_days_after)
    time_range = f"{start.isoformat()}/{end.isoformat()}"
    try:
        items = list(get_catalog().search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=time_range, limit=500).items())
        logger.info("Found %d Sentinel-1 RTC items in %s", len(items), time_range)
        return select_s1_pair(items, event.isoformat(), before.isoformat(), after.isoformat())
    except Exception:
        logger.exception("Sentinel-1 RTC STAC search failed")
        raise

def get_s1_imagery_pair(bbox, target_date):
    pair = search_s1(bbox, target_date)
    if not pair:
        return None
    before, after = pair
    scenes = []
    for item in pair:
        signed = pc.sign_item(item)
        preview = signed.assets.get("rendered_preview") or signed.assets.get("thumbnail")
        if preview is None:
            return None
        scenes.append({"id": item.id, "date": item.datetime.strftime("%Y-%m-%d %H:%M UTC"),
                       "orbit": item.properties.get("sat:relative_orbit", "N/A"), "url": preview.href})
    return {"pre": scenes[0], "post": scenes[1], "is_nearest": True}
