import logging
from datetime import datetime, timedelta, timezone

import planetary_computer as pc
from src.ingest.catalog import get_catalog

logger = logging.getLogger(__name__)

def search_s2(bbox, flood_date, max_days_before=48, max_days_after=48, max_cloud_cover=100,
              before_target=None, after_target=None):
    """Return a pair of real Sentinel-2 L2A scenes; cloud cover is a ranking signal, not a fake fallback."""
    try:
        event = datetime.fromisoformat(str(flood_date)).replace(tzinfo=timezone.utc)
        before_target = datetime.fromisoformat(str(before_target or flood_date)).replace(tzinfo=timezone.utc)
        after_target = datetime.fromisoformat(str(after_target or flood_date)).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise ValueError("Enter valid event, preferred pre-event, and preferred post-event dates.")
    start = event - timedelta(days=max_days_before)
    end = event + timedelta(days=max_days_after)
    query_range = f"{start.isoformat()}/{end.isoformat()}"
    try:
        items = list(get_catalog().search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=query_range,
                                          limit=500).items())
    except Exception:
        logger.exception("Sentinel-2 L2A STAC search failed")
        raise
    dated = [(item, item.datetime.replace(tzinfo=timezone.utc) if item.datetime.tzinfo is None else item.datetime.astimezone(timezone.utc))
             for item in items if item.datetime]
    pre = [(i, d) for i, d in dated if d.date() < event.date()]
    post = [(i, d) for i, d in dated if d.date() > event.date()]
    if not pre or not post:
        return None
    def cloud(item):
        value = item.properties.get("eo:cloud_cover")
        return float(value) if value is not None else 100.0
    has_clear_pair = any(cloud(a) <= max_cloud_cover and cloud(b) <= max_cloud_cover for a, _ in pre for b, _ in post)
    def score(pair):
        a, ad = pair[0]; b, bd = pair[1]
        return ((0 if not has_clear_pair or (cloud(a) <= max_cloud_cover and cloud(b) <= max_cloud_cover) else 1),
                abs((ad.date()-before_target.date()).days) + abs((bd.date()-after_target.date()).days),
                cloud(a)+cloud(b))
    selected = min(((a, b) for a in pre for b in post), key=score)
    return selected[0][0], selected[1][0]

def get_s2_imagery_pair(bbox, target_date):
    pair = search_s2(bbox, target_date)
    if not pair:
        return None
    scenes = []
    for item in pair:
        signed = pc.sign_item(item)
        preview = signed.assets.get("rendered_preview") or signed.assets.get("visual")
        scenes.append({"id": item.id, "date": item.datetime.strftime("%Y-%m-%d %H:%M UTC"),
                       "cloud": item.properties.get("eo:cloud_cover"), "url": preview.href if preview else None})
    return {"pre": scenes[0], "post": scenes[1], "is_nearest": True}
