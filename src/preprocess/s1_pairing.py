import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def _dt(item):
    value = getattr(item, "datetime", None)
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

def select_s1_pair(items, flood_date, before_target=None, after_target=None):
    """Choose real pre/post RTC scenes; prefer a matching relative orbit when available."""
    event = datetime.fromisoformat(str(flood_date)).replace(tzinfo=timezone.utc)
    before_target = datetime.fromisoformat(str(before_target or flood_date)).replace(tzinfo=timezone.utc)
    after_target = datetime.fromisoformat(str(after_target or flood_date)).replace(tzinfo=timezone.utc)
    before = [(item, _dt(item)) for item in items if _dt(item) and _dt(item).date() < event.date()]
    after = [(item, _dt(item)) for item in items if _dt(item) and _dt(item).date() > event.date()]
    if not before or not after:
        return None
    pairs = []
    for pre, pre_dt in before:
        for post, post_dt in after:
            same_track = (pre.properties.get("sat:relative_orbit"), pre.properties.get("sat:orbit_state")) == (
                post.properties.get("sat:relative_orbit"), post.properties.get("sat:orbit_state"))
            score = (not same_track, abs((pre_dt.date()-before_target.date()).days) +
                     abs((post_dt.date()-after_target.date()).days),
                     abs((event.date()-pre_dt.date()).days) + abs((post_dt.date()-event.date()).days))
            pairs.append((score, pre, post))
    _, pre, post = min(pairs, key=lambda row: row[0])
    if (pre.properties.get("sat:relative_orbit"), pre.properties.get("sat:orbit_state")) != (
            post.properties.get("sat:relative_orbit"), post.properties.get("sat:orbit_state")):
        logger.warning("Using nearest real Sentinel-1 scenes from different relative orbits.")
    return pre, post
