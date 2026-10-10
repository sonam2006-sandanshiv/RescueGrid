import logging
from src.ingest.catalog import get_catalog

logger = logging.getLogger(__name__)

def search_s2(bbox, flood_date, max_days_before=15, max_days_after=15, max_cloud_cover=30):
    """
    Queries Microsoft Planetary Computer for Sentinel-2 L2A.
    """
    catalog = get_catalog()
    from datetime import datetime, timedelta, timezone
    flood_dt = datetime.fromisoformat(flood_date).replace(tzinfo=timezone.utc)
    
    start_dt = flood_dt - timedelta(days=max_days_before)
    end_dt = flood_dt + timedelta(days=max_days_after)
    time_range = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    
    items = list(search.items())
    logger.info(f"Found {len(items)} S2 items with cloud cover < {max_cloud_cover}%")
    
    pre_items = [i for i in items if i.datetime.replace(tzinfo=timezone.utc) <= flood_dt]
    post_items = [i for i in items if i.datetime.replace(tzinfo=timezone.utc) > flood_dt]
    
    if not pre_items or not post_items:
        logger.warning("Could not find both pre and post event clear S2 images.")
        return None
        
    best_pre = min(pre_items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    best_post = min(post_items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    
    return (best_pre, best_post)
