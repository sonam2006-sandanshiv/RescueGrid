import logging
import geopandas as gpd
from shapely.geometry import box
from src.ingest.catalog import get_catalog
from src.preprocess.s1_pairing import select_s1_pair

logger = logging.getLogger(__name__)

def search_s1(bbox, flood_date, max_days_before=15, max_days_after=15):
    """
    Queries Microsoft Planetary Computer for Sentinel-1 GRD.
    """
    catalog = get_catalog()
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=f"{flood_date}T00:00:00Z/{flood_date}T00:00:00Z", # just placeholder, will do correct range
    )
    
    from datetime import datetime, timedelta, timezone
    flood_dt = datetime.fromisoformat(flood_date).replace(tzinfo=timezone.utc)
    start_dt = flood_dt - timedelta(days=max_days_before)
    end_dt = flood_dt + timedelta(days=max_days_after)
    time_range = f"{start_dt.isoformat()}/{end_dt.isoformat()}"
    
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=time_range,
        query={"sat:instrument_mode": {"eq": "IW"}}
    )
    
    items = list(search.items())
    logger.info(f"Found {len(items)} S1 items in range {time_range}")
    
    best_pair = select_s1_pair(items, flood_date)
    return best_pair
