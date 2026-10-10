import logging
from src.ingest.catalog import get_catalog

logger = logging.getLogger(__name__)

def search_dem(bbox):
    """
    Queries Microsoft Planetary Computer for Copernicus DEM GLO-30.
    """
    catalog = get_catalog()
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox
    )
    
    items = list(search.items())
    logger.info(f"Found {len(items)} DEM tiles for bbox {bbox}")
    
    return items
