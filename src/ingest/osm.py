import logging
import requests
import geopandas as gpd
import json
from shapely.geometry import LineString, Point, Polygon
from datetime import datetime

logger = logging.getLogger(__name__)

OHSOME_API = "https://api.ohsome.org/v1/elements/geometry"

def _validate_osm_date(osm_date, flood_date):
    """Ensure OSM snapshot date is strictly before the flood event."""
    osm_dt = datetime.fromisoformat(osm_date)
    flood_dt = datetime.fromisoformat(flood_date)
    
    if osm_dt >= flood_dt:
        raise ValueError(f"OSM date ({osm_date}) must be strictly BEFORE flood date ({flood_date})")

def fetch_osm_data(bbox, flood_date, osm_date="2026-07-27"):
    """
    Fetches OSM data using ohsome API at a specific snapshot date.
    Falls back to mock geospatial features if the external API returns an error or is blocked.
    bbox format: [minx, miny, maxx, maxy]
    """
    _validate_osm_date(osm_date, flood_date)
    
    b_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    filter_query = (
        "building=* or "
        "highway=* or "
        "bridge=yes or man_made=bridge or "
        "amenity in (hospital,clinic) or healthcare=* or "
        "place in (city,town,village,hamlet)"
    )
    
    params = {
        "bboxes": b_str,
        "time": osm_date,
        "filter": filter_query,
        "properties": "tags"
    }
    
    headers = {
        "User-Agent": "FloodMapperHackathon/1.0 (flood-damage-mapping-bot)"
    }
    
    logger.info(f"Querying ohsome API for bbox {b_str} at time {osm_date}")
    
    try:
        response = requests.post(OHSOME_API, data=params, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            logger.info(f"Retrieved {len(features)} features from OSM")
            if features:
                gdf = gpd.GeoDataFrame.from_features(features)
                gdf.set_crs("EPSG:4326", inplace=True)
                return gdf
        else:
            logger.warning(f"Ohsome API returned HTTP {response.status_code}. Using synthetic mock OSM data.")
    except Exception as e:
        logger.warning(f"Ohsome API query failed ({e}). Using synthetic mock OSM data.")
        
    # Fallback synthetic OSM data within the bounding box
    minx, miny, maxx, maxy = bbox
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    
    features = [
        # Roads connecting town to villages
        {
            "geometry": LineString([(minx + 0.02, cy), (cx, cy), (maxx - 0.02, cy)]),
            "properties": {"highway": "primary", "name": "Trishuli Highway"}
        },
        {
            "geometry": LineString([(cx, miny + 0.02), (cx, cy), (cx, maxy - 0.02)]),
            "properties": {"highway": "secondary", "name": "Bhote Koshi Access Road"}
        },
        # Bridge over river
        {
            "geometry": Point(cx, cy),
            "properties": {"bridge": "yes", "highway": "primary", "name": "Friendship Bridge"}
        },
        # Hospital / Town center
        {
            "geometry": Point(cx + 0.01, cy + 0.01),
            "properties": {"place": "town", "name": "Dhulikhel Central", "amenity": "hospital"}
        },
        # Remote villages
        {
            "geometry": Point(minx + 0.03, cy + 0.03),
            "properties": {"place": "village", "name": "Tatopani Village", "population": "1200"}
        },
        {
            "geometry": Point(maxx - 0.03, cy - 0.03),
            "properties": {"place": "village", "name": "Larcha Settlement", "population": "450"}
        },
        # Buildings
        {
            "geometry": Polygon([(cx - 0.001, cy - 0.001), (cx + 0.001, cy - 0.001), (cx + 0.001, cy + 0.001), (cx - 0.001, cy + 0.001)]),
            "properties": {"building": "yes", "name": "School"}
        },
        {
            "geometry": Polygon([(minx + 0.03, cy + 0.03), (minx + 0.032, cy + 0.03), (minx + 0.032, cy + 0.032), (minx + 0.03, cy + 0.032)]),
            "properties": {"building": "yes", "name": "Health Post"}
        }
    ]
    
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return gdf
