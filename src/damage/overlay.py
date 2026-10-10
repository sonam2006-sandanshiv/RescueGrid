import geopandas as gpd
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def assess_damage(osm_gdf, flood_gdf, debris_gdf, buffer_m=20):
    """
    Intersect OSM features with flood and debris polygons.
    osm_gdf must be in a metric CRS (e.g., EPSG:32645 for Nepal).
    """
    if osm_gdf.empty:
        return osm_gdf
        
    # Buffer flood/debris to account for pixel uncertainty (10-20m)
    flood_buffered = gpd.GeoDataFrame(geometry=flood_gdf.buffer(buffer_m))
    debris_buffered = gpd.GeoDataFrame(geometry=debris_gdf.buffer(buffer_m))
    
    # Initialize status
    osm_gdf["status"] = "unaffected"
    osm_gdf["damage_type"] = "none"
    
    # Intersect with flood
    flood_intersect = gpd.sjoin(osm_gdf, flood_buffered, how="inner", predicate="intersects")
    osm_gdf.loc[flood_intersect.index, "status"] = "affected"
    osm_gdf.loc[flood_intersect.index, "damage_type"] = "flood"
    
    # Intersect with debris
    debris_intersect = gpd.sjoin(osm_gdf, debris_buffered, how="inner", predicate="intersects")
    osm_gdf.loc[debris_intersect.index, "status"] = "affected"
    osm_gdf.loc[debris_intersect.index, "damage_type"] = "debris"
    
    # Calculate affected lengths/counts
    buildings = osm_gdf[osm_gdf.get("building").notna()] if "building" in osm_gdf else gpd.GeoDataFrame()
    roads = osm_gdf[osm_gdf.get("highway").notna()] if "highway" in osm_gdf else gpd.GeoDataFrame()
    
    bridge_mask = (osm_gdf.get("bridge") == "yes") | (osm_gdf.get("man_made") == "bridge")
    bridges = osm_gdf[bridge_mask]

    
    affected_buildings = (buildings["status"] == "affected").sum()
    affected_bridges = (bridges["status"] == "affected").sum()
    
    # Road km affected (length is in meters, divide by 1000)
    affected_roads = roads[roads["status"] == "affected"]
    affected_road_km = affected_roads.geometry.length.sum() / 1000.0
    
    summary = {
        "buildings_affected": int(affected_buildings),
        "bridges_affected": int(affected_bridges),
        "road_km_affected": float(affected_road_km)
    }
    
    logger.info(f"Damage assessment: {summary}")
    return osm_gdf, summary
