import pytest
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from src.damage.overlay import assess_damage

def test_assess_damage():
    # Synthetic OSM data
    osm_gdf = gpd.GeoDataFrame({
        "building": ["yes", None, None],
        "highway": [None, "residential", None],
        "bridge": [None, None, "yes"],
        "geometry": [
            Polygon([(0,0), (10,0), (10,10), (0,10)]), # building at 0,0
            LineString([(50,0), (50,100)]),            # road at x=50
            Polygon([(100,0), (110,0), (110,10), (100,10)]) # bridge at x=100
        ]
    }, crs="EPSG:32645")
    
    # Synthetic flood overlapping the road
    flood_gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])]
    }, crs="EPSG:32645")
    
    # Synthetic debris overlapping the bridge
    debris_gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(90, -10), (120, -10), (120, 20), (90, 20)])]
    }, crs="EPSG:32645")
    
    result_gdf, summary = assess_damage(osm_gdf, flood_gdf, debris_gdf, buffer_m=5)
    
    assert summary["buildings_affected"] == 0
    assert summary["bridges_affected"] == 1
    assert summary["road_km_affected"] > 0
    
    # Check statuses
    assert result_gdf.iloc[0]["status"] == "unaffected"
    assert result_gdf.iloc[1]["status"] == "affected"
    assert result_gdf.iloc[1]["damage_type"] == "flood"
    assert result_gdf.iloc[2]["status"] == "affected"
    assert result_gdf.iloc[2]["damage_type"] == "debris"
