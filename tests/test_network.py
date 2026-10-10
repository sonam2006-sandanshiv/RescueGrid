import pytest
import geopandas as gpd
from shapely.geometry import Point, LineString
from src.network.graph import build_road_graph
from src.network.cutoff import evaluate_settlements

def test_network_cutoff():
    # Synthetic dataset mimicking brief example: villages A/D reachable, B/C cutoff
    
    # Town at (0,0)
    # Village A at (10,0)
    # Village B at (30,0)
    # Village C at (40,0)
    # Village D at (0,20)
    
    # Roads: 
    # (0,0) to (10,0) [unaffected]
    # (10,0) to (30,0) [affected - flooded]
    # (30,0) to (40,0) [unaffected]
    # (0,0) to (0,20) [unaffected]
    
    osm_gdf = gpd.GeoDataFrame({
        "place": ["town", "village", "village", "village", "village", None, None, None, None],
        "name": ["Town", "Village A", "Village B", "Village C", "Village D", None, None, None, None],
        "amenity": [None]*9,
        "healthcare": [None]*9,
        "highway": [None, None, None, None, None, "residential", "residential", "residential", "residential"],
        "status": ["unaffected", "unaffected", "unaffected", "unaffected", "unaffected", 
                   "unaffected", "affected", "unaffected", "unaffected"],
        "geometry": [
            Point(0,0), Point(10,0), Point(30,0), Point(40,0), Point(0,20),
            LineString([(0,0), (10,0)]),
            LineString([(10,0), (30,0)]),
            LineString([(30,0), (40,0)]),
            LineString([(0,0), (0,20)])
        ]
    }, crs="EPSG:32645")
    
    G = build_road_graph(osm_gdf)
    results = evaluate_settlements(G, osm_gdf)
    
    # We should have 4 settlements (villages A,B,C,D) evaluated. Town is a destination.
    # Actually evaluate_settlements also includes towns in the 'settlements' list.
    
    for r in results:
        if r["name"] == "Village A":
            assert r["status"] == "reachable"
        elif r["name"] == "Village B":
            assert r["status"] == "cut off"
        elif r["name"] == "Village C":
            assert r["status"] == "cut off"
        elif r["name"] == "Village D":
            assert r["status"] == "reachable"
