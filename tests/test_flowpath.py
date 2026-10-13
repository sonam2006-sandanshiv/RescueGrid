import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.flowpath.trace import trace_flowpath, trace_downstream_path, list_settlements_along_flowpath

def test_trace_flowpath():
    dem = np.array([
        [100, 95, 90],
        [100, 90, 85],
        [100, 85, 80]
    ])
    path = trace_flowpath(dem, (0, 0))
    assert path[-1] == (2, 2)
    assert len(path) == 3

def test_trace_downstream_path():
    # Downward sloping DEM from North to South
    dem = np.linspace(4000, 1000, 100).reshape(10, 10)
    bounds = [85.80, 27.80, 86.00, 28.00]
    upstream = [85.90, 27.99] # Northern headwaters
    
    flowpath_gdf, coords = trace_downstream_path(upstream, dem, bounds)
    assert not flowpath_gdf.empty
    assert len(coords) >= 2
    assert flowpath_gdf.crs.to_string() == "EPSG:4326"

def test_list_settlements_along_flowpath():
    # Sloping DEM
    dem = np.linspace(3000, 1000, 100).reshape(10, 10)
    bounds = [85.80, 27.80, 86.00, 28.00]
    upstream = [85.90, 27.99]
    flowpath_gdf, _ = trace_downstream_path(upstream, dem, bounds)
    
    # Mock OSM settlements in WGS84
    mock_settlements = gpd.GeoDataFrame({
        "name": ["Tatopani", "Larcha", "Distant Ridge Village"],
        "place": ["village", "hamlet", "village"],
        "population": [1200, 450, 80],
        "geometry": [
            Point(85.90, 27.95),  # close to flowline
            Point(85.90, 27.85),  # further downstream on flowline
            Point(85.99, 27.99),  # way far east on ridge
        ]
    }, crs="EPSG:4326")
    
    results = list_settlements_along_flowpath(flowpath_gdf, mock_settlements, corridor_buffer_m=5000)
    assert len(results) >= 2
    # Check that Tatopani is upstream of Larcha
    names = [r["name"] for r in results]
    assert "Tatopani" in names
    assert "Larcha" in names
    if "Tatopani" in names and "Larcha" in names:
        assert names.index("Tatopani") < names.index("Larcha")
