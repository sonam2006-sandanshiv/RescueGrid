import numpy as np
from shapely.geometry import LineString, Point
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

def compute_d8_directions(dem):
    """
    Computes smoothed terrain for robust downhill descent.
    """
    from scipy.ndimage import uniform_filter
    return uniform_filter(dem, size=3)

def trace_flowpath(dem, start_idx, max_steps=1000):
    """
    Traces a flowpath from a starting grid coordinate (row, col) down the DEM
    using steepest gradient descent.
    """
    path = [start_idx]
    current = start_idx
    rows, cols = dem.shape
    
    for _ in range(max_steps):
        r, c = current
        min_elev = dem[r, c]
        next_pos = current
        
        # Search 8-connected neighborhood
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dem[nr, nc] < min_elev:
                        min_elev = dem[nr, nc]
                        next_pos = (nr, nc)
                        
        if next_pos == current:
            # Reached a local minimum / valley floor / boundary
            break
            
        path.append(next_pos)
        current = next_pos
        
    return path

def trace_downstream_path(upstream_coord, dem, bounds, max_steps=1000):
    """
    Given an upstream point (lon, lat) in WGS84, bounds [minx, miny, maxx, maxy],
    and a DEM grid (rows, cols), traces downhill along the valley corridor.
    Returns:
      flowpath_gdf: GeoDataFrame (EPSG:4326) with LineString
      path_coords: List of (lon, lat, elev) points along the path
    """
    minx, miny, maxx, maxy = bounds
    rows, cols = dem.shape
    lon, lat = upstream_coord[0], upstream_coord[1]
    
    # Clip starting coordinates to bounds
    clamped_lon = max(minx, min(maxx, lon))
    clamped_lat = max(miny, min(maxy, lat))
    
    # Calculate cell indices (assuming top-to-bottom row ordering, miny to maxy)
    col_idx = int(((clamped_lon - minx) / (maxx - minx)) * (cols - 1))
    # Row 0 is typically maxy (north)
    row_idx = int(((maxy - clamped_lat) / (maxy - miny)) * (rows - 1))
    
    col_idx = max(0, min(cols - 1, col_idx))
    row_idx = max(0, min(rows - 1, row_idx))
    
    # Trace steepest descent
    grid_path = trace_flowpath(dem, (row_idx, col_idx), max_steps=max_steps)
    
    # Convert grid points back to geographic coordinates
    coords = []
    for r, c in grid_path:
        pt_lon = minx + (c / (cols - 1)) * (maxx - minx)
        pt_lat = maxy - (r / (rows - 1)) * (maxy - miny)
        elev = float(dem[r, c])
        coords.append((pt_lon, pt_lat, elev))
        
    if len(coords) < 2:
        # Create minimal 2-point descent if at flat pit
        coords.append((coords[0][0], coords[0][1] - 0.001, coords[0][2] - 5))
        
    geom = LineString([(p[0], p[1]) for p in coords])
    flowpath_gdf = gpd.GeoDataFrame({"geometry": [geom], "length_pts": len(coords)}, crs="EPSG:4326")
    
    return flowpath_gdf, coords

def list_settlements_along_flowpath(flowpath_gdf, osm_gdf, corridor_buffer_m=250):
    """
    Bonus: Given the traced flood path, buffers by valley width (corridor_buffer_m)
    and lists all pre-event OSM settlements encountered downstream in topological order.
    Returns:
      List of dicts: [
        {"name": ..., "population": ..., "distance_km": ..., "elevation_drop_m": ..., "lat": ..., "lon": ...}
      ]
    """
    if flowpath_gdf.empty or osm_gdf.empty:
        return []
        
    # Project to metric CRS EPSG:32645 for accurate buffering and distance
    flow_metric = flowpath_gdf.to_crs(epsg=32645)
    osm_metric = osm_gdf.to_crs(epsg=32645)
    
    corridor = flow_metric.geometry.buffer(corridor_buffer_m).iloc[0]
    flow_line = flow_metric.geometry.iloc[0]
    
    place_col = osm_metric.get("place")
    if place_col is None:
        return []
        
    settlement_mask = place_col.isin(["city", "town", "village", "hamlet", "isolated_dwelling"])
    settlements = osm_metric[settlement_mask].copy()
    
    if settlements.empty:
        return []
        
    results = []
    for idx, row in settlements.iterrows():
        pt = row.geometry
        if pt is None:
            continue
        # Check if inside valley flood corridor
        if corridor.contains(pt) or corridor.distance(pt) < corridor_buffer_m:
            # Measure distance along flow path from upstream origin
            dist_along_m = flow_line.project(pt)
            dist_km = round(dist_along_m / 1000.0, 2)
            
            # WGS84 coordinates for mapping
            pt_wgs84 = gpd.GeoSeries([pt], crs="EPSG:32645").to_crs(epsg=4326).iloc[0]
            
            results.append({
                "name": row.get("name", f"Settlement #{idx}"),
                "place_type": row.get("place", "village"),
                "population": str(row.get("population", "Unknown")),
                "distance_km": dist_km,
                "lat": round(pt_wgs84.y, 5),
                "lon": round(pt_wgs84.x, 5),
            })
            
    # Sort from upstream to downstream (by distance along line)
    results.sort(key=lambda x: x["distance_km"])
    return results
