import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

def compute_d8_directions(dem):
    """
    Computes D8 flow directions.
    Directions:
    32 64 128
    16  X   1
     8  4   2
    """
    # Simple mock D8 for prototype (would use pysheds in prod)
    # 0 = North, 1 = NE, etc.
    # We'll just gradient descent
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(dem, size=3)
    return smoothed

def trace_flowpath(dem, start_idx, max_steps=1000):
    """
    Traces a flowpath from a starting point index down the DEM.
    """
    path = [start_idx]
    current = start_idx
    
    rows, cols = dem.shape
    
    for _ in range(max_steps):
        r, c = current
        
        # Look at 8 neighbors
        min_elev = dem[r, c]
        next_pos = current
        
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
            # Reached a pit
            break
            
        path.append(next_pos)
        current = next_pos
        
    return path

def trace_to_line(path, transform):
    """Converts a list of matrix indices to a GeoSeries LineString."""
    # Dummy implementation since we don't have affine transform loaded
    # In reality: coords = [transform * (c, r) for r, c in path]
    coords = [(c*10, r*10) for r, c in path]
    if len(coords) < 2:
        return None
    return gpd.GeoDataFrame({"geometry": [LineString(coords)]}, crs="EPSG:32645")
