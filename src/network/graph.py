import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import logging
import numpy as np

logger = logging.getLogger(__name__)

def build_road_graph(osm_gdf):
    """
    Builds a NetworkX graph from OSM roads.
    Edges are road segments, nodes are intersections/endpoints.
    Weights are edge lengths in meters.
    """
    G = nx.Graph()
    roads = osm_gdf[osm_gdf["highway"].notna()]
    
    if roads.empty:
        logger.warning("No roads found to build graph.")
        return G
        
    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        status = row.get("status", "unaffected")
        # If affected, we penalize the weight heavily to simulate it being cut
        penalty = 1e6 if status == "affected" else 1.0
            
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i+1]
                
                # compute distance (assuming metric CRS)
                dist = Point(p1).distance(Point(p2)) * penalty
                
                G.add_edge(p1, p2, weight=dist, id=idx, status=status)
                
    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G
    
def snap_to_graph(G, point):
    """Snaps a shapely Point to the nearest node in the graph."""
    if not G:
        return None
        
    nodes = np.array(G.nodes)
    p_coords = np.array((point.x, point.y))
    
    # Euclidean distance
    dists = np.linalg.norm(nodes - p_coords, axis=1)
    nearest_idx = np.argmin(dists)
    
    return tuple(nodes[nearest_idx])
