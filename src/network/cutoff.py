import networkx as nx
import pandas as pd
import geopandas as gpd
import logging
from src.network.graph import snap_to_graph


logger = logging.getLogger(__name__)

def evaluate_settlements(G, osm_gdf):
    """
    Evaluates cut-off status for settlements.
    G is the road graph (already built with pre-event data, penalized for damage).
    Returns a list of dictionaries with settlement stats.
    """
    place_col = osm_gdf.get("place")
    amenity_col = osm_gdf.get("amenity")
    healthcare_col = osm_gdf.get("healthcare")

    settlements = osm_gdf[place_col.isin(["city", "town", "village", "hamlet"])] if place_col is not None else gpd.GeoDataFrame()
    
    hosp_mask = (amenity_col.isin(["hospital", "clinic"])) if amenity_col is not None else False
    if healthcare_col is not None:
        hosp_mask = hosp_mask | healthcare_col.notna()
        
    hospitals = osm_gdf[hosp_mask] if isinstance(hosp_mask, pd.Series) else gpd.GeoDataFrame()
    towns = osm_gdf[place_col.isin(["city", "town"])] if place_col is not None else gpd.GeoDataFrame()

    
    # Snap all to graph
    hospital_nodes = [snap_to_graph(G, geom) for geom in hospitals.geometry if geom is not None]
    hospital_nodes = [n for n in hospital_nodes if n is not None]
    
    town_nodes = [snap_to_graph(G, geom) for geom in towns.geometry if geom is not None]
    town_nodes = [n for n in town_nodes if n is not None]
    
    destinations = set(hospital_nodes + town_nodes)
    if not destinations:
        logger.warning("No destinations (hospitals/towns) found in graph.")
    
    results = []
    
    for idx, row in settlements.iterrows():
        geom = row.geometry
        if geom is None:
            continue
            
        s_node = snap_to_graph(G, geom)
        if s_node is None:
            continue
            
        # We need to find the shortest path to ANY destination
        # We simulate "pre" status by ignoring the penalty, "post" by using the penalty.
        # But our graph currently just has a huge weight penalty.
        
        # Pre-event distance (ignoring penalties)
        def pre_weight(u, v, d):
            w = d.get('weight', 1.0)
            status = d.get('status', 'unaffected')
            return w / 1e6 if status == 'affected' else w
            
        # Post-event distance (using penalties)
        def post_weight(u, v, d):
            return d.get('weight', 1.0)
            
        pre_dist = float('inf')
        post_dist = float('inf')
        
        if destinations:
            try:
                # Multi-source shortest path could be faster, but this is simple
                dists_pre, _ = nx.multi_source_dijkstra(G, destinations, s_node, weight=pre_weight)
                pre_dist = dists_pre
            except nx.NetworkXNoPath:
                pre_dist = float('inf')
                
            try:
                dists_post, _ = nx.multi_source_dijkstra(G, destinations, s_node, weight=post_weight)
                post_dist = dists_post
            except nx.NetworkXNoPath:
                post_dist = float('inf')
        
        # Status classification
        if pre_dist == float('inf'):
            status = "no road access pre-event"
        elif post_dist >= 1e6:
            # The penalty was applied, meaning path goes through a flooded road
            status = "cut off"
        elif post_dist > pre_dist * 1.5: # 50% longer is a significant detour
            status = "detour"
        else:
            status = "reachable"
            
        results.append({
            "name": row.get("name", "Unknown"),
            "population": row.get("population", None),
            "status": status,
            "pre_dist_m": pre_dist if pre_dist != float('inf') else None,
            "post_dist_m": post_dist if post_dist < 1e6 else None,
            "geometry": geom
        })
        
    return results
