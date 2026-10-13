"""Road-based rescue route lookup using the public OSRM routing API."""
from __future__ import annotations
import requests
from shapely.geometry import LineString, shape

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

def get_routes(start, destination, candidate_flood=None, affected_roads=None, timeout=20):
    """Return the shortest route and any alternatives; flag mapped candidate intersections only."""
    start_lat, start_lon = map(float, start)
    end_lat, end_lon = map(float, destination)
    for lat, lon in ((start_lat, start_lon), (end_lat, end_lon)):
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Coordinates must be latitude, longitude in WGS84.")
    url = f"{OSRM_URL}/{start_lon},{start_lat};{end_lon},{end_lat}"
    response = requests.get(url, params={"alternatives": 3, "steps": "true", "geometries": "geojson",
                                         "overview": "full"}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message") or "No drivable road route was returned.")
    flood = shape(candidate_flood) if candidate_flood else None
    roads = shape(affected_roads) if affected_roads else None
    entries = []
    for route in data["routes"]:
        line = LineString(route["geometry"]["coordinates"])
        hits_flood = bool(flood and line.intersects(flood))
        hits_road = bool(roads and line.intersects(roads))
        entries.append((hits_flood or hits_road, route, hits_flood, hits_road))
    chosen = min(entries, key=lambda e: (e[0], e[1].get("duration", float("inf"))))
    route = chosen[1]
    maneuvers = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            instruction = maneuver.get("instruction") or f"{maneuver.get('type', 'Continue').capitalize()} {maneuver.get('modifier', '')}".strip()
            if instruction and (not maneuvers or maneuvers[-1] != instruction):
                maneuvers.append(instruction)
    return {"geometry": route["geometry"], "distance_km": route["distance"] / 1000,
            "duration_min": route["duration"] / 60, "alternatives": len(data["routes"]),
            "intersects_candidate_flood": chosen[2], "intersects_mapped_affected_roads": chosen[3],
            "possibly_affected": chosen[0], "instructions": maneuvers[:12],
            "safety": "Route safety is unverified. Confirm current road and bridge conditions with local teams."}
