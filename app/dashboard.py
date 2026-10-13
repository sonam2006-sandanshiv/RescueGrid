"""Fast, single-screen rescue workflow for satellite scene discovery."""
from __future__ import annotations

import os
import json
import hashlib
import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import folium
import planetary_computer as pc
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.analysis import analyze_event
from src.ingest.s1 import search_s1
from src.ingest.s2 import search_s2
from src.ingest.osm import fetch_osm_layers
from src.navigation import get_routes
from shapely.geometry import mapping, shape

st.set_page_config(page_title="Himalayan Flood Mapper", page_icon="🛰️", layout="wide")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
RUNS_DIR = PROJECT_ROOT / "runs"


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query: str) -> dict[str, Any] | None:
    parts = query.split(",")
    if len(parts) == 2:
        try:
            lat, lon = map(float, parts)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {"name": f"{lat:.5f}, {lon:.5f}", "lat": lat, "lon": lon}
        except ValueError:
            pass
    response = requests.get(NOMINATIM, params={"q": query, "format": "jsonv2", "limit": 1},
                            headers={"User-Agent": "HimalayanFloodMapper/1.0 (rescue mapping prototype)"}, timeout=8)
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    item = results[0]
    return {"name": item["display_name"], "lat": float(item["lat"]), "lon": float(item["lon"])}


def query_sensor(sensor: str, bbox: tuple[float, float, float, float], event: date,
                 before: date, after: date, window_days: int):
    if sensor == "Sentinel-1":
        return search_s1(bbox, event.isoformat(), max_days_before=window_days, max_days_after=window_days,
                         before_target=before.isoformat(), after_target=after.isoformat())
    return search_s2(bbox, event.isoformat(), max_days_before=window_days, max_days_after=window_days,
                     max_cloud_cover=100, before_target=before.isoformat(), after_target=after.isoformat())

def aoi_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """WGS84 bbox for a km radius; longitude width accounts for latitude."""
    import math
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / max(111.32 * abs(math.cos(math.radians(lat))), 1.0)
    return (max(-180.0, lon-lon_delta), max(-90.0, lat-lat_delta),
            min(180.0, lon+lon_delta), min(90.0, lat+lat_delta))


def _item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    raise TypeError(f"Unsupported STAC item: {type(item).__name__}")


def item_summary(item: Any) -> dict[str, Any]:
    item = _item_dict(item)
    props = item.get("properties", {})
    raw_date = props.get("datetime")
    scene_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date().isoformat()
    return {"id": item.get("id"), "date": scene_date,
            "cloud_cover_percent": props.get("eo:cloud_cover"),
            "orbit": props.get("sat:relative_orbit") or props.get(" sat:relative_orbit"),
            "pass": props.get("sat:orbit_state"), "assets": item.get("assets", {})}


def groq_copilot(report_data: dict[str, Any]) -> str:
    """Ask Groq for wording only; all dates and quantities below come from local map data."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in your environment to enable the report copilot.")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    response = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json={
        "model": model, "temperature": 0.1, "reasoning_effort": "low", "max_completion_tokens": 220,
        "messages": [{"role": "system", "content": (
            "Return a JSON object with exactly two string keys: English and Nepali. Each value is one short, "
            "careful sentence about candidate satellite changes needing field verification. Do not include "
            "digits, dates, quantities, counts, location names, damage claims, or facts not provided. Do not "
            "claim a flood is confirmed.")},
                    {"role": "user", "content": "Write a concise rescue update about reviewing candidate satellite change signals in the map."}]}, timeout=20)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice.get("message", {}).get("content", "")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("The report reached its small output limit. Use the verified map facts shown in the report.")
    try:
        note = json.loads(content)
        english, nepali = note.get("English", "").strip(), note.get("Nepali", "").strip()
        if not english or not nepali or any(char.isdigit() for char in english + nepali):
            raise ValueError("The wording did not meet the no-numbers requirement.")
    except (TypeError, ValueError):
        english = "Satellite imagery shows candidate changes that need field verification."
        nepali = "उपग्रह तस्बिरमा देखिएका सम्भावित परिवर्तनलाई स्थलमै पुष्टि गर्नुपर्छ।"

    lines = ["**Verified map facts**"]
    for sensor, pair in report_data.get("catalog_scene_pairs", {}).items():
        lines.append(f"- {sensor} scenes: {pair['before_date']} → {pair['after_date']}")
    stats = report_data.get("system_analysis_stats")
    if stats:
        labels = (
            ("candidate_flood_area_ha", "Candidate water", "ha"),
            ("disturbed_bare_surface_candidate_area_ha", "Disturbed surface candidate", "ha"),
            ("buildings_intersecting_candidate_water", "Buildings intersecting candidate water", ""),
            ("bridges_intersecting_candidate_water", "Bridges intersecting candidate water", ""),
            ("road_length_intersecting_candidate_water_km", "Road length intersecting candidate water", "km"),
            ("settlements_disconnected_from_towns_or_healthcare", "Settlements disconnected from mapped towns/healthcare", ""),
        )
        for key, label, unit in labels:
            if key in stats:
                value = "unknown" if stats[key] is None else str(stats[key])
                lines.append(f"- {label}: {value}{(' ' + unit) if unit and value != 'unknown' else ''}")
        lines.extend(f"- Map note: {warning}" for warning in stats.get("warnings", []))
    if not report_data.get("catalog_scene_pairs") and not stats:
        lines.append(f"- Event date: {report_data.get('event_date', 'unavailable')}")
    return f"**English**\n\n{english}\n\n**नेपाली**\n\n{nepali}\n\n" + "\n".join(lines)


def render_scene_card(sensor: str, label: str, item: dict[str, Any]) -> None:
    summary = item_summary(item)
    st.markdown(f"**{label} · {summary['date']}**")
    st.caption(summary["id"] or "Scene ID unavailable")
    if summary["cloud_cover_percent"] is not None:
        st.caption(f"Cloud cover: {float(summary['cloud_cover_percent']):.0f}%")
    if summary["orbit"] is not None:
        st.caption(f"Relative orbit {summary['orbit']} · {summary['pass'] or 'pass unavailable'}")
    assets = summary["assets"]
    preview = next((assets[key].get("href") for key in ("thumbnail", "rendered_preview", "visual")
                    if isinstance(assets.get(key), dict) and assets[key].get("href")), None)
    if preview and any(key in assets for key in ("thumbnail", "rendered_preview")):
        preview = pc.sign_url(preview)
        st.image(preview, caption=f"{sensor} catalog preview — {label}", use_container_width=True)
    hrefs = [(key, asset.get("href")) for key, asset in assets.items()
             if isinstance(asset, dict) and asset.get("href")]
    if hrefs:
        with st.expander(f"{sensor} source assets"):
            for key, href in hrefs[:10]:
                st.markdown(f"[{key}]({href})")


def metric_text(value: Any, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:,}{suffix}"


def affected_area_records(flood_data: dict[str, Any], settlement_path: Path,
                          fallback_name: str) -> list[dict[str, Any]]:
    """Build candidate-water component coordinates with nearby OSM place names when available."""
    settlements = []
    try:
        data = json.loads(settlement_path.read_text(encoding="utf-8"))
        for feature in data.get("features", []):
            if feature.get("geometry"):
                geom = shape(feature["geometry"])
                props = feature.get("properties", {})
                name = props.get("name") or props.get("name:en") or props.get("name:ne")
                if name:
                    settlements.append((name, geom))
    except (OSError, ValueError, TypeError):
        pass

    areas = []
    for feature in flood_data.get("features", []):
        try:
            geom = shape(feature["geometry"])
            parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for part in parts:
                if part.geom_type != "Polygon" or part.is_empty:
                    continue
                point = part.representative_point()
                closest = min(settlements, key=lambda row: row[1].distance(point), default=None)
                place = closest[0] if closest and closest[1].distance(point) <= 0.1 else None
                if not place and fallback_name and not any(char.isdigit() for char in fallback_name[:8]):
                    place = fallback_name.split(",")[0].strip()
                index = len(areas) + 1
                name = f"{place} · candidate water area {index}" if place else f"Candidate water area {index}"
                areas.append({"name": name, "latitude": float(point.y), "longitude": float(point.x)})
        except (KeyError, TypeError, ValueError):
            continue
    return areas[:20]


st.title("🛰️ Himalayan Flood Mapper")
st.caption("Search a place and event date. Compare Sentinel-1/2 before-and-after scenes and map candidate water, disturbed surfaces, and exposed access routes.")

with st.form("rescue_search"):
    saved_search = st.session_state.get("search_result")
    default_location = (f"{saved_search['location']['lat']:.5f}, {saved_search['location']['lon']:.5f}"
                        if saved_search else "")
    default_event = date.fromisoformat(saved_search["event_date"]) if saved_search else date.today()
    top = st.columns([4, 1, 1, 1])
    place = top[0].text_input("Location", value=default_location,
                              placeholder="Search a village, town, or latitude, longitude", key="search_location")
    event_date = top[1].date_input("Flood date", value=default_event, key="search_event_date")
    window_days = top[2].number_input("Search window (days)", min_value=7, max_value=180, value=48, step=7)
    radius_km = top[3].number_input("AOI radius (km)", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
    targets = st.columns(2)
    before_target = targets[0].date_input("Preferred pre-flood date", value=event_date - __import__("datetime").timedelta(days=12))
    after_target = targets[1].date_input("Preferred post-flood date", value=event_date + __import__("datetime").timedelta(days=12))
    submit = st.form_submit_button("Find satellite scenes", type="primary", use_container_width=True)

if submit:
    if not place.strip():
        st.error("Enter a location to search.")
    elif not before_target < event_date < after_target:
        st.error("Choose a pre-flood date before the event and a post-flood date after it.")
    elif (event_date-before_target).days > int(window_days) or (after_target-event_date).days > int(window_days):
        st.error("Preferred scene dates must fall inside the configured search window. Widen the window or move the preferred dates.")
    else:
        try:
            with st.spinner("Finding location and Sentinel scenes independently…"):
                location = geocode(place.strip())
                if location is None:
                    st.error("No matching place found. Try a nearby town or a latitude, longitude pair.")
                    st.stop()
                bbox = aoi_bbox(location["lat"], location["lon"], float(radius_km))
                found, errors = {}, {}
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {sensor: pool.submit(query_sensor, sensor, bbox, event_date, before_target,
                                                  after_target, int(window_days))
                               for sensor in ("Sentinel-1", "Sentinel-2")}
                    for sensor, future in futures.items():
                        try:
                            found[sensor] = future.result()
                            if not found[sensor]:
                                errors[sensor] = f"No real pre/post pair found within ±{int(window_days)} days. Widen the window or try another AOI."
                        except Exception as exc:
                            found[sensor] = None
                            errors[sensor] = f"Catalog search failed: {exc}"
                paired = {sensor: pair for sensor, pair in found.items() if pair and len(pair) == 2}
                for sensor in ("Sentinel-1", "Sentinel-2"):
                    st.session_state[f"accept_{sensor}"] = False
                st.session_state["search_result"] = {"location": location, "bbox": bbox,
                    "event_date": event_date.isoformat(), "before_target": before_target.isoformat(),
                    "after_target": after_target.isoformat(), "window_days": int(window_days),
                    "radius_km": float(radius_km), "found": found, "paired": paired, "errors": errors}
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("route_result", None)
        except requests.RequestException as exc:
            st.error(f"Location search service unavailable: {exc}")
        except Exception as exc:
            st.error(f"Location or catalog search could not complete: {exc}")

result = st.session_state.get("search_result")
if result:
    loc = result["location"]
    event_day = date.fromisoformat(result["event_date"])
    snapshot = date.fromordinal(event_day.toordinal()-1)
    st.subheader(loc["name"])
    st.caption(f"Event date: {result['event_date']} · preferred scenes {result['before_target']} / {result['after_target']} · ±{result['window_days']} days · AOI radius {result['radius_km']} km")
    map_col, data_col = st.columns([1, 2])
    with map_col:
        fmap = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=11, tiles="OpenStreetMap")
        folium.Marker([loc["lat"], loc["lon"]], tooltip=loc["name"], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(fmap)
        folium.Rectangle(bounds=[[result["bbox"][1], result["bbox"][0]], [result["bbox"][3], result["bbox"][2]]],
                         color="#f97316", fill=True, fill_opacity=.08, tooltip="Selected AOI · WGS84").add_to(fmap)
        st_folium(fmap, height=330, width=700, returned_objects=[])
    with data_col:
        for sensor in ("Sentinel-1", "Sentinel-2"):
            st.markdown(f"#### {sensor} · nearest pre/post scenes")
            if sensor in result["paired"]:
                left, right = st.columns(2)
                with left:
                    render_scene_card(sensor, "BEFORE", result["paired"][sensor][0])
                with right:
                    render_scene_card(sensor, "AFTER", result["paired"][sensor][1])
                before, after = item_summary(result["paired"][sensor][0]), item_summary(result["paired"][sensor][1])
                if before["date"] != result["before_target"] or after["date"] != result["after_target"]:
                    accepted = st.checkbox(f"Accept these nearest real {sensor} dates ({before['date']} / {after['date']})",
                                           key=f"accept_{sensor}", value=False)
                    if not accepted:
                        st.warning("Alternative dates need your acceptance before analysis.")
                else:
                    st.checkbox(f"Confirm selected {sensor} scenes", key=f"accept_{sensor}", value=True, disabled=True)
            else:
                st.warning(result["errors"].get(sensor, "No scenes found in this search window."))

    accepted_pairs = {sensor: pair for sensor, pair in result["paired"].items()
                      if st.session_state.get(f"accept_{sensor}", False)}
    if not st.session_state.get("analysis_result"):
        st.caption("Change polygons are candidate detections. Review the acquired dates and accept nearest-date alternatives before analysis.")
    if st.button("Analyze accepted imagery and map exposure", type="primary", disabled=not accepted_pairs):
        cache_identity = {"date": result["event_date"], "bbox": result["bbox"],
                          "scenes": {sensor: [item_summary(item)["id"] for item in pair]
                                     for sensor, pair in accepted_pairs.items()}}
        cache_key = hashlib.sha1(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:12]
        run_dir = RUNS_DIR / f"event-{result['event_date']}-{cache_key}"
        try:
            cached_path = run_dir / "analysis.json"
            stats = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
            cache_has_osm = bool(stats and stats.get("osm_feature_counts"))
            needs_analysis = stats is None or (os.environ.get("OHSOME_API_KEY", "").strip() and not cache_has_osm)
            if needs_analysis:
                osm_layers = {}
                if os.environ.get("OHSOME_API_KEY", "").strip():
                    try:
                        with st.spinner("Loading pre-event roads, buildings, bridges, and settlements…"):
                            osm_layers = fetch_osm_layers(tuple(result["bbox"]), snapshot,
                                                          date.fromisoformat(result["event_date"]), timeout=35)
                    except Exception as osm_exc:
                        st.warning(f"Pre-event OSM snapshot unavailable: {osm_exc}. Exposure and access will remain unknown.")
                with st.spinner("Downloading selected raster bands and mapping candidate change…"):
                    paired_items = {sensor: tuple(_item_dict(item) for item in pair)
                                    for sensor, pair in accepted_pairs.items()}
                    stats = analyze_event(paired_items, result["bbox"], osm_layers, run_dir)
            st.session_state["analysis_result"] = {"stats": stats, "run_dir": str(run_dir)}
        except Exception as exc:
            st.error(f"Satellite and map analysis could not complete: {exc}")

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        stats = analysis_result["stats"]
        run_dir = Path(analysis_result["run_dir"])
        st.markdown("### Event analysis")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Candidate water", metric_text(stats["candidate_flood_area_ha"], " ha"))
        metric_cols[1].metric("Disturbed surface", metric_text(stats["disturbed_bare_surface_candidate_area_ha"], " ha"))
        metric_cols[2].metric("Buildings / bridges exposed", f"{metric_text(stats['buildings_intersecting_candidate_water'])} / {metric_text(stats['bridges_intersecting_candidate_water'])}")
        metric_cols[3].metric("Road length exposed", metric_text(stats["road_length_intersecting_candidate_water_km"], " km"))
        metric_cols[4].metric("Settlements disconnected", metric_text(stats["settlements_disconnected_from_towns_or_healthcare"]))
        candidate_location = stats.get("candidate_water_location")
        if candidate_location:
            st.info(f"Candidate water centroid: {candidate_location['latitude']:.6f}, {candidate_location['longitude']:.6f} · "
                    f"{stats['candidate_flood_area_ha']:.2f} ha · sensor support: {stats.get('candidate_water_sensor_support', 'unknown')} · "
                    f"area-only screening severity: {stats.get('candidate_water_severity_estimate', 'unknown')} · suspected, not field verified")
        st.caption(f"Analyzed: {', '.join(stats['sensors_analyzed'])} · {stats['pixel_resolution_m']} m grid · OSM snapshot {snapshot.isoformat()}")
        for warning in stats["warnings"]:
            st.warning(warning)

        flood_path = run_dir / "flood_candidates.geojson"
        flood_data = {"type": "FeatureCollection", "features": []}
        if flood_path.exists():
            try:
                flood_data = json.loads(flood_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                st.warning("Candidate-water geometry could not be read; affected-area coordinates are unavailable.")
        affected_areas = affected_area_records(
            flood_data, run_dir / "nearby_settlements.geojson", loc.get("name", ""))

        st.markdown("### Rescuer location")
        st.caption("Choose the button to ask your browser for location permission. Your location is read only after you choose it.")
        gps = components.declare_component("rescue_gps", path=str(PROJECT_ROOT / "app" / "gps_component"))
        gps_value = gps(key="rescue_gps_location", default=None)
        current_gps = None
        if isinstance(gps_value, dict) and gps_value.get("lat") is not None and gps_value.get("lon") is not None:
            current_gps = (float(gps_value["lat"]), float(gps_value["lon"]))
            gps_id = (round(current_gps[0], 6), round(current_gps[1], 6))
            if st.session_state.get("_last_gps_location") != gps_id:
                st.session_state["route_start_lat"], st.session_state["route_start_lon"] = current_gps
                st.session_state["_last_gps_location"] = gps_id
            st.success(f"Current location: {current_gps[0]:.6f}, {current_gps[1]:.6f} (±{float(gps_value.get('accuracy_m', 0)):.0f} m)")
        else:
            st.info("Current location is not available yet. Choose ‘Allow location access for directions’ above, then allow it in your browser prompt. You can still enter a start point manually.")

        preview_col, impact_col = st.columns([1, 1])
        with preview_col:
            st.markdown("**Selected-scene before / after**")
            for sensor in stats["sensors_analyzed"]:
                a, b = st.columns(2)
                with a:
                    before_day = item_summary(result["paired"][sensor][0])["date"]
                    st.image(stats["previews"][sensor][0], caption=f"{sensor} before · {before_day}", use_container_width=True)
                with b:
                    after_day = item_summary(result["paired"][sensor][1])["date"]
                    st.image(stats["previews"][sensor][1], caption=f"{sensor} after · {after_day}", use_container_width=True)
        with impact_col:
            st.markdown("**Candidate impact map**")
            impact_map = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=12, tiles="OpenStreetMap")
            if flood_data["features"]:
                folium.GeoJson(flood_data, name="Candidate new water", style_function=lambda _: {
                    "color": "#ef4444", "fillColor": "#ef4444", "fillOpacity": .38, "weight": 2}).add_to(impact_map)
            debris_path = run_dir / "disturbed_surface_candidates.geojson"
            if debris_path.exists():
                debris_data = json.loads(debris_path.read_text(encoding="utf-8"))
                if debris_data["features"]:
                    folium.GeoJson(debris_data, name="Disturbed bare surface (debris/landslide candidate)",
                                   style_function=lambda _: {"color": "#a855f7", "fillColor": "#a855f7",
                                                             "fillOpacity": .25, "weight": 2}).add_to(impact_map)
            roads_path = run_dir / "affected_roads.geojson"
            if roads_path.exists():
                roads_data = json.loads(roads_path.read_text(encoding="utf-8"))
                if roads_data["features"]:
                    folium.GeoJson(roads_data, name="Roads crossing candidate water", style_function=lambda _: {
                        "color": "#f97316", "weight": 3}).add_to(impact_map)
            isolated_path = run_dir / "isolated_settlements.geojson"
            for filename, layer_name, color in (("exposed_buildings.geojson", "Buildings intersecting candidate area", "#dc2626"),
                                                ("exposed_bridges.geojson", "Bridges intersecting candidate area", "#7c3aed")):
                feature_path = run_dir / filename
                if feature_path.exists():
                    feature_data = json.loads(feature_path.read_text(encoding="utf-8"))
                    if feature_data["features"]:
                        folium.GeoJson(feature_data, name=layer_name, marker=folium.CircleMarker(radius=4, color=color,
                                     fill=True, fill_opacity=.8), style_function=lambda _, c=color: {"color": c, "weight": 3}).add_to(impact_map)
            if isolated_path.exists():
                isolated_data = json.loads(isolated_path.read_text(encoding="utf-8"))
                if isolated_data["features"]:
                    folium.GeoJson(isolated_data, name="Potentially disconnected settlements").add_to(impact_map)
            route_data = st.session_state.get("route_result")
            if route_data:
                folium.GeoJson(route_data["geometry"], name="OSRM route · safety unverified",
                               style_function=lambda _: {"color": "#2563eb", "weight": 5}).add_to(impact_map)
            for area in affected_areas:
                coords = f"{area['latitude']:.6f}, {area['longitude']:.6f}"
                folium.Marker([area["latitude"], area["longitude"]], tooltip=area["name"],
                              popup=folium.Popup(f"{area['name']}<br>Coordinates: {coords}", max_width=320),
                              icon=folium.Icon(color="red", icon="info-sign")).add_to(impact_map)
            if current_gps:
                folium.Marker([current_gps[0], current_gps[1]], tooltip="Your current location",
                              popup=f"Your current location<br>Coordinates: {current_gps[0]:.6f}, {current_gps[1]:.6f}",
                              icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(impact_map)
            folium.LayerControl().add_to(impact_map)
            st_folium(impact_map, height=370, width=700, returned_objects=[])
            if affected_areas:
                st.markdown("**Flood-affected candidate areas · approximate place names from OpenStreetMap where available**")
                st.dataframe(affected_areas, hide_index=True, use_container_width=True)
            else:
                st.info("No candidate-water polygon was detected for this scene pair, so there are no affected-area coordinates to list.")
        st.caption(f"Analysis files saved to `{analysis_result['run_dir']}`. Candidate-water polygons, exposure GeoJSON, mask GeoTIFF, and source-scene previews are included.")
        st.info(stats["note"])
        st.markdown("### Rescuer navigation")
        st.caption("Road directions use the public OSRM road graph. Mapped hazard intersections are screening signals only; route safety must be confirmed locally.")
        if affected_areas:
            first_area = affected_areas[0]
            suggested_name = first_area["name"]
            suggested_lat, suggested_lon = first_area["latitude"], first_area["longitude"]
        else:
            candidate = stats.get("candidate_water_location") or {}
            suggested_lat = float(candidate.get("latitude", loc["lat"]))
            suggested_lon = float(candidate.get("longitude", loc["lon"]))
            suggested_name = "AOI center (no candidate-water polygon)"
        destinations = [(f"System-suggested incident · {suggested_name} · {suggested_lat:.6f}, {suggested_lon:.6f}", suggested_lat, suggested_lon),
                        ("Selected AOI center", float(loc["lat"]), float(loc["lon"])),
                        ("Manual coordinates", None, None)]
        for filename, label in (("exposed_buildings.geojson", "Exposed building"),
                                ("exposed_bridges.geojson", "Exposed bridge"),
                                ("isolated_settlements.geojson", "Potentially isolated settlement")):
            path = run_dir / filename
            if path.exists():
                try:
                    features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
                    for index, feature in enumerate(features, 1):
                        point = shape(feature["geometry"]).representative_point()
                        destinations.append((f"{label} {index} · {point.y:.5f}, {point.x:.5f}", point.y, point.x))
                except Exception:
                    pass
        selected_destination = st.selectbox("Incident destination (suggested from mapped candidate water)", destinations,
                                              format_func=lambda d: d[0], index=0, key="route_destination_choice_v2")
        destination_choice = selected_destination[0]
        if st.session_state.get("_last_destination_choice_v2") != destination_choice:
            if selected_destination[1] is not None:
                st.session_state["route_dest_lat"] = float(selected_destination[1])
                st.session_state["route_dest_lon"] = float(selected_destination[2])
            st.session_state["_last_destination_choice_v2"] = destination_choice
        if isinstance(gps_value, dict):
            gps_location = (round(float(gps_value["lat"]), 6), round(float(gps_value["lon"]), 6))
            if st.session_state.get("_last_gps_location") != gps_location:
                st.session_state["route_start_lat"], st.session_state["route_start_lon"] = gps_location
                st.session_state["_last_gps_location"] = gps_location
        st.session_state.setdefault("route_start_lat", float(loc["lat"]))
        st.session_state.setdefault("route_start_lon", float(loc["lon"]))
        st.session_state.setdefault("route_dest_lat", float(suggested_lat))
        st.session_state.setdefault("route_dest_lon", float(suggested_lon))
        route_cols = st.columns(4)
        start_lat = route_cols[0].number_input("Start latitude", format="%.6f", key="route_start_lat")
        start_lon = route_cols[1].number_input("Start longitude", format="%.6f", key="route_start_lon")
        destination_lat = route_cols[2].number_input("Destination latitude", format="%.6f", key="route_dest_lat")
        destination_lon = route_cols[3].number_input("Destination longitude", format="%.6f", key="route_dest_lon")
        maps_url = "https://www.google.com/maps/dir/?" + urlencode({
            "api": "1", "origin": f"{start_lat:.6f},{start_lon:.6f}",
            "destination": f"{destination_lat:.6f},{destination_lon:.6f}", "travelmode": "driving",
        })
        st.link_button("Open selected incident destination in Google Maps", maps_url, type="primary")
        st.caption("Google Maps opens in a new tab with the selected coordinates. It cannot confirm flood or road safety.")
        if st.button("Get local route screening", key="get_rescue_route"):
            st.session_state.pop("route_result", None)
            try:
                flood_geom = None
                flood_path = run_dir / "flood_candidates.geojson"
                if flood_path.exists():
                    fc = json.loads(flood_path.read_text(encoding="utf-8"))
                    if fc.get("features"):
                        from shapely.geometry import shape
                        from shapely.ops import unary_union
                        flood_geom = mapping(unary_union([shape(f["geometry"]) for f in fc["features"]]))
                roads_geom = None
                affected_path = run_dir / "affected_roads.geojson"
                if affected_path.exists():
                    roads_fc = json.loads(affected_path.read_text(encoding="utf-8"))
                    if roads_fc.get("features"):
                        from shapely.geometry import shape
                        from shapely.ops import unary_union
                        roads_geom = mapping(unary_union([shape(f["geometry"]) for f in roads_fc["features"]]))
                with st.spinner("Requesting road route and available alternatives…"):
                    st.session_state["route_result"] = get_routes((start_lat, start_lon),
                        (destination_lat, destination_lon), flood_geom, roads_geom)
                st.rerun()
            except Exception as exc:
                st.error(f"Road directions unavailable: {exc}. Check coordinates or try again when routing service is reachable.")
        route_data = st.session_state.get("route_result")
        if route_data:
            st.write(f"**{route_data['distance_km']:.1f} km · {route_data['duration_min']:.0f} min · {route_data['alternatives']} route option(s)**")
            if route_data["possibly_affected"]:
                st.warning("The selected route intersects mapped candidate flooding or roads flagged as exposed. Use only after local validation.")
            st.error(route_data["safety"])
            if route_data["instructions"]:
                with st.expander("Turn-by-turn instructions"):
                    for step in route_data["instructions"]:
                        st.markdown(f"- {step}")
    if st.button("Draft source-grounded situation note", disabled=not accepted_pairs):
        facts = {"location": loc["name"], "event_date": result["event_date"], "catalog_scene_pairs": {}}
        for sensor, pair in accepted_pairs.items():
            before, after = item_summary(pair[0]), item_summary(pair[1])
            facts["catalog_scene_pairs"][sensor] = {"before_date": before["date"], "after_date": after["date"]}
            if before["orbit"] is not None:
                facts["catalog_scene_pairs"][sensor]["relative_orbit"] = before["orbit"]
                facts["catalog_scene_pairs"][sensor]["pass"] = before["pass"]
            if before["cloud_cover_percent"] is not None:
                facts["catalog_scene_pairs"][sensor]["before_cloud_percent"] = round(float(before["cloud_cover_percent"]))
            if after["cloud_cover_percent"] is not None:
                facts["catalog_scene_pairs"][sensor]["after_cloud_percent"] = round(float(after["cloud_cover_percent"]))
        if st.session_state.get("analysis_result"):
            stats = st.session_state["analysis_result"]["stats"]
            keys = ("candidate_flood_area_ha", "disturbed_bare_surface_candidate_area_ha",
                    "buildings_intersecting_candidate_water", "bridges_intersecting_candidate_water",
                    "road_length_intersecting_candidate_water_km",
                    "settlements_disconnected_from_towns_or_healthcare", "warnings")
            facts["system_analysis_stats"] = {key: copy.deepcopy(stats.get(key)) for key in keys}
        try:
            with st.spinner("Drafting a short note…"):
                st.markdown(groq_copilot(facts))
        except Exception as exc:
            st.error(f"Copilot unavailable: {exc}")
else:
    st.info("Enter a location and event date to find the nearest available Sentinel scenes.")
