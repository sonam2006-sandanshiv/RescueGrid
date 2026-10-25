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
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_folium import st_folium
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.analysis import analyze_event
from src.ingest.s1 import search_s1
from src.ingest.s2 import search_s2
from src.ingest.osm import fetch_osm_layers

st.set_page_config(page_title="Himalayan Flood Mapper", page_icon="🛰️", layout="wide")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
RUNS_DIR = PROJECT_ROOT / "runs"
gps_location = components.declare_component("rescue_gps", path=str(PROJECT_ROOT / "app" / "gps_component"))


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query: str) -> dict[str, Any] | None:
    parts = query.split(",")
    if len(parts) == 2:
        try:
            lat, lon = map(float, parts)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {"name": f"{lat:.5f}, {lon:.5f}", "lat": lat, "lon": lon,
                        "bbox": [lat-.05, lat+.05, lon-.05, lon+.05]}
        except ValueError:
            pass
    response = requests.get(NOMINATIM, params={"q": query, "format": "jsonv2", "limit": 1},
                            headers={"User-Agent": "HimalayanFloodMapper/1.0 (rescue mapping prototype)"}, timeout=8)
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    item = results[0]
    return {"name": item["display_name"], "lat": float(item["lat"]), "lon": float(item["lon"]),
            "bbox": [float(value) for value in item["boundingbox"]]}


def query_sensor(sensor: str, bbox: tuple[float, float, float, float], event: date):
    if sensor == "Sentinel-1":
        return search_s1(bbox, event.isoformat(), max_days_before=48, max_days_after=48)
    return search_s2(bbox, event.isoformat(), max_days_before=48, max_days_after=48,
                     max_cloud_cover=100)


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
            "pass": props.get("sat:orbit_state"), "platform": props.get("platform"),
            "datetime": raw_date, "collection": item.get("collection"),
            "asset_keys": sorted(item.get("assets", {}).keys()), "assets": item.get("assets", {})}


def write_run_manifest(run_dir: Path, result: dict[str, Any], stats: dict[str, Any], osm_snapshot: date) -> None:
    """Persist stable scene identifiers; never persist temporary signed asset URLs."""
    inputs = []
    for sensor, pair in result["paired"].items():
        for role, item in zip(("before", "after"), pair):
            summary = item_summary(item)
            inputs.append({"sensor": sensor, "role": role, "id": summary["id"],
                           "collection": summary["collection"], "platform": summary["platform"],
                           "datetime": summary["datetime"], "relative_orbit": summary["orbit"],
                           "orbit_direction": summary["pass"],
                           "cloud_cover_percent": summary["cloud_cover_percent"],
                           "asset_keys": summary["asset_keys"],
                           "stac_api": "https://planetarycomputer.microsoft.com/api/stac/v1"})
    manifest = {"event_date": result["event_date"], "aoi_bbox": result["bbox"],
                "osm_snapshot_date": osm_snapshot.isoformat(), "inputs": inputs,
                "analysis": stats, "reference_validation": "not run",
                "warning": "Signed asset URLs are intentionally omitted; IDs and collection are stable provenance."}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def groq_copilot(report_data: dict[str, Any]) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY in your environment to enable the report copilot.")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    response = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json={
        "model": model, "temperature": 0.1, "reasoning_effort": "low", "max_completion_tokens": 220,
        "messages": [{"role": "system", "content": (
            "Return a JSON object with exactly two string keys: English and Nepali. Each value is one short "
            "rescue note sentence. Use only supplied facts and numbers. Include both before/after sensor date pairs "
            "and the measured candidate-water area, calling it a candidate signal rather than confirmed flooding. "
            "State OSM exposure and access are unknown when their values are null. Do not infer damage, counts, "
            "other estimates, or recommendations.")},
                    {"role": "user", "content": str(report_data)}]}, timeout=20)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice.get("message", {}).get("content", "")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("The report reached its small output limit. Use the measured map values shown above.")
    if not content.strip():
        raise RuntimeError("The report service returned an empty note.")
    try:
        note = json.loads(content)
        english = note.get("English", "")
        nepali = note.get("Nepali") or note.get("Neapali", "")
        if english and nepali:
            return f"**English**\n\n{english}\n\n**नेपाली**\n\n{nepali}"
    except (TypeError, ValueError):
        pass
    return content


def scene_browser_url(sensor: str, scene_date: str, location: dict[str, Any]) -> str:
    """Open the official Copernicus Browser at this scene's AOI and acquisition date."""
    params = {
        "zoom": "12", "lat": f"{location['lat']:.6f}", "lng": f"{location['lon']:.6f}",
        "themeId": "DEFAULT-THEME", "fromTime": f"{scene_date}T00:00:00.000Z",
        "toTime": f"{scene_date}T23:59:59.999Z", "dateMode": "SINGLE",
    }
    # The Browser's public Sentinel-2 collection identifier is documented; omit
    # an uncertain S1 identifier so the exact AOI/date still opens reliably.
    if sensor == "Sentinel-2":
        params["datasetId"] = "S2_L2A_CDAS"
    return "https://browser.dataspace.copernicus.eu/?" + urlencode(params)


def render_scene_card(sensor: str, label: str, item: dict[str, Any], location: dict[str, Any]) -> None:
    summary = item_summary(item)
    st.markdown(f"**{label} · {summary['date']}**")
    st.caption(summary["id"] or "Scene ID unavailable")
    if summary["cloud_cover_percent"] is not None:
        st.caption(f"Cloud cover: {float(summary['cloud_cover_percent']):.0f}%")
    if summary["orbit"] is not None:
        st.caption(f"Relative orbit {summary['orbit']} · {summary['pass'] or 'pass unavailable'}")
    assets = summary["assets"]
    viewer_url = scene_browser_url(sensor, summary["date"], location)
    if sensor == "Sentinel-1":
        st.markdown(f"[Open this date and location in Copernicus Browser ↗]({viewer_url}) · select **Sentinel-1 GRD**")
    else:
        st.markdown(f"[Open this Sentinel-2 L2A scene in Copernicus Browser ↗]({viewer_url})")
    preview = next((assets[key].get("href") for key in ("thumbnail", "rendered_preview", "visual")
                    if isinstance(assets.get(key), dict) and assets[key].get("href")), None)
    if preview and any(key in assets for key in ("thumbnail", "rendered_preview")):
        try:
            st.image(preview, caption=f"{sensor} catalog preview — {label}", use_container_width=True)
        except Exception as exc:
            st.warning(f"The embedded scene preview could not load: {exc}. Use the Copernicus Browser link above to inspect the acquisition.")
    hrefs = [(key, asset.get("href")) for key, asset in assets.items()
             if isinstance(asset, dict) and asset.get("href")]
    if hrefs:
        with st.expander(f"{sensor} source assets"):
            for key, href in hrefs[:10]:
                st.markdown(f"[{key}]({href})")


def incident_point(geojson_path: Path, fallback: dict[str, Any]) -> tuple[float, float, str]:
    """Choose the representative point of the largest mapped candidate-water polygon."""
    try:
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        candidates = [shape(feature["geometry"]) for feature in data.get("features", [])
                      if feature.get("geometry")]
        candidates = [geometry for geometry in candidates if not geometry.is_empty]
        if candidates:
            geometry = max(candidates, key=lambda candidate: candidate.area)
            point = geometry.representative_point()
            return point.y, point.x, "center of the largest mapped candidate-water area"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return float(fallback["lat"]), float(fallback["lon"]), "selected AOI center (no candidate-water polygon)"


def metric_text(value: Any, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:,}{suffix}"


st.title("🛰️ Himalayan Flood Mapper")
st.caption("Search a place and event date. Compare Sentinel-1/2 before-and-after scenes and map candidate water, disturbed surfaces, and exposed access routes.")
st.caption("Map key: satellite layers are observations; dark red areas have cross-sensor support, amber areas are uncertain single-sensor water candidates, purple areas are unverified disturbed-surface candidates, and orange roads / magenta buildings / blue bridges may be exposed. No field-verified layers are loaded. Educational prototype, not an operational tool.")

with st.form("rescue_search"):
    place_col, date_col, button_col = st.columns([5, 2, 1])
    saved_search = st.session_state.get("search_result")
    default_location = (f"{saved_search['location']['lat']:.5f}, {saved_search['location']['lon']:.5f}"
                        if saved_search else "")
    default_date = date.fromisoformat(saved_search["event_date"]) if saved_search else date.today()
    place = place_col.text_input("Location", value=default_location,
                                  placeholder="Search a village, town, or coordinates", key="search_location")
    event_date = date_col.date_input("Event date", value=default_date, key="search_event_date")
    submit = button_col.form_submit_button("Search", type="primary", use_container_width=True)

if submit:
    if not place.strip():
        st.error("Enter a location to search.")
    else:
        try:
            with st.spinner("Finding location and satellite scenes…"):
                location = geocode(place.strip())
                if location is None:
                    st.error("No matching place found. Try a nearby town or a latitude, longitude pair.")
                    st.stop()
                south, north = location["bbox"][0], location["bbox"][1]
                west, east = location["bbox"][2], location["bbox"][3]
                # Keep town bounding boxes bounded so a broad place name cannot trigger a huge catalog query.
                lat, lon = location["lat"], location["lon"]
                # A focused local footprint keeps remote raster reads responsive.
                bbox = (max(west, lon-.02), max(south, lat-.02), min(east, lon+.02), min(north, lat+.02))
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    bbox = (lon-.02, lat-.02, lon+.02, lat+.02)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    s1_future = pool.submit(query_sensor, "Sentinel-1", bbox, event_date)
                    s2_future = pool.submit(query_sensor, "Sentinel-2", bbox, event_date)
                    found = {"Sentinel-1": s1_future.result(), "Sentinel-2": s2_future.result()}
                paired = {}
                errors = {}
                for sensor, pair in found.items():
                    if pair and len(pair) == 2:
                        paired[sensor] = pair
                    else:
                        errors[sensor] = "No before/after pair was available in the 48-day search window."
                st.session_state["search_result"] = {"location": location, "bbox": bbox,
                    "event_date": event_date.isoformat(), "found": found, "paired": paired, "errors": errors}
                st.session_state.pop("analysis_result", None)
        except requests.RequestException as exc:
            st.error(f"Search service unavailable: {exc}")
        except Exception as exc:
            st.error(f"Satellite catalog search failed: {exc}")

result = st.session_state.get("search_result")
if result:
    loc = result["location"]
    event_day = date.fromisoformat(result["event_date"])
    snapshot = date.fromordinal(event_day.toordinal()-1)
    st.subheader(loc["name"])
    st.caption(f"Event date: {result['event_date']} · nearest available scenes within 48 days before and after")
    map_col, data_col = st.columns([1, 2])
    with map_col:
        fmap = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=11, tiles="OpenStreetMap")
        folium.Marker([loc["lat"], loc["lon"]], tooltip=loc["name"], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(fmap)
        folium.Rectangle(bounds=[[result["bbox"][1], result["bbox"][0]], [result["bbox"][3], result["bbox"][2]]],
                         color="#f97316", fill=True, fill_opacity=.08, tooltip="Satellite search area").add_to(fmap)
        st_folium(fmap, height=330, width=700, returned_objects=[])
    with data_col:
        for sensor in ("Sentinel-1", "Sentinel-2"):
            st.markdown(f"#### {sensor} · nearest pre/post scenes")
            if sensor in result["paired"]:
                left, right = st.columns(2)
                with left:
                    render_scene_card(sensor, "BEFORE", result["paired"][sensor][0], loc)
                with right:
                    render_scene_card(sensor, "AFTER", result["paired"][sensor][1], loc)
            else:
                st.info(result["errors"].get(sensor, "No scenes found in this search window."))

    if not st.session_state.get("analysis_result"):
        st.caption("Scene pairs are ready. Pixel, infrastructure, and access estimates are calculated when you start analysis.")
    if st.button("Analyze imagery, infrastructure, and access", type="primary", disabled=not result["paired"]):
        cache_identity = {"date": result["event_date"], "bbox": result["bbox"],
                          "scenes": {sensor: [item_summary(item)["id"] for item in pair]
                                     for sensor, pair in result["paired"].items()}}
        cache_key = hashlib.sha1(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:12]
        run_dir = RUNS_DIR / f"event-{result['event_date']}-{cache_key}"
        try:
            cached_path = run_dir / "analysis.json"
            stats = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
            cache_has_osm = bool(stats and stats.get("osm_feature_counts"))
            if stats is None or (os.environ.get("OHSOME_API_KEY", "").strip() and not cache_has_osm):
                with st.spinner("Downloading selected raster bands, mapping water change, and checking pre-event roads…"):
                    osm_layers = fetch_osm_layers(tuple(result["bbox"]), snapshot,
                                                  date.fromisoformat(result["event_date"]), timeout=35)
                    paired_items = {sensor: tuple(_item_dict(item) for item in pair)
                                    for sensor, pair in result["paired"].items()}
                    stats = analyze_event(paired_items, result["bbox"], osm_layers, run_dir)
            write_run_manifest(run_dir, result, stats, snapshot)
            st.session_state["analysis_result"] = {"stats": stats, "run_dir": str(run_dir)}
        except Exception as exc:
            st.warning(f"Pre-event OSM lookup did not finish: {exc}. Running image analysis without exposure and road-connectivity layers.")
            try:
                stats = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8")) if (run_dir / "analysis.json").exists() else None
                if stats is None:
                    with st.spinner("Analyzing the available satellite imagery…"):
                        paired_items = {sensor: tuple(_item_dict(item) for item in pair)
                                        for sensor, pair in result["paired"].items()}
                        stats = analyze_event(paired_items, result["bbox"], {}, run_dir)
                if "OSM data unavailable; building, bridge, road, and isolation results are incomplete." not in stats["warnings"]:
                    stats["warnings"].append("OSM data unavailable; building, bridge, road, and isolation results are incomplete.")
                write_run_manifest(run_dir, result, stats, snapshot)
                st.session_state["analysis_result"] = {"stats": stats, "run_dir": str(run_dir)}
            except Exception as analysis_exc:
                st.error(f"Image analysis could not complete: {analysis_exc}")
        except Exception as exc:
            st.error(f"Analysis could not complete: {exc}")

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        stats = analysis_result["stats"]
        run_dir = Path(analysis_result["run_dir"])
        st.markdown("### Event analysis")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Candidate water", metric_text(stats["candidate_flood_area_ha"], " ha"))
        metric_cols[1].metric("Disturbed surface", metric_text(stats["disturbed_bare_surface_candidate_area_ha"], " ha"))
        metric_cols[2].metric("Potentially exposed buildings / bridges", f"{metric_text(stats['buildings_intersecting_candidate_water'])} / {metric_text(stats['bridges_intersecting_candidate_water'])}")
        metric_cols[3].metric("Roads crossing candidate water", metric_text(stats["road_length_intersecting_candidate_water_km"], " km"))
        metric_cols[4].metric("Potentially disconnected settlements", metric_text(stats["settlements_disconnected_from_towns_or_healthcare"]))
        st.caption(f"Analyzed: {', '.join(stats['sensors_analyzed'])} · {stats['pixel_resolution_m']} m grid · OSM snapshot {snapshot.isoformat()}")
        for warning in stats["warnings"]:
            st.warning(warning)
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
            flood_path = run_dir / "flood_candidates.geojson"
            if flood_path.exists():
                flood_data = json.loads(flood_path.read_text(encoding="utf-8"))
                if flood_data["features"]:
                    folium.GeoJson(flood_data, name="Water candidates by support", style_function=lambda feature: {
                        "color": "#991b1b" if feature["properties"].get("evidence_support") == "cross_sensor" else "#f59e0b",
                        "fillColor": "#991b1b" if feature["properties"].get("evidence_support") == "cross_sensor" else "#f59e0b",
                        "fillOpacity": .42, "weight": 2}, tooltip=folium.GeoJsonTooltip(
                            fields=["evidence_support", "confidence_interpretation", "area_ha"])).add_to(impact_map)
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
            buildings_path = run_dir / "exposed_buildings.geojson"
            if buildings_path.exists():
                buildings_data = json.loads(buildings_path.read_text(encoding="utf-8"))
                if buildings_data["features"]:
                    folium.GeoJson(buildings_data, name="Potentially exposed buildings", style_function=lambda _: {
                        "color": "#d946ef", "fillColor": "#d946ef", "fillOpacity": .45, "weight": 2}).add_to(impact_map)
            bridges_path = run_dir / "exposed_bridges.geojson"
            if bridges_path.exists():
                bridges_data = json.loads(bridges_path.read_text(encoding="utf-8"))
                if bridges_data["features"]:
                    folium.GeoJson(bridges_data, name="Potentially exposed bridges", style_function=lambda _: {
                        "color": "#2563eb", "weight": 4}).add_to(impact_map)
            isolated_path = run_dir / "isolated_settlements.geojson"
            if isolated_path.exists():
                isolated_data = json.loads(isolated_path.read_text(encoding="utf-8"))
                if isolated_data["features"]:
                    folium.GeoJson(isolated_data, name="Potentially disconnected settlements").add_to(impact_map)
            dest_lat, dest_lon, destination_source = incident_point(flood_path, loc)
            folium.Marker([dest_lat, dest_lon], tooltip="Suggested incident destination",
                          popup=f"Suggested incident destination: {destination_source}",
                          icon=folium.Icon(color="red", icon="exclamation-sign", prefix="glyphicon")).add_to(impact_map)
            folium.LayerControl().add_to(impact_map)
            st_folium(impact_map, height=370, width=700, returned_objects=[])
        st.caption(f"Analysis files saved to `{analysis_result['run_dir']}`. Candidate-water polygons, potentially exposed asset GeoJSON, three-state class GeoTIFF (1 candidate, 2 observed/no candidate, 3 unknown), confidence GeoTIFF, and source-scene previews are included.")
        st.info(stats["note"])

        st.markdown("### Rescuer navigation")
        st.caption(f"Suggested incident destination: {destination_source} · {dest_lat:.5f}, {dest_lon:.5f}. This is a mapped candidate, not a field-verified incident location.")
        st.info("For a GPS start point, choose the button below and allow location access when your browser asks. If access is denied or unavailable, enter your starting coordinates. Opening Google Maps shares the start and destination coordinates with Google.")
        gps_fix = gps_location(key="rescue_gps_location")
        if isinstance(gps_fix, dict) and gps_fix.get("lat") is not None and gps_fix.get("lon") is not None:
            fix_id = f"{gps_fix['lat']:.6f},{gps_fix['lon']:.6f}"
            if st.session_state.get("gps_fix_id") != fix_id:
                st.session_state["start_lat"] = float(gps_fix["lat"])
                st.session_state["start_lon"] = float(gps_fix["lon"])
                st.session_state["gps_fix_id"] = fix_id
        st.caption("Use the suggested incident site, or edit its coordinates if responders have a verified destination.")
        start_col, dest_col = st.columns(2)
        with start_col:
            st.markdown("**Responder start point**")
            start_lat = st.number_input("Start latitude", min_value=-90.0, max_value=90.0,
                                        value=float(st.session_state.get("start_lat", loc["lat"])),
                                        format="%.6f", key="start_lat")
            start_lon = st.number_input("Start longitude", min_value=-180.0, max_value=180.0,
                                        value=float(st.session_state.get("start_lon", loc["lon"])),
                                        format="%.6f", key="start_lon")
        with dest_col:
            st.markdown("**Suggested incident site**")
            dest_lat = st.number_input("Destination latitude", min_value=-90.0, max_value=90.0,
                                       value=float(dest_lat), format="%.6f", key="incident_dest_lat")
            dest_lon = st.number_input("Destination longitude", min_value=-180.0, max_value=180.0,
                                       value=float(dest_lon), format="%.6f", key="incident_dest_lon")
        maps_url = "https://www.google.com/maps/dir/?" + urlencode({
            "api": "1", "origin": f"{start_lat:.6f},{start_lon:.6f}",
            "destination": f"{dest_lat:.6f},{dest_lon:.6f}", "travelmode": "driving",
        })
        st.link_button("Open driving directions in Google Maps", maps_url, type="primary")
        st.warning("Google Maps provides route guidance; it does not verify flood or road conditions. Confirm route safety with local responders before travel.")
    if st.button("Draft source-grounded situation note", disabled=not result["paired"]):
        facts = {"location": loc["name"], "event_date": result["event_date"], "catalog_scene_pairs": {}}
        for sensor, pair in result["paired"].items():
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

st.divider()
st.caption("Contains modified Copernicus Sentinel data 2026. © OpenStreetMap contributors. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. Educational prototype, not an operational tool.")
