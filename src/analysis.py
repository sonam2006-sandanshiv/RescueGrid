"""Pixel-level flood change, OSM exposure, and access analysis.

All reported figures are derived from the selected source scenes and pre-event
OSM snapshot. Water is a candidate detection; optical/radar change is not ground truth.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import LineString, MultiLineString, Point, shape, mapping
from shapely.ops import unary_union
import networkx as nx


S1_ALIASES = {"vv": ("vv", "sigma0-vv", "sigma0_vv"),
              "vh": ("vh", "sigma0-vh", "sigma0_vh")}
S1_MASK_ALIASES = ("mask", "layover_shadow_mask", "layover-shadow-mask")
S1_MASK_ALIASES = ("mask", "layover_shadow_mask", "layover-shadow-mask")
S2_ALIASES = {"blue": ("b02", "b2", "blue"), "green": ("b03", "b3", "green"),
              "red": ("b04", "b4", "red"), "nir": ("b08", "b8", "nir"),
              "swir": ("b11", "b11_20m", "swir16"), "scl": ("scl", "scl_20m", "scene-classification")}


def _asset(item: dict[str, Any], aliases: tuple[str, ...]) -> str:
    assets = item.get("assets", {})
    for alias in aliases:
        if alias in assets and assets[alias].get("href"):
            href = assets[alias]["href"]
            return _signed_asset(href)
    lowered = {str(key).lower(): value for key, value in assets.items()}
    for alias in aliases:
        if lowered.get(alias) and lowered[alias].get("href"):
            href = lowered[alias]["href"]
            return _signed_asset(href)
    raise ValueError(f"Scene {item.get('id')} has no usable {aliases[0]} raster asset")


def _rtc_mask_valid(item: dict[str, Any], values: np.ndarray) -> np.ndarray:
    """Decode an RTC validity mask only when STAC classes document its codes."""
    assets = item.get("assets", {})
    asset = next((value for key, value in assets.items() if str(key).lower() in S1_MASK_ALIASES), None)
    classes = []
    for band in (asset or {}).get("raster:bands", []):
        classes.extend(band.get("classification:classes", []))
    valid_codes = []
    for category in classes:
        label = " ".join(str(category.get(key, "")) for key in ("name", "description", "title")).lower()
        code = category.get("value")
        if code is None:
            continue
        invalid_words = ("shadow", "layover", "invalid", "no data", "nodata")
        if any(word in label for word in invalid_words):
            continue
        if "valid" in label or "unaffected" in label or "clear" in label:
            valid_codes.append(code)
    if not valid_codes:
        raise ValueError("RTC mask has no STAC-documented valid-pixel class")
    return np.isfinite(values) & np.isin(values, valid_codes)


def _signed_asset(href: str) -> str:
    if (".blob.core.windows.net/" not in href or "planetarycomputer" in href
            or "sig=" in href.lower() or "se=" in href.lower()):
        return href
    headers = {}
    subscription_key = os.environ.get("PC_SDK_SUBSCRIPTION_KEY", "").strip()
    if subscription_key:
        headers["Ocp-Apim-Subscription-Key"] = subscription_key
    response = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign",
                            params={"href": href}, headers=headers, timeout=12)
    response.raise_for_status()
    return response.json().get("href", href)


def _grid(bbox: tuple[float, float, float, float], resolution_m: float = 20,
          max_pixels: int = 250_000):
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    crs = rasterio.crs.CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=11)
    width = max(1, math.ceil((right-left)/resolution_m))
    height = max(1, math.ceil((top-bottom)/resolution_m))
    if width * height > max_pixels:
        resolution_m *= math.sqrt(width * height / max_pixels)
        width = max(1, math.ceil((right-left)/resolution_m))
        height = max(1, math.ceil((top-bottom)/resolution_m))
    transform = from_origin(left, top, resolution_m, resolution_m)
    return crs, transform, width, height, resolution_m


def _read(item: dict[str, Any], aliases: tuple[str, ...], crs, transform, width: int, height: int):
    href = _asset(item, aliases)
    resampling = Resampling.nearest if aliases in (S2_ALIASES["scl"], S1_MASK_ALIASES) else Resampling.bilinear
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES",
                      GDAL_HTTP_TIMEOUT="20", GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(href) as src:
            vrt_options = {"crs": crs, "transform": transform, "width": width, "height": height,
                           "resampling": resampling, "nodata": src.nodata}
            with WarpedVRT(src, **vrt_options) as vrt:
                data = vrt.read(1, masked=True).astype("float32").filled(np.nan)
    return data


def _s1_water(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    valid = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
    if not valid.any():
        raise ValueError("Sentinel-1 backscatter assets contain no valid overlapping pixels")
    # Planetary Computer sentinel-1-rtc assets are terrain-corrected linear
    # gamma0 power. Convert power with 10*log10; the catalog adapter avoids
    # substituting uncalibrated amplitude-DN assets from unrelated collections.
    pre_db = 10 * np.log10(np.maximum(pre, 1e-8))
    post_db = 10 * np.log10(np.maximum(post, 1e-8))
    drop = pre_db - post_db
    # Conservative open-water candidate: low post-event VV plus a meaningful
    # decrease from the same-track pre-event observation.
    threshold = float(np.nanpercentile(post_db[valid], 35))
    candidate = valid & (post_db <= threshold) & (drop > 2.5)
    return candidate, valid, f"Sentinel-1 RTC gamma0 VV decrease >2.5 dB in low-backscatter 35th-percentile pixels (threshold {threshold:.1f} dB)"


def _s2_water(pre_green, pre_nir, pre_swir, post_green, post_nir, post_swir, pre_scl, post_scl):
    arrays = (pre_green, pre_nir, pre_swir, post_green, post_nir, post_swir)
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    # Mask clouds, cirrus, shadow, snow, and saturated pixels when SCL is present.
    if pre_scl is None or post_scl is None:
        # A clear before/after comparison cannot be assumed when either SCL is absent.
        return np.zeros(valid.shape, dtype=bool), np.zeros(valid.shape, dtype=bool), "Sentinel-2 unknown: scene classification layer unavailable"
    invalid_scl = [0, 1, 3, 8, 9, 10, 11]
    valid &= np.isfinite(pre_scl) & ~np.isin(np.rint(pre_scl).astype("int16"), invalid_scl)
    valid &= np.isfinite(post_scl) & ~np.isin(np.rint(post_scl).astype("int16"), invalid_scl)
    def mndwi(green, swir):
        return (green-swir) / (green+swir+1e-6)
    pre_index, post_index = mndwi(pre_green, pre_swir), mndwi(post_green, post_swir)
    # NDWI guard helps avoid classifying dark roofs/roads as open water.
    post_ndwi = (post_green-post_nir)/(post_green+post_nir+1e-6)
    candidate = valid & (post_index > .15) & (post_ndwi > -.05) & (pre_index < .05)
    return candidate, valid, "Sentinel-2 cloud-masked new-water candidate (MNDWI/NDWI change)"


def _s2_bare_surface_change(pre_red, pre_nir, post_red, post_nir, post_swir, pre_scl, post_scl):
    arrays = (pre_red, pre_nir, post_red, post_nir, post_swir)
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    if pre_scl is not None and post_scl is not None:
        invalid_scl = [0, 1, 3, 8, 9, 10, 11]
        valid &= np.isfinite(pre_scl) & ~np.isin(np.rint(pre_scl).astype("int16"), invalid_scl)
        valid &= np.isfinite(post_scl) & ~np.isin(np.rint(post_scl).astype("int16"), invalid_scl)
    else:
        return np.zeros(valid.shape, dtype=bool), "Disturbed-surface unknown: SCL unavailable for pre/post cloud screening"
    pre_ndvi = (pre_nir-pre_red)/(pre_nir+pre_red+1e-6)
    post_ndvi = (post_nir-post_red)/(post_nir+post_red+1e-6)
    # Vegetated before / low vegetation and elevated SWIR after is a disturbed
    # bare-surface signal. It may represent debris or a landslide, but also
    # harvest, erosion, or seasonal change; it is kept separate from water.
    candidate = valid & (pre_ndvi > .35) & (post_ndvi < .15) & ((pre_ndvi-post_ndvi) > .25)
    return candidate, "Sentinel-2 vegetation-loss/bare-surface change candidate; debris is unverified"


def _polygonize(mask: np.ndarray, transform, crs, minimum_pixels: int = 4):
    geoms, kept = [], np.zeros(mask.shape, dtype="uint8")
    # Ignore isolated speckle in the reported vector footprint.
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    keep_ids = np.where(sizes >= minimum_pixels)[0]
    keep_ids = keep_ids[keep_ids != 0]
    kept[np.isin(labels, keep_ids)] = 1
    for geom, value in shapes(kept, mask=kept.astype(bool), transform=transform, connectivity=8):
        if value:
            geoms.append(shape(transform_geom(crs, "EPSG:4326", geom)))
    return unary_union(geoms) if geoms else None, kept


def _load_geojson(payload: dict[str, Any]):
    return [(feature, shape(feature["geometry"])) for feature in payload.get("features", [])
            if feature.get("geometry")]


def _write_preview(path: Path, bands: list[np.ndarray], valid: np.ndarray | None = None):
    rgb = []
    for band in bands:
        mask = np.isfinite(band)
        if valid is not None: mask &= valid
        if not mask.any():
            rgb.append(np.zeros(band.shape, dtype="uint8")); continue
        low, high = np.nanpercentile(band[mask], (2, 98))
        scaled = np.clip((band-low)/(high-low+1e-6), 0, 1)
        rgb.append(np.where(mask, scaled*255, 0).astype("uint8"))
    with rasterio.open(path, "w", driver="PNG", width=rgb[0].shape[1], height=rgb[0].shape[0],
                       count=3, dtype="uint8") as dst:
        for index, channel in enumerate(rgb, 1): dst.write(channel, index)


def _road_access(roads, settlements, health, impact):
    lines = []
    for feature, geom in roads:
        if isinstance(geom, LineString): lines.append((feature, geom))
        elif isinstance(geom, MultiLineString): lines.extend((feature, part) for part in geom.geoms)
    graph = nx.Graph()
    edge_records = []
    # OSM way segments are connected by rounded endpoints. Flood-intersecting
    # ways are removed in the scenario graph to estimate access loss.
    for feature, line in lines:
        coords = list(line.coords)
        if len(coords) < 2: continue
        start, end = tuple(round(v, 5) for v in coords[0][:2]), tuple(round(v, 5) for v in coords[-1][:2])
        length = max(1.0, line.length * 111_000 * math.cos(math.radians(line.centroid.y)))
        graph.add_edge(start, end, length=length, affected=bool(impact and line.intersects(impact)))
        edge_records.append((start, end, feature, line, length))
    safe_graph = graph.copy()
    safe_graph.remove_edges_from([(a, b) for a, b, attrs in graph.edges(data=True) if attrs["affected"]])
    baseline_components = {node: index for index, component in enumerate(nx.connected_components(graph)) for node in component}
    safe_components = {node: index for index, component in enumerate(nx.connected_components(safe_graph)) for node in component}
    destinations = []
    for _, geom in health:
        point = geom if isinstance(geom, Point) else geom.representative_point()
        if graph: destinations.append(min(graph, key=lambda node: (node[0]-point.x)**2+(node[1]-point.y)**2))
    for feature, geom in settlements:
        if feature.get("properties", {}).get("place") not in ("city", "town"):
            continue
        point = geom if isinstance(geom, Point) else geom.representative_point()
        if graph: destinations.append(min(graph, key=lambda node: (node[0]-point.x)**2+(node[1]-point.y)**2))
    impacted_roads = [(feature, line, length) for _, _, feature, line, length in edge_records
                      if impact and line.intersects(impact)]
    isolated = []
    for feature, geom in settlements:
        point = geom if isinstance(geom, Point) else geom.representative_point()
        if not graph or not destinations: continue
        origin = min(graph, key=lambda node: (node[0]-point.x)**2+(node[1]-point.y)**2)
        reachable_components = {baseline_components[target] for target in destinations if target in baseline_components}
        safe_reachable_components = {safe_components[target] for target in destinations if target in safe_components}
        was_reachable = baseline_components.get(origin) in reachable_components
        remains_reachable = safe_components.get(origin) in safe_reachable_components
        if was_reachable and not remains_reachable:
            isolated.append(feature)
    return impacted_roads, isolated, len(set(destinations))


def analyze_event(paired: dict[str, tuple[dict[str, Any], dict[str, Any]]], bbox,
                  osm_layers: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Download selected bands, classify new water, intersect OSM layers, and save map outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    crs, transform, width, height, resolution = _grid(tuple(bbox))
    warnings, masks, methods, previews, evidences, debris_mask = [], {}, {}, {}, {}, None
    if "Sentinel-1" in paired:
        pre_item, post_item = paired["Sentinel-1"]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                before_future = pool.submit(_read, pre_item, S1_ALIASES["vv"], crs, transform, width, height)
                after_future = pool.submit(_read, post_item, S1_ALIASES["vv"], crs, transform, width, height)
                before, after = before_future.result(), after_future.result()
            candidate, valid, method = _s1_water(before, after)
            try:
                before_mask = _read(pre_item, S1_MASK_ALIASES, crs, transform, width, height)
                after_mask = _read(post_item, S1_MASK_ALIASES, crs, transform, width, height)
                valid &= _rtc_mask_valid(pre_item, before_mask)
                valid &= _rtc_mask_valid(post_item, after_mask)
                candidate &= valid
                method += "; shadow/layover/invalid pixels excluded by RTC mask"
            except ValueError:
                warnings.append("Sentinel-1 RTC layover/shadow mask is absent or its class codes are undocumented; terrain-geometry uncertainty remains")
            masks["Sentinel-1"], methods["Sentinel-1"] = candidate, method
            evidences["Sentinel-1"] = valid
            for role, array in (("before", before), ("after", after)):
                with rasterio.open(output_dir / f"sentinel1_{role}_vv.tif", "w", driver="GTiff", height=height,
                                   width=width, count=1, dtype="float32", crs=crs, transform=transform,
                                   nodata=np.nan, compress="deflate") as dst:
                    dst.write(array.astype("float32"), 1)
            previews["Sentinel-1"] = []
            for role, array in (("before", before), ("after", after)):
                valid = np.isfinite(array)
                median = np.nanmedian(array[valid])
                visual = (20 if median > 10 else 10)*np.log10(np.maximum(array, 1e-8)) if median > 0 else array
                _write_preview(output_dir / f"sentinel1_{role}.png", [visual]*3, valid)
                previews["Sentinel-1"].append(str(output_dir / f"sentinel1_{role}.png"))
        except Exception as exc:
            warnings.append(f"Sentinel-1 analysis unavailable: {exc}")
    if "Sentinel-2" in paired:
        pre_item, post_item = paired["Sentinel-2"]
        try:
            # Each COG band is an independent HTTP range read; fetching them
            # concurrently avoids serial minute-long waits on public storage.
            bands = ("green", "nir", "swir", "red")
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {(role, key): pool.submit(_read, item, S2_ALIASES[key], crs,
                                                     transform, width, height)
                           for role, item in (("before", pre_item), ("after", post_item))
                           for key in bands}
                arrays = {name: future.result() for name, future in futures.items()}
            before = [arrays[("before", key)] for key in ("green", "nir", "swir")]
            after = [arrays[("after", key)] for key in ("green", "nir", "swir")]
            pre_red, post_red = arrays[("before", "red")], arrays[("after", "red")]
            try:
                pre_scl = _read(pre_item, S2_ALIASES["scl"], crs, transform, width, height)
                post_scl = _read(post_item, S2_ALIASES["scl"], crs, transform, width, height)
            except Exception:
                pre_scl = post_scl = None
            s2_candidate, s2_valid, s2_method = _s2_water(*before, *after, pre_scl, post_scl)
            if s2_valid.any():
                masks["Sentinel-2"], methods["Sentinel-2"] = s2_candidate, s2_method
                evidences["Sentinel-2"] = s2_valid
            else:
                warnings.append(s2_method)
            if pre_scl is not None and post_scl is not None:
                debris_mask, methods["disturbed_surface"] = _s2_bare_surface_change(
                    pre_red, before[1], post_red, after[1], after[2], pre_scl, post_scl)
            else:
                warnings.append("Disturbed-surface change unknown because SCL cloud screening was unavailable")
            previews["Sentinel-2"] = []
            for role, rgb in (("before", [before[2], before[1], before[0]]),
                              ("after", [after[2], after[1], after[0]])):
                _write_preview(output_dir / f"sentinel2_{role}.png", rgb)
                previews["Sentinel-2"].append(str(output_dir / f"sentinel2_{role}.png"))
        except Exception as exc:
            warnings.append(f"Sentinel-2 analysis unavailable: {exc}")

    if not masks:
        raise RuntimeError("No selected sensor had the required readable bands. " + " ".join(warnings))
    # A detection from either sensor remains visible; corroboration is recorded
    # separately. Cloud/no-data and invalid RTC geometry remain class 3 unknown.
    combined = np.logical_or.reduce(list(masks.values()))
    observed = np.logical_or.reduce(list(evidences.values()))
    corroborated = np.logical_and(masks["Sentinel-1"], masks["Sentinel-2"]) if len(masks) > 1 else np.zeros(combined.shape, bool)
    flood_geom, cleaned = _polygonize(combined, transform, crs)
    impact_s2, _ = _polygonize(masks["Sentinel-2"], transform, crs) if "Sentinel-2" in masks else (None, None)
    debris_geom, debris_cleaned = _polygonize(debris_mask, transform, crs) if debris_mask is not None else (None, None)
    flood_area_m2 = int(cleaned.sum()) * resolution * resolution
    debris_area_m2 = int(debris_cleaned.sum()) * resolution * resolution if debris_cleaned is not None else 0
    impact_parts = [geom for geom in (flood_geom, debris_geom) if geom is not None]
    impact = unary_union(impact_parts) if impact_parts else None
    if flood_geom is None:
        warnings.append("No candidate new-water pixels passed the selected thresholds.")
    # 1 = candidate water, 2 = observed/no candidate, 3 = unknown/insufficient
    # evidence. A valid non-flood pixel is never conflated with nodata.
    pixel_class = np.full((height, width), 3, dtype="uint8")
    pixel_class[observed] = 2
    pixel_class[combined] = 1
    confidence = np.zeros((height, width), dtype="uint8")
    confidence[combined & ~corroborated] = 1
    confidence[corroborated] = 2
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
               "dtype": "uint8", "crs": crs, "transform": transform, "nodata": None, "compress": "deflate"}
    with rasterio.open(output_dir / "candidate_flood_mask.tif", "w", **profile) as dst:
        dst.write(np.where(cleaned, 1, pixel_class).astype("uint8"), 1)
        dst.set_band_description(1, "1 candidate water; 2 observed with no candidate; 3 no usable observation/unknown")
    with rasterio.open(output_dir / "candidate_confidence.tif", "w", **profile) as dst:
        dst.write(confidence, 1)
        dst.set_band_description(1, "0 no candidate; 1 single-sensor candidate; 2 cross-sensor corroborated candidate")

    layers = {name: _load_geojson(data) for name, data in osm_layers.items()}
    buildings = layers.get("buildings", [])
    roads = layers.get("roads", [])
    bridges = layers.get("bridges", [])
    settlements = layers.get("settlements", [])
    health = layers.get("health", [])
    exposed_buildings = [(feature, geom) for feature, geom in buildings if impact and geom.intersects(impact)]
    exposed_bridges = [(feature, geom) for feature, geom in bridges if impact and geom.intersects(impact)]
    impacted_roads, isolated, destination_count = _road_access(roads, settlements, health, impact)
    if not osm_layers:
        warnings.append("Pre-event OSM unavailable; exposure and access values are unknown, not zero.")
    elif not destination_count:
        warnings.append("No mapped town or healthcare destination could be connected to the road graph; settlement isolation is unknown.")

    def feature_collection(entries, geom_index=1):
        features = []
        for entry in entries:
            feature, geom = entry[0], entry[geom_index]
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": feature.get("properties", {})})
        return {"type": "FeatureCollection", "features": features}

    water_support_features = []
    for support, candidate_pixels in (("cross_sensor", corroborated),
                                      ("single_sensor", combined & ~corroborated)):
        candidate_geom, support_clean = _polygonize(candidate_pixels, transform, crs)
        if candidate_geom is not None:
            water_support_features.append({"type": "Feature", "geometry": mapping(candidate_geom),
                "properties": {"classification": "candidate_new_water", "evidence_support": support,
                    "confidence_interpretation": "cross-sensor corroboration; not a calibrated probability" if support == "cross_sensor" else "single-sensor candidate; treat as uncertain",
                    "area_ha": round(int(support_clean.sum()) * resolution * resolution / 10000, 2),
                    "method": methods.get("Sentinel-1", methods.get("Sentinel-2"))}})

    exports = {
        "flood_candidates.geojson": {"type": "FeatureCollection", "features": water_support_features},
        "exposed_buildings.geojson": feature_collection(exposed_buildings),
        "exposed_bridges.geojson": feature_collection(exposed_bridges),
        "affected_roads.geojson": feature_collection([(f, g) for f, g, _ in impacted_roads]),
        "isolated_settlements.geojson": feature_collection([(f, shape(f["geometry"])) for f in isolated]),
    }
    if impact_s2 is not None:
        exports["sentinel2_water_candidates.geojson"] = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(impact_s2), "properties": {"classification": "optical_new_water_candidate"}}]}
    if debris_geom is not None:
        exports["disturbed_surface_candidates.geojson"] = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(debris_geom), "properties": {
                "classification": "vegetation_loss_bare_surface_candidate",
                "area_ha": round(debris_area_m2/10000, 2), "interpretation": "may include debris or landslide; unverified"}}]}
    for name, content in exports.items():
        (output_dir / name).write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    road_km = sum(length for _, _, length in impacted_roads) / 1000
    stats = {"candidate_flood_area_ha": round(flood_area_m2 / 10000, 2),
             "disturbed_bare_surface_candidate_area_ha": round(debris_area_m2 / 10000, 2) if debris_mask is not None else None,
             "candidate_pixels": int(cleaned.sum()), "pixel_resolution_m": round(resolution, 1),
             "buildings_intersecting_candidate_water": len(exposed_buildings) if "buildings" in osm_layers else None,
             "bridges_intersecting_candidate_water": len(exposed_bridges) if "bridges" in osm_layers else None,
             "road_length_intersecting_candidate_water_km": round(road_km, 2) if "roads" in osm_layers else None,
             "settlements_disconnected_from_towns_or_healthcare": len(isolated) if destination_count else None,
             "sensors_analyzed": list(masks), "methods": methods, "warnings": warnings, "previews": previews,
             "observed_pixels": int(observed.sum()), "unknown_pixels": int((~observed).sum()),
             "single_sensor_candidate_pixels": int((combined & ~corroborated).sum()),
             "cross_sensor_corroborated_pixels": int(corroborated.sum()),
             "osm_feature_counts": {name: len(features) for name, features in layers.items()},
             "note": "Satellite pixels are labelled candidate water, observed without a candidate, or unknown. Water and disturbed bare-surface signals are estimates; bare-ground change is not confirmed debris. OSM intersections are potentially exposed assets, not verified damage. Permanent water, seasonal channels, vegetation flooding, and terrain bias require local validation."}
    (output_dir / "analysis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
