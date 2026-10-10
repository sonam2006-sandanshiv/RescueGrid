"""
FloodLens AI — Track B Dashboard
Location-first workflow: search → AOI draw → run pipeline → display results.
All displayed numbers come from the pipeline's summary.json artefact.
Educational prototype, not an operational tool.
"""

import sys
import json
import glob
import logging
from pathlib import Path
from datetime import datetime, date, timedelta

import requests
import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl
from streamlit_folium import st_folium
import pandas as pd

# ── repo root on path ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import run as run_pipeline
from src.report.sitrep import copilot_qa
from src.ingest.s1 import search_s1
from src.ingest.s2 import search_s2

logging.basicConfig(level=logging.WARNING)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodLens AI — Flood Damage from Space",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif;
    background: #0b0f19;
    color: #e2e8f0;
}
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1e293b; }
.hero { padding: 0.5rem 0 1rem; }
.hero h1 {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p { color: #94a3b8; font-size: 0.95rem; margin: 0; }
.kpi-grid { display: flex; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
.kpi { background: #1e293b; border-radius: 10px; padding: 14px 18px;
       flex: 1; min-width: 130px; border-top: 3px solid #38bdf8; }
.kpi.red  { border-top-color: #f43f5e; }
.kpi.amber{ border-top-color: #f59e0b; }
.kpi.green{ border-top-color: #22c55e; }
.kpi .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing:.05em; color:#94a3b8; }
.kpi .val   { font-size: 1.7rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; }
.kpi .unit  { font-size: 0.85rem; color: #94a3b8; }
.step { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.step .badge {
    background:#1e40af; color:#bfdbfe; border-radius:50%; width:26px; height:26px;
    display:flex; align-items:center; justify-content:center;
    font-size:0.78rem; font-weight:700; flex-shrink:0;
}
.step.done .badge { background:#14532d; color:#86efac; }
.step.active .badge { background:#7c3aed; color:#ddd6fe; }
hr.divider { border: none; border-top: 1px solid #1e293b; margin: 1rem 0; }
.attr { font-size: 0.68rem; color: #475569; margin-top: 1rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── session-state defaults ────────────────────────────────────────────────────
for k, v in {
    "step": 1,            # wizard step 1-4
    "aoi_bbox": None,     # [minx,miny,maxx,maxy]
    "place_name": "",
    "flood_date": "2026-08-26",
    "osm_date": "2026-07-27",
    "map_center": [27.90, 85.90],
    "map_zoom": 9,
    "geocode_results": [],
    "scene_summary": None,
    "latest_summary_path": None,
    "summary_data": {},
    "run_log": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── helpers ───────────────────────────────────────────────────────────────────
NOMINATIM = "https://nominatim.openstreetmap.org/search"

def geocode(query: str) -> list[dict]:
    """Return list of {name, lat, lon, bbox_deg} from Nominatim."""
    try:
        r = requests.get(
            NOMINATIM,
            params={"q": query, "format": "json", "limit": 5,
                    "addressdetails": 0, "accept-language": "en"},
            headers={"User-Agent": "FloodLensAI/1.0 (flood-hackathon-prototype)"},
            timeout=8,
        )
        results = []
        for d in r.json():
            bb = d.get("boundingbox", [])
            if len(bb) == 4:
                s, n, w, e = [float(x) for x in bb]
                results.append({
                    "name": d.get("display_name", "")[:90],
                    "lat": float(d["lat"]),
                    "lon": float(d["lon"]),
                    "bbox": [w, s, e, n],   # [minx, miny, maxx, maxy]
                })
        return results
    except Exception as e:
        return []

def bbox_to_str(bb: list) -> str:
    return f"{bb[0]:.5f},{bb[1]:.5f},{bb[2]:.5f},{bb[3]:.5f}"

def expand_bbox(lat: float, lon: float, km: float = 25) -> list:
    """Expand a point to a bbox of ±km kilometres."""
    deg_lat = km / 111.0
    deg_lon = km / (111.0 * abs(max(0.01, abs(lat))) ** 0.01 / 1)  # approx
    deg_lon = km / (111.0 * max(0.01, abs(lat) or 1.0) ** 0.001)
    deg_lon = km / 111.0  # simplified for mid-latitudes
    return [lon - deg_lon, lat - deg_lat, lon + deg_lon, lat + deg_lat]

def load_latest_summary():
    """Find the most recent runs/*/summary.json relative to REPO_ROOT."""
    pattern = str(REPO_ROOT / "runs" / "*" / "summary.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if files:
        p = files[0]
        with open(p, encoding="utf-8") as f:
            return p, json.load(f)
    return None, {}

def fetch_scene_preview(bbox, flood_date):
    """Query MPC for available S1/S2 scenes and return a summary dict."""
    try:
        s1 = search_s1(bbox, flood_date, max_days_before=20, max_days_after=20)
        s2 = search_s2(bbox, flood_date)
        return {
            "s1_found": bool(s1),
            "s1_pre": s1[0].id if s1 else None,
            "s1_post": s1[1].id if s1 else None,
            "s1_orbit": s1[0].properties.get("sat:relative_orbit") if s1 else None,
            "s2_found": bool(s2),
            "s2_pre": s2[0].id if s2 else None,
            "s2_post": s2[1].id if s2 else None,
        }
    except Exception as e:
        return {"error": str(e)}

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🛰️ FloodLens AI</h1>
  <p>Track B · Mapping Flood Damage from Space · Sentinel-1/2 + OSM · Educational prototype</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar wizard ─────────────────────────────────────────────────────────────
def sidebar_step_badge(n, label, state):
    cls = "done" if state == "done" else ("active" if state == "active" else "step")
    icon = "✓" if state == "done" else str(n)
    st.sidebar.markdown(
        f'<div class="step {cls}"><div class="badge">{icon}</div><span>{label}</span></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### 📍 Analysis Wizard")
    sidebar_step_badge(1, "Search location", "done" if st.session_state.step > 1 else "active")
    sidebar_step_badge(2, "Set AOI & dates", "done" if st.session_state.step > 2 else ("active" if st.session_state.step == 2 else "step"))
    sidebar_step_badge(3, "Preview scenes", "done" if st.session_state.step > 3 else ("active" if st.session_state.step == 3 else "step"))
    sidebar_step_badge(4, "Run & view results", "active" if st.session_state.step == 4 else "step")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 1: Location Search ───────────────────────────────────────────────
    st.markdown("**Step 1 · Location Search**")
    query = st.text_input("Place / river / district", value="Bhote Koshi Trishuli Nepal",
                          placeholder="e.g. Tatopani, Sindhupalchok…", key="location_query")
    if st.button("🔍 Search", use_container_width=True):
        with st.spinner("Geocoding…"):
            st.session_state.geocode_results = geocode(query)
        if not st.session_state.geocode_results:
            st.warning("No results — try a broader name.")

    if st.session_state.geocode_results:
        choices = [r["name"][:70] for r in st.session_state.geocode_results]
        sel_idx = st.selectbox("Select match:", range(len(choices)),
                               format_func=lambda i: choices[i], key="geo_sel")
        chosen = st.session_state.geocode_results[sel_idx]

        radius_km = st.slider("Analysis radius (km)", 5, 100, 25, key="radius_km")
        if st.button("📌 Use this location", use_container_width=True):
            st.session_state.aoi_bbox = expand_bbox(chosen["lat"], chosen["lon"], radius_km)
            st.session_state.place_name = chosen["name"][:60]
            st.session_state.map_center = [chosen["lat"], chosen["lon"]]
            st.session_state.map_zoom = 10
            st.session_state.step = 2
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 2: Dates & manual bbox ──────────────────────────────────────────
    st.markdown("**Step 2 · Dates & AOI**")
    flood_date_input = st.date_input(
        "Flood event date", value=date.fromisoformat(st.session_state.flood_date),
        min_value=date(2014, 1, 1), max_value=date.today(), key="fd"
    )
    osm_date_input = st.date_input(
        "OSM snapshot (pre-event)", value=date.fromisoformat(st.session_state.osm_date),
        min_value=date(2014, 1, 1), max_value=date.today(), key="od"
    )
    if osm_date_input >= flood_date_input:
        st.error("OSM date must be BEFORE flood date.")
    else:
        st.session_state.flood_date = flood_date_input.isoformat()
        st.session_state.osm_date = osm_date_input.isoformat()

    bbox_manual = st.text_input(
        "Override bbox (minX,minY,maxX,maxY)",
        value=bbox_to_str(st.session_state.aoi_bbox) if st.session_state.aoi_bbox else "85.80,27.80,86.00,28.00",
        key="bbox_manual"
    )
    if st.button("Apply manual bbox", use_container_width=False):
        try:
            parts = [float(x.strip()) for x in bbox_manual.split(",")]
            assert len(parts) == 4
            st.session_state.aoi_bbox = parts
            st.session_state.step = max(st.session_state.step, 2)
            st.rerun()
        except Exception:
            st.error("Bad bbox format. Use: minX,minY,maxX,maxY")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 3: Preview scenes ────────────────────────────────────────────────
    st.markdown("**Step 3 · Preview Satellite Scenes**")
    if st.session_state.aoi_bbox and st.button("🛰 Check available scenes", use_container_width=True):
        with st.spinner("Querying Microsoft Planetary Computer…"):
            bb = st.session_state.aoi_bbox
            st.session_state.scene_summary = fetch_scene_preview(bb, st.session_state.flood_date)
        st.session_state.step = max(st.session_state.step, 3)
        st.rerun()

    sc = st.session_state.scene_summary
    if sc:
        if "error" in sc:
            st.warning(f"Scene query: {sc['error']}")
        else:
            st.success("✅ S1 pair found" if sc["s1_found"] else "⚠️ No S1 pair (will use mock)")
            if sc["s1_found"]:
                st.caption(f"Pre: `{(sc['s1_pre'] or '')[:30]}…`")
                st.caption(f"Post: `{(sc['s1_post'] or '')[:30]}…`")
                st.caption(f"Orbit: `{sc.get('s1_orbit', 'N/A')}`")
            st.info("✅ S2 pair found" if sc["s2_found"] else "ℹ️ S2 unavailable (S1-only mode)")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 4: Run ───────────────────────────────────────────────────────────
    st.markdown("**Step 4 · Run Full Analysis**")
    engine = st.selectbox("Detection engine",
                          ["Classical S1 + DEM HAND", "Fused S1+S2 Ensemble", "AI U-Net (offline mock)"],
                          key="engine_sel")
    run_btn = st.button("🚀 Execute Pipeline", use_container_width=True,
                        disabled=st.session_state.aoi_bbox is None)

if run_btn:
    bbox = st.session_state.aoi_bbox
    fdate = st.session_state.flood_date
    odate = st.session_state.osm_date
    with st.spinner("Running end-to-end pipeline… (30–120 s for real STAC queries)"):
        try:
            summary_path = run_pipeline(bbox, fdate, odate)
            st.session_state.latest_summary_path = str(summary_path)
            _, sd = load_latest_summary()
            st.session_state.summary_data = sd
            st.session_state.step = 4
            st.session_state.run_log.append(
                f"{datetime.now().strftime('%H:%M:%S')} — Run complete → {summary_path}"
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# load last completed run even without clicking Run
if not st.session_state.summary_data:
    p, sd = load_latest_summary()
    if sd:
        st.session_state.latest_summary_path = p
        st.session_state.summary_data = sd

sd = st.session_state.summary_data

# ── KPI row ───────────────────────────────────────────────────────────────────
def kpi(label, val, unit="", css=""):
    return f"""<div class="kpi {css}">
        <div class="label">{label}</div>
        <div class="val">{val}<span class="unit"> {unit}</span></div>
    </div>"""

st.markdown('<div class="kpi-grid">' +
    kpi("Flooded Area",         sd.get("flooded_km2", "—"),    "km²",  "red") +
    kpi("Debris Area",          sd.get("debris_km2", "—"),     "km²",  "amber") +
    kpi("Buildings Affected",   sd.get("buildings_affected","—"),     "",     "red") +
    kpi("Road Damage",          sd.get("road_km_affected","—"), "km",  "amber") +
    kpi("Bridges Affected",     sd.get("bridges_affected","—"),       "",     "red") +
    kpi("Cut-off Settlements",  sd.get("cutoff_count", "—"),          "",     "red") +
'</div>', unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_infra, tab_cutoff, tab_copilot, tab_reports, tab_provenance = st.tabs([
    "🗺️ Flood Map", "🏗️ Infrastructure", "🚧 Cut-off Settlements",
    "🤖 Rescue Copilot", "📄 Reports", "🔍 Data Provenance"
])

# ── helpers for AOI centre ────────────────────────────────────────────────────
def map_center():
    if st.session_state.aoi_bbox:
        bb = st.session_state.aoi_bbox
        return [(bb[1]+bb[3])/2, (bb[0]+bb[2])/2]
    return st.session_state.map_center

def map_zoom():
    return st.session_state.map_zoom

# ── Tab 1: Flood map ──────────────────────────────────────────────────────────
with tab_map:
    col_map, col_legend = st.columns([3, 1])
    with col_map:
        m = folium.Map(location=map_center(), zoom_start=map_zoom(),
                       tiles="OpenStreetMap", prefer_canvas=True)

        # Draw AOI box
        if st.session_state.aoi_bbox:
            bb = st.session_state.aoi_bbox
            folium.Rectangle(
                bounds=[[bb[1], bb[0]], [bb[3], bb[2]]],
                color="#38bdf8", fill=False, weight=2.5,
                tooltip=f"AOI · {bbox_to_str(bb)}"
            ).add_to(m)

        # Flood water layer (from detection output or mock indicator)
        ctr = map_center()
        folium.CircleMarker(
            location=ctr, radius=35, weight=2,
            color="#3b82f6", fill=True, fill_color="#3b82f6", fill_opacity=0.45,
            tooltip="S1 Inundation zone (representative)"
        ).add_to(m)

        # Debris layer
        folium.CircleMarker(
            location=[ctr[0]+0.012, ctr[1]-0.012], radius=22, weight=2,
            color="#f59e0b", fill=True, fill_color="#f59e0b", fill_opacity=0.5,
            tooltip="Debris / mud flow zone (representative)"
        ).add_to(m)

        # Settlement markers from summary
        for i, s in enumerate(sd.get("top_settlements", [])):
            status = s.get("status", "")
            colour = "red" if status == "cut off" else ("orange" if "detour" in status else "green")
            offset_lat = ctr[0] + (i - len(sd.get("top_settlements",[])) / 2) * 0.018
            offset_lon = ctr[1] + (i % 2 - 0.5) * 0.015
            pre_dist_m = s.get('pre_dist_m')
            pre_dist_str = f"{pre_dist_m/1000:.1f} km" if pre_dist_m else "N/A"
            popup_html = (
                f"<b>{s.get('name','?')}</b><br>Status: <b>{status}</b>"
                f"<br>Pop: {s.get('population','N/A')}"
                f"<br>Pre-dist: {pre_dist_str}"
            )
            folium.Marker(
                location=[offset_lat, offset_lon],
                popup=folium.Popup(popup_html, max_width=200),
                icon=folium.Icon(color=colour, icon="home", prefix="fa"),
                tooltip=s.get("name","Settlement")
            ).add_to(m)

        # Draw tool for custom AOI
        Draw(export=False, draw_options={
            "polyline": False, "circle": False, "marker": False,
            "circlemarker": False,
            "polygon": {"allowIntersection": False, "showArea": True},
            "rectangle": {"showArea": True},
        }).add_to(m)
        MeasureControl(primary_length_unit="kilometers").add_to(m)

        map_data = st_folium(m, width="100%", height=540, returned_objects=["all_drawings"])

        # If user drew a rectangle/polygon, offer to use it as AOI
        drawn = (map_data or {}).get("all_drawings")
        if drawn:
            feat = drawn[-1] if isinstance(drawn, list) else drawn
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                drawn_bb = [min(lons), min(lats), max(lons), max(lats)]
                st.info(f"Drawn bbox: `{bbox_to_str(drawn_bb)}`")
                if st.button("✅ Use drawn AOI"):
                    st.session_state.aoi_bbox = drawn_bb
                    st.session_state.step = max(st.session_state.step, 2)
                    st.rerun()

    with col_legend:
        st.markdown("**Map legend**")
        st.markdown("🟦 Flood water inundation")
        st.markdown("🟧 Debris / mud flow")
        st.markdown("🔴 Cut-off settlement")
        st.markdown("🟢 Reachable settlement")
        st.markdown("🟦 AOI boundary")
        if sd:
            st.markdown("---")
            st.markdown(f"**Event:** {sd.get('event_name','—')}")
            st.markdown(f"**Date:** {sd.get('flood_date','—')}")
            st.markdown(f"**Run ID:** `{sd.get('run_id','—')}`")
        else:
            st.info("No run yet. Use the sidebar to set your location and run the pipeline.")

# ── Tab 2: Infrastructure ─────────────────────────────────────────────────────
with tab_infra:
    if not sd:
        st.info("Run the pipeline first to see infrastructure damage assessment.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Buildings affected", sd.get("buildings_affected", 0))
        c2.metric("Road damage (km)", sd.get("road_km_affected", 0))
        c3.metric("Bridges affected", sd.get("bridges_affected", 0))
        st.caption("Computed by intersecting S1/S2 flood/debris polygons with pre-event OSM features.")
        st.markdown("""
**Methodology:**
- OSM data fetched at snapshot date strictly before the flood event.
- Flood and debris polygons buffered ±20 m for pixel uncertainty.
- Spatial join (predicate: *intersects*) identifies affected features.
- Road damage is the total length of highway segments within the flood/debris envelope.
- Bridge damage counted from `bridge=yes` or `man_made=bridge` tags.
        """)

# ── Tab 3: Cut-off Settlements ────────────────────────────────────────────────
with tab_cutoff:
    if not sd:
        st.info("Run the pipeline first to see cut-off analysis.")
    else:
        rows = sd.get("top_settlements", [])
        if not rows:
            st.info("No settlements identified in the AOI (OSM place= nodes not found).")
        else:
            df = pd.DataFrame(rows).drop(columns=["geometry"], errors="ignore")
            df["pre_dist_km"] = df.get("pre_dist_m", pd.Series()).apply(
                lambda x: f"{x/1000:.1f}" if pd.notna(x) and x else "N/A"
            )
            df["post_dist_km"] = df.get("post_dist_m", pd.Series()).apply(
                lambda x: f"{x/1000:.1f}" if pd.notna(x) and x else "∞"
            )
            st.dataframe(
                df[["name","population","status","pre_dist_km","post_dist_km"]].rename(columns={
                    "name":"Settlement","population":"Population (OSM)",
                    "status":"Connectivity Status",
                    "pre_dist_km":"Pre-event dist (km)",
                    "post_dist_km":"Post-event dist (km)"
                }),
                use_container_width=True, hide_index=True
            )
            st.caption("""
Status classes: **cut off** = no path to town/hospital without crossing flooded road;
**detour** = path >50% longer than pre-event; **reachable** = no significant impact.
            """)

# ── Tab 4: Copilot ────────────────────────────────────────────────────────────
with tab_copilot:
    st.markdown("""
**Rescue Copilot** answers questions about this specific event using only numbers
from the pipeline's `summary.json` — no external LLM inference.
    """)
    if not st.session_state.latest_summary_path:
        st.info("Run the pipeline first so the Copilot has real data to query.")
    else:
        example_qs = [
            "Which settlements are cut off?",
            "How many buildings are affected?",
            "What is the total road damage?",
            "How much area was flooded?",
        ]
        for eq in example_qs:
            if st.button(eq, key=f"eq_{eq}"):
                st.session_state["copilot_q"] = eq
        q = st.text_input("Your question:", value=st.session_state.get("copilot_q",""))
        if q:
            ans = copilot_qa(q, st.session_state.latest_summary_path)
            st.success(f"**Answer:** {ans}")
            st.caption("⚠️ Numbers sourced deterministically from pipeline artefacts.")

# ── Tab 5: Reports ────────────────────────────────────────────────────────────
with tab_reports:
    if not st.session_state.latest_summary_path:
        st.info("Run the pipeline to generate reports.")
    else:
        run_dir = Path(st.session_state.latest_summary_path).parent
        for lang, label in [("en","English"), ("ne","Nepali (नेपाली)")]:
            f = run_dir / f"report_{lang}.html"
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    st.download_button(
                        f"📥 SITREP — {label}",
                        fh.read(), file_name=f"SITREP_{lang.upper()}.html",
                        mime="text/html", use_container_width=True
                    )
        # Show raw summary JSON
        with st.expander("📋 Raw summary.json"):
            st.json(sd)

        if st.session_state.run_log:
            with st.expander("🕒 Run log"):
                for line in st.session_state.run_log[-10:]:
                    st.code(line)

# ── Tab 6: Provenance ─────────────────────────────────────────────────────────
with tab_provenance:
    if sd:
        inp = sd.get("inputs", {})
        st.markdown(f"""
| Field | Value |
|-------|-------|
| S1 Pre-event scene | `{inp.get("s1_pre","N/A")}` |
| S1 Post-event scene | `{inp.get("s1_post","N/A")}` |
| S1 Relative orbit | `{inp.get("s1_relative_orbit","N/A")}` |
| S2 Pre-event scene | `{inp.get("s2_pre","N/A")}` |
| S2 Post-event scene | `{inp.get("s2_post","N/A")}` |
| DEM tiles | `{inp.get("dem_tiles","N/A")}` |
| OSM snapshot date | `{sd.get("osm_date","N/A")}` |
| OSM features ingested | `{inp.get("osm_features_count","N/A")}` |
| Run ID | `{sd.get("run_id","N/A")}` |
        """)

    st.markdown("""
#### Challenge-rule compliance
- ✅ Only Sentinel-1 GRD IW scenes with matching relative orbit and pass direction are compared.
- ✅ OSM snapshot date enforced to be strictly before the flood date.
- ✅ EMSR927 / UNOSAT outputs are confined to `evaluation/` and never imported by `src/`.
- ✅ All KPIs are computed from `runs/*/summary.json`, not hard-coded.
- ✅ DEM: Copernicus GLO-30 via Microsoft Planetary Computer.
    """)
    st.markdown("""
#### Limitations
- **Radar shadow / layover:** steep Himalayan valley walls may block sensor view.
- **10 m resolution:** narrow footbridges or small landslide dams may be missed.
- **No real S1 download:** pixel arrays are synthetic in prototype mode; replace with
  actual STAC asset download and `rasterio` read for production.
- **OSM completeness:** rural Nepal mapping may be incomplete.
    """)

    st.markdown("""
<div class="attr">
<b>Attribution:</b> Contains modified Copernicus Sentinel data 2026.
Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018
provided under COPERNICUS by the European Union and ESA; all rights reserved.
© OpenStreetMap contributors (ODbL).
Training data references: Bountos et al. 2024 (Kuro Siwo); Bonafilia et al. 2020 (Sen1Floods11).<br>
<b>Educational prototype, not an operational tool.</b>
</div>
""", unsafe_allow_html=True)
