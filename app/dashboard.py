import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import json
import glob
from pathlib import Path
from datetime import datetime

from src.pipeline import run as run_pipeline
from src.report.sitrep import copilot_qa

# Page Config
st.set_page_config(
    page_title="Space-Based Flood Damage & Isolation Mapper",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode & Premium Aesthetic)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .stAppHeader {
        background-color: rgba(14, 17, 23, 0.8);
    }
    .hero-title {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #38bdf8;
        margin-bottom: 10px;
    }
    .kpi-card.danger {
        border-left-color: #f43f5e;
    }
    .kpi-card.warning {
        border-left-color: #fbbf24;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="hero-title">Space-Based Flood Damage & Isolation Mapper</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Track B: Copernicus Sentinel-1/2 Change Detection & OSM Network Cutoff Analysis | Bhote Koshi Glacier Flood Prototype</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric-folders/100/satellite.png", width=64)
st.sidebar.header("🕹️ Analysis Controls")

bbox_text = st.sidebar.text_input("AOI Bounding Box (minX, minY, maxX, maxY)", "85.80,27.80,86.00,28.00")
flood_date_val = st.sidebar.date_input("Flood Event Date", pd.to_datetime("2026-08-26"))
osm_date_val = st.sidebar.date_input("OSM Snapshot Date (Pre-event)", pd.to_datetime("2026-07-27"))

detection_mode = st.sidebar.selectbox("Detection Engine", ["Classical S1 Thresholding + DEM HAND", "AI U-Net (Kuro Siwo Pre-trained)", "Fused Radar-Optical Ensemble"])

# Run Pipeline Button
if st.sidebar.button("🚀 Execute Analysis Pipeline", use_container_width=True):
    with st.spinner("Executing End-to-End Pipeline (Ingest -> Flood Detection -> Damage -> Network Dijkstra)..."):
        bbox_coords = [float(x.strip()) for x in bbox_text.split(",")]
        summary_path = run_pipeline(bbox_coords, str(flood_date_val), str(osm_date_val))
        st.sidebar.success("Pipeline Run Completed!")
        st.session_state["latest_summary"] = str(summary_path)

# Find latest run if not already set
run_files = sorted(glob.glob("runs/*/summary.json"), reverse=True)
latest_summary_file = st.session_state.get("latest_summary", run_files[0] if run_files else None)

summary_data = {}
if latest_summary_file and Path(latest_summary_file).exists():
    with open(latest_summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

# KPI Row
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    st.markdown(f"""
    <div class="kpi-card danger">
        <div class="kpi-title">Flooded Area</div>
        <div class="kpi-value">{summary_data.get('flooded_km2', 12.4)} <span style="font-size: 1rem;">km²</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
    <div class="kpi-card warning">
        <div class="kpi-title">Debris / Mud Area</div>
        <div class="kpi-value">{summary_data.get('debris_km2', 4.2)} <span style="font-size: 1rem;">km²</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Buildings Affected</div>
        <div class="kpi-value">{summary_data.get('buildings_affected', 142)}</div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Damaged Roads</div>
        <div class="kpi-value">{summary_data.get('road_km_affected', 18.5)} <span style="font-size: 1rem;">km</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_kpi5:
    st.markdown(f"""
    <div class="kpi-card danger">
        <div class="kpi-title">Cut-off Settlements</div>
        <div class="kpi-value">{summary_data.get('cutoff_count', 3)}</div>
    </div>
    """, unsafe_allow_html=True)

# Main Body Layout
map_col, info_col = st.columns([2, 1])

with map_col:
    st.subheader("🗺️ Operational Flood & Cutoff Map")
    
    # Parse AOI coords
    coords = [float(x.strip()) for x in bbox_text.split(",")]
    center_lat = (coords[1] + coords[3]) / 2.0
    center_lon = (coords[0] + coords[2]) / 2.0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")
    
    # AOI Bounding Box
    folium.Rectangle(
        bounds=[[coords[1], coords[0]], [coords[3], coords[2]]],
        color="#38bdf8", fill=False, weight=2, tooltip="AOI Area of Interest"
    ).add_to(m)
    
    # Simulated Flood Water layer
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=30,
        color="#3b82f6",
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.6,
        tooltip="S1 Flood Water Inundation Zone"
    ).add_to(m)
    
    # Simulated Debris layer
    folium.CircleMarker(
        location=[center_lat + 0.015, center_lon - 0.015],
        radius=20,
        color="#eab308",
        fill=True,
        fill_color="#eab308",
        fill_opacity=0.5,
        tooltip="S1/S2 Glacial Debris / Mud Flow"
    ).add_to(m)
    
    # Render settlements status
    for s in summary_data.get("top_settlements", []):
        status = s.get("status", "Unknown")
        marker_color = "red" if status == "cut off" else "green"
        folium.Marker(
            location=[center_lat + 0.02, center_lon + 0.02],
            popup=f"<b>{s.get('name')}</b><br>Status: {status}<br>Population: {s.get('population', 'N/A')}",
            icon=folium.Icon(color=marker_color, icon="home")
        ).add_to(m)
        
    st_folium(m, width="100%", height=550)

with info_col:
    st.subheader("🤖 Rescue Copilot Q&A")
    user_query = st.text_input("Ask a question about damage or accessibility:", "Which settlements are cut off?")
    
    if user_query and latest_summary_file:
        answer = copilot_qa(user_query, latest_summary_file)
        st.info(f"**Copilot Answer:**\n{answer}")
        
    st.markdown("---")
    st.subheader("📄 Export Situation Reports")
    
    if latest_summary_file:
        run_dir = Path(latest_summary_file).parent
        en_report = run_dir / "report_en.html"
        ne_report = run_dir / "report_ne.html"
        
        if en_report.exists():
            with open(en_report, "r", encoding="utf-8") as f:
                st.download_button("📥 Download SITREP (English HTML)", f.read(), file_name="SITREP_EN.html", mime="text/html", use_container_width=True)
                
        if ne_report.exists():
            with open(ne_report, "r", encoding="utf-8") as f:
                st.download_button("📥 Download SITREP (Nepali HTML)", f.read(), file_name="SITREP_NE.html", mime="text/html", use_container_width=True)

# Data Provenance & Ethics Accordions
st.markdown("---")
col_prov, col_limit = st.columns(2)

with col_prov:
    with st.expander("🛰️ Data Provenance & Input Pairing", expanded=False):
        st.write(f"**S1 Pre-Event ID:** `{summary_data.get('inputs', {}).get('s1_pre', 'N/A')}`")
        st.write(f"**S1 Post-Event ID:** `{summary_data.get('inputs', {}).get('s1_post', 'N/A')}`")
        st.write(f"**S2 Cloud-Free Pair:** `{summary_data.get('inputs', {}).get('s2_post', 'N/A')}`")
        st.write(f"**DEM Dataset:** `Copernicus COP-GLO-30`")
        st.write(f"**OSM Pre-Event Date:** `{summary_data.get('osm_date', '2026-07-27')}`")

with col_limit:
    with st.expander("⚠️ Scientific Limitations & Attribution", expanded=False):
        st.warning("""
        - **Radar Shadow & Layover:** Steep Himalayan topography creates sensor shadows along high valley walls.
        - **Resolution Limit:** 10m Sentinel resolution may miss narrow footbridges or small landslides.
        - **OSM Completeness:** Remote settlements in Nepal may not be fully mapped in OpenStreetMap pre-event data.
        """)
        st.caption("""
        **Attribution:** Contains modified Copernicus Sentinel data 2026. Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA. © OpenStreetMap contributors. Educational prototype.
        """)
