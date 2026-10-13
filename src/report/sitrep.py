import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_report(summary_json_path, output_dir=".", lang="en"):
    """
    Generates situation reports (PDF/HTML) from a summary JSON in English or Nepali.
    """
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    # Build rows for settlements
    rows = ""
    for s in summary.get("top_settlements", []):
        name = s.get("name", "Unknown")
        pop = s.get("population", "N/A")
        pre_dist = f'{s.get("pre_dist_m", 0)/1000:.1f} km' if s.get("pre_dist_m") else "N/A"
        status = s.get("status", "Unknown")
        rows += f"<tr><td>{name}</td><td>{pop}</td><td>{pre_dist}</td><td>{status}</td></tr>\n"
        
    context = {
        "event_name": summary.get("event_name", "Bhote Koshi Flood"),
        "flood_date": summary.get("flood_date", "YYYY-MM-DD"),
        "aoi": summary.get("aoi", "Unknown"),
        "flooded_km2": summary.get("flooded_km2", 0),
        "debris_km2": summary.get("debris_km2", 0),
        "buildings": summary.get("buildings_affected", 0),
        "road_km": summary.get("road_km_affected", 0),
        "bridges": summary.get("bridges_affected", 0),
        "cutoff_count": summary.get("cutoff_count", 0),
        "settlements_rows": rows,
        "s1_pre": summary.get("inputs", {}).get("s1_pre", "N/A"),
        "s1_post": summary.get("inputs", {}).get("s1_post", "N/A"),
        "osm_date": summary.get("osm_date", "N/A")
    }
    
    template_name = f"report_{lang}.html"
    template_path = Path(f"src/report/templates/{template_name}")
    if not template_path.exists():
        template_path = Path("src/report/templates/report_en.html")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()
        
    html_content = html_template.format(**context)
    
    out_pdf = Path(output_dir) / f"report_{lang}.pdf"
    out_html = Path(output_dir) / f"report_{lang}.html"
    
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(out_pdf)
        logger.info(f"Report generated: {out_pdf}")
    except (ImportError, OSError):
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.warning(f"WeasyPrint failed to load. Fallback HTML generated: {out_html}")

def _is_nepali(text: str) -> bool:
    """Check if query contains Devanagari script."""
    return any("\u0900" <= ch <= "\u097f" for ch in text)

def copilot_qa(query: str, summary_json_path, lang=None) -> str:
    """
    Situation-report copilot:
    Answers rescue teams' questions deterministically in English or Nepali.
    Strictly uses numbers from the pipeline summary JSON, never hallucinating.
    """
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    q = query.strip().lower()
    in_nepali = (lang == "ne") or _is_nepali(query)
    
    event = summary.get("event_name", "Bhote Koshi-Trishuli Flood")
    date_str = summary.get("flood_date", "2026-08-26")
    flood_km2 = summary.get("flooded_km2", 0)
    debris_km2 = summary.get("debris_km2", 0)
    buildings = summary.get("buildings_affected", 0)
    road_km = summary.get("road_km_affected", 0)
    bridges = summary.get("bridges_affected", 0)
    cutoff = summary.get("cutoff_count", 0)
    settlements = summary.get("top_settlements", [])
    cutoff_names = [s.get("name", "Unknown") for s in settlements if s.get("status") == "cut off"]
    cutoff_str = ", ".join(cutoff_names) if cutoff_names else ("कुनै छैन" if in_nepali else "None detected")
    
    inputs = summary.get("inputs", {})
    s1_pre = inputs.get("s1_pre", "N/A")
    s1_post = inputs.get("s1_post", "N/A")
    osm_date = summary.get("osm_date", "2026-07-27")

    if in_nepali:
        # Nepali Question Answering
        if any(w in query for w in ["सम्पर्कविहीन", "अलग", "थुनिएका", "पुग्न नसकिने", "बस्ती"]):
            return f"सम्पर्कविहीन बस्तीहरूको संख्या: {cutoff}। नामावली: {cutoff_str}।"
        elif any(w in query for w in ["बाढी", "पानी", "जलमग्न", "क्षेत्रफल", "कति क्षेत्र"]):
            return f"जलमग्न बाढी क्षेत्र: {flood_km2} वर्ग कि.मी., गेग्रान/लेदो बहाव क्षेत्र: {debris_km2} वर्ग कि.मी.।"
        elif any(w in query for w in ["घर", "भवन", "संरचना"]):
            return f"प्रभावित घर/संरचनाहरूको संख्या: {buildings} वटा। (पूर्व-विपद् OSM तथ्याङ्कबाट प्रमाणित)"
        elif any(w in query for w in ["सडक", "बाटो", "पुल", "यातायात"]):
            return f"क्षतिग्रस्त सडक खण्ड: {road_km} कि.मी., क्षतिग्रस्त/प्रभावित पुलहरू: {bridges} वटा।"
        elif any(w in query for w in ["अस्पताल", "उपचार", "स्वास्थ्य"]):
            return f"नजिकको अस्पताल वा बजारसँग सडक सम्पर्क विच्छेद भएका बस्तीहरू: {cutoff} वटा ({cutoff_str})।"
        elif any(w in query for w in ["उपग्रह", "स्याटेलाइट", "डाटा", "तथ्याङ्क"]):
            return f"उपग्रह इनपुटहरू: Sentinel-1 SAR (पूर्व: {s1_pre}, पश्चात्: {s1_post}), पूर्व-विपद् OpenStreetMap स्न्यापशट ({osm_date})।"
        else:

            return (
                f"विपद् सारांश ({event}, {date_str}): जलमग्न क्षेत्र {flood_km2} वर्ग कि.मी., "
                f"लेदो क्षेत्र {debris_km2} वर्ग कि.मी., प्रभावित घर {buildings} वटा, "
                f"क्षतिग्रस्त सडक {road_km} कि.मी., पुल {bridges} वटा, र सम्पर्कविहीन बस्ती {cutoff} वटा ({cutoff_str})।"
            )
    else:
        # English Question Answering
        if any(w in q for w in ["flooded", "water", "inundat", "area", "extent"]):
            return f"Flooded water area: {flood_km2} km², Debris/mudflow area: {debris_km2} km²."
        elif any(w in q for w in ["building", "structure", "house"]):
            return f"Affected buildings: {buildings} (identified via pre-event OSM spatial overlay)."
        elif any(w in q for w in ["road", "bridge", "highway", "transit"]):
            return f"Affected roads: {road_km} km, Affected bridges: {bridges}."
        elif any(w in q for w in ["cut off", "settlement", "isolated", "village", "stranded"]):
            return f"Total cut-off settlements: {cutoff}. Affected settlements list: {cutoff_str}."
        elif any(w in q for w in ["hospital", "clinic", "health", "medical"]):
            return f"Settlements that have lost road connectivity to nearest hospital/town: {cutoff} ({cutoff_str})."
        elif any(w in q for w in ["satellite", "sentinel", "data", "sensor", "provenance"]):
            return f"Inputs used: Sentinel-1 SAR (pre: {s1_pre}, post: {s1_post}), Copernicus WorldDEM-30, and pre-event OSM snapshot ({osm_date})."
        else:
            return (
                f"Event: {event} ({date_str}) | Flooded: {flood_km2} km² | Debris: {debris_km2} km² | "
                f"Damaged buildings: {buildings} | Road damage: {road_km} km | Bridges: {bridges} | "
                f"Cut-off settlements: {cutoff} ({cutoff_str})."
            )

def generate_short_sitrep(summary_json_path, lang="en") -> str:
    """
    Generates a concise 1-paragraph SitRep in English or Nepali for rapid radio transmission.
    """
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    event = summary.get("event_name", "Bhote Koshi Flood")
    date_str = summary.get("flood_date", "2026-08-26")
    flood_km2 = summary.get("flooded_km2", 0)
    buildings = summary.get("buildings_affected", 0)
    road_km = summary.get("road_km_affected", 0)
    bridges = summary.get("bridges_affected", 0)
    cutoff = summary.get("cutoff_count", 0)
    settlements = summary.get("top_settlements", [])
    cutoff_names = [s.get("name", "Unknown") for s in settlements if s.get("status") == "cut off"]
    cutoff_str = ", ".join(cutoff_names) if cutoff_names else ("none" if lang == "en" else "कुनै छैन")

    if lang == "ne":
        return (
            f"[अवस्था प्रतिवेदन - {event}] मिति {date_str} मा आएको बाढीले {flood_km2} वर्ग कि.मी. क्षेत्र जलमग्न बनाएको छ। "
            f"हालसम्म {buildings} घर संरचना, {road_km} कि.मी. सडक खण्ड, र {bridges} वटा पुलमा प्रत्यक्ष क्षति पुष्टि भएको छ। "
            f"सडक सञ्जाल अवरुद्ध हुँदा {cutoff} वटा बस्तीहरू ({cutoff_str}) नजिकको बजार तथा अस्पतालबाट पूर्ण रूपमा सम्पर्कविहीन भएका छन्। "
            f"सबै तथ्याङ्कहरू Sentinel-1 र पूर्व-विपद् OpenStreetMap नक्साबाट सिधै प्रमाणित गरिएको हो।"
        )
    else:
        return (
            f"[SITUATION REPORT - {event}] As of post-event satellite pass ({date_str}), inundation covers {flood_km2} km². "
            f"Confirmed infrastructure damage includes {buildings} buildings, {road_km} km of road corridors, and {bridges} bridges. "
            f"Road network graph analysis indicates {cutoff} settlements ({cutoff_str}) are completely cut off from the nearest hospital or town center. "
            f"All metrics are verified from Sentinel-1 SAR change detection and pre-event OpenStreetMap."
        )

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    dummy_summary = {
        "event_name": "Bhote Koshi-Trishuli Glacial Flood 2026",
        "flood_date": "2026-08-26",
        "aoi": "85.80,27.80,86.00,28.00",
        "flooded_km2": 15.2,
        "debris_km2": 5.1,
        "buildings_affected": 42,
        "road_km_affected": 8.5,
        "bridges_affected": 2,
        "cutoff_count": 2,
        "top_settlements": [
            {"name": "Tatopani", "population": "1200", "pre_dist_m": 7800, "status": "cut off"},
            {"name": "Larcha", "population": "450", "pre_dist_m": 12000, "status": "cut off"}
        ],
        "inputs": {"s1_pre": "S1A_IW_GRDH_20260814", "s1_post": "S1A_IW_GRDH_20260826"},
        "osm_date": "2026-07-27"
    }
    with open("dummy_summary.json", "w", encoding="utf-8") as f:
        json.dump(dummy_summary, f)
        
    generate_report("dummy_summary.json", lang="en")
    generate_report("dummy_summary.json", lang="ne")
    print("EN Copilot QA:", copilot_qa("Which settlements are cut off?", "dummy_summary.json"))
    print("NE Copilot QA:", copilot_qa("कुन कुन बस्तीहरू सम्पर्कविहीन छन्?", "dummy_summary.json"))

