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

def copilot_qa(query, summary_json_path):
    """
    Deterministic Copilot Q&A tool function that queries summary JSON metrics directly.
    """
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    q = query.lower()
    if "flooded" in q or "water" in q or "area" in q:
        return f"Flooded area: {summary.get('flooded_km2', 0)} km², Debris area: {summary.get('debris_km2', 0)} km²."
    elif "building" in q or "structure" in q:
        return f"Affected buildings: {summary.get('buildings_affected', 0)}."
    elif "road" in q or "bridge" in q:
        return f"Affected roads: {summary.get('road_km_affected', 0)} km, Affected bridges: {summary.get('bridges_affected', 0)}."
    elif "cut off" in q or "settlement" in q or "isolated" in q:
        count = summary.get("cutoff_count", 0)
        settlements = summary.get("top_settlements", [])
        names = [s.get("name", "Unknown") for s in settlements if s.get("status") == "cut off"]
        return f"Total cut-off settlements: {count}. Cut-off list: {', '.join(names) if names else 'None'}."
    else:
        return f"Event: {summary.get('event_name')}, Flooded area: {summary.get('flooded_km2')} km², Damaged buildings: {summary.get('buildings_affected')}, Cut-off settlements: {summary.get('cutoff_count')}."

if __name__ == "__main__":
    dummy_summary = {
        "event_name": "Test Flood",
        "flood_date": "2026-08-26",
        "aoi": "85.0,27.8,85.5,28.3",
        "flooded_km2": 15.2,
        "debris_km2": 5.1,
        "buildings_affected": 42,
        "road_km_affected": 8.5,
        "bridges_affected": 2,
        "cutoff_count": 3,
        "top_settlements": [
            {"name": "Village A", "population": "500", "pre_dist_m": 10500, "status": "cut off"}
        ],
        "inputs": {"s1_pre": "S1_PRE_ID", "s1_post": "S1_POST_ID"},
        "osm_date": "2026-07-27"
    }
    with open("dummy_summary.json", "w", encoding="utf-8") as f:
        json.dump(dummy_summary, f)
        
    generate_report("dummy_summary.json", lang="en")
    generate_report("dummy_summary.json", lang="ne")

