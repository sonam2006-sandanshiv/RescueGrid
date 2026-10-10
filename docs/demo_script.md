# Demo Script (3-minute)

## 0:00 - 0:30: Introduction & Dashboard
- **Visual**: Show the Streamlit Dashboard.
- **Narration**: "Welcome to our Flood Damage Mapping prototype. For Track B, our goal is to answer where the flood hit, what was damaged, and who is cut off. We achieve this autonomously using open data. Let's run the Bhote Koshi case study."
- **Action**: Click 'Run Pipeline' on the dashboard.

## 0:30 - 1:15: Methodology & Rules
- **Visual**: Show the backend logs scrolling / manifest generation.
- **Narration**: "The system queries the Copernicus STAC API for Sentinel-1 and 2. It strictly enforces the same-orbit rule for radar, rejecting mismatched tracks. We also pull OpenStreetMap data via the ohsome API, enforcing a strict pre-event snapshot date to avoid data leakage."

## 1:15 - 2:00: AI & Classical Fusion
- **Visual**: Toggle map layers between Classical, U-Net, and Fusion.
- **Narration**: "We use a ResNet34 U-Net trained on the Kuro Siwo dataset. Since Himalayan terrain is extreme, we fuse this with a classical heuristic baseline that masks out steep slopes and high elevations using the Copernicus DEM, minimizing false positives from radar shadow."

## 2:00 - 2:45: Damage & Network Routing
- **Visual**: Show the Cut-off Settlements layer (Red/Amber/Green).
- **Narration**: "We buffer the flood extent by 20m and intersect it with OSM infrastructure. We then build a NetworkX road graph. By penalizing flooded edges, we run Dijkstra's algorithm to determine exactly which settlements have lost road access to the nearest hospital or town."

## 2:45 - 3:00: Reporting & Conclusion
- **Visual**: Open the generated HTML/PDF Situation Report.
- **Narration**: "Finally, a localized situation report is generated with hard, verifiable metrics—never invented by an LLM. Thank you."
