# RescueGrid

RescueGrid gives rescue teams one fast workflow: search for a place, choose an event date, compare the nearest Sentinel-1 and Sentinel-2 scenes, and map candidate changes.

## Run the dashboard

1. Use Python 3.11 or newer and install `requirements.txt`.
2. Copy `.env.example` to `.env`. Add `GROQ_API_KEY` for the optional bilingual situation note and `OHSOME_API_KEY` for historical roads, bridges, buildings, settlements, and healthcare layers.
3. Start the app with `streamlit run app/dashboard.py`.
4. Search a place or enter `latitude,longitude`, choose the event date, and run imagery analysis.

The dashboard uses Sentinel-1 on the same relative orbit and pass before and after the event. It selects Sentinel-2 observations nearest the date, with cloud cover as a tie-breaker. It compares radar backscatter and optical water/vegetation signals on a bounded local grid, then overlays candidate areas with the pre-event OSM snapshot. The Groq report is generated only when requested, uses only scene metadata and measured map statistics, and is capped at 220 completion tokens with low reasoning effort.

Analysis exports are saved under `runs/`. Candidate-water and disturbed-ground detections are screening estimates; imagery alone cannot confirm damage or debris. If historical OSM data is unavailable, exposure and road-access metrics are shown as unknown rather than zero. Upstream tracing and travel-time routing are not part of this focused dashboard.

The repository's training, evaluation, report, and CLI modules remain available separately. The dashboard does not call the demo's synthetic fallback paths.
