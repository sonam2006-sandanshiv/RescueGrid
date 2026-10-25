# Himalayan Flood Mapper

Search a location and event date, find the nearest Sentinel-1 and Sentinel-2 scenes, map candidate new water, estimate exposed OSM infrastructure, and flag settlements whose mapped road access to a town or healthcare point may be cut.

## Project layout

- frontend/ — static web client for Vercel.
- backend/ — FastAPI adapter for Render.
- src/ — existing satellite, raster, OSM, and analysis modules used by both entry points.
- app/dashboard.py — existing Streamlit interface, retained for local use and compatibility.

The deployment folders are additive. The existing app/ and src/ paths have not been moved.

## Run

1. Install Python 3.11 and install `requirements.txt`.
2. Copy `.env.example` to `.env`; set `GROQ_API_KEY` for the optional situation-report copilot. For historic OSM layers, create a free ohsome API key and set `OHSOME_API_KEY`. The selected satellite storage provider may require account/key access; configure `PC_SDK_SUBSCRIPTION_KEY` or CDSE S3 credentials if prompted.
3. Start the interface with `streamlit run app/dashboard.py`.
4. Search a place or `latitude,longitude`, choose an event date, and select **Analyze imagery, infrastructure, and access**.

The CLI accepts a WGS84 bbox: `python -m src.cli run --bbox 85.0,27.8,85.5,28.3 --date 2026-08-26`.

## What the results mean

The workflow searches Planetary Computer for Sentinel-1 RTC and Sentinel-2 L2A, reads scene bands onto a bounded UTM grid, and maps candidate new water and disturbed bare surfaces. Sentinel-1 pairs must share relative orbit and pass direction. Sentinel-2 pairs must share an MGRS tile and be within 12 days of the event; otherwise optical comparison is unavailable. Candidate areas are intersected with buildings, bridges, and roads from the OpenStreetMap history snapshot immediately before the event. A road graph estimates settlements disconnected from mapped towns and healthcare destinations after candidate-impact-crossing road links are removed. Maps, masks, scene previews, summary statistics, and source provenance are saved under `runs/`.

The Groq copilot runs only when requested, receives the scene metadata and computed statistics, and is capped at 300 completion tokens. All numbers come from the analysis output; the model is instructed not to invent counts.

For data provenance, the pipeline searches Planetary Computer's `sentinel-1-rtc` collection for linear gamma0 scenes and applies the same-relative-orbit/pass selector. Scene timestamps, platform, collection, orbit, and asset keys are written to each run manifest; temporary signed asset URLs are omitted. Sentinel-2 L2A uses Planetary Computer first and Element84 as catalog fallback. The public STAC search works without a key; `PC_SDK_SUBSCRIPTION_KEY` can reduce anonymous data-read throttling. Uncalibrated Sentinel-1 GRD amplitude values are not silently substituted into the gamma0 threshold detector.

Candidate raster values are `1` candidate water, `2` observed without candidate, and `3` unknown/insufficient evidence. `candidate_confidence.tif` records `1` for single-sensor support and `2` for cross-sensor corroboration. Cloud/no-data and supplied RTC layover/shadow masks are treated as unknown. The detector still requires independent validation for seasonal rivers, vegetation inundation, and Himalayan terrain. Use an aligned, projected-metre grid and a manifest of expert-labelled Sen1Floods11 held-out chips with `python -m evaluation.metrics <manifest.json>` to compute precision, recall, F1, IoU, false-alarm area, missed flood area, and prediction coverage. No benchmark metrics are reported until those labeled rasters are present.

The provided `flood_unet_best.pth` exposes a state-dict layout compatible with an SMP ResNet-18 U-Net. Load it with optional packages from `requirements-ai.txt` and `src.ai.infer`, but inference requires the checkpoint's training means, standard deviations, and validation-selected probability threshold. Those settings are not stored in the weights, so the system refuses to guess them. This checkpoint has not been independently scored in this workspace.

These are screening estimates, not verified damage. Water thresholds may confuse terrain shadow, wet soil, snow, or dark surfaces; bare-surface change may be harvest or seasonal change rather than debris. Sentinel-2 cloud masking depends on an SCL asset being available. OSM completeness and the endpoint-connected road graph limit damage and isolation accuracy; the graph does not model road condition, bridge capacity, travel time, or turns at unsplit junctions. Upstream flow tracing is not included. Missing imagery or OSM data is reported as a warning rather than silently treated as zero impact.

© OpenStreetMap contributors. Contains modified Copernicus Sentinel data 2026. See [ATTRIBUTION.md](ATTRIBUTION.md).
