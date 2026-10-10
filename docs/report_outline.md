# 6-Page Situation Report Outline

## 1. Problem Statement
- Goal: Rapidly map flood damage to support emergency response.
- Case Study: Bhote Koshi–Trishuli glacier-collapse flood (August 2026).

## 2. Data Sources
- **Satellite Data**: Sentinel-1 GRD (SAR), Sentinel-2 L2A (Optical).
- **Topography**: Copernicus DEM GLO-30 (Slope and HAND masks).
- **Infrastructure**: OpenStreetMap via ohsome API (strict pre-event temporal filtering).

## 3. Methodology (Classical & Fusion)
- **SAR Processing**: Fast Lee speckle filtering, VV drop thresholding, VH texture variance for debris.
- **Optical Processing**: NDWI/NDVI thresholds (cloud-free pixels).
- **Fusion**: Confidence-based overrides where S2 is cloud-free.

## 4. AI Model (Kuro Siwo U-Net)
- **Architecture**: ResNet34 U-Net trained on 4-channel SAR.
- **Evaluation**: Zero-shot generalization tested on held-out Himalayan regions.
- **Performance**: High recall on open water, lower precision in shadow regions.

## 5. Damage & Network Analysis
- **Overlay**: Geospatial intersection with 10-20m buffer accounting for SAR resolution.
- **Cut-off Assessment**: NetworkX Dijkstra routing using pre-event OSM road graph. Penalized weights applied to flooded road segments.
- **Results**: X settlements completely cut off from nearest hospital/town.

## 6. Limitations & Comparison (EMSR927)
- Honest review of failure modes (radar shadow, sub-pixel features).
- High agreement with EMSR927 on main river corridor, divergence in steep tributaries.
