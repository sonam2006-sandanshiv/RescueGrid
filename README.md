# Track B: Mapping Flood Damage from Space

End-to-end, reproducible prototype for flood damage mapping using Sentinel-1, Sentinel-2, Copernicus DEM, and OpenStreetMap.

## Quick Start
1. Clone the repository
2. Set up credentials: `cp .env.example .env` and fill in your CDSE credentials
3. `make setup`
4. `make run --bbox 85.0,27.8,85.5,28.3 --date 2026-08-26`
5. `make dashboard`

See `docs/` for more information.
