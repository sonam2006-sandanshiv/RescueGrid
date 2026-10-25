# RescueGrid backend

This FastAPI service adapts the existing src.pipeline.run workflow. It validates a small WGS84 area, queues one analysis at a time, returns the run manifest and summary, and serves only the known GeoJSON map layers.

## Deploy to Render

The repository includes render.yaml. Create a new Blueprint from the GitHub repository, then set:

- FRONTEND_ORIGINS: the exact Vercel origin, for example https://rescuegrid.vercel.app (no trailing slash).
- PC_SDK_SUBSCRIPTION_KEY: optional Planetary Computer subscription key.
- OHSOME_API_KEY: optional ohsome API key for historical OSM data.

The service uses the repository root, installs requirements.txt, starts backend.main:app, and checks /api/health. Keep keys in Render environment settings; do not put them in frontend files.

## Local API

From the repository root:

~~~powershell
uvicorn backend.main:app --reload --port 8000
~~~

Open http://localhost:8000/docs for the API schema. POST /api/analyze accepts:

~~~json
{
  "bbox": [85.0, 27.8, 85.04, 27.84],
  "flood_date": "2026-08-26"
}
~~~

It returns a job ID; poll GET /api/jobs/{job_id} until complete. Job state is in memory and output files use the existing runs/ directory, so this starter deployment is intended for a single Render instance. It is not a durable job queue.
