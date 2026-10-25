"""Small API adapter around the existing satellite analysis pipeline."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
from src.pipeline import run

RUNS_ROOT = Path(os.environ.get("RUNS_DIR", PROJECT_ROOT / "runs")).resolve()
MAX_BBOX_SPAN_DEGREES = 0.25
MAX_ACTIVE_JOBS = 2
LAYER_FILES = {
    "water_candidates": "flood_candidates.geojson",
    "exposed_buildings": "exposed_buildings.geojson",
    "exposed_bridges": "exposed_bridges.geojson",
    "affected_roads": "affected_roads.geojson",
    "isolated_settlements": "isolated_settlements.geojson",
    "sentinel2_candidates": "sentinel2_water_candidates.geojson",
    "disturbed_surface": "disturbed_surface_candidates.geojson",
}

app = FastAPI(title="RescueGrid API", version="1.0.0")
LOG = logging.getLogger("rescuegrid.api")
origins = [origin.strip().rstrip("/") for origin in os.environ.get(
    "FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rescuegrid-analysis")


class AnalysisRequest(BaseModel):
    bbox: list[float] = Field(description="WGS84 west, south, east, north")
    flood_date: date

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox must contain west, south, east, north")
        west, south, east, north = value
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox coordinates must be ordered WGS84 bounds")
        if east - west > MAX_BBOX_SPAN_DEGREES or north - south > MAX_BBOX_SPAN_DEGREES:
            raise ValueError("Keep each bbox side at or below 0.25 degrees")
        return value


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "rescuegrid-api"}


@app.post("/api/analyze", status_code=202)
def create_analysis(request: AnalysisRequest) -> dict[str, str]:
    """Queue analysis off the request thread so the browser can poll its status."""
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        active = sum(job["status"] in {"queued", "running"} for job in _jobs.values())
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="Analysis service is busy. Try again in a moment.")
        _jobs[job_id] = {"status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    _executor.submit(_run_analysis, job_id, tuple(request.bbox), request.flood_date)
    return {"job_id": job_id, "status": "queued", "status_url": "/api/jobs/" + job_id}


def _run_analysis(job_id: str, bbox: tuple[float, float, float, float], flood_date: date) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        run_dir = run(bbox, flood_date, output_root=RUNS_ROOT).resolve()
        if run_dir.parent != RUNS_ROOT:
            raise RuntimeError("Analysis output escaped the configured runs directory")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        layers = {
            key: "/api/runs/" + run_dir.name + "/layers/" + key
            for key, filename in LAYER_FILES.items()
            if (run_dir / filename).is_file()
        }
        result = {"run_id": run_dir.name, "manifest": manifest, "layers": layers}
        with _jobs_lock:
            _jobs[job_id].update(status="completed", result=result,
                                 completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        # Keep detailed exceptions in Render logs; return a short message to the browser.
        LOG.exception("Analysis job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=str(exc)[:500],
                                 completed_at=datetime.now(timezone.utc).isoformat())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found; it may have expired after a service restart")
        return dict(job, job_id=job_id)


@app.get("/api/runs/{run_id}/layers/{layer_name}")
def get_layer(run_id: str, layer_name: str) -> Any:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id) or layer_name not in LAYER_FILES:
        raise HTTPException(status_code=404, detail="Layer not found")
    run_dir = (RUNS_ROOT / run_id).resolve()
    if run_dir.parent != RUNS_ROOT:
        raise HTTPException(status_code=404, detail="Layer not found")
    path = run_dir / LAYER_FILES[layer_name]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Layer not available for this analysis")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read analysis layer") from exc
