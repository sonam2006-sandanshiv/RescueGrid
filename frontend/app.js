const API_STORAGE_KEY = "rescuegrid-api-url";
const form = document.querySelector("#analysis-form");
const apiInput = document.querySelector("#api-url");
const statusBox = document.querySelector("#status");
const resultsBox = document.querySelector("#results"); 
const analyzeButton = document.querySelector("#analyze-button");
const map = L.map("map", { scrollWheelZoom: true }).setView([27.95, 85.0], 7);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

apiInput.value = localStorage.getItem(API_STORAGE_KEY) || "http://localhost:8000";
document.querySelector("#flood-date").value = new Date().toISOString().slice(0, 10);
let areaRectangle;
let resultLayers = [];

function setStatus(message, kind = "info") {
  statusBox.hidden = false;
  statusBox.dataset.kind = kind;
  statusBox.textContent = message;
}

function selectedBounds() {
  return ["west", "south", "east", "north"].map((id) => {
    const value = Number(document.getElementById(id).value);
    if (!Number.isFinite(value)) throw new Error("Enter all four search bounds.");
    return value;
  });
}

function drawBounds(bounds, fit = false) {
  const [west, south, east, north] = bounds;
  if (areaRectangle) map.removeLayer(areaRectangle);
  areaRectangle = L.rectangle([[south, west], [north, east]], {
    color: "#dd9950", weight: 2, fillColor: "#e4aa58", fillOpacity: 0.12,
  }).addTo(map);
  if (fit) map.fitBounds(areaRectangle.getBounds(), { padding: [28, 28], maxZoom: 12 });
}

map.on("click", (event) => {
  const { lat, lng } = event.latlng;
  const bounds = [lng - 0.02, lat - 0.02, lng + 0.02, lat + 0.02];
  ["west", "south", "east", "north"].forEach((id, index) => {
    document.getElementById(id).value = bounds[index].toFixed(5);
  });
  drawBounds(bounds);
});

document.querySelectorAll(".bounds-grid input").forEach((input) => {
  input.addEventListener("change", () => {
    try { drawBounds(selectedBounds()); }
    catch { /* Partial entry is allowed until the form is submitted. */ }
  });
});

function apiBase() {
  const value = apiInput.value.trim().replace(/\/+$/, "");
  if (!value) throw new Error("Enter the public Render API URL first.");
  const url = new URL(value);
  if (url.protocol !== "https:" && url.hostname !== "localhost" && url.hostname !== "127.0.0.1") {
    throw new Error("The API URL must use HTTPS outside local development.");
  }
  localStorage.setItem(API_STORAGE_KEY, value);
  return value;
}

async function apiFetch(base, path, options = {}) {
  const response = await fetch(base + path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || ("API returned " + response.status));
  return body;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForJob(base, jobId) {
  for (;;) {
    const job = await apiFetch(base, "/api/jobs/" + encodeURIComponent(jobId));
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "Analysis failed. Check the Render service logs.");
    setStatus(job.status === "queued" ? "Analysis queued. Waiting for the Render worker…" : "Reading scenes and building the map…");
    await sleep(1800);
  }
}

function addMetric(container, label, value, suffix = "") {
  const card = document.createElement("div");
  card.className = "metric";
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value === null || value === undefined ? "Unknown" : (value + suffix);
  card.append(name, number);
  container.append(card);
}

function styleFor(feature, name) {
  const support = feature && feature.properties && feature.properties.evidence_support;
  if (name === "water_candidates") return support === "cross_sensor"
    ? { color: "#9b3029", fillColor: "#9b3029", fillOpacity: 0.42, weight: 2 }
    : { color: "#d48c31", fillColor: "#d48c31", fillOpacity: 0.38, weight: 2 };
  const colors = {
    exposed_buildings: "#b747a7", exposed_bridges: "#315db4", affected_roads: "#dd762e",
    isolated_settlements: "#6c5aa6", sentinel2_candidates: "#6a9f66", disturbed_surface: "#8954af",
  };
  const color = colors[name] || "#4b6658";
  return { color, fillColor: color, fillOpacity: 0.3, weight: name === "affected_roads" ? 4 : 2 };
}

async function showLayers(base, result) {
  resultLayers.forEach((layer) => map.removeLayer(layer));
  resultLayers = [];
  const names = {
    water_candidates: "Candidate water",
    exposed_buildings: "Potentially exposed buildings",
    exposed_bridges: "Potentially exposed bridges",
    affected_roads: "Roads crossing candidate water",
    isolated_settlements: "Potentially disconnected settlements",
    sentinel2_candidates: "Sentinel-2 water candidates",
    disturbed_surface: "Disturbed surface candidates",
  };
  for (const [key, path] of Object.entries(result.layers || {})) {
    const geojson = await apiFetch(base, path);
    const layer = L.geoJSON(geojson, {
      style: (feature) => styleFor(feature, key),
      pointToLayer: (_feature, latlng) => L.circleMarker(latlng, {
        radius: 6, color: styleFor(null, key).color, fillOpacity: 0.75,
      }),
      onEachFeature: (feature, item) => {
        const props = feature.properties || {};
        const details = Object.entries(props).slice(0, 5).map(([k, v]) => k + ": " + v).join(" · ");
        if (details) item.bindTooltip((names[key] || key) + " · " + details);
      },
    }).addTo(map);
    resultLayers.push(layer);
  }
  if (resultLayers.length) {
    const group = L.featureGroup(resultLayers);
    if (group.getBounds().isValid()) map.fitBounds(group.getBounds(), { padding: [24, 24], maxZoom: 13 });
  }
}

async function renderResult(base, result) {
  const manifest = result.manifest || {};
  const stats = manifest.analysis || {};
  const metrics = document.querySelector("#metrics");
  metrics.replaceChildren();
  addMetric(metrics, "Candidate water", stats.candidate_flood_area_ha, " ha");
  addMetric(metrics, "Disturbed surface candidate", stats.disturbed_bare_surface_candidate_area_ha, " ha");
  addMetric(metrics, "Potentially exposed buildings", stats.buildings_intersecting_candidate_water);
  addMetric(metrics, "Potentially exposed bridges", stats.bridges_intersecting_candidate_water);
  addMetric(metrics, "Roads crossing candidate water", stats.road_length_intersecting_candidate_water_km, " km");
  addMetric(metrics, "Potentially disconnected settlements", stats.settlements_disconnected_from_towns_or_healthcare);
  document.querySelector("#run-id").textContent = result.run_id || "";
  const warningBox = document.querySelector("#warnings");
  warningBox.replaceChildren();
  const warnings = [...(manifest.warnings || []), ...(stats.warnings || [])];
  [...new Set(warnings)].forEach((warning) => {
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = warning;
    warningBox.append(item);
  });
  document.querySelector("#interpretation").textContent = stats.note || manifest.note || "Scene discovery completed. No pixel analysis was available for this search.";
  resultsBox.hidden = false;
  await showLayers(base, result);
  resultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  analyzeButton.disabled = true;
  resultsBox.hidden = true;
  try {
    const base = apiBase();
    const bbox = selectedBounds();
    const floodDate = document.querySelector("#flood-date").value;
    if (!floodDate) throw new Error("Choose the event date.");
    drawBounds(bbox, true);
    setStatus("Sending the bounded area to the analysis service…");
    const queued = await apiFetch(base, "/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox, flood_date: floodDate }),
    });
    const result = await waitForJob(base, queued.job_id);
    setStatus("Analysis finished. Review candidate layers and warnings before using these estimates.");
    await renderResult(base, result);
  } catch (error) {
    setStatus(error.message || "Could not reach the backend service.", "error");
  } finally {
    analyzeButton.disabled = false;
  }
});

document.querySelector("#health-check").addEventListener("click", async () => {
  try {
    const base = apiBase();
    await apiFetch(base, "/api/health");
    setStatus("Render API is reachable.");
  } catch (error) {
    setStatus(error.message || "API health check failed.", "error");
  }
});
