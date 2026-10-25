"""Score flood candidate rasters against independently supplied labels.

Reference values must be water=1, non-water=0, unknown/nodata=-1. Prediction
values follow the project export: candidate=1, observed-negative=2, unknown=3.
Unknown prediction pixels are excluded from confusion counts and separately
reported as coverage; they are never treated as dry land.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def score_arrays(prediction: np.ndarray, reference: np.ndarray, pixel_area_m2: float,
                 reference_unknown: int = -1, prediction_unknown: int = 3) -> dict[str, Any]:
    """Compute pixel metrics and false-alarm/missed area for aligned rasters."""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference arrays must have identical shapes")
    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be positive")
    label_valid = (reference != reference_unknown) & np.isin(reference, [0, 1])
    scored = label_valid & (prediction != prediction_unknown) & np.isin(prediction, [1, 2])
    unknown_prediction = label_valid & (prediction == prediction_unknown)
    truth = reference[scored] == 1
    predicted = prediction[scored] == 1
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(np.count_nonzero(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {"tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "tn_pixels": tn,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "false_positive_rate": false_positive_rate,
            "false_alarm_area_ha": fp * pixel_area_m2 / 10_000,
            "missed_flood_area_ha": fn * pixel_area_m2 / 10_000,
            "reference_valid_pixels": int(label_valid.sum()),
            "scored_pixels": int(scored.sum()),
            "unknown_prediction_pixels": int(unknown_prediction.sum()),
            "prediction_coverage": float(scored.sum() / label_valid.sum()) if label_valid.any() else None,
            "pixel_area_m2": float(pixel_area_m2)}


def score_geotiffs(prediction_path: Path, reference_path: Path) -> dict[str, Any]:
    """Compare aligned GeoTIFFs; reject grids that would require resampling."""
    with rasterio.open(prediction_path) as pred_ds, rasterio.open(reference_path) as ref_ds:
        if (pred_ds.crs != ref_ds.crs or pred_ds.transform != ref_ds.transform
                or pred_ds.width != ref_ds.width or pred_ds.height != ref_ds.height):
            raise ValueError("Prediction and reference GeoTIFFs must use the same grid; reproject explicitly first")
        if not pred_ds.crs or not pred_ds.crs.is_projected:
            raise ValueError("Area metrics require a projected CRS in metres; reproject geographic grids before scoring")
        if pred_ds.crs.linear_units.lower() not in ("metre", "meter", "metres", "meters"):
            raise ValueError("Area metrics require metre-based projected units")
        pred, ref = pred_ds.read(1), ref_ds.read(1)
        area = abs(pred_ds.transform.a * pred_ds.transform.e)
        result = score_arrays(pred, ref, area)
    result.update({"prediction_file": str(prediction_path), "reference_file": str(reference_path),
                   "reference_crs": str(pred_ds.crs) if pred_ds.crs else None})
    return result


def evaluate_manifest(manifest_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Score a JSON list of event/chip pairs and aggregate counts exactly."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    chips = []
    for row in document.get("chips", []):
        result = score_geotiffs(Path(row["prediction"]), Path(row["reference"]))
        result["event"] = row["event"]
        result["chip_id"] = row.get("chip_id")
        chips.append(result)
    if not chips:
        raise ValueError("Evaluation manifest has no chips")
    totals = {key: sum(chip[key] for chip in chips) for key in
              ("tp_pixels", "fp_pixels", "fn_pixels", "tn_pixels", "false_alarm_area_ha", "missed_flood_area_ha", "scored_pixels", "reference_valid_pixels", "unknown_prediction_pixels")}
    tp, fp, fn = totals["tp_pixels"], totals["fp_pixels"], totals["fn_pixels"]
    totals.update({"precision": tp/(tp+fp) if tp+fp else None,
                   "recall": tp/(tp+fn) if tp+fn else None,
                   "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                   "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
                   "false_positive_rate": fp/(fp+totals["tn_pixels"]) if fp+totals["tn_pixels"] else None,
                   "prediction_coverage": totals["scored_pixels"]/totals["reference_valid_pixels"] if totals["reference_valid_pixels"] else None})
    result = {"metrics": totals, "chips": chips,
              "events": _event_metrics(chips),
              "reference_provenance": document.get("reference_provenance"),
              "warning": "Metrics describe only the supplied labeled chips and valid-prediction coverage; they do not certify general Himalayan performance."}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _event_metrics(chips: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate confusion counts per independent flood event."""
    events: dict[str, dict[str, Any]] = {}
    for chip in chips:
        event = str(chip["event"])
        current = events.setdefault(event, {key: 0 for key in ("tp_pixels", "fp_pixels", "fn_pixels", "tn_pixels", "scored_pixels", "reference_valid_pixels", "unknown_prediction_pixels", "false_alarm_area_ha", "missed_flood_area_ha")})
        for key in current:
            current[key] += chip[key]
    for stats in events.values():
        tp, fp, fn = stats["tp_pixels"], stats["fp_pixels"], stats["fn_pixels"]
        stats.update({"precision": tp/(tp+fp) if tp+fp else None,
                      "recall": tp/(tp+fn) if tp+fn else None,
                      "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                      "iou": tp/(tp+fp+fn) if tp+fp+fn else None,
                      "false_positive_rate": fp/(fp+stats["tn_pixels"]) if fp+stats["tn_pixels"] else None,
                      "prediction_coverage": stats["scored_pixels"]/stats["reference_valid_pixels"] if stats["reference_valid_pixels"] else None})
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON with reference_provenance and chips[] paths")
    parser.add_argument("--output", type=Path, default=Path("reports/detector_eval.json"))
    args = parser.parse_args()
    result = evaluate_manifest(args.manifest, args.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
