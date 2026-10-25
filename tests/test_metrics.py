import numpy as np
import pytest
from evaluation.metrics import score_arrays, _event_metrics
from src.ai.infer import classify_probabilities
from src.analysis import _rtc_mask_valid, _s1_water, _s2_water


def test_metrics_include_false_alarms_misses_and_unknown_coverage():
    truth = np.array([[1, 1, 0, -1]])
    pred = np.array([[1, 2, 1, 1]])
    result = score_arrays(pred, truth, 100)
    assert result["tp_pixels"] == 1
    assert result["fp_pixels"] == 1
    assert result["fn_pixels"] == 1
    assert result["precision"] == pytest.approx(.5)
    assert result["recall"] == pytest.approx(.5)
    assert result["f1"] == pytest.approx(.5)
    assert result["iou"] == pytest.approx(1/3)
    assert result["false_alarm_area_ha"] == pytest.approx(.01)
    assert result["missed_flood_area_ha"] == pytest.approx(.01)
    assert result["prediction_coverage"] == pytest.approx(1)


def test_metrics_ignore_unknown_predictions_instead_of_counting_as_dry():
    result = score_arrays(np.array([[3]]), np.array([[1]]), 100)
    assert result["scored_pixels"] == 0
    assert result["tp_pixels"] == result["fn_pixels"] == 0
    assert result["prediction_coverage"] == 0


def test_metrics_require_matching_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        score_arrays(np.zeros((2, 2)), np.zeros((2, 3)), 100)


def test_event_aggregates_keep_separate_counts_and_rates():
    chips = [
        {"event": "Nepal", "tp_pixels": 3, "fp_pixels": 1, "fn_pixels": 2, "tn_pixels": 4,
         "scored_pixels": 10, "reference_valid_pixels": 10, "unknown_prediction_pixels": 0,
         "false_alarm_area_ha": .01, "missed_flood_area_ha": .02},
        {"event": "Bolivia", "tp_pixels": 1, "fp_pixels": 0, "fn_pixels": 1, "tn_pixels": 8,
         "scored_pixels": 10, "reference_valid_pixels": 10, "unknown_prediction_pixels": 0,
         "false_alarm_area_ha": 0, "missed_flood_area_ha": .01},
    ]
    result = _event_metrics(chips)
    assert result["Nepal"]["iou"] == pytest.approx(3/6)
    assert result["Bolivia"]["iou"] == pytest.approx(.5)


def test_classification_preserves_uncertain_probability_band_and_missing_pixels():
    result = classify_probabilities(np.array([[.1, .48, .91, np.nan]]), .5, .4, .6)
    assert result.tolist() == [[2, 3, 1, 3]]


def test_ai_probability_threshold_must_be_supplied_from_validation():
    with pytest.raises(ValueError, match="validation-selected"):
        classify_probabilities(np.array([[.9]]), None)


def test_s1_compares_linear_power_as_gamma0_db_and_uses_same_pixel_baseline():
    pre = np.array([[0.2, 0.02, 0.2, np.nan]], dtype=float)
    post = np.array([[0.02, 0.02, 0.2, 0.1]], dtype=float)
    candidate, valid, method = _s1_water(pre, post)
    assert candidate.tolist() == [[True, False, False, False]]
    assert valid.tolist() == [[True, True, True, False]]
    assert "gamma0" in method


def test_s2_without_scene_classification_is_unknown_not_dry():
    arrays = [np.ones((1, 1), dtype=float) for _ in range(6)]
    candidate, valid, method = _s2_water(*arrays, None, None)
    assert not candidate.any() and not valid.any()
    assert "unknown" in method


def test_rtc_mask_class_values_are_not_guessed_without_stac_classes():
    mask = np.array([[0, 1, 2, 255]])
    with pytest.raises(ValueError, match="documented"):
        _rtc_mask_valid({"assets": {"mask": {"href": "mask.tif"}}}, mask)
    item = {"assets": {"mask": {"href": "mask.tif", "raster:bands": [{
        "classification:classes": [
            {"value": 0, "name": "valid sample"},
            {"value": 1, "name": "shadow"},
            {"value": 2, "name": "layover"},
            {"value": 255, "name": "invalid"},
        ]}]}}}
    assert _rtc_mask_valid(item, mask).tolist() == [[True, False, False, False]]
