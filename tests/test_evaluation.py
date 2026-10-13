import pytest
from evaluation.emsr927_comparison import evaluate_against_emsr927
from evaluation.metrics import compute_iou, compute_precision_recall_f1
import numpy as np

def test_metrics_calculation():
    m1 = np.array([True, True, False, False])
    m2 = np.array([True, False, False, False])
    iou = compute_iou(m1, m2)
    assert round(iou, 2) == 0.50
    prec, rec, f1 = compute_precision_recall_f1(m1, m2)
    assert prec == 0.5
    assert rec == 1.0
    assert round(f1, 2) == 0.67

def test_emsr927_benchmark():
    report = evaluate_against_emsr927()
    assert "metrics" in report
    assert "water" in report["metrics"]
    assert "debris" in report["metrics"]
    assert report["metrics"]["water"]["spatial_iou"] > 0.60
    assert report["metrics"]["infrastructure"]["bridge_agreement_rate"] == 1.0
