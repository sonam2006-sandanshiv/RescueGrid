"""CPU inference adapter for the supplied Flood U-Net checkpoint.

Checkpoint weights are not enough to recover training normalization or a
decision threshold. Callers must supply both from the model's training record;
this adapter never guesses them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def load_configured_checkpoint():
    """Load the checkpoint path selected through ``FLOOD_UNET_CHECKPOINT``."""
    checkpoint = os.environ.get("FLOOD_UNET_CHECKPOINT", "").strip()
    if not checkpoint:
        raise RuntimeError("Set FLOOD_UNET_CHECKPOINT to the supplied flood_unet_best.pth file")
    return load_checkpoint(Path(checkpoint))


def load_checkpoint(path: Path):
    """Load a plain state dict for an SMP ResNet-18 U-Net, using safe loading."""
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional dependencies in requirements-ai.txt for checkpoint inference") from exc
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "encoder.conv1.weight" not in state or "segmentation_head.0.weight" not in state:
        raise ValueError("Checkpoint is not a recognized plain SMP ResNet-18 U-Net state dict")
    input_channels = int(state["encoder.conv1.weight"].shape[1])
    classes = int(state["segmentation_head.0.weight"].shape[0])
    if input_channels != 2:
        raise ValueError(f"Checkpoint expects {input_channels} input channels; the scene adapter supports VV and VH (2)")
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=input_channels,
                    classes=classes, activation=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_probabilities(model: Any, vv_db: np.ndarray, vh_db: np.ndarray,
                          mean: tuple[float, float] | None,
                          std: tuple[float, float] | None) -> np.ndarray:
    """Run CPU prediction for aligned VV/VH dB arrays with supplied normalization."""
    if mean is None or std is None or len(mean) != 2 or len(std) != 2 or any(value <= 0 for value in std):
        raise ValueError("Checkpoint training means and positive standard deviations for VV/VH are required")
    if vv_db.shape != vh_db.shape or vv_db.ndim != 2:
        raise ValueError("VV and VH inputs must be same-shaped 2D arrays")
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    normalized = np.stack(((vv_db-mean[0])/std[0], (vh_db-mean[1])/std[1]), axis=0)
    normalized[:, ~valid] = 0
    import torch
    with torch.inference_mode():
        output = model(torch.from_numpy(normalized[None].astype("float32")))
        if output.ndim != 4 or output.shape[1] != 1:
            raise ValueError("Checkpoint must produce one binary flood logit channel")
        probability = torch.sigmoid(output[0, 0]).cpu().numpy()
    probability[~valid] = np.nan
    return probability


def classify_probabilities(probability: np.ndarray, threshold: float | None,
                           unknown_low: float | None = None,
                           unknown_high: float | None = None) -> np.ndarray:
    """Make tri-state predictions only with a threshold specified by validation."""
    if threshold is None or not 0 < threshold < 1:
        raise ValueError("A validation-selected probability threshold is required")
    if (unknown_low is None) != (unknown_high is None):
        raise ValueError("Set both ambiguity bounds or neither")
    if unknown_low is not None and not (0 <= unknown_low < unknown_high <= 1):
        raise ValueError("Ambiguity bounds must satisfy 0 <= low < high <= 1")
    labels = np.full(probability.shape, 3, dtype="uint8")
    valid = np.isfinite(probability)
    labels[valid] = 2
    labels[valid & (probability >= threshold)] = 1
    if unknown_low is not None:
        labels[valid & (probability >= unknown_low) & (probability <= unknown_high)] = 3
    return labels
