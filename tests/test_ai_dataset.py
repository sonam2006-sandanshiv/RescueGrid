import pytest
import torch
from src.ai.dataset_sen1floods11 import Sen1Floods11Dataset
from src.ai.dataset_kurosiwo import KuroSiwoDataset

def test_sen1floods11_dataset_loader():
    ds = Sen1Floods11Dataset(split="train")
    assert len(ds) > 0
    img, mask = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape[0] == 4 # 4 channels
    assert mask.ndim == 2 # 2D mask

def test_kurosiwo_dataset_loader():
    ds = KuroSiwoDataset("data/kurosiwo", split="train")
    assert len(ds) > 0
    img, mask = ds[0]
    assert img.shape[0] == 4
    assert mask.ndim == 2
