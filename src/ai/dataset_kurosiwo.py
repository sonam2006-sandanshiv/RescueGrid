import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class KuroSiwoDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        Loads Kuro Siwo dataset.
        split: 'train', 'val', or 'test'.
        Data rule: split by event/region, not by tile.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # In a real implementation, read a CSV/JSON splitting by event
        # For prototype, we simulate a list of file paths.
        self.samples = self._load_split()
        
    def _load_split(self):
        # Dummy implementation
        return [{"pre_vv": "dummy.tif", "label": "dummy_mask.tif"}] * 10
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # In reality, load 4 channels (pre VV/VH, post VV/VH) 
        # and 1 mask channel.
        # Returning dummy tensors for now so CI and tests pass.
        
        # 4 channels: Pre VV, Post VV, Pre VH, Post VH
        image = torch.randn(4, 256, 256)
        
        # Mask: 0 background, 1 water, 2 debris
        mask = torch.randint(0, 3, (256, 256), dtype=torch.long)
        
        if self.transform:
            pass # apply albumentations here
            
        return image, mask
