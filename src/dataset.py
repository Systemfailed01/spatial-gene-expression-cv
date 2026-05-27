import torch
from torch.utils.data import Dataset


class SpatialDataset(Dataset):
    """Paired H&E patch and gene expression dataset for 10x Visium data."""

    def __init__(self, patches, expression):
        # patches: (N, H, W, C) float32 tensor
        # expression: (N, n_genes) float32 tensor
        self.patches    = patches.permute(0, 3, 1, 2) / 255.0
        self.expression = expression

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.expression[idx]
