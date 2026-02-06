import torch
import numpy as np

#-------------------------------------------------------------------------
def extract_geometric_features(mask):
    """
    mask: (B, 1, H, W)
    returns: (B, 3)
    """
    features = []
    for m in mask:
        m = m.squeeze().cpu().numpy()
        h, w = m.shape

        area_ratio = m.sum() / (h * w)
        thickness_ratio = m.sum(axis=0).max() / h
        bbox_height = np.any(m, axis=1).sum() / h

        features.append([area_ratio, thickness_ratio, bbox_height])

    return torch.tensor(features, dtype=torch.float32)
