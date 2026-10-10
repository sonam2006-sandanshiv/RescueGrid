import numpy as np

def compute_iou(pred_mask, ref_mask):
    """Computes Intersection over Union."""
    intersection = np.logical_and(pred_mask, ref_mask).sum()
    union = np.logical_or(pred_mask, ref_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_precision_recall_f1(pred_mask, ref_mask):
    tp = np.logical_and(pred_mask, ref_mask).sum()
    fp = np.logical_and(pred_mask, ~ref_mask).sum()
    fn = np.logical_and(~pred_mask, ref_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1
