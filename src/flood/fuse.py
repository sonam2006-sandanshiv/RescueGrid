import numpy as np

def fuse_s1_s2(s1_classes, s1_confidence, s2_classes, s2_confidence, cloud_mask):
    """
    Fuse S1 and S2 flood/debris classes.
    S2 takes precedence where confidence is higher and it's not clouded.
    Classes: 0: none, 1: flood, 2: debris, 3: nodata
    """
    fused_classes = np.copy(s1_classes)
    fused_confidence = np.copy(s1_confidence)
    
    if s2_classes is None:
        return fused_classes, fused_confidence
        
    # Valid S2 pixels
    valid_s2 = (~cloud_mask) & (s2_classes != 3)
    
    # Simple logic: If S2 sees something with higher confidence, use S2
    s2_override = valid_s2 & (s2_confidence > s1_confidence)
    
    fused_classes[s2_override] = s2_classes[s2_override]
    fused_confidence[s2_override] = s2_confidence[s2_override]
    
    return fused_classes, fused_confidence
