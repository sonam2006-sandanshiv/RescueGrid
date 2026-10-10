import numpy as np
from src.preprocess.speckle import lee_filter
from src.preprocess.masks import calculate_slope, mock_hand

def detect_flood_s1(pre_vv, post_vv, pre_vh, post_vh, dem, config):
    """
    Classical heuristic flood and debris detection from Sentinel-1 and DEM.
    Returns class grid: {0: none, 1: flood water, 2: debris/mud, 3: uncertain/nodata}
    """
    # 1. Speckle filtering
    pre_vv_f = lee_filter(pre_vv)
    post_vv_f = lee_filter(post_vv)
    post_vh_f = lee_filter(post_vh)
    
    # 2. Topo masks
    # Assuming cell size of ~10m for typical Sentinel
    slope = calculate_slope(dem, 10, 10)
    hand = mock_hand(dem)
    
    slope_mask = slope > config.get("slope_threshold_deg", 15)
    hand_mask = hand > config.get("hand_threshold_m", 20)
    invalid_mask = slope_mask | hand_mask
    
    # 3. Water detection (dark in post-VV AND dropped from pre-VV)
    vv_water_thresh = config.get("vv_water_threshold_db", -18)
    vv_drop_thresh = config.get("vv_drop_db", 3)
    
    water_mask = (post_vv_f < vv_water_thresh) & ((pre_vv_f - post_vv_f) > vv_drop_thresh)
    water_mask = water_mask & (~invalid_mask)
    
    # 4. Debris/mud detection (high variance change, rough texture)
    from scipy.ndimage import uniform_filter
    
    # Texture change: local standard deviation of VH
    post_vh_var = uniform_filter(post_vh_f**2, 5) - uniform_filter(post_vh_f, 5)**2
    pre_vh_var = uniform_filter(pre_vh**2, 5) - uniform_filter(pre_vh, 5)**2
    
    # Debris often shows as an increase in texture (roughness) and backscatter, not dark like water
    debris_mask = (post_vh_var > pre_vh_var * 2.0) & (post_vv_f > -15) & (~water_mask)
    
    # 5. Compile output
    # 0 none, 1 flood water, 2 debris/mud, 3 no-data
    out_classes = np.zeros_like(post_vv, dtype=np.uint8)
    out_classes[debris_mask] = 2
    out_classes[water_mask] = 1
    
    # Optional: flag shadow/layover as 3 (uncertain) based on slope direction, 
    # but for now we just flag nodata areas.
    out_classes[np.isnan(post_vv)] = 3
    
    # Confidence mask (dummy implementation for prototype)
    confidence = np.ones_like(post_vv, dtype=np.float32) * 0.8
    confidence[slope_mask] = 0.4
    
    return out_classes, confidence
