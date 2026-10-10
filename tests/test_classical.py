import numpy as np
from src.flood.classical import detect_flood_s1

def test_detect_flood_s1():
    size = (50, 50)
    # Background: -10 dB
    pre_vv = np.full(size, -10.0)
    post_vv = np.full(size, -10.0)
    pre_vh = np.full(size, -15.0)
    post_vh = np.full(size, -15.0)
    dem = np.zeros(size)
    
    # Insert water in top-left (10x10)
    # Water is dark in VV (-22 dB) and drops significantly
    post_vv[0:10, 0:10] = -22.0
    
    # Insert debris in bottom-right (10x10)
    # Debris has high variance in post VH
    # We will simulate high variance by making it a checkerboard of high/low values
    import itertools
    for i, j in itertools.product(range(40, 50), range(40, 50)):
        if (i+j) % 2 == 0:
            post_vh[i, j] = -5.0
        else:
            post_vh[i, j] = -25.0
            
    config = {
        "vv_water_threshold_db": -18,
        "vv_drop_db": 3,
        "slope_threshold_deg": 15,
        "hand_threshold_m": 20
    }
    
    out_classes, confidence = detect_flood_s1(pre_vv, post_vv, pre_vh, post_vh, dem, config)
    
    # Check water
    assert np.all(out_classes[2:8, 2:8] == 1)
    
    # Check background
    assert np.all(out_classes[20:30, 20:30] == 0)
    
    # Check debris
    # Since variance uses a 5x5 filter, the edges of the 10x10 might not be perfect, check center
    assert np.all(out_classes[43:47, 43:47] == 2)
