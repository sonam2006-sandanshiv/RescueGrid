import pytest
import numpy as np
from src.flowpath.trace import trace_flowpath

def test_trace_flowpath():
    # Simple sloping DEM
    dem = np.array([
        [100, 95, 90],
        [100, 90, 85],
        [100, 85, 80]
    ])
    
    path = trace_flowpath(dem, (0,0))
    # Should trace down to (2,2)
    assert path[-1] == (2,2)
    assert len(path) == 3
