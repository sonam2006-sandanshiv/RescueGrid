import numpy as np

def calculate_slope(elevation_grid, cell_size_x, cell_size_y):
    """
    Computes slope in degrees from a DEM.
    """
    dz_dx = np.gradient(elevation_grid, cell_size_x, axis=1)
    dz_dy = np.gradient(elevation_grid, cell_size_y, axis=0)
    
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg

def mock_hand(elevation_grid, drainage_network=None):
    """
    Mocks Height Above Nearest Drainage (HAND) for the heuristic model
    if actual D8 accumulation is too slow.
    In a real scenario, this would use whitebox tools or pysheds.
    """
    # A rough mock: just assume valleys are local minima
    from scipy.ndimage import minimum_filter
    valleys = minimum_filter(elevation_grid, size=15)
    hand = elevation_grid - valleys
    return np.maximum(hand, 0)
