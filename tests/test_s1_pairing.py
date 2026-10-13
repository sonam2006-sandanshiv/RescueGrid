import pytest
from datetime import datetime, timezone, timedelta
from src.preprocess.s1_pairing import select_s1_pair

class MockItem:
    def __init__(self, dt, orbit, direction):
        self.datetime = dt
        self.properties = {
            "sat:relative_orbit": orbit,
            "sat:orbit_state": direction
        }

def test_select_s1_pair():
    flood_date = "2026-08-26T00:00:00"
    base_dt = datetime(2026, 8, 26, tzinfo=timezone.utc)
    
    pre1 = MockItem(base_dt - timedelta(days=12), 100, "ascending")
    pre2 = MockItem(base_dt - timedelta(days=5), 101, "descending")
    post1 = MockItem(base_dt + timedelta(days=2), 100, "ascending") # 14 days apart from pre1
    post2 = MockItem(base_dt + timedelta(days=7), 101, "descending") # 12 days apart from pre2
    
    items = [pre1, pre2, post1, post2]
    
    pair = select_s1_pair(items, flood_date)
    assert pair is not None
    assert pair[0] == pre2
    assert pair[1] == post2
    
def test_select_s1_pair_falls_back_to_nearest_real_cross_orbit_pair():
    flood_date = "2026-08-26T00:00:00"
    base_dt = datetime(2026, 8, 26, tzinfo=timezone.utc)
    
    pre1 = MockItem(base_dt - timedelta(days=12), 100, "ascending")
    post1 = MockItem(base_dt + timedelta(days=2), 101, "ascending")
    
    items = [pre1, post1]
    
    pair = select_s1_pair(items, flood_date)
    assert pair == (pre1, post1)
