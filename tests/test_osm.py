import pytest
from src.ingest.osm import _validate_osm_date

def test_osm_date_guard_passes():
    # Should not raise exception
    _validate_osm_date("2026-07-27", "2026-08-26")
    
def test_osm_date_guard_fails_same_day():
    with pytest.raises(ValueError):
        _validate_osm_date("2026-08-26", "2026-08-26")

def test_osm_date_guard_fails_after():
    with pytest.raises(ValueError):
        _validate_osm_date("2026-08-27", "2026-08-26")
