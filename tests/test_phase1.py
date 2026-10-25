from datetime import date
from pathlib import Path

import pytest

from src.ingest.osm import validate_osm_date
from src.preprocess.s1_pairing import select_s1_pair
from src.ingest import s1
from src.preprocess.s1_pairing import select_s2_pair


def scene(day, orbit=33, direction="ascending", ident="x"):
    return {"id": ident, "properties": {"datetime": day + "T05:00:00Z",
            "sat:relative_orbit": orbit, "sat:orbit_state": direction}}


def test_s1_pair_selects_nearest_12_day_same_track():
    items = [scene("2026-08-14", ident="pre12"), scene("2026-08-26", orbit=44, ident="wrong-orbit"),
             scene("2026-09-07", ident="post12"), scene("2026-09-07", direction="descending", ident="wrong-pass")]
    pre, post = select_s1_pair(items, date(2026, 8, 26))
    assert (pre["id"], post["id"]) == ("pre12", "post12")


def test_s1_pair_refuses_mismatched_tracks():
    with pytest.raises(ValueError, match="same relative orbit"):
        select_s1_pair([scene("2026-08-14"), scene("2026-08-26", orbit=44)], date(2026, 8, 20))


def test_osm_date_is_strictly_pre_event():
    validate_osm_date(date(2026, 7, 27), date(2026, 8, 26))
    for invalid in (date(2026, 8, 26), date(2026, 8, 27)):
        with pytest.raises(ValueError, match="strictly earlier"):
            validate_osm_date(invalid, date(2026, 8, 26))


def test_src_does_not_reference_forbidden_reference_paths():
    src = Path(__file__).resolve().parents[1] / "src"
    forbidden = ("evaluation/reference", "EMSR927", "UNOSAT")
    for path in src.rglob("*.py"):
        content = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker.lower() not in content, f"{path} contains forbidden marker {marker}"


def test_s1_query_uses_planetary_computer_rtc_not_placeholder_grd(monkeypatch):
    captured = {}
    def fake_search(collection, bbox, start, end, **kwargs):
        captured.update(collection=collection, bbox=bbox, start=start, end=end, **kwargs)
        return []
    monkeypatch.setattr(s1, "search_planetary_computer", fake_search)
    s1.query((85, 27, 86, 28), date(2026, 8, 26), max_days_before=48, max_days_after=48)
    assert captured["collection"] == "sentinel-1-rtc"
    assert captured["start"] == date(2026, 7, 9)
    assert captured["end"] == date(2026, 10, 13)


def test_s2_pair_is_near_event_and_same_mgrs_tile():
    def s2(day, tile, cloud, ident):
        return {"id": f"S2_{tile}_{ident}", "properties": {
            "datetime": day + "T05:00:00Z", "eo:cloud_cover": cloud}}
    items = [s2("2026-08-20", "T45RUL", 20, "pre-near"),
             s2("2026-08-25", "T45RUL", 30, "pre-cloudier"),
             s2("2026-09-28", "T45RUL", 1, "post-too-late"),
             s2("2026-08-30", "T45RUM", 2, "post-other-tile"),
             s2("2026-08-31", "T45RUL", 18, "post-near")]
    pre, post = select_s2_pair(items, date(2026, 8, 26))
    assert pre["id"].endswith("pre-near")
    assert post["id"].endswith("post-near")


def test_s2_pair_refuses_distant_or_tile_mismatched_scenes():
    items = [
        {"id": "S2_T45RUL_pre", "properties": {"datetime": "2026-08-14T00:00:00Z", "eo:cloud_cover": 1}},
        {"id": "S2_T45RUM_post", "properties": {"datetime": "2026-09-28T00:00:00Z", "eo:cloud_cover": 1}},
    ]
    with pytest.raises(ValueError, match="within 12 days"):
        select_s2_pair(items, date(2026, 8, 26))
