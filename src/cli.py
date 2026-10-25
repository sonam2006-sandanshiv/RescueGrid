"""Run a full event analysis from a WGS84 bounding box."""
from __future__ import annotations

import argparse
import logging
from datetime import date
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Himalayan Flood Mapper")
    parser.add_argument("--bbox", required=True, help="west,south,east,north")
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--osm-date", type=date.fromisoformat)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    bbox = tuple(float(x) for x in args.bbox.split(","))
    print(run(bbox, args.date, args.osm_date))


if __name__ == "__main__":
    main()
