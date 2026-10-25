"""STAC discovery for Sentinel-1 and Sentinel-2 scenes."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

LOG = logging.getLogger(__name__)
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
FALLBACK_STAC = "https://earth-search.aws.element84.com/v1"
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_stac(collection: str, bbox: tuple[float, float, float, float], start: date,
                end: date, *, endpoints: tuple[str, ...] = (CDSE_STAC, FALLBACK_STAC),
                limit: int = 200, timeout: int = 60) -> list[dict[str, Any]]:
    """Search STAC Item Search endpoints, using fallback on transport/API errors."""
    body = {"collections": [collection], "bbox": list(bbox),
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "limit": limit}
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            response = requests.post(f"{endpoint.rstrip('/')}/search", json=body, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            # A capped first page is sufficient to choose the nearest before/
            # after pair and avoids walking hundreds of scenes for broad AOIs.
            return payload.get("features", [])
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f"{endpoint}: {exc}")
            LOG.warning("STAC search failed at %s: %s", endpoint, exc)
    raise RuntimeError("All STAC endpoints failed: " + "; ".join(errors))


def scene_datetime(item: dict[str, Any]) -> datetime:
    """Parse STAC item's standard datetime field as UTC."""
    raw = item.get("properties", {}).get("datetime")
    if not raw:
        raise ValueError(f"STAC item {item.get('id')} has no properties.datetime")
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def query_event_scenes(collection: str, bbox: tuple[float, float, float, float], event: date,
                       before_days: int = 48, after_days: int = 48, **kwargs: Any) -> list[dict[str, Any]]:
    """Query a symmetric temporal window around the event date."""
    return search_stac(collection, bbox, event - timedelta(days=before_days),
                       event + timedelta(days=after_days), **kwargs)


def search_planetary_computer(collection: str, bbox: tuple[float, float, float, float],
                              start: date, end: date, *, limit: int = 200,
                              timeout: int = 60) -> list[dict[str, Any]]:
    """Search Planetary Computer and sign returned blob assets for rasterio.

    This uses the public STAC API and optional ``PC_SDK_SUBSCRIPTION_KEY``.
    STAC discovery itself does not require a key; PC throttles anonymous reads.
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError as exc:
        raise RuntimeError("Planetary Computer search needs pystac-client and planetary-computer") from exc
    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC,
                                        modifier=planetary_computer.sign_inplace,
                                        timeout=timeout)
    search = catalog.search(collections=[collection], bbox=list(bbox),
                            datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
                            max_items=limit)
    return [item.to_dict() for item in search.items()]
