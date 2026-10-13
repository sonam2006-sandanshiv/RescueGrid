import pystac_client


def get_catalog():
    """Return unsigned STAC items so raster assets can be freshly signed when read."""
    return pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
