import pystac_client
import planetary_computer

def get_catalog():
    """Returns the Planetary Computer STAC catalog client."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog
