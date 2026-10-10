# Limitations of the Prototype

This flood damage mapping system is an **educational prototype** developed for "Track B: Mapping Flood Damage from Space". It is **not** an operational tool and must not be used for life-safety decisions.

## Known Technical Limitations
- **Temporal Resolution**: Sentinel-1 has a 12-day revisit time (or 6-day with both satellites, though S1B is offline). This means the post-event image may miss peak inundation, capturing only residual water or debris.
- **Topographic Distortions**: The Himalayan region features extremely steep terrain. This causes severe radar shadow and layover effects in Sentinel-1 SAR imagery, obscuring valleys where floods often occur.
- **Spatial Resolution**: Sentinel S1/S2 data operates at 10-20m resolution. Narrow features like footpaths, suspension bridges, or individual buildings are often sub-pixel and may be missed entirely.
- **Optical Cloud Cover**: Sentinel-2 imagery is frequently blocked by clouds during monsoon season, limiting fusion capabilities.
- **OSM Completeness**: OpenStreetMap data in rural Nepal is often incomplete. Buildings, local trails, or clinics might not be mapped, leading to underestimation of impact.
- **AI Domain Shift**: The U-Net model is trained primarily on global datasets (Kuro Siwo, Sen1Floods11). Performance in the specific domain of steep Himalayan valleys may degrade compared to the test set metrics.
- **Unverified Passability**: A road intersecting a flood polygon is marked as "affected" and heavily penalized in the network routing. However, this does not guarantee the road is impassable on the ground, nor does it detect landslides outside the immediate river corridor.
