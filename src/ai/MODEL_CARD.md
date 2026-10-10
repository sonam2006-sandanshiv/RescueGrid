# Flood & Debris Segmentation U-Net

## Model Details
- **Architecture**: U-Net (segmentation-models-pytorch) with ResNet34 encoder.
- **Inputs**: 4 channels (Sentinel-1 VV Pre, Sentinel-1 VV Post, Sentinel-1 VH Pre, Sentinel-1 VH Post).
- **Outputs**: 3 classes (0: None, 1: Flood Water, 2: Debris/Mud/Uncertain).
- **Training Data**: Kuro Siwo (Bountos et al., 2024, NeurIPS).
- **Evaluation**: Split strictly by event/region to ensure no geographic leakage. Tested on held-out unseen Himalayan regions.

## Intended Use
- **Primary Use**: Educational prototype for post-disaster rapid mapping (Track B: Mapping Flood Damage from Space).
- **Out of Scope**: Not an operational tool. Not for life-safety decisions.

## Metrics
- **Overall IoU**: TBD
- **Flood F1**: TBD
- **Debris F1**: TBD

## Known Limitations and Failure Modes
- **Domain Shift**: Trained largely on global datasets. Performance in steep Himalayan valleys (radar shadow/layover) may degrade.
- **Class Imbalance**: Debris is significantly underrepresented in most datasets compared to water.
- **Resolution**: At 10-20m resolution, small or narrow flood extents might be missed.
