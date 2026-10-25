# Historical flood detector benchmark

## Result

The current Sentinel-1 RTC change threshold was evaluated on one chip from the official Sen1Floods11 v1.1 held-out test split: India, chip `900498`. The hand-labeled reference acquisition is 2016-08-12. The chip covers approximately 5 km by 5 km.

The selected satellite observations were same-track Sentinel-1A RTC scenes:

- Before: 2016-07-19 23:46:35 UTC, relative orbit 77, descending.
- Event/reference date: 2016-08-12 23:46:36 UTC, relative orbit 77, descending.

The detector applied the existing rule unchanged: post-event VV in the lowest 35th percentile and a decrease greater than 2.5 dB from the pre-event observation. The two RTC scenes and the reference were evaluated on a common 20 m UTM grid (59,330 scored pixels, about 2,373 ha).

| Metric | Measured result |
| --- | ---: |
| Pixel accuracy | 73.48% |
| Precision | 78.99% |
| Recall | 2.35% |
| F1 | 4.56% |
| IoU | 2.33% |
| False positive rate | 0.23% |
| False alarm area | 4.00 ha |
| Missed reference flood area | 625.28 ha |
| Reference-valid pixel coverage | 100% |

The positive reference was constructed as `LabelHand water AND NOT JRCWaterHand permanent water`; the remaining valid hand-labeled water is used as a flood-only proxy. Reference unknown pixels remain excluded. This avoids counting mapped permanent water as new flooding, but it is still a proxy: the hand-label layer describes water extent and the subtraction depends on the JRC permanent-water layer.

## Interpretation and limits

On this chip the threshold was conservative: most flagged pixels agreed with the derived reference, but it detected only a small share of reference flood pixels. The 73.46% pixel accuracy is dominated by non-flood pixels and should not be read alone; recall, F1, IoU, false alarms, and missed area are more informative here.

This is one held-out chip from one event, not a regional accuracy estimate. Sen1Floods11 is an independent, georeferenced benchmark with hand-labeled chips and published event metadata ([dataset documentation](https://github.com/cloudtostreet/Sen1Floods11/blob/master/docs/README.md), [Bonafilia et al., CVPR Workshops 2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html)).

The RTC scenes did not expose a STAC-documented layover/shadow validity mask, so this run cannot establish terrain-wide reliability. The reported pixel coverage means the arrays had valid values, not that radar shadow, steep terrain, or vegetation effects were ruled out. No model checkpoint accuracy is reported; the provided checkpoint lacks the training normalization and validation threshold needed for a defensible inference run.

Machine-readable counts, acquisition IDs, grid, and thresholds are in [`runs/benchmark-india-900498-2016-08/metrics.json`](../runs/benchmark-india-900498-2016-08/metrics.json). The predicted tri-state raster is [`runs/benchmark-india-900498-2016-08/prediction.tif`](../runs/benchmark-india-900498-2016-08/prediction.tif). The source reference rasters used for this run are under [`evaluation/data/sen1floods11`](data/sen1floods11).
