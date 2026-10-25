# `flood_unet_best.pth` checkpoint record

## What is known

The local checkpoint file is a PyTorch zip-format state dictionary. A static inspection of its tensor names found an encoder/decoder and `segmentation_head.0` consistent with a ResNet-18 U-Net state dictionary. The file itself does not contain a model card, input normalization values, training split, class definition, or probability threshold.

## Use in this prototype

`src/ai/infer.py` provides safe `weights_only=True` loading and CPU inference for aligned Sentinel-1 VV/VH dB grids. It requires training means and standard deviations and a threshold selected using validation data. Ambiguous probabilities and no-data remain class `3` unknown. The detector does not automatically blend this checkpoint into the classical map until those model settings and independent held-out metrics are provided.

## Validation status

Not yet independently scored. No checkpoint accuracy, precision, recall, F1, or IoU is claimed. Before operational use, attach the training metadata and score against a dataset split that was not used to tune the model. The official Sen1Floods11 hand-labeled held-out set is one eligible external benchmark, with the Bolivia event a separate geographic holdout; see the [dataset repository](https://github.com/cloudtostreet/Sen1Floods11) and [Bonafilia et al. (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf).
