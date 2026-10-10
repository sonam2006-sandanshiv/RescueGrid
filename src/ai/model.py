import torch
import segmentation_models_pytorch as smp

def create_model(encoder_name="resnet34", in_channels=4, classes=3):
    """
    Creates a U-Net model for flood and debris segmentation.
    in_channels: 4 (pre_vv, post_vv, pre_vh, post_vh)
    classes: 3 (0: none, 1: flood, 2: debris/uncertain)
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None, # training from scratch or use imagenet if 3 channels, but we have 4
        in_channels=in_channels,
        classes=classes,
    )
    return model

def get_loss_fn():
    """
    Returns a combined Dice and Focal loss to handle class imbalance.
    """
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    
    def criterion(y_pred, y_true):
        return dice_loss(y_pred, y_true) + focal_loss(y_pred, y_true)
        
    return criterion
