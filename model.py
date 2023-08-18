from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def build_default_model(num_classes: int, in_channels: int,
                            img_sz: int) -> FasterRCNN:
    """Returns a FasterRCNN model.

    Note that the model returned will have (num_classes + 1) output
    classes. +1 for the null class (zeroth index)

    Returns:
        FasterRCNN: a FasterRCNN model.
    """
    pretrained = True
    backbone_arch = "resnet152"
    backbone = resnet_fpn_backbone(backbone_arch, pretrained)

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes + 1,
        min_size=img_sz,
        max_size=img_sz,
        image_mean=image_mean,
        image_std=image_std)
    return model