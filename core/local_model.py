from __future__ import annotations

from typing import Callable

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def get_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def _try_load_pretrained_weights(arch: str, pretrained: bool):
    if not pretrained:
        return None

    try:
        if arch == "mobilenet_v3_small":
            from torchvision.models import MobileNet_V3_Small_Weights

            return MobileNet_V3_Small_Weights.IMAGENET1K_V1
        if arch == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights

            return EfficientNet_B0_Weights.IMAGENET1K_V1
        if arch == "resnet18":
            from torchvision.models import ResNet18_Weights

            return ResNet18_Weights.IMAGENET1K_V1
    except Exception:
        # Offline or missing cached weights.
        return None
    return None


def build_model(
    arch: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    from torch import nn
    from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18

    if arch not in {"mobilenet_v3_small", "resnet18", "efficientnet_b0"}:
        raise ValueError(f"Unsupported arch: {arch}")

    weights = _try_load_pretrained_weights(arch=arch, pretrained=pretrained)

    if arch == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone_filter: Callable[[str], bool] = lambda name: name.startswith("features")
    elif arch == "efficientnet_b0":
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        backbone_filter = lambda name: name.startswith("features")
    else:
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        backbone_filter = lambda name: not name.startswith("fc")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if backbone_filter(name):
                param.requires_grad = False

    return model
