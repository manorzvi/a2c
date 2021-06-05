from typing import Tuple
from torchvision import transforms


class AtariBaseTransform:
    def __init__(
            self,
            normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __call__(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )


TRANSFORMS = {
    "AtariBaseTransform": AtariBaseTransform,
}