
import numpy as np
import torch
import torchvision.transforms.v2 as t
from typing import Any, Dict, List


class ColorJitter_custom(torch.nn.Module):
    def forward(self, img, mask):
        trans = t.ColorJitter(brightness=.5, hue=.3)
        out = trans(img[:3])
        trans_img = torch.cat((out, img[3:]), 0)
        return trans_img, mask


class RandomColorJitter_custom(ColorJitter_custom):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        apply_transform = (torch.rand(size=(1,)) < self.p).item()
        params = dict(apply_transform=apply_transform)
        return params

    def transform(self, inpt: Any, params: Dict[str, Any]):
        if not params["apply_transform"]:
            return inpt
        else:
            return super().transform(inpt, params)


class RandomAutocontrast_custom(torch.nn.Module):
    def forward(self, img, mask):
        trans = t.RandomAutocontrast()
        out = trans(img[:3])
        trans_img = torch.cat((out, img[3:]), 0)
        return trans_img, mask


class AdjustSharpness_custom(torch.nn.Module):
    def forward(self, img, mask):
        trans = t.RandomAdjustSharpness(sharpness_factor=2)
        out = trans(img[:3])
        trans_img = torch.cat((out, img[3:]), 0)
        return trans_img, mask


class RandomAdjustSharpness_custom(AdjustSharpness_custom):
    def __init__(self, p=1):
        self.p = p
        super().__init__()

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        apply_transform = (torch.rand(size=(1,)) < self.p).item()
        params = dict(apply_transform=apply_transform)
        return params

    def transform(self, inpt: Any, params: Dict[str, Any]):
        if not params["apply_transform"]:
            return inpt
        else:
            return super().transform(inpt, params)
