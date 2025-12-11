import torch
import torch.nn as nn
from kornia.filters import sobel, canny


class ThresholdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        print(x.shape)
        green = x[:, 1, :, :].unsqueeze(1)
        sobimg = sobel(green)
        return 50 * (sobimg - self.threshold)


class ThresholdCanny(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([0.5]))
        self.low_threshold = nn.Parameter(torch.tensor([0.1]))
        self.high_threshold = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x):
        green = x[:, 1, :, :].unsqueeze(1)
        cannyimg = canny(green)
        return 50 * (cannyimg - self.threshold)
