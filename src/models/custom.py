import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class ResNet34_UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet34(pretrained=pretrained)

        # Encoder
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64
        self.pool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.decoder0 = DecoderBlock(64, 64, 32)

        # Final classifier
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # extra skip from early features
        d0 = self.decoder0(d1, e0)

        out = self.final(d0)
        return out
