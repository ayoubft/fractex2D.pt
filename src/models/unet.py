from collections import OrderedDict
import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    A U-Net architecture for image segmentation with configurable input/output
    channels and initial feature maps.
    """

    def __init__(self, in_channels=4, out_channels=1, init_features=32):
        super().__init__()
        features = init_features

        # Encoder blocks
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        # Final 1x1 convolution
        self.conv = nn.Conv2d(features, out_channels, 1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        """
        Constructs a two-convolution block with BatchNorm and ReLU activation.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (f"{name}_conv1", nn.Conv2d(
                        in_channels, features, kernel_size=3, padding=1, bias=False
                    )),
                    (f"{name}_bn1", nn.BatchNorm2d(features)),
                    (f"{name}_relu1", nn.ReLU(inplace=True)),
                    (f"{name}_conv2", nn.Conv2d(
                        features, features, kernel_size=3, padding=1, bias=False
                    )),
                    (f"{name}_bn2", nn.BatchNorm2d(features)),
                    (f"{name}_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
