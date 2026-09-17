"""SAM2 (Hiera + FPN neck) as a dense binary segmentation backbone.

SAM2's own mask decoder is promptable (point/box/mask prompts, tuned for
interactive + video segmentation) and doesn't fit automatic, dense,
per-pixel binary segmentation. Instead this discards everything except
`Sam2Model.vision_encoder` (the Hiera backbone + its built-in FPN neck,
which already emits a 3-level feature pyramid at 1/4, 1/8 and 1/16 scale,
256 channels each) and attaches a small trainable top-down FPN decoder that
upsamples back to full resolution -- the same "pretrained encoder + light
decoder head" pattern used for the sm_deeplabv3plus/sm_unet baselines, just
with SAM2's SA-1B-pretrained encoder instead of an ImageNet-pretrained
ResNet.

Pretrained weights are pulled from the Hugging Face Hub on first use
(`facebook/sam2.1-hiera-{tiny,small,base-plus,large}`), so there is no
vendored SAM2 source tree and no checkpoint file to ship. This needs
transformers >= 4.56, which is where `Sam2Model` landed.

Like every other model in this repo, `forward` returns probabilities in
[0, 1] at exactly the input resolution: `src/train2.py` thresholds the raw
output and never applies a sigmoid of its own.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _FPNDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, decoder_channels: int = 128, out_channels: int = 1):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels, kernel_size=1) for _ in range(3)
        ])
        self.smooth = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
        self.up_head = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, decoder_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, decoder_channels // 2),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_channels // 2, out_channels, kernel_size=1)

    def forward(self, fpn_hidden_states):
        # fpn_hidden_states: finest -> coarsest, e.g. [1/4, 1/8, 1/16] of input
        p_fine, p_mid, p_coarse = fpn_hidden_states
        p3 = self.lateral[2](p_coarse)
        p2 = self.lateral[1](p_mid) + F.interpolate(p3, size=p_mid.shape[-2:], mode="nearest")
        p1 = self.lateral[0](p_fine) + F.interpolate(p2, size=p_fine.shape[-2:], mode="nearest")
        p1 = self.smooth(p1)
        x = self.up_head(p1)  # 1/4 scale -> full resolution (2x upsample twice)
        return self.head(x)


class SAM2SegNet(nn.Module):
    """SAM2's Hiera encoder with a light FPN decoder head for binary segmentation.

    Args:
        in_channels: must be 3 -- SAM2's patch embedding is RGB-pretrained.
        out_channels: number of output mask channels (1 for fracture/no-fracture).
        checkpoint: Hugging Face Hub id of the SAM2 variant to load.
        decoder_channels: width of the trainable FPN decoder.
        freeze_encoder: if True, train the decoder only (linear-probe style).
        img_size: if set, bilinearly resize the input up to this size before the
            encoder and resize the prediction back down. Benchmark tiles are
            256 px while SAM2 was pretrained at 1024, so this trades compute for
            a closer match to the pretraining scale. Hiera handles arbitrary
            input sizes, so leaving this as None is valid.
        normalize: None to feed the dataset's plain [0, 1] RGB (what every other
            model in this repo sees), or "imagenet" to apply ImageNet mean/std
            inside forward, matching SAM2's own preprocessing. Doing it here
            rather than in the dataset keeps the shared data pipeline identical
            for all benchmark rows.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        checkpoint: str = "facebook/sam2.1-hiera-tiny",
        decoder_channels: int = 128,
        freeze_encoder: bool = False,
        img_size: Optional[int] = None,
        normalize: Optional[str] = None,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("SAM2's pretrained encoder expects 3-channel RGB input")
        if normalize not in (None, "none", "imagenet"):
            raise ValueError(f"normalize must be None or 'imagenet', got {normalize!r}")

        self.img_size = img_size
        self.normalize = None if normalize == "none" else normalize
        self.encoder = Sam2Model.from_pretrained(checkpoint).vision_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        # Sam2's neck always projects every FPN level to this many channels,
        # regardless of backbone size (tiny/small/base_plus/large).
        self.decoder = _FPNDecoder(in_channels=256, decoder_channels=decoder_channels,
                                   out_channels=out_channels)

        # Buffers so .to(device)/state_dict round-tripping keeps working.
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if self.img_size is not None and (h, w) != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        if self.normalize == "imagenet":
            x = (x - self._mean) / self._std
        enc_out = self.encoder(x)
        logits = self.decoder(enc_out.fpn_hidden_states)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)
