from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _norm_groups(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetDehazer(nn.Module):
    """Small U-Net baseline with the same callable interface as DehazingDiT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        channels = [base_channels * (2 ** i) for i in range(depth)]

        encoders = []
        prev_channels = in_channels
        for channels_i in channels:
            encoders.append(ConvBlock(prev_channels, channels_i))
            prev_channels = channels_i
        self.encoders = nn.ModuleList(encoders)

        self.downs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i], kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(_norm_groups(channels[i]), channels[i]),
                    nn.SiLU(),
                )
                for i in range(depth - 1)
            ]
        )

        self.up_convs = nn.ModuleList(
            [
                nn.Conv2d(channels[i + 1], channels[i], kernel_size=3, padding=1)
                for i in reversed(range(depth - 1))
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(channels[i] * 2, channels[i])
                for i in reversed(range(depth - 1))
            ]
        )
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, x_hazy: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or x_hazy.ndim != 4:
            raise RuntimeError("UNetDehazer expects 4D image tensors.")
        if z.shape != x_hazy.shape:
            raise RuntimeError(f"z shape {tuple(z.shape)} must match x_hazy shape {tuple(x_hazy.shape)}")

        x = torch.cat([z, x_hazy], dim=1)
        if x.shape[1] != self.in_channels:
            raise RuntimeError(f"UNetDehazer expected {self.in_channels} input channels, got {x.shape[1]}")

        skips: List[torch.Tensor] = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if idx < len(self.downs):
                x = self.downs[idx](x)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.output(x)
