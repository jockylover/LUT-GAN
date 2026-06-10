import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, use_spectral_norm: bool = False):
        """
        input concat(content, image) shape [B,6,H,W]
        """
        super().__init__()

        def _maybe_sn(layer: nn.Module) -> nn.Module:
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        def CBlock(in_c, out_c, norm=True):
            layers = [_maybe_sn(nn.Conv2d(in_c, out_c, 4, 2, 1))]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            CBlock(in_channels, 64, norm=False),
            CBlock(64, 128),
            CBlock(128, 256),
            CBlock(256, 512),
            _maybe_sn(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, num_scales: int = 2, use_spectral_norm: bool = False):
        super().__init__()
        num_scales = max(1, int(num_scales))
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels=in_channels, use_spectral_norm=use_spectral_norm)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor):
        preds = []
        feat = x
        for idx, disc in enumerate(self.discriminators):
            preds.append(disc(feat))
            if idx < len(self.discriminators) - 1:
                feat = self.downsample(feat)
        return preds
