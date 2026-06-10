import warnings

import torch.nn as nn
import torchvision.models as models

BACKBONE_OUT = {
    "resnet18": 512,
    "resnet50": 2048,
}


class StyleEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, pretrained: bool = True, arch: str = "resnet18"):
        """
        arch: resnet18 (lighter) or resnet50 (richer features)
        """
        super().__init__()
        arch = arch.lower()
        if arch not in BACKBONE_OUT:
            raise ValueError(f"Unsupported encoder arch: {arch}")

        base = None
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            try:
                base = models.resnet18(weights=weights)
            except Exception as exc:
                if pretrained:
                    warnings.warn(f"Failed to load pretrained weights, using random init: {exc}")
                    base = models.resnet18(weights=None)
                else:
                    raise
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            try:
                base = models.resnet50(weights=weights)
            except Exception as exc:
                if pretrained:
                    warnings.warn(f"Failed to load pretrained weights, using random init: {exc}")
                    base = models.resnet50(weights=None)
                else:
                    raise

        self.features = nn.Sequential(*list(base.children())[:-1])  # drop final fc layer
        self.fc = nn.Linear(BACKBONE_OUT[arch], latent_dim)

    def forward(self, x):
        h = self.features(x)           # [B,C,1,1]
        h = h.view(h.size(0), -1)      # [B,C]
        z = self.fc(h)                 # [B,latent_dim]
        return z
