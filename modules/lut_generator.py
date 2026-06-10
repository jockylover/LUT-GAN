import torch
import torch.nn as nn


class LUTGenerator(nn.Module):
    def __init__(self, latent_dim: int = 256, size: int = 17, delta_scale: float = 0.1, zero_mean: bool = True):
        super().__init__()
        self.size = size
        self.delta_scale = delta_scale
        self.zero_mean = zero_mean
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, size * size * size * 3),
        )

    def forward(self, z, identity_lut: torch.Tensor):
        """
        z: [B,latent_dim]
        identity_lut: [N,N,N,3]
        return: [B,N,N,N,3]
        """
        B = z.size(0)
        N = self.size
        delta = self.fc(z)                      # [B,N^3*3]
        delta = delta.view(B, N, N, N, 3)
        delta = torch.tanh(delta) * self.delta_scale  # limit residual magnitude
        if self.zero_mean:
            # Zero-mean the residual to avoid global brightness shifts
            delta = delta - delta.mean(dim=(1, 2, 3, 4), keepdim=True)

        lut = identity_lut.unsqueeze(0) + delta
        lut = torch.clamp(lut, 0, 1)
        # Anchor extreme corners so black/white remain stable
        lut[:, 0, 0, 0, :] = 0.0
        lut[:, -1, -1, -1, :] = 1.0
        return lut
