import torch

def create_identity_lut(size: int, device=None):
    """
    Generate an identity 3D LUT: [N,N,N,3], RGB uniformly sampled in [0, 1]
    """
    if device is None:
        device = "cpu"
    rgbs = torch.linspace(0, 1, size, device=device)
    r, g, b = torch.meshgrid(rgbs, rgbs, rgbs, indexing="ij")
    lut = torch.stack([r, g, b], dim=-1)  # [N,N,N,3]
    return lut


def apply_lut_3d(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """
    Apply a 3D LUT to a batch of images (batched trilinear interpolation, differentiable).

    img: [B,3,H,W], value range [0,1]
    lut: [B,N,N,N,3], value range [0,1]
    return: [B,3,H,W]
    """
    B, C, H, W = img.shape
    _, N, _, _, _ = lut.shape
    device = img.device

    # Map pixels to [0, N-1]
    x = img.permute(0, 2, 3, 1) * (N - 1)      # [B,H,W,3]

    x0 = torch.floor(x).long().clamp(0, N - 2) # lower bound index
    x1 = x0 + 1                                 # upper bound index
    wx = x - x0.float()                         # interpolation weights 0..1

    r0, g0, b0 = x0[..., 0], x0[..., 1], x0[..., 2]
    r1, g1, b1 = x1[..., 0], x1[..., 1], x1[..., 2]

    wx_r = wx[..., 0].unsqueeze(-1)  # [B,H,W,1]
    wx_g = wx[..., 1].unsqueeze(-1)
    wx_b = wx[..., 2].unsqueeze(-1)

    b_idx = torch.arange(B, device=device)[:, None, None]  # [B,1,1]

    # 8 corner samples
    c000 = lut[b_idx, r0, g0, b0]  # [B,H,W,3]
    c001 = lut[b_idx, r0, g0, b1]
    c010 = lut[b_idx, r0, g1, b0]
    c011 = lut[b_idx, r0, g1, b1]
    c100 = lut[b_idx, r1, g0, b0]
    c101 = lut[b_idx, r1, g0, b1]
    c110 = lut[b_idx, r1, g1, b0]
    c111 = lut[b_idx, r1, g1, b1]

    # R-axis interpolation
    c00 = c000 * (1 - wx_r) + c100 * wx_r
    c01 = c001 * (1 - wx_r) + c101 * wx_r
    c10 = c010 * (1 - wx_r) + c110 * wx_r
    c11 = c011 * (1 - wx_r) + c111 * wx_r

    # G-axis interpolation
    c0 = c00 * (1 - wx_g) + c10 * wx_g
    c1 = c01 * (1 - wx_g) + c11 * wx_g

    # B-axis interpolation
    out = c0 * (1 - wx_b) + c1 * wx_b  # [B,H,W,3]

    return out.permute(0, 3, 1, 2)     # [B,3,H,W]
