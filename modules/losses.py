import torch

def lut_tv_loss(lut: torch.Tensor) -> torch.Tensor:
    """
    lut: [B,N,N,N,3] 3D Total Variation smoothing
    """
    loss = 0.0
    loss = loss + (lut[:, 1:, :, :, :] - lut[:, :-1, :, :, :]).abs().mean()
    loss = loss + (lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]).abs().mean()
    loss = loss + (lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]).abs().mean()
    return loss


def lut_identity_loss(
    lut: torch.Tensor,
    identity_lut: torch.Tensor,
    shadow_thresh: float = 0.2,
    shadow_boost: float = 3.0,
) -> torch.Tensor:
    """
    lut: [B,N,N,N,3]
    identity_lut: [N,N,N,3]
    shadow_thresh: luminance threshold to upweight dark bins
    shadow_boost: extra weight applied to bins darker than shadow_thresh
    """
    target = identity_lut.to(lut.device).unsqueeze(0)  # [1,N,N,N,3]

    # Upweight dark bins so the LUT stays faithful in shadows
    luma = (identity_lut.to(lut.device) *
            torch.tensor([0.299, 0.587, 0.114], device=lut.device)).sum(-1)  # [N,N,N]
    shadow_mask = (luma < shadow_thresh).float()
    weight = (1.0 + shadow_boost * shadow_mask).unsqueeze(0).unsqueeze(-1)  # [1,N,N,N,1]

    return ((lut - target).abs() * weight).mean()
