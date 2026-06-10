import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _parse_lut_size(lines) -> int:
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    raise ValueError("LUT_3D_SIZE not found in .cube file")


def load_cube(path: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Load a .cube 3D LUT file into a tensor of shape [N, N, N, 3] with values in [0, 1].
    """
    with open(path, "r") as f:
        raw_lines = f.readlines()

    size = _parse_lut_size(raw_lines)
    values = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("TITLE") or upper.startswith("LUT_3D_SIZE") or upper.startswith("DOMAIN_MIN") or upper.startswith("DOMAIN_MAX"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        values.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if len(values) != size ** 3:
        raise ValueError(f"Unexpected LUT length: got {len(values)}, expected {size ** 3}")

    lut = torch.tensor(values, dtype=torch.float32, device=device).view(size, size, size, 3)
    return lut


def resample_lut(lut: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resample a LUT tensor to a new size using trilinear interpolation.
    """
    if lut.shape[0] == size:
        return lut
    lut_5d = lut.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,N,N,N]
    lut_resized = F.interpolate(lut_5d, size=(size, size, size), mode="trilinear", align_corners=True)
    lut_resized = lut_resized.squeeze(0).permute(1, 2, 3, 0).contiguous()
    return lut_resized


def find_matching_lut(style_name: str, lut_root: str) -> Optional[str]:
    """
    Match a style name to a LUT file in lut_root, ignoring extension and case.
    """
    if not lut_root or not os.path.isdir(lut_root):
        return None
    target = style_name.strip().lower()
    for fname in os.listdir(lut_root):
        if not os.path.isfile(os.path.join(lut_root, fname)):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".cube", ".CUBE"):
            continue
        if stem.strip().lower() == target:
            return os.path.join(lut_root, fname)
    return None
