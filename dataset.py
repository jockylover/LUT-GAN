import glob
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from preprocess import build_image_transform, build_style_transform
from lut_io import find_matching_lut, load_cube, resample_lut


class PairedStyleDataset(Dataset):
    """
    content_dir: directory of original content images
    target_dir: directory of color-graded images (used as both style and target);
                paired by filename, e.g., 001.jpg exists in both
    """
    def __init__(
        self,
        content_dir: str,
        target_dir: str,
        image_size: int = 256,
        style_norm: str = "none",
        use_target_as_style: bool = True,
    ):
        self.pairs: List[Tuple[str, str]] = []
        self.use_target_as_style = use_target_as_style

        content_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            content_paths.extend(glob.glob(os.path.join(content_dir, ext)))
        content_paths = sorted(content_paths)

        for c_path in content_paths:
            fname = os.path.basename(c_path)
            t_path = os.path.join(target_dir, fname)
            if os.path.exists(t_path):
                self.pairs.append((c_path, t_path))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid pairs between {content_dir} and {target_dir}")

        self.style_transform = build_style_transform(image_size, style_norm)
        self.image_transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple:
        c_path, t_path = self.pairs[idx]
        content_img = Image.open(c_path).convert("RGB")
        target_img = Image.open(t_path).convert("RGB")

        content = self.image_transform(content_img)
        target = self.image_transform(target_img)

        if self.use_target_as_style:
            style = self.style_transform(target_img)
        else:
            style = self.style_transform(content_img)

        return style, content, target


class MultiStyleLUTDataset(Dataset):
    """
    Multi-style paired dataset:
      content_dir: original images
      style_root:  style reference images organized by style name
      target_root: styled targets organized by style name
    """
    def __init__(
        self,
        content_dir: str,
        style_root: str,
        target_root: str,
        image_size: int = 256,
        style_norm: str = "none",
        lut_root: Optional[str] = None,
        lut_size: Optional[int] = None,
        use_target_as_style: bool = False,
    ):
        self.samples: List[Tuple[str, str, str]] = []
        self.style_to_paths: Dict[str, List[str]] = {}
        self.lut_cache: Dict[str, Optional[Tuple[torch.Tensor, float]]] = {}
        self.lut_size = lut_size
        self.use_target_as_style = use_target_as_style

        content_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            content_paths.extend(glob.glob(os.path.join(content_dir, ext)))
        content_paths = sorted(content_paths)

        styles = []
        for name in os.listdir(target_root):
            target_dir = os.path.join(target_root, name)
            style_dir = os.path.join(style_root, name)
            if os.path.isdir(target_dir) and os.path.isdir(style_dir):
                styles.append(name)
        styles = sorted(styles)
        if not styles:
            raise RuntimeError(f"No style folders found in {target_root}")

        for style_name in styles:
            target_dir = os.path.join(target_root, style_name)
            style_dir = os.path.join(style_root, style_name)
            style_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                style_paths.extend(glob.glob(os.path.join(style_dir, ext)))
            style_paths = sorted(style_paths)
            if not style_paths:
                continue
            self.style_to_paths[style_name] = style_paths

            for c_path in content_paths:
                fname = os.path.basename(c_path)
                t_path = os.path.join(target_dir, fname)
                if os.path.exists(t_path):
                    self.samples.append((style_name, c_path, t_path))

        if len(self.samples) == 0:
            raise RuntimeError("No valid multi-style pairs found")

        self.style_transform = build_style_transform(image_size, style_norm)
        self.image_transform = build_image_transform(image_size)

        if lut_root and lut_size:
            for style_name in styles:
                lut_path = find_matching_lut(style_name, lut_root)
                if lut_path:
                    lut = load_cube(lut_path)
                    lut = resample_lut(lut, lut_size)
                    self.lut_cache[style_name] = (lut, 1.0)
                else:
                    self.lut_cache[style_name] = (
                        torch.zeros(lut_size, lut_size, lut_size, 3),
                        0.0,
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_style_path(self, style_name: str, avoid_name: str) -> str:
        paths = self.style_to_paths[style_name]
        if len(paths) <= 1:
            return paths[0]
        for _ in range(5):
            cand = random.choice(paths)
            if os.path.basename(cand) != avoid_name:
                return cand
        return random.choice(paths)

    def __getitem__(self, idx: int) -> Tuple:
        style_name, c_path, t_path = self.samples[idx]
        content_img = Image.open(c_path).convert("RGB")
        target_img = Image.open(t_path).convert("RGB")

        if self.use_target_as_style:
            style_img = target_img
        else:
            style_path = self._pick_style_path(style_name, os.path.basename(t_path))
            style_img = Image.open(style_path).convert("RGB")

        style = self.style_transform(style_img)
        content = self.image_transform(content_img)
        target = self.image_transform(target_img)

        if self.lut_cache:
            lut, mask = self.lut_cache.get(style_name, (None, 0.0))
            if lut is None or self.lut_size is None:
                lut = torch.zeros(0)
                mask = 0.0
            return style, content, target, lut, torch.tensor(mask, dtype=content.dtype)

        return style, content, target
