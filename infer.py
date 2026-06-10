import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from modules import (
    StyleEncoder,
    LUTGenerator,
    create_identity_lut,
    apply_lut_3d,
)
from preprocess import build_style_transform


def parse_args():
    p = argparse.ArgumentParser(description="Inference: style + content -> stylized image")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    p.add_argument("--style", type=str, required=True, help="Path to style image")
    p.add_argument("--content", type=str, required=True, help="Path to content image")
    p.add_argument("--output", type=str, required=True, help="Path to save output image")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_style(path, image_size, style_norm):
    tfm = build_style_transform(image_size, style_norm)
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1,3,H,W]


def load_content(path):
    tfm = transforms.ToTensor()
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1,3,H,W]


def save_image(tensor, path):
    tensor = tensor.clamp(0, 1).squeeze(0)
    img = transforms.ToPILImage()(tensor.cpu())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt["args"]
    lut_size = ckpt_args["lut_size"]
    latent_dim = ckpt_args["latent_dim"]
    encoder_arch = ckpt_args.get("encoder_arch", "resnet18")
    delta_scale = ckpt_args.get("delta_scale", 0.1)
    zero_mean = ckpt_args.get("zero_mean", True)
    style_norm = ckpt_args.get("style_norm", "none")

    identity_lut = create_identity_lut(lut_size, device=device)

    encoder = StyleEncoder(latent_dim=latent_dim, arch=encoder_arch)
    generator = LUTGenerator(latent_dim=latent_dim, size=lut_size, delta_scale=delta_scale, zero_mean=zero_mean)

    encoder.load_state_dict(ckpt["encoder"])
    generator.load_state_dict(ckpt["generator"])

    encoder.to(device).eval()
    generator.to(device).eval()

    style_img = load_style(args.style, args.image_size, style_norm).to(device)
    content_img = load_content(args.content).to(device)

    with torch.no_grad():
        z = encoder(style_img)
        lut = generator(z, identity_lut)
        out = apply_lut_3d(content_img, lut)

    save_image(out, args.output)
    print("Saved stylized image to", args.output)


if __name__ == "__main__":
    main()
