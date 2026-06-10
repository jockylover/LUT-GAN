import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import (
    StyleEncoder,
    LUTGenerator,
    MultiScalePatchDiscriminator,
    lut_tv_loss,
    lut_identity_loss,
    create_identity_lut,
    apply_lut_3d,
)
from dataset import PairedStyleDataset, MultiStyleLUTDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train LUT-GAN on paired or multi-style datasets")
    parser.add_argument("--content_dir", type=str, required=True,
                        help="Path to content (original) images")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="Path to target (styled) images for single-style training")
    parser.add_argument("--style_root", type=str, default=None,
                        help="Root folder with style reference images (multi-style)")
    parser.add_argument("--target_root", type=str, default=None,
                        help="Root folder with target images (multi-style)")
    parser.add_argument("--lut_root", type=str, default=None,
                        help="Optional folder with LUT .cube files for LUT supervision")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lut_size", type=int, default=33)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--encoder_arch", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"],
                        help="Backbone for style encoder")
    parser.add_argument("--pretrained_encoder", action="store_true",
                        help="Use ImageNet pretrained weights for the style encoder")
    parser.add_argument("--style_norm", type=str, default="auto",
                        choices=["auto", "none", "imagenet"],
                        help="Normalization for style input; auto matches pretrained_encoder")
    parser.add_argument("--delta_scale", type=float, default=0.1,
                        help="Residual LUT scale (higher = stronger LUTs)")
    parser.add_argument("--no_zero_mean", action="store_true",
                        help="Disable zero-mean residual in LUT generator")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_rec", type=float, default=100.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_tv", type=float, default=0.1)
    parser.add_argument("--lambda_id", type=float, default=0.2)
    parser.add_argument("--lambda_lut", type=float, default=1.0,
                        help="LUT supervision loss weight (requires lut_root)")
    parser.add_argument("--style_from_target", action="store_true",
                        help="Use target image as style input (multi-style)")
    parser.add_argument("--shadow_thresh", type=float, default=0.2,
                        help="Luminance threshold to upweight dark LUT bins")
    parser.add_argument("--shadow_boost", type=float, default=3.0,
                        help="Weight multiplier for bins darker than shadow_thresh")
    parser.add_argument("--num_scales", type=int, default=2,
                        help="Number of scales for multi-scale PatchGAN (>=1)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional cap per epoch (for quick tests)")
    parser.add_argument("--progress_update", type=int, default=10,
                        help="Update progress postfix every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Optional checkpoint to warm-start encoder/generator")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_style_norm(args) -> str:
    if args.style_norm == "auto":
        return "imagenet" if args.pretrained_encoder else "none"
    return args.style_norm


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(args.save_dir, exist_ok=True)

    style_norm = resolve_style_norm(args)

    # Dataset & loader
    if args.style_root or args.target_root:
        if not args.style_root or not args.target_root:
            raise ValueError("Both --style_root and --target_root are required for multi-style training")
        dataset = MultiStyleLUTDataset(
            content_dir=args.content_dir,
            style_root=args.style_root,
            target_root=args.target_root,
            image_size=args.image_size,
            style_norm=style_norm,
            lut_root=args.lut_root,
            lut_size=args.lut_size,
            use_target_as_style=args.style_from_target,
        )
    else:
        if not args.target_dir:
            raise ValueError("--target_dir is required for single-style training")
        dataset = PairedStyleDataset(
            content_dir=args.content_dir,
            target_dir=args.target_dir,
            image_size=args.image_size,
            style_norm=style_norm,
            use_target_as_style=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    print("Num pairs:", len(dataset))

    # Identity LUT
    identity_lut = create_identity_lut(args.lut_size, device=device)  # [N,N,N,3]

    # Models
    encoder = StyleEncoder(
        latent_dim=args.latent_dim,
        pretrained=args.pretrained_encoder,
        arch=args.encoder_arch,
    ).to(device)
    generator = LUTGenerator(
        latent_dim=args.latent_dim,
        size=args.lut_size,
        delta_scale=args.delta_scale,
        zero_mean=not args.no_zero_mean,
    ).to(device)
    discriminator = None
    if args.lambda_adv > 0:
        discriminator = MultiScalePatchDiscriminator(
            in_channels=6,
            num_scales=max(1, args.num_scales),
            use_spectral_norm=True,
        ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        generator.load_state_dict(ckpt["generator"])
        if discriminator is not None:
            disc_state = ckpt.get("discriminator")
            if isinstance(disc_state, dict):
                discriminator.load_state_dict(disc_state)
        print(f"Loaded weights from {args.resume}")

    optimizer_G = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D = None
    if discriminator is not None:
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.lr, betas=(0.5, 0.999)
        )

    criterion_l1 = nn.L1Loss()

    steps_per_epoch = len(dataloader)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = steps_per_epoch * args.epochs
    progress_interval = max(1, args.progress_update)
    progress = tqdm(total=total_steps, desc="Training", dynamic_ncols=True, file=sys.stdout)
    start_time = time.time()

    for epoch in range(args.epochs):
        encoder.train()
        generator.train()
        if discriminator is not None:
            discriminator.train()

        running_G, running_D = 0.0, 0.0
        step_count = 0
        epoch_start = time.time()

        for batch in dataloader:
            step_count += 1
            if len(batch) == 3:
                style_img, content_img, target_img = batch
                lut_gt = None
                lut_mask = None
            else:
                style_img, content_img, target_img, lut_gt, lut_mask = batch

            style_img = style_img.to(device)
            content_img = content_img.to(device)
            target_img = target_img.to(device)

            B = style_img.size(0)

            # ----- 1. Generator forward -----
            z = encoder(style_img)                               # [B,latent_dim]
            lut = generator(z, identity_lut)                     # [B,N,N,N,3]
            fake_img = apply_lut_3d(content_img, lut)            # [B,3,H,W]

            # ----- 2. Train discriminator D -----
            loss_D = torch.tensor(0.0, device=device)
            if discriminator is not None and optimizer_D is not None:
                optimizer_D.zero_grad()

                real_pair = torch.cat([content_img, target_img], dim=1)   # [B,6,H,W]
                fake_pair = torch.cat([content_img, fake_img.detach()], dim=1)

                pred_real_list = discriminator(real_pair)
                pred_fake_list = discriminator(fake_pair)

                # LSGAN loss averaged across scales
                loss_D_real = sum(((pred_real - 1) ** 2).mean() for pred_real in pred_real_list) / len(pred_real_list)
                loss_D_fake = sum((pred_fake ** 2).mean() for pred_fake in pred_fake_list) / len(pred_fake_list)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimizer_D.step()

            # ----- 3. Train generator G -----
            optimizer_G.zero_grad()

            loss_G_adv = torch.tensor(0.0, device=device)
            if discriminator is not None:
                fake_pair_for_G = torch.cat([content_img, fake_img], dim=1)
                pred_fake_for_G_list = discriminator(fake_pair_for_G)
                loss_G_adv = sum(((pred_fake_for_G - 1) ** 2).mean() for pred_fake_for_G in pred_fake_for_G_list) / len(pred_fake_for_G_list)
            loss_G_rec = criterion_l1(fake_img, target_img)
            loss_G_tv = lut_tv_loss(lut)
            loss_G_id = lut_identity_loss(
                lut,
                identity_lut,
                shadow_thresh=args.shadow_thresh,
                shadow_boost=args.shadow_boost,
            )

            loss_G_lut = torch.tensor(0.0, device=device)
            if lut_gt is not None and lut_mask is not None and args.lambda_lut > 0:
                lut_gt = lut_gt.to(device)
                lut_mask = lut_mask.to(device).view(B, 1, 1, 1, 1)
                loss_G_lut = ((lut - lut_gt).abs() * lut_mask).mean()

            loss_G = (args.lambda_adv * loss_G_adv +
                      args.lambda_rec * loss_G_rec +
                      args.lambda_tv * loss_G_tv +
                      args.lambda_id * loss_G_id +
                      args.lambda_lut * loss_G_lut)

            loss_G.backward()
            optimizer_G.step()

            running_G += loss_G.item()
            running_D += loss_D.item()

            progress.update(1)
            if (progress.n % progress_interval == 0) or (progress.n == total_steps):
                elapsed = time.time() - start_time
                eta = (elapsed / progress.n) * (total_steps - progress.n) if progress.n else 0.0
                progress.set_postfix({
                    "epoch": f"{epoch+1}/{args.epochs}",
                    "G": f"{loss_G.item():.3f}",
                    "D": f"{loss_D.item():.3f}",
                    "eta": format_duration(eta),
                })

            if args.max_steps and step_count >= args.max_steps:
                break

        denom = max(1, step_count)
        avg_G = running_G / denom
        avg_D = running_D / denom
        elapsed_epoch = time.time() - epoch_start
        avg_epoch_time = (time.time() - start_time) / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_epochs = avg_epoch_time * remaining_epochs
        print(
            f"[Epoch {epoch+1}/{args.epochs}] G_loss={avg_G:.4f}, D_loss={avg_D:.4f} "
            f"| epoch_time={format_duration(elapsed_epoch)} "
            f"| eta={format_duration(eta_epochs)}"
        )

        ckpt_path = os.path.join(args.save_dir, f"lutgan_epoch_{epoch+1}.pth")
        ckpt_args = vars(args)
        ckpt_args["style_norm"] = style_norm
        ckpt_args["delta_scale"] = args.delta_scale
        ckpt_args["zero_mean"] = not args.no_zero_mean
        torch.save({
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "epoch": epoch + 1,
            "args": ckpt_args,
        }, ckpt_path)
        print("Saved checkpoint to", ckpt_path)

    progress.close()


if __name__ == "__main__":
    main()
