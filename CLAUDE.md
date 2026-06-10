# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LUT-GAN: given a **style image**, an encoder produces a latent vector, a generator turns it
into a **3D color LUT**, and that LUT is applied to a **content image** via differentiable
trilinear interpolation. The LUT can be exported as a standard `.cube` file. There is a Gradio
demo plus CLI tools for inference, LUT export, evaluation, and training.

## Environment & commands

A Windows virtualenv lives at `lut/`. Either activate it or call its python directly:

- Activate: `.\lut\Scripts\activate`
- Direct: `lut\Scripts\python.exe <script>.py ...`

There is no test suite, linter, or build step — scripts are run directly.

Common entry points (all take `--device cpu` to override CUDA auto-detect):

- Demo: `python app.py --ckpt checkpoints/best.pth --port 7860` (UI text is Chinese)
- Inference: `python infer.py --ckpt <pth> --style <img> --content <img> --output <out>`
- Export LUT: `python export_lut.py --ckpt <pth> --style <img> --lut_path outputs/x.cube`
- Eval vs GT LUT: `python eval_lut.py --ckpt <pth> --style <img> --lut <gt.cube> --content <img>` (prints LUT L1 + output PSNR)
- Train (single-style): `python train.py --content_dir <c> --target_dir <t> --save_dir checkpoints`
- Train (multi-style): `python train.py --content_dir <c> --style_root <s> --target_root <t> [--lut_root style_luts]`
- Full 2-stage pipeline: `.\train_long.ps1` (stage1 = LUT-supervised pretrain with `--lambda_adv 0`, stage2 = GAN fine-tune, then eval; logs to `logs/`)

## Important repo facts

- **Checkpoints are self-describing.** `train.py` saves `{"encoder", "generator",
  "discriminator", "epoch", "args"}`. Inference/export/eval read `ckpt["args"]` to reconstruct
  the exact model (`lut_size`, `latent_dim`, `encoder_arch`, `delta_scale`, `zero_mean`,
  `style_norm`). When changing model construction, keep these arg keys in sync across
  `train.py` and the loaders in `app.py`/`infer.py`/`export_lut.py`/`eval_lut.py`, or old
  checkpoints break.
- `app.py` begins with monkeypatches for `huggingface_hub.HfFolder`, gradio networking, and
  `gradio_client` schema parsing — compatibility shims for the pinned versions in `lut/`.
  Don't remove them without testing the demo actually launches.

## Architecture (data flow)

`style_img → StyleEncoder → z → LUTGenerator(z, identity_lut) → lut → apply_lut_3d(content, lut) → stylized`

Modules live in `modules/` (re-exported via `modules/__init__.py`):

- `style_encoder.py` — `StyleEncoder`: ResNet18/50 backbone (fc dropped) → `Linear` to `latent_dim`.
- `lut_generator.py` — `LUTGenerator`: MLP predicts a **residual** added to the identity LUT.
  Residual is `tanh`-bounded by `delta_scale`, optionally zero-meaned (`zero_mean`) to avoid
  global brightness shifts, then clamped to [0,1]; the black `(0,0,0)` and white corners are
  hard-anchored so endpoints stay stable.
- `lut3d.py` — `create_identity_lut(size)` builds the `[N,N,N,3]` identity grid;
  `apply_lut_3d` is a hand-written batched, differentiable trilinear sampler (each item in the
  batch has its own LUT).
- `discriminator.py` — `MultiScalePatchDiscriminator`: a **conditional** PatchGAN. Input is
  `concat(content, image)` = 6 channels; multiple scales via avg-pool downsampling; LSGAN loss.
- `losses.py` — `lut_tv_loss` (3D total-variation smoothness) and `lut_identity_loss`
  (keep LUT near identity, with `shadow_boost` upweighting dark bins).

Generator/encoder train together under one Adam optimizer; the discriminator has its own.

### Training losses (weights are `--lambda_*`)

`loss_G = lambda_adv*adv + lambda_rec*L1(fake,target) + lambda_tv*tv + lambda_id*identity + lambda_lut*lut_supervision`

`lambda_adv=0` disables the discriminator entirely (no `optimizer_D`). `lambda_lut` only
applies when a `--lut_root` of `.cube` files is supplied and matched per style.

### Datasets (`dataset.py`)

- `PairedStyleDataset` — single style. Content and target images **paired by identical
  filename**; the target doubles as the style input. Returns `(style, content, target)`.
- `MultiStyleLUTDataset` — `style_root/<style_name>/` and `target_root/<style_name>/`
  subfolders, paired to `content_dir` by filename. Optionally loads a ground-truth `.cube` per
  style (`lut_io.find_matching_lut` matches by folder name) and returns
  `(style, content, target, lut_gt, lut_mask)`; the mask is 0 when no LUT was found.
  `train.py` branches on `len(batch)` to detect which form it got.

`preprocess.py` builds the transforms; `style_norm` (`none`/`imagenet`) must match how the
encoder was trained (`auto` → imagenet iff `--pretrained_encoder`). `lut_io.py` handles
`.cube` read + trilinear resample to the model's `lut_size`.

## Data layout

- `data/base_content/` — content images; `data/lut_multi/{style_images,targets}/<style>/` — multi-style training data.
- `style_luts/` — reference `.cube` LUTs (used as supervision targets, matched by name).
- `checkpoints/` — `.pth` files and per-run subfolders; `best.pth` is the demo default fallback.
- `outputs/` — generated images and exported `.cube` files; `logs/` — training logs.
- `FiveK_original/` — Adobe FiveK source images.
