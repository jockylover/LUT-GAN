LUT-GAN Project
================

This repository contains a lightweight LUT-based style transfer model with a Gradio demo, CLI tools for inference/export, and training scripts.

Quick start (Windows, existing venv)
------------------------------------
- Activate the provided virtual environment: `.\lut\Scripts\activate`
- Launch the Gradio demo (uses default checkpoint path `checkpoints/best.pth`): `python app.py`
- Open the printed URL in your browser if it does not auto-launch.

If you prefer your own environment, install Python 3.9+ and run:
`pip install torch torchvision pillow tqdm gradio`

Running the demo
----------------
- Command: `python app.py --ckpt checkpoints/lutgan_epoch_10.pth --port 7860`
- Inputs: upload a style image (required) and optionally a content image (keeps original resolution).
- Outputs: stylized image preview and optional exported 3D LUT (`outputs/style_<timestamp>.cube`).

CLI inference (stylize one image)
---------------------------------
`python infer.py --ckpt <path/to/ckpt.pth> --style <style.jpg> --content <content.jpg> --output <out.jpg> --image_size 256`
- Uses a resized style image for encoding; content is also resized to `image_size` in this script.
- Device auto-selects CUDA if available; override with `--device cpu`.

Export a LUT from a style image
-------------------------------
`python export_lut.py --ckpt <path/to/ckpt.pth> --style <style.jpg> --lut_path outputs/style.cube --device cuda`
- Produces a `.cube` LUT file matching the trained LUT size.

Training
--------
`python train.py --content_dir <content_images> --target_dir <styled_images> --save_dir checkpoints --epochs 10 --batch_size 2`
- Content/target images are paired by filename (e.g., `001.jpg` must exist in both folders).
- Key hyperparameters: `--lut_size` (default 17), `--latent_dim` (default 256), `--lambda_*` weights for losses.
- Checkpoints are saved under `checkpoints/lutgan_epoch_<n>.pth` with args embedded for reuse.

Files and modules
-----------------
- `app.py`: Gradio UI for interactive LUT generation and export.
- `infer.py`: CLI stylization for a single style/content pair.
- `export_lut.py`: CLI tool to export a 3D LUT from a style image.
- `train.py`: Training loop for paired style datasets.
- `dataset.py`: Paired dataset loader (content/target matched by filename).
- `modules/`: Model components (`StyleEncoder`, `LUTGenerator`, LUT ops, losses, discriminator).

Troubleshooting
---------------
- If the demo cannot bind the port, it will auto-increment up to 10 times.
- Ensure the checkpoint path exists; pass a specific `.pth` with `--ckpt` if the default is missing.
- For CUDA issues, run with `--device cpu`.
