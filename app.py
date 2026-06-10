import argparse
import os
import time
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

"""
Compat shim: some huggingface_hub versions do not expose HfFolder, but newer
gradio releases still import it. Create a lightweight stub if missing.
"""
try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolder:
            @staticmethod
            def get_token():
                return os.environ.get("HF_TOKEN")

            @staticmethod
            def save_token(token: str):
                os.environ["HF_TOKEN"] = token

        huggingface_hub.HfFolder = _HfFolder  # type: ignore[attr-defined]
except Exception:
    huggingface_hub = None  # gradio import will still fail if totally missing

import gradio as gr

# Force localhost check to succeed to avoid launch errors in restrictive environments
try:
    import gradio.networking as _gnet

    _gnet.url_ok = lambda url: True  # type: ignore[assignment]
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Gradio client schema compatibility fix (handles bool schemas gracefully)
# --------------------------------------------------------------------------- #
try:
    import gradio_client.utils as gc_utils

    _orig_get_type = getattr(gc_utils, "get_type", None)
    _orig_json_schema_to_python_type = getattr(gc_utils, "_json_schema_to_python_type", None)

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        if _orig_get_type:
            return _orig_get_type(schema)
        return "Any"

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        if _orig_json_schema_to_python_type:
            return _orig_json_schema_to_python_type(schema, defs)
        return "Any"

    gc_utils.get_type = _safe_get_type  # type: ignore[attr-defined]
    gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore[attr-defined]
    gc_utils.json_schema_to_python_type = lambda schema: _safe_json_schema_to_python_type(  # type: ignore[attr-defined]
        schema, schema.get("$defs") if isinstance(schema, dict) else None
    )
except Exception:
    pass

from modules import StyleEncoder, LUTGenerator, create_identity_lut, apply_lut_3d


def parse_args():
    parser = argparse.ArgumentParser(description="LUT-GAN Gradio demo")
    parser.add_argument("--ckpt", type=str, default="checkpoints/lutgan_epoch_80.pth",
                        help="Path to checkpoint .pth")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Style image resize for encoder (content keeps original size)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu; auto-detect if not set")
    parser.add_argument("--port", type=int, default=7860, help="Port for gradio; auto-fallback if busy")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for gradio")
    return parser.parse_args()


def build_style_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def _resolve_ckpt_path(raw_path: str) -> str:
    """
    Resolve a checkpoint path. Accepts:
    - direct file path
    - directory containing .pth files (picks latest by mtime)
    - basename of a .pth inside checkpoints/ (searches recursively)
    """
    if os.path.isfile(raw_path):
        return raw_path

    candidates = []
    if os.path.isdir(raw_path):
        search_roots = [raw_path]
    else:
        # look under checkpoints/ recursively for matching basename
        search_roots = [os.path.join("checkpoints")]
        target_name = os.path.basename(raw_path)
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".pth"):
                    continue
                if not os.path.isdir(raw_path):
                    if fname != target_name:
                        continue
                candidates.append(os.path.join(dirpath, fname))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint not found: {raw_path}")
    # pick latest by modification time
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_models(ckpt_path: str, device: str) -> Tuple[StyleEncoder, LUTGenerator, torch.Tensor, dict, str]:
    ckpt_path = _resolve_ckpt_path(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]
    latent_dim = ckpt_args["latent_dim"]
    lut_size = ckpt_args["lut_size"]

    identity_lut = create_identity_lut(lut_size, device=device)
    # pretrained=False avoids network downloads; weights are loaded from the checkpoint
    encoder = StyleEncoder(latent_dim=latent_dim, pretrained=False)
    generator = LUTGenerator(latent_dim=latent_dim, size=lut_size)
    encoder.load_state_dict(ckpt["encoder"])
    generator.load_state_dict(ckpt["generator"])
    encoder.to(device).eval()
    generator.to(device).eval()
    return encoder, generator, identity_lut, ckpt_args, ckpt_path


def save_cube_lut(lut: torch.Tensor, path: str):
    """
    lut: [N,N,N,3], values in [0,1]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    N = lut.shape[0]
    lut_np = lut.detach().cpu().numpy().reshape(-1, 3)
    with open(path, "w") as f:
        f.write("# Generated by LUT-GAN\n")
        f.write(f"LUT_3D_SIZE {N}\n")
        for r, g, b in lut_np:
            f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def make_runner(encoder, generator, identity_lut, style_preprocess, content_preprocess, device, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    def run(style_img: Optional[Image.Image], content_img: Optional[Image.Image], export_lut: bool):
        if style_img is None:
            return None, None, "请先上传风格图"
        try:
            style = style_preprocess(style_img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = encoder(style)
                lut = generator(z, identity_lut)          # [1,N,N,N,3]
                out_pil = None
                if content_img is not None:
                    content = content_preprocess(content_img).unsqueeze(0).to(device)
                    out = apply_lut_3d(content, lut)      # [1,3,H,W]
                    out_pil = to_pil(out.squeeze(0).clamp(0, 1).cpu())
            lut_path = None
            if export_lut:
                stamp = int(time.time())
                lut_path = os.path.join(output_dir, f"style_{stamp}.cube")
                save_cube_lut(lut[0], lut_path)
            msg = "完成"
            if out_pil is None:
                msg = "已生成 LUT"
            if lut_path:
                msg += f" | LUT 已保存: {lut_path}"
            return out_pil, lut_path, msg
        except Exception as e:
            return None, None, f"出错: {e}"

    return run


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    style_preprocess = build_style_transform(args.image_size)
    # Content keeps original resolution; only normalize to tensor
    content_preprocess = transforms.ToTensor()
    encoder, generator, identity_lut, _, resolved_ckpt = load_models(args.ckpt, device)
    print(f"Loaded checkpoint from {resolved_ckpt} on {device}")
    runner = make_runner(encoder, generator, identity_lut, style_preprocess, content_preprocess, device)

    with gr.Blocks(title="LUT-GAN Demo") as demo:
        gr.Markdown("## LUT-GAN: 从风格图生成 LUT 并应用到内容图（内容图可选，保持原图尺寸）")
        with gr.Row():
            style_in = gr.Image(type="pil", label="风格图")
            content_in = gr.Image(type="pil", label="内容图（可选，仅应用 LUT 时使用）")
        export_checkbox = gr.Checkbox(label="导出 LUT (.cube)", value=True)
        run_btn = gr.Button("生成 / 导出")
        out_img = gr.Image(type="pil", label="风格化结果")
        lut_file = gr.File(label="导出 LUT 文件")
        status = gr.Textbox(label="状态", interactive=False)
        reset_btn = gr.ClearButton([style_in, content_in, export_checkbox, out_img, lut_file, status])

        run_btn.click(fn=runner, inputs=[style_in, content_in, export_checkbox],
                      outputs=[out_img, lut_file, status])

    port = args.port
    last_err = None
    for _ in range(10):
        try:
            demo.launch(
                server_name=args.host,
                server_port=port,
                share=False,
                inbrowser=False,
                show_error=True,
            )
            return
        except OSError as e:
            last_err = e
            port += 1
            continue
    raise RuntimeError(f"Failed to launch Gradio after trying 10 ports starting at {args.port}") from last_err


if __name__ == "__main__":
    main()
