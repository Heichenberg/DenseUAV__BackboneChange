# -*- coding: utf-8 -*-
"""Visualize GeoToken-V2 query-token attention maps on satellite images.

This script reads aux["attn"] from GeoToken-V2 DivHead-style heads. It does
not visualize query parameters or CAG gates; only cross-attention maps are
shown.
"""

from __future__ import print_function

import argparse
import math
import os
import random
import sys
from types import SimpleNamespace

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.taskflow import make_model  # noqa: E402
from tool.utils import load_network  # noqa: E402


# ========================= 用户配置区 =========================
# 平时只需要修改这里，然后运行：
#   python tools/visualize_query_tokens.py
#
# checkpoint 必须对应能返回 aux["attn"] 的 head，例如：
#   GeoTokenV2CAGDivHead / MSGE_GeoTokenV2CAGDivHead
USER_CONFIG = {
    "checkpoint": "checkpoints/VMamba-Tiny-MSGEBlock+GeoTokenV2CAGDivHead0.5-120-bts16-sp2_blr0.0009_hlr0.003_blr0.0009_hlr0.003/net_119.pth",
    "satellite_dir": "/home/cjr/GIT_REPO/Compare_Trial/Dataset/DenseUAV-DSS/test/gallery_satellite",
    "selected_images": None,
    "num_samples": 2,
    "output": "query_token_attention.png",
    "device": "cuda",
    "seed": 666,
    "image_size": None,
    "dpi": 300,
    "colormap": "jet",
    "alpha": 0.5,
    "save_pdf": False,
}
# ==============================================================


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ATTN_ERROR = (
    'This checkpoint/head does not return query attention. Please use a DivHead '
    'or enable attention output for visualization.'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize GeoToken-V2 query token attention maps."
    )
    parser.add_argument("--checkpoint", default=USER_CONFIG["checkpoint"])
    parser.add_argument("--satellite_dir", default=USER_CONFIG["satellite_dir"])
    parser.add_argument(
        "--selected_images",
        nargs="+",
        default=USER_CONFIG["selected_images"],
        help="Optional satellite image paths, file names, or stems.",
    )
    parser.add_argument("--num_samples", default=USER_CONFIG["num_samples"], type=int)
    parser.add_argument("--output", default=USER_CONFIG["output"])
    parser.add_argument("--device", default=USER_CONFIG["device"])
    parser.add_argument("--seed", default=USER_CONFIG["seed"], type=int)
    parser.add_argument("--image_size", default=USER_CONFIG["image_size"], type=int)
    parser.add_argument("--dpi", default=USER_CONFIG["dpi"], type=int)
    parser.add_argument("--colormap", default=USER_CONFIG["colormap"])
    parser.add_argument("--alpha", default=USER_CONFIG["alpha"], type=float)
    parser.add_argument(
        "--save_pdf",
        action="store_true",
        default=USER_CONFIG["save_pdf"],
        help="Also save a PDF next to the PNG output.",
    )
    return parser.parse_args()


def validate_args(args):
    missing = []
    for field in ["checkpoint", "satellite_dir"]:
        value = getattr(args, field)
        if not value or str(value).startswith("/path/to/"):
            missing.append(field)
    if missing:
        raise ValueError(
            "请先在 tools/visualize_query_tokens.py 顶部 USER_CONFIG 中填写这些配置："
            + ", ".join(missing)
        )
    if not os.path.isfile(resolve_path(args.checkpoint)):
        raise FileNotFoundError("Checkpoint not found: {}".format(resolve_path(args.checkpoint)))
    if not os.path.isdir(resolve_path(args.satellite_dir)):
        raise FileNotFoundError("Satellite dir not found: {}".format(resolve_path(args.satellite_dir)))
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load_opt_from_checkpoint(checkpoint_path, args):
    ckpt_path = resolve_path(checkpoint_path)
    opt_path = os.path.join(os.path.dirname(ckpt_path), "opts.yaml")
    if not os.path.isfile(opt_path):
        raise FileNotFoundError("opts.yaml not found next to checkpoint: {}".format(opt_path))

    with open(opt_path, "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) or {}
    opt = SimpleNamespace(**config)
    opt.checkpoint = ckpt_path
    if not hasattr(opt, "load_from") or opt.load_from is None:
        opt.load_from = ""
    if not hasattr(opt, "h"):
        opt.h = 256
    if not hasattr(opt, "w"):
        opt.w = 256
    if args.image_size is not None:
        opt.h = args.image_size
        opt.w = args.image_size
    return opt


def load_network_for_visualization(opt):
    try:
        return load_network(opt)
    except RuntimeError as exc:
        message = str(exc)
        if "Missing key(s) in state_dict" not in message:
            raise
        print("Strict checkpoint loading failed; retrying with strict=False for visualization.")
        print(message)
        model = make_model(opt)
        checkpoint = torch.load(opt.checkpoint, map_location="cpu")
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print("Missing keys kept as initialized:", list(missing_keys))
        if unexpected_keys:
            print("Unexpected keys ignored:", list(unexpected_keys))
        return model


def build_transform(height, width):
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def collect_images(root):
    root = resolve_path(root)
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, filename))
    paths.sort()
    return paths


def resolve_selected_images(selected, all_paths):
    if not selected:
        return None
    abs_to_path = {os.path.abspath(path): path for path in all_paths}
    name_to_paths = {}
    stem_to_paths = {}
    for path in all_paths:
        name_to_paths.setdefault(os.path.basename(path), []).append(path)
        stem_to_paths.setdefault(os.path.splitext(os.path.basename(path))[0], []).append(path)

    chosen = []
    used = set()
    for item in selected:
        candidates = []
        abs_item = os.path.abspath(resolve_path(item))
        if abs_item in abs_to_path:
            candidates = [abs_to_path[abs_item]]
        elif item in name_to_paths:
            candidates = name_to_paths[item]
        elif item in stem_to_paths:
            candidates = stem_to_paths[item]
        if not candidates:
            raise ValueError("Could not find selected satellite image: {}".format(item))
        path = next((candidate for candidate in candidates if candidate not in used), candidates[0])
        chosen.append(path)
        used.add(path)
    return chosen


def choose_images(args):
    all_paths = collect_images(args.satellite_dir)
    if not all_paths:
        raise RuntimeError("No satellite images found in {}".format(resolve_path(args.satellite_dir)))
    selected = resolve_selected_images(args.selected_images, all_paths)
    if selected is not None:
        return selected[: args.num_samples]

    rng = random.Random(args.seed)
    paths = list(all_paths)
    rng.shuffle(paths)
    return paths[: args.num_samples]


def run_satellite_forward(model, image_tensor):
    _, satellite_outputs = model(None, image_tensor)
    return satellite_outputs


def get_attention_from_outputs(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
        raise RuntimeError(ATTN_ERROR)
    aux = outputs[2]
    if not isinstance(aux, dict) or "attn" not in aux:
        raise RuntimeError(ATTN_ERROR)
    attn = aux["attn"]
    if not torch.is_tensor(attn):
        raise RuntimeError('aux["attn"] exists but is not a tensor.')
    if attn.ndim != 3:
        raise RuntimeError(
            'Expected aux["attn"] shape [B, K, H*W], got {}'.format(tuple(attn.shape))
        )
    if attn.shape[0] != 1:
        raise RuntimeError("This visualization expects batch size 1, got {}".format(attn.shape[0]))
    if attn.shape[1] != 8:
        print("Warning: expected 8 query tokens, got {}. Visualizing actual K.".format(attn.shape[1]))
    return attn.detach().float().cpu()


def infer_attention_side(hw):
    side = int(math.sqrt(hw))
    if side * side != hw:
        raise RuntimeError(
            "Cannot reshape attention with H*W={}; it is not a perfect square, "
            "and this script could not infer feature-map H/W.".format(hw)
        )
    return side


def normalize_map(attn_map):
    min_v = float(attn_map.min())
    max_v = float(attn_map.max())
    return (attn_map - min_v) / (max_v - min_v + 1e-12)


def apply_colormap(attn_map, colormap):
    cmap = cm.get_cmap(colormap)
    colored = cmap(attn_map)[:, :, :3]
    return (colored * 255.0).astype(np.uint8)


def make_overlay(rgb_image, attn_vector, colormap, alpha, display_size):
    hw = int(attn_vector.numel())
    side = infer_attention_side(hw)
    attn_map = attn_vector.view(1, 1, side, side)
    attn_map = F.interpolate(
        attn_map, size=(display_size, display_size), mode="bilinear", align_corners=False
    )
    attn_map = normalize_map(attn_map[0, 0].numpy())
    heatmap = apply_colormap(attn_map, colormap)

    base = resize_square(rgb_image, display_size)
    base_arr = np.asarray(base).astype(np.float32)
    overlay = base_arr * (1.0 - alpha) + heatmap.astype(np.float32) * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def pil_resize_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def resize_square(image, size):
    return image.resize((size, size), pil_resize_filter())


def find_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_center_text(draw, xy, text, font, fill=(20, 20, 20)):
    x, y, w, h = xy
    tw, th = text_size(draw, text, font)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=font, fill=fill)


def prepare_sample_visuals(model, transform, image_path, args, device, display_size):
    original = Image.open(image_path).convert("RGB")
    image_tensor = transform(original).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = run_satellite_forward(model, image_tensor)
    attn = get_attention_from_outputs(outputs)[0]

    visuals = [resize_square(original, display_size)]
    for query_idx in range(attn.shape[0]):
        visuals.append(
            make_overlay(original, attn[query_idx], args.colormap, args.alpha, display_size)
        )
    return visuals, attn.shape[0]


def save_visualization(samples, image_paths, args):
    display_size = 150
    tile_gap = 16
    sample_gap = 34
    label_w = 95
    title_h = 54
    row_label_h = 24
    row_h = row_label_h + display_size
    margin_l = 28
    margin_r = 30
    margin_b = 28
    cols = 5
    max_tiles = max(len(visuals) for visuals in samples)
    rows_per_sample = int(math.ceil(max_tiles / float(cols)))

    width = margin_l + label_w + cols * display_size + (cols - 1) * tile_gap + margin_r
    sample_h = rows_per_sample * row_h + (rows_per_sample - 1) * 10
    height = title_h + len(samples) * sample_h + (len(samples) - 1) * sample_gap + margin_b

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(22, bold=True)
    label_font = find_font(17, bold=True)
    small_font = find_font(15, bold=False)

    draw_center_text(
        draw,
        (0, 14, width, 28),
        "GeoToken-V2 Query Token Attention Visualization",
        title_font,
    )

    for sample_idx, visuals in enumerate(samples):
        sample_top = title_h + sample_idx * (sample_h + sample_gap)
        draw.text(
            (margin_l, sample_top + row_label_h + display_size - 12),
            "Sample {}".format(sample_idx + 1),
            font=label_font,
            fill=(20, 20, 20),
            anchor="lm",
        )

        x0 = margin_l + label_w
        for tile_idx, visual in enumerate(visuals):
            row = tile_idx // cols
            col = tile_idx % cols
            x = x0 + col * (display_size + tile_gap)
            y = sample_top + row * (row_h + 10)
            header = "Original" if tile_idx == 0 else "Q{}".format(tile_idx)
            draw_center_text(draw, (x, y, display_size, row_label_h), header, small_font)
            canvas.paste(visual, (x, y + row_label_h))

    output = resolve_path(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    canvas.save(output, dpi=(args.dpi, args.dpi))
    if args.save_pdf:
        pdf_output = os.path.splitext(output)[0] + ".pdf"
        canvas.save(pdf_output, "PDF", resolution=args.dpi)
    return output


def main():
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    opt = load_opt_from_checkpoint(args.checkpoint, args)
    print("Using checkpoint: {}".format(resolve_path(args.checkpoint)))
    print("Using head: {}".format(getattr(opt, "head", "unknown")))
    print("Input image size: {}x{}".format(opt.h, opt.w))

    model = load_network_for_visualization(opt)
    model.eval()
    model.to(device)

    transform = build_transform(opt.h, opt.w)
    image_paths = choose_images(args)
    print("Selected satellite images:")
    for path in image_paths:
        print("  {}".format(path))

    display_size = 150
    samples = []
    for path in image_paths:
        visuals, num_queries = prepare_sample_visuals(
            model, transform, path, args, device, display_size
        )
        samples.append(visuals)

    output = save_visualization(samples, image_paths, args)
    print("Saved visualization to {}".format(output))


if __name__ == "__main__":
    main()
