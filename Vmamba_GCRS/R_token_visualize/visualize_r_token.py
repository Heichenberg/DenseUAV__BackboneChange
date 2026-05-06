#!/usr/bin/env python3
"""
This script explicitly loads the trained VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE model.
Images are randomly sampled from the DenseUAV dataset.
R token is not attention; it is a masked pooling token. The visualization shows its effective spatial region, feature response, and gated contribution.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import datasets, transforms

# Ensure repository root is on PYTHONPATH so `from models...` works
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.taskflow import make_model
from models.Head.GeoTokenHead import GeoTokenHeadV1


def parse_args():
    parser = argparse.ArgumentParser(description="R token region visualization for VMamba-DenseUAV")
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--checkpoint", default="latest_checkpoint.pth", type=str)
    parser.add_argument("--test_dir", required=True, type=str)
    parser.add_argument("--view", default="both", choices=["query_drone", "gallery_satellite", "both"])
    parser.add_argument("--num_samples", default=8, type=int)
    parser.add_argument("--output_dir", default="Vmamba_GCRS/R_token_visualize/outputs", type=str)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    parser.add_argument("--show_gate", type=int, default=1, help="1 to show context_gate in title, 0 to hide")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(gpu_ids: str):
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        return torch.device("cpu")
    ids = [int(x) for x in gpu_ids.split(",") if x.strip()]
    ids = [x for x in ids if x >= 0]
    if not ids:
        return torch.device("cpu")
    torch.cuda.set_device(ids[0])
    return torch.device(f"cuda:{ids[0]}")


def load_opt_from_yaml(checkpoint_dir: Path):
    opts_path = checkpoint_dir / "opts.yaml"
    if not opts_path.exists():
        raise FileNotFoundError(f"opts.yaml not found: {opts_path}")
    with open(opts_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_model(checkpoint_dir: Path, checkpoint_name: str, device):
    cfg = load_opt_from_yaml(checkpoint_dir)
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    cfg["load_from"] = ""
    cfg["use_gpu"] = torch.cuda.is_available() and str(device).startswith("cuda")
    opt = SimpleNamespace(**cfg)

    model = make_model(opt)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, opt, checkpoint_path


def find_geo_head(model):
    model_ref = model.module if hasattr(model, "module") else model
    for m in model_ref.modules():
        if isinstance(m, GeoTokenHeadV1):
            return m
    return None


def validate_gcr5_d192_gate(model, opt):
    alias_ok = getattr(opt, "backbone", "") == "VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE"
    geo_head = find_geo_head(model)
    if geo_head is None:
        raise RuntimeError("GeoTokenHeadV1 not found in model; cannot perform R token region visualization.")

    checks = [
        (geo_head.context_size == 5, f"context_size expected 5, got {geo_head.context_size}"),
        (geo_head.context_dim == 192, f"context_dim expected 192, got {geo_head.context_dim}"),
        (geo_head.context_gate_logit is not None, "context_gate_logit is None; expected gated context token."),
        ("context" in geo_head.active_tokens, f"active_tokens must include context, got {geo_head.active_tokens}"),
    ]

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        detail = "\n".join([
            f"backbone alias in opts.yaml: {getattr(opt, 'backbone', '<missing>')}",
            f"active_tokens: {geo_head.active_tokens}",
            f"context_size: {geo_head.context_size}",
            f"context_dim: {geo_head.context_dim}",
            f"context_gate_logit_exists: {geo_head.context_gate_logit is not None}",
        ] + failed)
        raise RuntimeError(
            "This script must run with VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE (or exact equivalent head config).\n"
            + detail
        )
    if not alias_ok:
        print(
            "[Warning] opts.yaml backbone alias is not VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE, "
            "but head configuration matches the required equivalent setup. Continue."
        )
    return geo_head


def make_datasets(test_dir, h, w):
    test_dir = Path(test_dir)
    data_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    raw_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
    ])

    datasets_map = {}
    for v in ["query_drone", "gallery_satellite"]:
        root = test_dir / v
        if not root.exists():
            raise FileNotFoundError(f"view directory not found: {root}")
        datasets_map[v] = datasets.ImageFolder(str(root), data_transform)

    return datasets_map, raw_transform


def sample_items(dataset, num_samples, seed):
    total = len(dataset.samples)
    if total == 0:
        return []
    n = min(num_samples, total)
    rng = random.Random(seed)
    return rng.sample(range(total), n)


def to_numpy_img(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def upsample_map(map_2d, target_h, target_w):
    t = torch.from_numpy(map_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze().cpu().numpy()


def draw_and_save(
    out_path,
    orig_img,
    center_overlay,
    context_overlay,
    r_heat_overlay,
    gate_value,
    view_name,
    class_id,
    sample_idx,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(orig_img)
    axes[1].imshow(center_overlay, cmap="Blues", alpha=0.40)
    axes[1].set_title("Center 3x3 region")
    axes[1].axis("off")

    axes[2].imshow(orig_img)
    axes[2].imshow(context_overlay, cmap="Oranges", alpha=0.45)
    axes[2].set_title("R token effective region")
    axes[2].axis("off")

    axes[3].imshow(orig_img)
    axes[3].imshow(r_heat_overlay, cmap="jet", alpha=0.50)
    title = "R token feature-norm response"
    if gate_value is not None:
        title += f"\nGated R contribution (context_gate={gate_value:.4f})"
    axes[3].set_title(title)
    axes[3].axis("off")

    fig.suptitle(
        f"R token region visualization | view={view_name} | id={class_id:04d} | idx={sample_idx:03d}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_single_sample(model, head, tensor_img, view_name, device):
    x = tensor_img.unsqueeze(0).to(device)
    with torch.no_grad():
        if view_name == "query_drone":
            _drone_res, _sat_res = model(None, x)
        elif view_name == "gallery_satellite":
            _drone_res, _sat_res = model(x, None)
        else:
            raise ValueError(f"Unsupported view {view_name}")

    required = ["last_x_proj", "last_center_mask", "last_context_mask", "last_structure_attn"]
    for key in required:
        if not hasattr(head, key):
            raise RuntimeError(
                f"GeoTokenHeadV1 has no '{key}'. Please add debug cache in GeoTokenHead.py first."
            )

    if head.last_x_proj is None or head.last_context_mask is None:
        raise RuntimeError("Debug cache is empty after forward; check head forward/cache logic.")

    x_proj = head.last_x_proj[0].detach().cpu()          # [C,H,W]
    center_mask = head.last_center_mask[0, 0].detach().cpu()   # [H,W]
    context_mask = head.last_context_mask[0, 0].detach().cpu() # [H,W]
    gate_value = head.get_context_gate_value()

    feature_norm = torch.norm(x_proj, p=2, dim=0)        # [H,W]
    r_feature_norm = feature_norm * context_mask          # [H,W]

    return center_mask.numpy(), context_mask.numpy(), r_feature_norm.numpy(), tuple(x_proj.shape[1:])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.gpu_ids)

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, opt, checkpoint_path = load_model(checkpoint_dir, args.checkpoint, device)
    geo_head = validate_gcr5_d192_gate(model, opt)

    datasets_map, raw_transform = make_datasets(args.test_dir, opt.h, opt.w)
    views = ["query_drone", "gallery_satellite"] if args.view == "both" else [args.view]

    summary_lines = [
        f"checkpoint_dir: {checkpoint_dir}",
        f"checkpoint: {checkpoint_path}",
        f"test_dir: {args.test_dir}",
        f"view: {args.view}",
        f"num_samples: {args.num_samples}",
        f"context_size: {geo_head.context_size}",
        f"context_dim: {geo_head.context_dim}",
        f"context_gate: {geo_head.context_gate_logit is not None}",
        f"active_tokens: {geo_head.active_tokens}",
    ]

    global_counter = 0
    for view_name in views:
        dataset = datasets_map[view_name]
        chosen = sample_items(dataset, args.num_samples, args.seed + (0 if view_name == "query_drone" else 1000))

        for i, sample_idx in enumerate(chosen):
            img_path, class_id = dataset.samples[sample_idx]
            pil_raw = Image.open(img_path).convert("RGB")
            pil_show = raw_transform(pil_raw)
            orig_img = to_numpy_img(pil_show)

            tensor_img, _ = dataset[sample_idx]
            center_mask_2d, context_mask_2d, r_feature_norm_2d, feat_hw = run_single_sample(
                model, geo_head, tensor_img, view_name, device
            )

            h_img, w_img = orig_img.shape[:2]
            center_overlay = upsample_map(center_mask_2d, h_img, w_img)
            context_overlay = upsample_map(context_mask_2d, h_img, w_img)
            r_heat_overlay = upsample_map(r_feature_norm_2d, h_img, w_img)

            max_v = float(r_heat_overlay.max())
            if max_v > 1e-12:
                r_heat_overlay = r_heat_overlay / max_v

            gate_value = geo_head.get_context_gate_value() if int(args.show_gate) == 1 else None

            filename = f"{view_name}_id{class_id:04d}_idx{sample_idx:03d}_R_token_region.png"
            out_path = output_dir / filename
            draw_and_save(
                out_path,
                orig_img,
                center_overlay,
                context_overlay,
                r_heat_overlay,
                gate_value,
                view_name,
                class_id,
                sample_idx,
            )

            summary_lines.append(
                f"[{global_counter}] view={view_name} class_id={class_id} sample_idx={sample_idx} feature_hw={feat_hw} image_path={img_path} output={out_path}"
            )
            global_counter += 1

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved visualizations to: {output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
