#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.Head.GeoTokenHead import GeoTokenHeadV1
from models.taskflow import make_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize G, C, and GC5 heatmaps for GeoTokenHeadV1 GC5.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--test_dir", default="/home/cjr/GIT_REPO/Dataset/DenseUAV/test")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional class IDs, e.g. 002612 002454")
    parser.add_argument(
        "--from_result_mat",
        action="store_true",
        help="Use pytorch_result_1.mat in checkpoint_dir and pick top1 matched pairs.",
    )
    parser.add_argument("--match_status", default="correct", choices=["correct", "wrong", "any"])
    parser.add_argument("--num_pairs", type=int, default=2)
    parser.add_argument("--drone_image", default="H80.JPG")
    parser.add_argument("--satellite_image", default="H80.tif")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--alpha", type=float, default=0.48)
    return parser.parse_args()


def setup_device(gpu_ids):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    ids = [int(x) for x in gpu_ids.split(",") if x.strip()]
    ids = [x for x in ids if x >= 0]
    if not ids:
        return torch.device("cpu")
    torch.cuda.set_device(ids[0])
    return torch.device("cuda:{}".format(ids[0]))


def load_cfg(checkpoint_dir):
    opts_path = Path(checkpoint_dir) / "opts.yaml"
    if not opts_path.exists():
        raise FileNotFoundError("opts.yaml not found: {}".format(opts_path))
    with opts_path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def normalize_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_model(checkpoint_dir, checkpoint_name, device):
    cfg = load_cfg(checkpoint_dir)
    cfg["load_from"] = ""
    cfg["use_gpu"] = device.type == "cuda"
    opt = SimpleNamespace(**cfg)
    model = make_model(opt)

    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_path))
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = normalize_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint:", checkpoint_path)
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))

    model = model.to(device)
    model.eval()
    return model, opt


def find_geo_head(model):
    model_ref = model.module if hasattr(model, "module") else model
    for module in model_ref.modules():
        if isinstance(module, GeoTokenHeadV1):
            return module
    raise RuntimeError("GeoTokenHeadV1 not found in model.")


def choose_pairs(test_dir, ids, num_pairs, drone_image, satellite_image, seed):
    test_dir = Path(test_dir)
    drone_root = test_dir / "query_drone"
    sat_root = test_dir / "gallery_satellite"
    if ids:
        candidates = [str(x).zfill(6) for x in ids]
    else:
        drone_ids = {p.name for p in drone_root.iterdir() if p.is_dir()}
        sat_ids = {p.name for p in sat_root.iterdir() if p.is_dir()}
        candidates = sorted(drone_ids & sat_ids)
        random.Random(seed).shuffle(candidates)

    pairs = []
    for class_id in candidates:
        drone_path = drone_root / class_id / drone_image
        sat_path = sat_root / class_id / satellite_image
        if drone_path.exists() and sat_path.exists():
            pairs.append((class_id, drone_path, sat_path))
        if len(pairs) >= num_pairs:
            break
    if len(pairs) < num_pairs:
        raise RuntimeError("Only found {} valid pairs; need {}.".format(len(pairs), num_pairs))
    return pairs


def choose_pairs_from_result_mat(checkpoint_dir, num_pairs, match_status):
    mat_path = Path(checkpoint_dir) / "pytorch_result_1.mat"
    if not mat_path.exists():
        raise FileNotFoundError("pytorch_result_1.mat not found: {}".format(mat_path))

    import scipy.io

    result = scipy.io.loadmat(str(mat_path))
    query_feature = torch.from_numpy(result["query_f"]).float()
    gallery_feature = torch.from_numpy(result["gallery_f"]).float()
    query_label = result["query_label"][0]
    gallery_label = result["gallery_label"][0]
    query_path = [str(x).strip() for x in result["query_path"].reshape(-1)]
    gallery_path = [str(x).strip() for x in result["gallery_path"].reshape(-1)]

    top1 = torch.mm(query_feature, gallery_feature.t()).argmax(dim=1).cpu().numpy()
    top1_correct = gallery_label[top1] == query_label

    pairs = []
    used_ids = set()
    for idx, gallery_idx in enumerate(top1):
        is_correct = bool(top1_correct[idx])
        if match_status == "correct" and not is_correct:
            continue
        if match_status == "wrong" and is_correct:
            continue
        class_id = str(int(query_label[idx])).zfill(6)
        if class_id in used_ids:
            continue
        used_ids.add(class_id)
        pairs.append((class_id, Path(query_path[idx]), Path(gallery_path[gallery_idx])))
        if len(pairs) >= num_pairs:
            break

    if len(pairs) < num_pairs:
        raise RuntimeError(
            "Only found {} '{}' top1 pairs; need {}.".format(len(pairs), match_status, num_pairs)
        )
    return pairs


def image_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def resize_raw_image(path, h, w):
    return Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)


def normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom > 1e-8:
        arr = arr / denom
    return arr


def upsample_map(map_2d, h, w):
    t = torch.from_numpy(map_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def gc5_maps_from_head(head):
    if tuple(head.active_tokens) != ("global", "center"):
        raise RuntimeError("Expected GC5 active_tokens=('global', 'center'), got {}".format(head.active_tokens))
    x_proj = head.last_x_proj[0].detach().float().cpu()      # [C,H,W]
    center_mask = head.last_center_mask[0, 0].detach().float().cpu()

    emb_weight = head.embedding.weight.detach().float().cpu()  # [embedding_dim, 2C]
    channels = x_proj.shape[0]
    global_weight = emb_weight[:, :channels].abs().mean(dim=0).view(channels, 1, 1)
    center_weight = emb_weight[:, channels:channels * 2].abs().mean(dim=0).view(channels, 1, 1)

    # Spatial contribution proxies for the two pooled tokens.
    g_map = (x_proj.abs() * global_weight).sum(dim=0) / float(x_proj.shape[1] * x_proj.shape[2])
    c_map = (x_proj.abs() * center_weight).sum(dim=0) * center_mask / center_mask.sum().clamp(min=1.0)
    gc5_map = g_map + c_map

    return normalize_map(g_map.numpy()), normalize_map(c_map.numpy()), normalize_map(gc5_map.numpy())


def run_image(model, head, image_path, view_name, transform, device, h, w):
    raw = resize_raw_image(image_path, h, w)
    tensor = transform(raw).unsqueeze(0).to(device)
    with torch.no_grad():
        if view_name == "drone":
            model(tensor, None)
        elif view_name == "satellite":
            model(None, tensor)
        else:
            raise ValueError(view_name)
    g_map, c_map, gc5_map = gc5_maps_from_head(head)
    return raw, upsample_map(g_map, h, w), upsample_map(c_map, h, w), upsample_map(gc5_map, h, w)


def draw_pair(out_path, rows, alpha):
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 8))
    headers = ["Original", "G", "C", "GC5"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=13)

    for row_idx, row in enumerate(rows):
        label, raw, g_map, c_map, gc5_map = row
        raw_np = np.asarray(raw)
        maps = [None, g_map, c_map, gc5_map]
        for col in range(4):
            ax = axes[row_idx, col]
            ax.imshow(raw_np)
            if maps[col] is not None:
                ax.imshow(maps[col], cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
            ax.set_ylabel(label if col == 0 else "", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    device = setup_device(args.gpu_ids)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "gc5_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, opt = load_model(checkpoint_dir, args.checkpoint, device)
    head = find_geo_head(model)
    transform = image_transform(opt.h, opt.w)
    if args.from_result_mat:
        pairs = choose_pairs_from_result_mat(checkpoint_dir, args.num_pairs, args.match_status)
    else:
        pairs = choose_pairs(args.test_dir, args.ids, args.num_pairs, args.drone_image, args.satellite_image, args.seed)

    saved = []
    for pair_idx, (class_id, drone_path, sat_path) in enumerate(pairs, start=1):
        drone_row = ("drone {} {}".format(class_id, drone_path.name),) + run_image(
            model, head, drone_path, "drone", transform, device, opt.h, opt.w
        )
        sat_row = ("satellite {} {}".format(class_id, sat_path.name),) + run_image(
            model, head, sat_path, "satellite", transform, device, opt.h, opt.w
        )
        out_path = output_dir / "gc5_pair_{:02d}_{}.png".format(pair_idx, class_id)
        draw_pair(out_path, [drone_row, sat_row], args.alpha)
        saved.append(out_path)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("checkpoint_dir: {}\n".format(checkpoint_dir))
        f.write("checkpoint: {}\n".format(args.checkpoint))
        f.write("test_dir: {}\n".format(args.test_dir))
        f.write("active_tokens: {}\n".format(head.active_tokens))
        f.write("center_size: {}\n".format(head.center_size))
        for path in saved:
            f.write("saved: {}\n".format(path))
    print("Saved:")
    for path in saved:
        print(path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
