# -*- coding: utf-8 -*-
"""Visualize Top-K satellite retrieval results for three DenseUAV models.

The ranking logic follows test.py/evaluate_gpu.py:
- query_drone features use the drone branch, gallery_satellite features use
  the satellite branch;
- original and horizontally flipped images are forwarded and summed;
- features are L2-normalized, with the same [B, D, parts] handling as test.py;
- a satellite result is correct when gallery_label == query_label, matching
  evaluate_gpu.py's good_index definition.
"""

from __future__ import print_function

import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tool.utils import load_network  # noqa: E402
from models.taskflow import make_model  # noqa: E402


# ========================= 用户配置区 =========================
# 平时只需要修改这里，然后运行：
#   python tools/visualize_top5_compare.py
#
# 路径可以写相对当前项目根目录的路径，也可以写绝对路径。
# selected_queries:
#   - None 表示自动选择 query；
#   - 也可以写成 ["0001.jpg", "0007.jpg", "0012.jpg", "0030.jpg"]；
#   - 支持 query 图片路径、文件名、文件 stem，或数字 label。
USER_CONFIG = {
    "model_a_checkpoint": "checkpoints/VMamba-Tiny-MSGEBlock+GeoTokenV2CAGDivHead0.5-120-bts16-sp2_blr0.0009_hlr0.003_blr0.0009_hlr0.003/net_119.pth",
    "model_b_checkpoint": "checkpoints/ViTS-224-FSRA/net_119.pth",
    "model_c_checkpoint": "checkpoints/vmamba_tiny_singlebranchcnn/net_119.pth",
    "model_a_name": "Ours",
    "model_b_name": "FSRA",
    "model_c_name": "VMamba",
    "query_dir": "/home/cjr/GIT_REPO/Compare_Trial/Dataset/DenseUAV-DSS/test/query_drone",
    "gallery_dir": "/home/cjr/GIT_REPO/Compare_Trial/Dataset/DenseUAV-DSS/test/gallery_satellite",
    "num_queries": 4,
    "topk": 5,
    "output": "retrieval_top5_compare.png",
    "device": "cuda",
    "seed": 666,
    "dpi": 300,
    "batchsize": None,
    "num_worker": 4,
    "candidate_queries": 64,
    "selected_queries": None,
    "save_pdf": False,
}
# ==============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a paper-style Top-5 retrieval comparison figure."
    )
    parser.add_argument("--model_a_checkpoint", default=USER_CONFIG["model_a_checkpoint"])
    parser.add_argument("--model_b_checkpoint", default=USER_CONFIG["model_b_checkpoint"])
    parser.add_argument("--model_c_checkpoint", default=USER_CONFIG["model_c_checkpoint"])
    parser.add_argument("--model_a_name", default=USER_CONFIG["model_a_name"])
    parser.add_argument("--model_b_name", default=USER_CONFIG["model_b_name"])
    parser.add_argument("--model_c_name", default=USER_CONFIG["model_c_name"])
    parser.add_argument(
        "--query_dir",
        default=USER_CONFIG["query_dir"],
        help="ImageFolder-style UAV query dir.",
    )
    parser.add_argument(
        "--gallery_dir",
        default=USER_CONFIG["gallery_dir"],
        help="ImageFolder-style satellite gallery dir.",
    )
    parser.add_argument("--num_queries", default=USER_CONFIG["num_queries"], type=int)
    parser.add_argument("--topk", default=USER_CONFIG["topk"], type=int)
    parser.add_argument("--output", default=USER_CONFIG["output"])
    parser.add_argument("--device", default=USER_CONFIG["device"])
    parser.add_argument("--seed", default=USER_CONFIG["seed"], type=int)
    parser.add_argument("--dpi", default=USER_CONFIG["dpi"], type=int)
    parser.add_argument("--batchsize", default=USER_CONFIG["batchsize"], type=int)
    parser.add_argument("--num_worker", default=USER_CONFIG["num_worker"], type=int)
    parser.add_argument(
        "--candidate_queries",
        default=USER_CONFIG["candidate_queries"],
        type=int,
        help="Number of random candidates evaluated during automatic query selection.",
    )
    parser.add_argument(
        "--selected_queries",
        nargs="+",
        default=USER_CONFIG["selected_queries"],
        help="Optional query paths, file names, stems, or numeric labels.",
    )
    parser.add_argument(
        "--save_pdf",
        action="store_true",
        default=USER_CONFIG["save_pdf"],
        help="Also save a PDF next to the PNG output.",
    )
    return parser.parse_args()


def validate_args(args):
    path_fields = [
        "model_a_checkpoint",
        "model_b_checkpoint",
        "model_c_checkpoint",
        "query_dir",
        "gallery_dir",
    ]
    missing = []
    for field in path_fields:
        value = getattr(args, field)
        if not value or str(value).startswith("/path/to/"):
            missing.append(field)
    if missing:
        raise ValueError(
            "请先在 tools/visualize_top5_compare.py 顶部 USER_CONFIG 中填写这些配置："
            + ", ".join(missing)
        )


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load_opt_from_checkpoint(checkpoint_path, fallback_args):
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
    if not hasattr(opt, "block"):
        opt.block = 1
    if not hasattr(opt, "h"):
        opt.h = 256
    if not hasattr(opt, "w"):
        opt.w = 256
    if fallback_args.batchsize is not None:
        opt.batchsize = fallback_args.batchsize
    elif not hasattr(opt, "batchsize"):
        opt.batchsize = 128
    opt.num_worker = fallback_args.num_worker
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


def pil_resize_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def build_dataset(root, height, width):
    transform = transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return datasets.ImageFolder(resolve_path(root), transform)


def get_labels_paths(image_folder):
    labels = []
    paths = []
    for path, _ in image_folder.imgs:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return np.asarray(labels), paths


def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1, device=img.device).long()
    return img.index_select(3, inv_idx)


def branch_forward(model, input_img, view_index):
    if view_index == 1:
        outputs, _ = model(input_img, None)
    elif view_index == 3:
        _, outputs = model(None, input_img)
    else:
        raise ValueError("Unsupported view index: {}".format(view_index))
    return outputs[1]


def extract_feature(model, dataloader, view_index, opt, device):
    features = torch.FloatTensor()
    for img, _ in tqdm(dataloader, desc="extract view {}".format(view_index), leave=False):
        img = img.to(device)
        ff = None
        for i in range(2):
            input_img = fliplr(img) if i == 1 else img
            outputs = branch_forward(model, Variable(input_img), view_index)
            ff = outputs if ff is None else ff + outputs

        if len(ff.shape) == 3:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.detach().cpu()), 0)
    return features


def make_loader(dataset, batch_size, num_worker):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker
    )


def load_model_and_features(model_spec, args, device):
    opt = load_opt_from_checkpoint(model_spec["checkpoint"], args)
    query_dataset = build_dataset(args.query_dir, opt.h, opt.w)
    gallery_dataset = build_dataset(args.gallery_dir, opt.h, opt.w)
    query_loader = make_loader(query_dataset, opt.batchsize, opt.num_worker)
    gallery_loader = make_loader(gallery_dataset, opt.batchsize, opt.num_worker)

    model = load_network_for_visualization(opt)
    model.eval()
    model.to(device)

    with torch.no_grad():
        query_feature = extract_feature(model, query_loader, 3, opt, device)
        gallery_feature = extract_feature(model, gallery_loader, 1, opt, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    query_label, query_path = get_labels_paths(query_dataset)
    gallery_label, gallery_path = get_labels_paths(gallery_dataset)
    return {
        "name": model_spec["name"],
        "query_feature": query_feature,
        "gallery_feature": gallery_feature,
        "query_label": query_label,
        "gallery_label": gallery_label,
        "query_path": query_path,
        "gallery_path": gallery_path,
    }


def topk_indices(query_feature, gallery_feature, query_index, topk):
    query = query_feature[query_index].view(-1, 1)
    score = torch.mm(gallery_feature, query).squeeze(1).numpy()
    return np.argsort(score)[::-1][:topk]


def correct_count(result, query_index, indices):
    query_label = result["query_label"][query_index]
    gallery_labels = result["gallery_label"][indices]
    return int(np.sum(gallery_labels == query_label))


def resolve_selected_queries(selected, query_paths, query_labels):
    selected_indices = []
    used = set()
    abs_to_index = {os.path.abspath(path): i for i, path in enumerate(query_paths)}
    name_to_indices = {}
    stem_to_indices = {}
    label_to_indices = {}
    for i, path in enumerate(query_paths):
        name_to_indices.setdefault(os.path.basename(path), []).append(i)
        stem_to_indices.setdefault(os.path.splitext(os.path.basename(path))[0], []).append(i)
        label_to_indices.setdefault(str(int(query_labels[i])), []).append(i)

    for item in selected:
        candidates = []
        abs_item = os.path.abspath(resolve_path(item))
        if abs_item in abs_to_index:
            candidates = [abs_to_index[abs_item]]
        elif item in name_to_indices:
            candidates = name_to_indices[item]
        elif item in stem_to_indices:
            candidates = stem_to_indices[item]
        elif item in label_to_indices:
            candidates = label_to_indices[item]
        if not candidates:
            raise ValueError("Could not find selected query: {}".format(item))
        chosen = next((idx for idx in candidates if idx not in used), candidates[0])
        selected_indices.append(chosen)
        used.add(chosen)
    return selected_indices


def choose_queries(results, args):
    query_paths = results[0]["query_path"]
    query_labels = results[0]["query_label"]
    if args.selected_queries:
        indices = resolve_selected_queries(args.selected_queries, query_paths, query_labels)
        return indices[: args.num_queries]

    rng = random.Random(args.seed)
    all_indices = list(range(len(query_paths)))
    rng.shuffle(all_indices)
    candidates = all_indices[: min(args.candidate_queries, len(all_indices))]

    ranked = []
    topk_cache = {}
    for qidx in candidates:
        counts = []
        has_diff = False
        row_topks = []
        for result in results:
            idxs = topk_indices(
                result["query_feature"], result["gallery_feature"], qidx, args.topk
            )
            row_topks.append(tuple(idxs.tolist()))
            counts.append(correct_count(result, qidx, idxs))
        has_diff = len(set(row_topks)) > 1
        score = (
            counts[0],
            int(counts[2] < args.topk),
            int(counts[1] != counts[0] or counts[2] != counts[0]),
            int(has_diff),
            -qidx,
        )
        ranked.append((score, qidx))
        topk_cache[qidx] = row_topks

    ranked.sort(reverse=True)
    chosen = [qidx for _, qidx in ranked[: args.num_queries]]
    if len(chosen) < args.num_queries:
        for qidx in all_indices:
            if qidx not in chosen:
                chosen.append(qidx)
            if len(chosen) == args.num_queries:
                break
    return chosen


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


def fit_image(path, size):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), pil_resize_filter())
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_center_text(draw, xy, text, font, fill=(20, 20, 20)):
    x, y, w, h = xy
    tw, th = text_size(draw, text, font)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=font, fill=fill)


def save_visualization(results, selected_indices, args):
    topk = args.topk
    num_queries = len(selected_indices)

    query_size = 110
    sat_size = 90
    row_h = 102
    block_gap = 28
    title_h = 76
    margin_l = 36
    margin_r = 34
    query_x = margin_l
    divider_x = query_x + query_size + 36
    sat_x = divider_x + 34
    sat_gap = 14
    model_x = sat_x + topk * sat_size + (topk - 1) * sat_gap + 32
    model_w = 170
    width = model_x + model_w + margin_r
    height = title_h + num_queries * (3 * row_h) + (num_queries - 1) * block_gap + 28

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = find_font(22, bold=True)
    rank_font = find_font(16, bold=False)
    model_font = find_font(18, bold=True)

    draw_center_text(draw, (query_x, 18, query_size, 28), "UAV查询影像", title_font)
    sat_area_w = topk * sat_size + (topk - 1) * sat_gap
    draw_center_text(
        draw,
        (sat_x, 18, sat_area_w, 28),
        "Top-{}卫星影像检索结果".format(topk),
        title_font,
    )
    for k in range(topk):
        x = sat_x + k * (sat_size + sat_gap)
        draw_center_text(draw, (x, 49, sat_size, 22), "Top-{}".format(k + 1), rank_font)

    content_top = title_h
    content_bottom = height - 20
    draw.line((divider_x, content_top - 8, divider_x, content_bottom), fill=(80, 80, 80), width=2)

    green = (28, 150, 72)
    red = (210, 55, 45)
    for block_i, qidx in enumerate(selected_indices):
        block_top = title_h + block_i * (3 * row_h + block_gap)
        query_y = block_top + (3 * row_h - query_size) // 2
        q_img = fit_image(results[0]["query_path"][qidx], query_size)
        image.paste(q_img, (query_x, query_y))

        for model_i, result in enumerate(results):
            row_top = block_top + model_i * row_h
            sat_y = row_top + (row_h - sat_size) // 2
            idxs = topk_indices(result["query_feature"], result["gallery_feature"], qidx, topk)
            query_label = result["query_label"][qidx]
            for rank, gidx in enumerate(idxs):
                sat_img = fit_image(result["gallery_path"][gidx], sat_size)
                sat_img_x = sat_x + rank * (sat_size + sat_gap)
                image.paste(sat_img, (sat_img_x, sat_y))
                is_correct = result["gallery_label"][gidx] == query_label
                color = green if is_correct else red
                for offset in range(3):
                    rect = (
                        sat_img_x - offset,
                        sat_y - offset,
                        sat_img_x + sat_size - 1 + offset,
                        sat_y + sat_size - 1 + offset,
                    )
                    draw.rectangle(rect, outline=color)
            draw_center_text(
                draw,
                (model_x, row_top + (row_h - 30) // 2, model_w, 30),
                result["name"],
                model_font,
            )

    output = resolve_path(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    image.save(output, dpi=(args.dpi, args.dpi))
    if args.save_pdf:
        pdf_output = os.path.splitext(output)[0] + ".pdf"
        image.save(pdf_output, "PDF", resolution=args.dpi)
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

    model_specs = [
        {"checkpoint": args.model_a_checkpoint, "name": args.model_a_name},
        {"checkpoint": args.model_b_checkpoint, "name": args.model_b_name},
        {"checkpoint": args.model_c_checkpoint, "name": args.model_c_name},
    ]

    results = []
    for spec in model_specs:
        print("Loading and extracting features for {}...".format(spec["name"]))
        results.append(load_model_and_features(spec, args, device))

    selected_indices = choose_queries(results, args)
    if len(selected_indices) == 0:
        raise RuntimeError("No query image was selected.")

    print("Selected queries:")
    for idx in selected_indices:
        print("  label={} path={}".format(results[0]["query_label"][idx], results[0]["query_path"][idx]))

    output = save_visualization(results, selected_indices, args)
    print("Saved visualization to {}".format(output))


if __name__ == "__main__":
    main()
