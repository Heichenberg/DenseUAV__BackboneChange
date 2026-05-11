#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level response and sensitivity analysis for DenseUAV + VMamba +
GeoTokenHeadV1/GC5R checkpoints.

Run from the repository root:
    python token_analysis.py --checkpoint_dir checkpoints/xxx

Or copy/run it inside one checkpoint experiment directory:
    python token_analysis.py

All outputs are written to:
    checkpoints/xxx/token_analysis/
"""

import argparse
import csv
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch/torchvision is required to run token_analysis.py. "
        "Activate the DenseUAV training environment first. Missing: {}".format(exc)
    )

from models.taskflow import make_model


LOGGER_NAME = "token_analysis"
RELATED_MODULE_KEYWORDS = ("head", "token", "geo", "mixer", "gate", "pool")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SampleRecord:
    image: torch.Tensor
    path: str
    label: str
    sample_id: str


class UAVImageDataset(Dataset):
    """Simple path-preserving UAV image dataset.

    The script analyzes drone/UAV samples. It accepts either a DenseUAV root
    containing query_drone/drone folders or a folder of images.
    """

    def __init__(self, data_root: str, image_size: int, max_samples: Optional[int] = None, explicit_paths: Optional[Sequence[str]] = None):
        self.data_root = str(data_root)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if explicit_paths is None:
            self.samples = self._discover_samples(self.data_root)
        else:
            self.samples = []
            for path in explicit_paths:
                label = self._label_from_path(path)
                self.samples.append((str(path), label))

        self.samples = sorted(self.samples, key=lambda item: (str(item[1]), item[0]))
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError("No UAV images found under {}".format(self.data_root))

    @staticmethod
    def _label_from_path(path: str) -> str:
        parent = os.path.basename(os.path.dirname(path))
        return parent if parent else "unknown"

    @classmethod
    def _discover_samples(cls, root: str) -> List[Tuple[str, str]]:
        root_path = Path(root)
        candidates = [
            root_path / "test" / "query_drone",
            root_path / "query_drone",
            root_path / "train" / "drone",
            root_path / "drone",
        ]
        search_roots = [p for p in candidates if p.is_dir()]
        if not search_roots and root_path.is_dir():
            search_roots = [root_path]

        samples: List[Tuple[str, str]] = []
        for search_root in search_roots:
            for ext in IMAGE_EXTS:
                for path in search_root.rglob("*" + ext):
                    samples.append((str(path), cls._label_from_path(str(path))))
            if samples:
                break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleRecord:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        sample_id = "{:05d}".format(index)
        return SampleRecord(tensor, path, str(label), sample_id)


def collate_records(batch: Sequence[SampleRecord]) -> Dict[str, object]:
    return {
        "image": torch.stack([item.image for item in batch], dim=0),
        "path": [item.path for item in batch],
        "label": [item.label for item in batch],
        "sample_id": [item.sample_id for item in batch],
    }


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(str(output_dir / "token_analysis.log"), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def read_config(checkpoint_dir: Path) -> SimpleNamespace:
    config_names = ("opts.yaml", "config.yaml", "args.yaml")
    data = {}
    for name in config_names:
        path = checkpoint_dir / name
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.load(handle, Loader=yaml.FullLoader)
            if isinstance(loaded, dict):
                data.update(loaded)
            break

    opt_txt = checkpoint_dir / "opt.txt"
    if not data and opt_txt.is_file():
        for line in opt_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

    opt = SimpleNamespace(**data)
    defaults = {
        "checkpoint": "",
        "h": 224,
        "w": 224,
        "droprate": 0.5,
        "nclasses": 1,
        "num_bottleneck": 512,
        "head": "GeoTokenHeadV1",
        "head_pool": "avg",
        "backbone": "VMamba-Tiny-_GeoTokenHeadV1_GC5R_D192",
        "backbone_weight": "",
        "load_from": "no",
        "batchsize": 16,
        "num_worker": 4,
    }
    for key, value in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    return opt


def find_checkpoint(checkpoint_dir: Path) -> Path:
    preferred = ["best_checkpoint.pth", "best.pth", "latest_checkpoint.pth", "last.pth"]
    for name in preferred:
        path = checkpoint_dir / name
        if path.is_file():
            return path

    net_paths = sorted(checkpoint_dir.glob("net_*.pth"))
    if net_paths:
        return net_paths[-1]

    pths = sorted(
        [p for p in checkpoint_dir.glob("*.pth") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pths:
        return pths[0]
    raise FileNotFoundError("No checkpoint .pth file found in {}".format(checkpoint_dir))


def load_model(checkpoint_dir: Path, device: torch.device, logger: logging.Logger):
    opt = read_config(checkpoint_dir)
    checkpoint_path = find_checkpoint(checkpoint_dir)
    opt.checkpoint = str(checkpoint_path)
    if getattr(opt, "backbone_weight", ""):
        logger.info("Skipping pretrained backbone_weight while building analysis model; checkpoint weights will be loaded.")
        opt.backbone_weight = ""
    logger.info("Using checkpoint: {}".format(checkpoint_path))

    model = make_model(opt)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint {}. Check that opts.yaml matches the saved model. {}".format(checkpoint_path, exc)
        )
    if missing:
        logger.warning("Missing keys while loading checkpoint: {}".format(missing[:20]))
    if unexpected:
        logger.warning("Unexpected keys while loading checkpoint: {}".format(unexpected[:20]))
    model.to(device)
    model.eval()
    return model, opt


def infer_data_root(args, opt) -> str:
    if args.data_root:
        return args.data_root
    opt_data_dir = getattr(opt, "data_dir", "")
    if opt_data_dir:
        path = Path(opt_data_dir)
        if path.name == "train":
            candidate = path.parent / "test"
            if candidate.is_dir():
                return str(path.parent)
        return str(path)
    default = Path("../Dataset/DenseUAV").resolve()
    return str(default)


def to_numpy_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if target_size is not None:
        image = image.resize(target_size, Image.BICUBIC)
    return np.asarray(image).astype(np.float32) / 255.0


def normalize_map(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(response.min())
    max_value = float(response.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(response, dtype=np.float32)
    return (response - min_value) / (max_value - min_value)


def center_ratio(response: torch.Tensor) -> float:
    if response.ndim == 3:
        response = response.squeeze(0)
    height, width = response.shape[-2], response.shape[-1]
    ch = max(1, height // 5)
    cw = max(1, width // 5)
    row_start = max(0, (height - ch) // 2)
    col_start = max(0, (width - cw) // 2)
    center = response[..., row_start : row_start + ch, col_start : col_start + cw]
    denom = response.sum().clamp(min=1e-12)
    return float((center.sum() / denom).detach().cpu().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim > 1:
        a = a.flatten()
    if b.ndim > 1:
        b = b.flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).detach().cpu().item())


class TokenAnalyzer:
    def __init__(
        self,
        model,
        dataloader,
        output_dir,
        device,
        adjacent_csv: str = "",
        image_size: int = 224,
        save_raw_maps: bool = False,
        data_root: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.device = device
        self.adjacent_csv = adjacent_csv
        self.image_size = image_size
        self.save_raw_maps = save_raw_maps
        self.data_root = data_root
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.records: List[Dict[str, object]] = []
        self.backbone_features: Optional[torch.Tensor] = None
        self._last_backbone_feature: Optional[torch.Tensor] = None
        self._fallback_warned = False
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if hasattr(self.model, "backbone"):
            self._hooks.append(self.model.backbone.register_forward_hook(self._backbone_hook))
        else:
            self.logger.warning("Model has no .backbone attribute; response fallback may be unavailable.")

    def _backbone_hook(self, _module, _inputs, output):
        if torch.is_tensor(output) and output.ndim == 4:
            self._last_backbone_feature = output.detach()

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _warn_fallback(self):
        if not self._fallback_warned:
            self.logger.warning("[Warning] Using fallback response construction because explicit token outputs were not found.")
            self._fallback_warned = True

    def _head_module(self):
        head = getattr(self.model, "head", None)
        if head is not None and hasattr(head, "head"):
            return head.head
        return head

    def _print_related_modules(self):
        related = []
        for name, module in self.model.named_modules():
            lowered = name.lower()
            cls_name = module.__class__.__name__.lower()
            if any(key in lowered or key in cls_name for key in RELATED_MODULE_KEYWORDS):
                related.append("{}: {}".format(name or "<root>", module.__class__.__name__))
        if related:
            self.logger.info("Related model modules:\n{}".format("\n".join(related[:120])))
        else:
            self.logger.info("No modules matched keywords: {}".format(", ".join(RELATED_MODULE_KEYWORDS)))

    def _extract_geo_tokens(self, output_embedding: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        head = self._head_module()
        feature = getattr(head, "last_x_proj", None)
        if feature is None:
            feature = self._last_backbone_feature
            self._warn_fallback()
        if feature is None or feature.ndim != 4:
            self._print_related_modules()
            raise RuntimeError(
                "Could not capture a [B,C,H,W] feature map. Current model forward does not expose token outputs. "
                "Please return a debug dict from GeoTokenHead forward or check hook module names."
            )

        feature = feature.detach()
        batch, channels, height, width = feature.shape
        abs_response = feature.abs().mean(dim=1)

        center_mask = getattr(head, "last_center_mask", None)
        context_mask = getattr(head, "last_context_mask", None)
        structure_response = getattr(head, "last_structure_attn", None)
        if center_mask is None or center_mask.shape[-2:] != (height, width):
            self._warn_fallback()
            center_mask = feature.new_zeros((batch, 1, height, width))
            ch = max(1, height // 5)
            cw = max(1, width // 5)
            rs = max(0, (height - ch) // 2)
            cs = max(0, (width - cw) // 2)
            center_mask[:, :, rs : rs + ch, cs : cs + cw] = 1.0
        if context_mask is None or context_mask.shape[-2:] != (height, width):
            context_mask = 1.0 - center_mask

        center_mask = center_mask.detach().to(feature.device)
        context_mask = context_mask.detach().to(feature.device)

        g_response = abs_response
        c_response = abs_response * center_mask.squeeze(1)
        r_response = abs_response * context_mask.squeeze(1)
        if structure_response is not None and structure_response.shape[-2:] == (height, width):
            r_response = r_response + structure_response.detach().squeeze(1).to(feature.device)
        fused_response = g_response + c_response + r_response

        g_embedding = feature.mean(dim=(2, 3))
        c_embedding = masked_average(feature, center_mask)
        if context_mask.sum().item() > 0:
            r_embedding = masked_average(feature, context_mask)
        else:
            r_embedding = g_embedding

        gc5r_embedding = output_embedding.detach() if output_embedding is not None else torch.cat([g_embedding, c_embedding, r_embedding], dim=1)
        return {
            "G_response": g_response.detach(),
            "C_response": c_response.detach(),
            "GC5R_response": fused_response,
            "G_embedding": g_embedding.detach(),
            "C_embedding": c_embedding.detach(),
            "GC5R_embedding": gc5r_embedding.detach(),
        }

    def _forward_batch(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._last_backbone_feature = None
        drone_res, _sat_res = self.model(images, None)
        output_embedding = None
        if isinstance(drone_res, (list, tuple)) and len(drone_res) >= 2 and torch.is_tensor(drone_res[1]):
            output_embedding = drone_res[1]
        elif torch.is_tensor(drone_res):
            output_embedding = drone_res
        return self._extract_geo_tokens(output_embedding)

    def collect_token_outputs(self):
        self.records.clear()
        self.model.eval()
        count = 0
        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                outputs = self._forward_batch(images)
                batch_size = images.shape[0]
                for i in range(batch_size):
                    record = {
                        "sample_id": batch["sample_id"][i],
                        "image_path": batch["path"][i],
                        "label": batch["label"][i],
                        "G_response": outputs["G_response"][i].detach().cpu(),
                        "C_response": outputs["C_response"][i].detach().cpu(),
                        "GC5R_response": outputs["GC5R_response"][i].detach().cpu(),
                        "G_embedding": outputs["G_embedding"][i].detach().cpu(),
                        "C_embedding": outputs["C_embedding"][i].detach().cpu(),
                        "GC5R_embedding": outputs["GC5R_embedding"][i].detach().cpu(),
                    }
                    self.records.append(record)
                    count += 1
        self.logger.info("Collected token outputs for {} UAV samples.".format(count))

    def make_response_maps(self):
        if not self.records:
            self.collect_token_outputs()

        out_dir = self.output_dir / "response_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = out_dir / "heatmaps"
        overlay_dir = out_dir / "overlays"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = self.output_dir / "raw_maps"
        if self.save_raw_maps:
            raw_dir.mkdir(parents=True, exist_ok=True)

        for idx, record in enumerate(self.records):
            image_path = str(record["image_path"])
            original = to_numpy_image(image_path)
            height, width = original.shape[:2]
            maps = {
                "G response": record["G_response"],
                "C response": record["C_response"],
                "GC5R fused response": record["GC5R_response"],
            }
            overlays = []
            heatmaps = []
            for name, tensor in maps.items():
                response = tensor_to_resized_map(tensor, (height, width))
                heat = apply_colormap(response)
                overlay = (0.55 * original + 0.45 * heat).clip(0, 1)
                heatmaps.append((name, heat))
                overlays.append((name, overlay))
                safe = name.replace(" ", "_").replace("/", "_")
                Image.fromarray((heat * 255).astype(np.uint8)).save(heatmap_dir / "sample_{}_{}.png".format(record["sample_id"], safe))
                Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_dir / "sample_{}_{}.png".format(record["sample_id"], safe))
                if self.save_raw_maps:
                    np.save(str(raw_dir / "sample_{}_{}.npy".format(record["sample_id"], safe)), response)
                    Image.fromarray((heat * 255).astype(np.uint8)).save(raw_dir / "sample_{}_{}_heatmap.png".format(record["sample_id"], safe))
                    Image.fromarray((overlay * 255).astype(np.uint8)).save(raw_dir / "sample_{}_{}_overlay.png".format(record["sample_id"], safe))

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(original)
            axes[0].set_title("Original")
            for ax, (name, overlay) in zip(axes[1:], overlays):
                ax.imshow(overlay)
                ax.set_title(name)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(str(out_dir / "sample_{}.png".format(record["sample_id"])), dpi=180)
            plt.close(fig)

        self.logger.info("Saved response map visualizations to {}".format(out_dir))

    def compute_center_response_ratio(self):
        if not self.records:
            self.collect_token_outputs()

        csv_path = self.output_dir / "center_response_ratio.csv"
        rows = []
        for record in self.records:
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "image_path": record["image_path"],
                    "label": record["label"],
                    "G_ratio": center_ratio(record["G_response"]),
                    "C_ratio": center_ratio(record["C_response"]),
                    "GC5R_ratio": center_ratio(record["GC5R_response"]),
                }
            )
        write_csv(csv_path, rows, ["sample_id", "image_path", "label", "G_ratio", "C_ratio", "GC5R_ratio"])

        values = {
            "G": np.asarray([row["G_ratio"] for row in rows], dtype=np.float32),
            "C": np.asarray([row["C_ratio"] for row in rows], dtype=np.float32),
            "GC5R": np.asarray([row["GC5R_ratio"] for row in rows], dtype=np.float32),
        }
        labels = list(values.keys())
        means = [float(values[key].mean()) for key in labels]
        stds = [float(values[key].std(ddof=1)) if values[key].size > 1 else 0.0 for key in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, yerr=stds, capsize=5, color=["#4E79A7", "#F28E2B", "#59A14F"])
        ax.set_xlabel("Token")
        ax.set_ylabel("Center Response Ratio")
        ax.set_title("Center-aware response ratio (mean +/- std)")
        ax.set_ylim(0, min(1.0, max(means) + max(stds) + 0.15))
        fig.tight_layout()
        fig.savefig(str(self.output_dir / "center_response_ratio.png"), dpi=180)
        plt.close(fig)

        self.logger.info("Saved center response ratio CSV and plot.")

    def compute_adjacent_sensitivity(self):
        triplets = self._load_adjacent_triplets()
        if not triplets:
            self.logger.warning(
                "Adjacent-point sensitivity skipped: no adjacent_csv was provided and automatic T-1/T/T+1 construction failed."
            )
            return

        needed_paths = []
        for center_path, prev_path, next_path, _labels in triplets:
            needed_paths.extend([center_path, prev_path, next_path])
        path_to_record = self._collect_specific_paths(needed_paths)

        rows = []
        for center_path, prev_path, next_path, labels in triplets:
            center = path_to_record.get(os.path.abspath(center_path))
            prev = path_to_record.get(os.path.abspath(prev_path))
            nxt = path_to_record.get(os.path.abspath(next_path))
            if center is None or prev is None or nxt is None:
                continue
            rows.append(self._adjacent_row(center, prev, "T-1"))
            rows.append(self._adjacent_row(center, nxt, "T+1"))

        if not rows:
            self.logger.warning("Adjacent-point sensitivity skipped: triplet images could not be loaded.")
            return

        csv_path = self.output_dir / "adjacent_sensitivity.csv"
        fieldnames = ["center_sample", "neighbor_sample", "geo_relation", "sim_G", "sim_C", "sim_GC5R"]
        write_csv(csv_path, rows, fieldnames)
        self._plot_adjacent(rows)
        self.logger.info("Saved adjacent-point sensitivity CSV and plot.")

    def _adjacent_row(self, center: Dict[str, object], neighbor: Dict[str, object], relation: str) -> Dict[str, object]:
        return {
            "center_sample": center["image_path"],
            "neighbor_sample": neighbor["image_path"],
            "geo_relation": relation,
            "sim_G": cosine(center["G_embedding"], neighbor["G_embedding"]),
            "sim_C": cosine(center["C_embedding"], neighbor["C_embedding"]),
            "sim_GC5R": cosine(center["GC5R_embedding"], neighbor["GC5R_embedding"]),
        }

    def _load_adjacent_triplets(self):
        if self.adjacent_csv:
            return read_adjacent_csv(self.adjacent_csv)
        return auto_adjacent_triplets(self.data_root)

    def _collect_specific_paths(self, paths: Sequence[str]) -> Dict[str, Dict[str, object]]:
        unique_paths = sorted({os.path.abspath(path) for path in paths if path and os.path.exists(path)})
        if not unique_paths:
            return {}
        dataset = UAVImageDataset(self.data_root or ".", self.image_size, max_samples=None, explicit_paths=unique_paths)
        loader = DataLoader(dataset, batch_size=max(1, min(len(dataset), self.dataloader.batch_size or 1)), shuffle=False, num_workers=0, collate_fn=collate_records)
        path_to_record = {}
        with torch.no_grad():
            for batch in loader:
                outputs = self._forward_batch(batch["image"].to(self.device))
                for i, path in enumerate(batch["path"]):
                    record = {
                        "image_path": path,
                        "label": batch["label"][i],
                        "G_embedding": outputs["G_embedding"][i].detach().cpu(),
                        "C_embedding": outputs["C_embedding"][i].detach().cpu(),
                        "GC5R_embedding": outputs["GC5R_embedding"][i].detach().cpu(),
                    }
                    path_to_record[os.path.abspath(path)] = record
        return path_to_record

    def _plot_adjacent(self, rows: Sequence[Dict[str, object]]):
        data = [
            np.asarray([float(row["sim_G"]) for row in rows], dtype=np.float32),
            np.asarray([float(row["sim_C"]) for row in rows], dtype=np.float32),
            np.asarray([float(row["sim_GC5R"]) for row in rows], dtype=np.float32),
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, labels=["G", "C", "GC5R"], showmeans=True)
        ax.set_xlabel("Token")
        ax.set_ylabel("Adjacent Cosine Similarity")
        ax.set_title("Adjacent-point token similarity")
        ax.set_ylim(-1.0, 1.0)
        fig.tight_layout()
        fig.savefig(str(self.output_dir / "adjacent_sensitivity.png"), dpi=180)
        plt.close(fig)

    def run_all(self):
        try:
            self.collect_token_outputs()
            self.make_response_maps()
            self.compute_center_response_ratio()
            self.compute_adjacent_sensitivity()
        finally:
            self.close()


def masked_average(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = feature * mask
    denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
    return weighted.sum(dim=(2, 3)) / denom


def tensor_to_resized_map(tensor: torch.Tensor, size_hw: Tuple[int, int]) -> np.ndarray:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    resized = F.interpolate(tensor.float(), size=size_hw, mode="bilinear", align_corners=False)
    return normalize_map(resized.squeeze().detach().cpu().numpy())


def apply_colormap(response: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    return cmap(normalize_map(response))[..., :3].astype(np.float32)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_adjacent_csv(path: str):
    triplets = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"center_path", "prev_path", "next_path", "center_label", "prev_label", "next_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("adjacent_csv missing columns: {}".format(sorted(missing)))
        for row in reader:
            triplets.append(
                (
                    row["center_path"],
                    row["prev_path"],
                    row["next_path"],
                    (row["center_label"], row["prev_label"], row["next_label"]),
                )
            )
    return triplets


def auto_adjacent_triplets(data_root: str):
    if not data_root:
        return []
    dataset_samples = UAVImageDataset._discover_samples(data_root)
    by_label: Dict[str, List[str]] = {}
    for path, label in dataset_samples:
        by_label.setdefault(label, []).append(path)
    labels = sorted(by_label.keys(), key=natural_label_key)
    triplets = []
    for idx in range(1, len(labels) - 1):
        prev_label, center_label, next_label = labels[idx - 1], labels[idx], labels[idx + 1]
        if not (is_numeric_like(prev_label) and is_numeric_like(center_label) and is_numeric_like(next_label)):
            continue
        if int(prev_label) + 1 != int(center_label) or int(center_label) + 1 != int(next_label):
            continue
        center_path = sorted(by_label[center_label])[0]
        prev_path = sorted(by_label[prev_label])[0]
        next_path = sorted(by_label[next_label])[0]
        triplets.append((center_path, prev_path, next_path, (center_label, prev_label, next_label)))
        if len(triplets) >= 64:
            break
    return triplets


def is_numeric_like(value: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(value)))


def natural_label_key(value: str):
    value = str(value)
    return (0, int(value)) if is_numeric_like(value) else (1, value)


def parse_args():
    parser = argparse.ArgumentParser(description="DenseUAV GeoTokenHead token-level response analysis")
    parser.add_argument(
        "--checkpoint_dir",
        default="",
        help="Path to checkpoints/experiment directory. Defaults to current directory when it looks like a checkpoint folder.",
    )
    parser.add_argument("--data_root", default="", help="DenseUAV dataset root. Defaults to opts.yaml data_dir when available.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_samples", default=32, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adjacent_csv", default="", help="CSV with center_path,prev_path,next_path,center_label,prev_label,next_label")
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--save_raw_maps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("checkpoint_dir does not exist: {}".format(checkpoint_dir))

    output_dir = checkpoint_dir / "token_analysis"
    logger = setup_logger(output_dir)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, opt = load_model(checkpoint_dir, device, logger)
    data_root = infer_data_root(args, opt)
    logger.info("Using data root: {}".format(data_root))

    image_size = int(args.image_size)
    if image_size <= 0:
        image_size = int(getattr(opt, "h", 224))
    dataset = UAVImageDataset(data_root, image_size=image_size, max_samples=args.num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_records,
    )

    analyzer = TokenAnalyzer(
        model=model,
        dataloader=loader,
        output_dir=output_dir,
        device=device,
        adjacent_csv=args.adjacent_csv,
        image_size=image_size,
        save_raw_maps=args.save_raw_maps,
        data_root=data_root,
        logger=logger,
    )
    analyzer.run_all()
    logger.info("Token analysis complete. Outputs saved to {}".format(output_dir))


def resolve_checkpoint_dir(checkpoint_dir_arg: str) -> Path:
    if checkpoint_dir_arg:
        return Path(checkpoint_dir_arg).expanduser().resolve()
    cwd = Path.cwd().resolve()
    has_config = any((cwd / name).is_file() for name in ("opts.yaml", "config.yaml", "args.yaml", "opt.txt"))
    has_weights = any(cwd.glob("*.pth"))
    if has_config and has_weights:
        return cwd
    raise SystemExit(
        "Please provide --checkpoint_dir, or run this script inside a checkpoint experiment folder "
        "that contains opts.yaml/config.yaml and a .pth checkpoint."
    )


if __name__ == "__main__":
    main()
