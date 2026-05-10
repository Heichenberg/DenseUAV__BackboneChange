import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset


class FSSFeatureDataset(Dataset):
    """Sample one satellite image and one drone image for each train location ID."""

    def __init__(self, source_dataset, samples_per_id=1):
        self.source_dataset = source_dataset
        self.samples_per_id = max(1, int(samples_per_id))
        self.num_ids = len(source_dataset)

    def __len__(self):
        return self.num_ids * self.samples_per_id

    def _sample_image(self, name, cls_name):
        img_path = np.random.choice(self.source_dataset.dict_path[name][cls_name], 1)[0]
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, index):
        idx = int(index % self.num_ids)
        cls_name = self.source_dataset.map_dict[idx]
        sat_img = self._sample_image("satellite", cls_name)
        drone_img = self._sample_image("drone", cls_name)
        sat_img = self.source_dataset.transforms_satellite(sat_img)
        drone_img = self.source_dataset.transforms_drone_street(drone_img)
        return sat_img, drone_img, idx


def should_update_fss(epoch, opt, fss_ratio=None, has_neighbors=True):
    if getattr(opt, "train_strategy", "origin") != "dss":
        return False
    ratio = getattr(opt, "dss_fss_ratio", 0.0) if fss_ratio is None else fss_ratio
    if ratio <= 0:
        return False
    interval = int(getattr(opt, "dss_fss_update_interval", 10))
    if interval <= 0:
        return False
    if getattr(opt, "dss_stage_mode", "fixed") == "loss_adaptive":
        last_update_epoch = getattr(opt, "_last_fss_update_epoch", None)
        if not has_neighbors or last_update_epoch is None:
            return True
        return epoch - last_update_epoch >= interval
    start_epoch = int(getattr(opt, "dss_fss_start_epoch", getattr(opt, "dss_start_epoch", 0)))
    if epoch < start_epoch:
        return False
    return (epoch - start_epoch) % interval == 0


def _cls_hash(cls_names):
    payload = "\n".join(cls_names).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _extract_feature(branch_output):
    if branch_output is None:
        raise ValueError("FSS feature extraction received an empty model branch output")
    if isinstance(branch_output, (list, tuple)):
        if len(branch_output) < 2:
            raise ValueError("FSS expected branch output [cls, feature], got {}".format(type(branch_output)))
        feature = branch_output[1]
    else:
        feature = branch_output
    if isinstance(feature, (list, tuple)):
        feature = torch.stack(list(feature), dim=-1).mean(dim=-1)
    if feature.ndim > 2:
        feature = feature.flatten(2).mean(dim=-1)
    if feature.ndim != 2:
        raise ValueError("FSS expected a 2D feature tensor, got shape {}".format(tuple(feature.shape)))
    return feature


def _default_device(model, opt):
    if getattr(opt, "use_gpu", False) and torch.cuda.is_available():
        return torch.device("cuda")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _fss_cache_paths(opt, epoch, topk):
    cache_dir = os.path.join("checkpoints", opt.name, "dss_cache")
    base = "fss_neighbors_epoch{:03d}_top{}".format(epoch, topk)
    return (
        cache_dir,
        os.path.join(cache_dir, base + ".npy"),
        os.path.join(cache_dir, base + "_meta.json"),
    )


def build_fss_neighbors(model, source_dataset, opt, device=None, logger=None, epoch=None):
    started = time.time()
    device = _default_device(model, opt) if device is None else torch.device(device)
    topk = int(getattr(opt, "dss_fss_topk", 64))
    topk = min(max(1, topk), len(source_dataset) - 1)
    samples_per_id = max(1, int(getattr(opt, "dss_fss_samples_per_id", 1)))
    epoch = int(getattr(opt, "_current_epoch", 0) if epoch is None else epoch)

    fss_dataset = FSSFeatureDataset(source_dataset, samples_per_id=samples_per_id)
    loader = DataLoader(
        fss_dataset,
        batch_size=getattr(opt, "batchsize", 16),
        shuffle=False,
        num_workers=getattr(opt, "num_worker", 0),
        pin_memory=getattr(opt, "use_gpu", False),
    )

    was_training = model.training
    model.eval()
    feature_sum = None
    counts = torch.zeros(len(source_dataset), dtype=torch.float32)
    use_autocast = getattr(opt, "autocast", False) and device.type == "cuda"

    with torch.no_grad():
        for sat_imgs, drone_imgs, indices in loader:
            sat_imgs = sat_imgs.to(device, non_blocking=True)
            drone_imgs = drone_imgs.to(device, non_blocking=True)
            indices = indices.to(torch.long)
            with autocast(enabled=use_autocast):
                drone_output, sat_output = model(drone_imgs, sat_imgs)
                drone_feature = F.normalize(_extract_feature(drone_output), dim=1)
                sat_feature = F.normalize(_extract_feature(sat_output), dim=1)
                location_feature = F.normalize((drone_feature + sat_feature) * 0.5, dim=1)
            location_feature = location_feature.detach().cpu().float()
            if feature_sum is None:
                feature_sum = torch.zeros(len(source_dataset), location_feature.shape[1], dtype=torch.float32)
            feature_sum.index_add_(0, indices.cpu(), location_feature)
            counts.index_add_(0, indices.cpu(), torch.ones(indices.numel(), dtype=torch.float32))

    if was_training:
        model.train(True)

    if feature_sum is None:
        raise ValueError("FSS could not extract any features from the training dataset")

    counts = counts.clamp_min(1.0).unsqueeze(1)
    features = F.normalize(feature_sum / counts, dim=1)
    sim = torch.mm(features, features.t())
    sim.fill_diagonal_(-float("inf"))
    neighbors = torch.topk(sim, k=topk, dim=1, largest=True).indices.cpu().numpy().astype(np.int64)

    cache_dir, neighbors_path, meta_path = _fss_cache_paths(opt, epoch, topk)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(neighbors_path, neighbors)
    meta = {
        "split": "train",
        "epoch": epoch,
        "topk": topk,
        "num_ids": len(source_dataset),
        "cls_hash": _cls_hash(source_dataset.cls_names),
        "feature_dim": int(features.shape[1]),
        "samples_per_id": samples_per_id,
        "neighbors_path": os.path.abspath(neighbors_path),
        "elapsed_sec": time.time() - started,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    if logger is not None:
        logger.info(
            "FSS neighbors updated: epoch=%d shape=%s path=%s elapsed=%.1fs",
            epoch,
            tuple(neighbors.shape),
            neighbors_path,
            meta["elapsed_sec"],
        )
    return neighbors
