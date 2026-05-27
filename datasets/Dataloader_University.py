import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import glob
import math
import json
import hashlib


class Dataloader_University(Dataset):
    def __init__(self, root, transforms, names=['satellite', 'drone'], max_ids=0, id_subset_file=""):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        if id_subset_file:
            with open(id_subset_file, "r", encoding="utf-8") as handle:
                selected = [line.strip() for line in handle if line.strip()]
            cls_names = [name for name in cls_names if name in set(selected)]
        elif max_ids > 0:
            cls_names = cls_names[:max_ids]
        # 获取所有图片的相对路径分别放到对应的类别中
        # {satelite:{0839:[0839.jpg],0840:[0840.jpg]}}
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in cls_names:
                img_list = os.listdir(os.path.join(root, name, cls_name))
                img_path_list = [os.path.join(
                    root, name, cls_name, img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
            # dict_path[name+"/"+cls_name] = img_path_list

        # 获取设置名字与索引之间的镜像
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    # 从对应的类别中抽一张出来
    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_satellite(img)

        # img = self.sample_from_cls("street",cls_nums)
        # img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s, img_d, index

    def __len__(self):
        return len(self.cls_names)


class DataLoader_Inference(Dataset):
    def __init__(self, root, transforms):
        super(DataLoader_Inference, self).__init__()
        self.root = root
        self.imgs = glob.glob(root+"/*.tif")
        self.tranforms = transforms
        sorted(self.imgs)
        self.labels = [os.path.basename(img).split(".tif")[
            0] for img in self.imgs]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        return self.tranforms(img), self.labels[index]

    def __len__(self):
        return len(self.imgs)


class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8, sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0, self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        return self.data_len * self.sample_num


def _parse_gps_token(token):
    token = token.strip()
    if token.startswith(("E", "N", "W", "S")):
        sign = -1.0 if token[0] in ("W", "S") else 1.0
        return sign * float(token[1:])
    return float(token)


def _haversine_matrix(coords):
    coords = np.asarray(coords, dtype=np.float64)
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class DSSSampler_University(object):
    """
    Dynamic Sampling Strategy v1.

    This sampler keeps the existing Dataset contract intact: it only yields
    location indices. Each yielded index is still converted by the dataset into
    one random satellite-drone positive pair. DSS changes which location IDs
    share a batch so metric losses see harder negative IDs.
    """

    def __init__(
        self,
        data_source,
        batchsize=8,
        sample_num=1,
        gps_file="",
        gds_topk=64,
        dss_start_epoch=0,
        gds_ratio=0.5,
        fss_ratio=0.0,
        rs_ratio=0.5,
        cache_dir="",
        fss_neighbors=None,
        tact_gps=False,
        tact_smooth=False,
        total_epoch=1,
    ):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num
        self.gps_file = gps_file
        self.gds_topk = gds_topk
        self.dss_start_epoch = dss_start_epoch
        self.base_gds_ratio = float(gds_ratio)
        self.base_fss_ratio = float(fss_ratio)
        self.base_rs_ratio = float(rs_ratio)
        self.tact_gps = bool(tact_gps)
        self.tact_smooth = bool(tact_smooth)
        if self.tact_gps and self.tact_smooth:
            raise ValueError("TACT-GPS and TACT-Smooth cannot be enabled at the same time")
        self.total_epoch = int(total_epoch)
        self.current_gds_ratio = 0.3 if (self.tact_gps or self.tact_smooth) else self.base_gds_ratio
        self.current_fss_ratio = 0.0 if (self.tact_gps or self.tact_smooth) else self.base_fss_ratio
        self.current_rs_ratio = 0.7 if (self.tact_gps or self.tact_smooth) else self.base_rs_ratio
        self.gds_ratio = self.current_gds_ratio
        self.fss_ratio = self.current_fss_ratio
        self.rs_ratio = self.current_rs_ratio
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.epoch = 0
        self._last_tact_log_epoch = None
        self.fss_neighbors = self._validate_neighbors(fss_neighbors, "FSS") if fss_neighbors is not None else None
        self.gds_neighbors = self._build_gds_neighbors(data_source)
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.tact_gps or self.tact_smooth:
            progress = self._tact_progress(epoch)
            p_gps = 0.3 + 0.6 * progress
        if self.tact_gps:
            self.current_gds_ratio = p_gps
            self.current_fss_ratio = 0.0
            self.current_rs_ratio = 1.0 - p_gps
            self.gds_ratio = self.current_gds_ratio
            self.fss_ratio = self.current_fss_ratio
            self.rs_ratio = self.current_rs_ratio
            if self._last_tact_log_epoch != epoch:
                print("[TACT-GPS] epoch={}/{}, gds_ratio={:.3f}, fss_ratio={:.3f}, rs_ratio={:.3f}".format(
                    epoch,
                    self.total_epoch,
                    self.current_gds_ratio,
                    self.current_fss_ratio,
                    self.current_rs_ratio,
                ))
                self._last_tact_log_epoch = epoch
        elif self.tact_smooth:
            lambda_gps = (1.0 + math.cos(math.pi * progress)) / 2.0
            target_gds_ratio = p_gps * lambda_gps
            target_fss_ratio = p_gps * (1.0 - lambda_gps)
            target_rs_ratio = 1.0 - p_gps
            fss_available = self.fss_neighbors is not None
            if fss_available:
                effective_gds_ratio = target_gds_ratio
                effective_fss_ratio = target_fss_ratio
            else:
                effective_gds_ratio = target_gds_ratio + target_fss_ratio
                effective_fss_ratio = 0.0
            effective_rs_ratio = target_rs_ratio

            self.current_gds_ratio = effective_gds_ratio
            self.current_fss_ratio = effective_fss_ratio
            self.current_rs_ratio = effective_rs_ratio
            self.gds_ratio = self.current_gds_ratio
            self.fss_ratio = self.current_fss_ratio
            self.rs_ratio = self.current_rs_ratio
            if self._last_tact_log_epoch != epoch:
                print(
                    "[TACT-Smooth] epoch={}/{}, progress={:.3f}, p={:.3f}, lambda={:.3f}, "
                    "target_gds={:.3f}, target_fss={:.3f}, target_rs={:.3f}, "
                    "effective_gds={:.3f}, effective_fss={:.3f}, effective_rs={:.3f}, "
                    "fss_available={}".format(
                        epoch,
                        self.total_epoch,
                        progress,
                        p_gps,
                        lambda_gps,
                        target_gds_ratio,
                        target_fss_ratio,
                        target_rs_ratio,
                        self.current_gds_ratio,
                        self.current_fss_ratio,
                        self.current_rs_ratio,
                        fss_available,
                    )
                )
                self._last_tact_log_epoch = epoch

    def update_fss_neighbors(self, neighbors=None):
        self.fss_neighbors = self._validate_neighbors(neighbors, "FSS") if neighbors is not None else None
        if self.tact_smooth:
            self.set_epoch(self.epoch)

    def _tact_progress(self, epoch):
        if self.total_epoch <= 1:
            return 0.0
        progress = epoch / float(self.total_epoch - 1)
        return max(0.0, min(1.0, progress))

    def set_dss_ratios(self, gds_ratio=None, fss_ratio=None, rs_ratio=None):
        if gds_ratio is not None:
            self.base_gds_ratio = float(gds_ratio)
        if fss_ratio is not None:
            self.base_fss_ratio = float(fss_ratio)
        if rs_ratio is not None:
            self.base_rs_ratio = float(rs_ratio)
        if not self.tact_gps and not self.tact_smooth:
            self.current_gds_ratio = self.base_gds_ratio
            self.current_fss_ratio = self.base_fss_ratio
            self.current_rs_ratio = self.base_rs_ratio
            self.gds_ratio = self.current_gds_ratio
            self.fss_ratio = self.current_fss_ratio
            self.rs_ratio = self.current_rs_ratio

    def _default_cache_dir(self):
        if not self.gps_file:
            return ""
        dataset_root = os.path.dirname(os.path.abspath(self.gps_file))
        return os.path.join(dataset_root, "dss_cache")

    def _cls_hash(self, cls_names):
        payload = "\n".join(cls_names).encode("utf-8")
        return hashlib.md5(payload).hexdigest()[:12]

    def _gds_cache_paths(self, data_source, topk):
        if not self.cache_dir:
            return None, None
        cls_hash = self._cls_hash(data_source.cls_names)
        base = "gds_neighbors_train_top{}_n{}_{}.npy".format(topk, self.data_len, cls_hash)
        neighbors_path = os.path.join(self.cache_dir, base)
        meta_path = neighbors_path.replace(".npy", "_meta.json")
        return neighbors_path, meta_path

    def _validate_neighbors(self, neighbors, name):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if neighbors.ndim != 2:
            raise ValueError("{} neighbors must be a 2D array, got shape {}".format(name, neighbors.shape))
        if neighbors.shape[0] != self.data_len:
            raise ValueError("{} neighbors row count {} does not match dataset size {}".format(
                name, neighbors.shape[0], self.data_len))
        return neighbors

    def _load_gds_cache(self, data_source, topk):
        neighbors_path, meta_path = self._gds_cache_paths(data_source, topk)
        if not neighbors_path or not os.path.exists(neighbors_path) or not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        expected_hash = self._cls_hash(data_source.cls_names)
        if (
            meta.get("topk") != topk
            or meta.get("num_ids") != self.data_len
            or meta.get("cls_hash") != expected_hash
        ):
            return None
        return self._validate_neighbors(np.load(neighbors_path), "GDS")

    def _save_gds_cache(self, data_source, topk, neighbors):
        neighbors_path, meta_path = self._gds_cache_paths(data_source, topk)
        if not neighbors_path:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(neighbors_path, neighbors)
            meta = {
                "split": "train",
                "topk": topk,
                "num_ids": self.data_len,
                "cls_hash": self._cls_hash(data_source.cls_names),
                "gps_file": os.path.abspath(self.gps_file),
                "cls_names": data_source.cls_names,
            }
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
        except OSError as exc:
            print("Warning: failed to write GDS cache to {}: {}".format(self.cache_dir, exc))

    def _read_gps_coords(self, data_source):
        if not self.gps_file:
            return None
        if not os.path.exists(self.gps_file):
            raise FileNotFoundError("DSS gps file not found: {}".format(self.gps_file))

        coords_by_cls = {}
        with open(self.gps_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                rel_path, lon_token, lat_token = parts[:3]
                path_parts = rel_path.replace("\\", "/").split("/")
                if len(path_parts) < 3:
                    continue
                cls_name = path_parts[-2]
                if cls_name in coords_by_cls:
                    continue
                lon = _parse_gps_token(lon_token)
                lat = _parse_gps_token(lat_token)
                coords_by_cls[cls_name] = (lat, lon)

        coords = []
        missing = []
        for cls_name in data_source.cls_names:
            if cls_name not in coords_by_cls:
                missing.append(cls_name)
                coords.append((math.nan, math.nan))
            else:
                coords.append(coords_by_cls[cls_name])
        if missing:
            raise ValueError("Missing GPS coordinates for {} train IDs, first missing: {}".format(
                len(missing), missing[:5]))
        return np.asarray(coords, dtype=np.float64)

    def _build_gds_neighbors(self, data_source):
        coords = self._read_gps_coords(data_source)
        if coords is None or self.data_len <= 1:
            return None
        topk = min(max(1, self.gds_topk), self.data_len - 1)
        cached = self._load_gds_cache(data_source, topk)
        if cached is not None:
            return cached
        dist = _haversine_matrix(coords)
        np.fill_diagonal(dist, np.inf)
        neighbors = np.argsort(dist, axis=1)[:, :topk].astype(np.int64)
        self._save_gds_cache(data_source, topk, neighbors)
        return neighbors

    def _random_indices(self, count, exclude=None):
        if count <= 0:
            return []
        exclude = set() if exclude is None else set(exclude)
        pool = [idx for idx in range(self.data_len) if idx not in exclude]
        if not pool:
            return []
        replace = len(pool) < count
        return np.random.choice(pool, count, replace=replace).astype(np.int64).tolist()

    def _sample_candidates(self, candidates, count, used):
        if count <= 0 or candidates is None:
            return []
        candidates = [int(idx) for idx in candidates if int(idx) not in used]
        if not candidates:
            return []
        replace = len(candidates) < count
        return np.random.choice(candidates, count, replace=replace).astype(np.int64).tolist()

    def _random_batch(self):
        replace = self.data_len < self.batchsize
        return np.random.choice(self.data_len, self.batchsize, replace=replace).astype(np.int64).tolist()

    def sample_batch(self, anchor=None):
        if not self.last_stats:
            self.last_stats = {"epoch": self.epoch, "batches": 0, "anchor": 0, "gds": 0, "fss": 0, "rs": 0}
        if self.epoch < self.dss_start_epoch or self.gds_neighbors is None:
            self.last_stats["rs"] += self.batchsize
            return self._random_batch()

        anchor = int(np.random.randint(self.data_len) if anchor is None else anchor)
        batch = [anchor]
        used = {anchor}
        remaining = self.batchsize - 1
        gds_ratio = self.current_gds_ratio
        rs_ratio = self.current_rs_ratio
        fss_available = self.fss_neighbors is not None and self.current_fss_ratio > 0
        fss_ratio = self.current_fss_ratio if fss_available else 0.0
        ratio_sum = max(gds_ratio + fss_ratio + rs_ratio, 1e-12)
        gds_count = int(round(remaining * gds_ratio / ratio_sum))
        fss_count = int(round(remaining * fss_ratio / ratio_sum))

        gds_samples = self._sample_candidates(self.gds_neighbors[anchor], gds_count, used)
        batch.extend(gds_samples)
        used.update(gds_samples)

        fss_samples = self._sample_candidates(
            None if self.fss_neighbors is None else self.fss_neighbors[anchor],
            fss_count,
            used,
        )
        batch.extend(fss_samples)
        used.update(fss_samples)

        rs_needed = self.batchsize - len(batch)
        rs_samples = self._random_indices(rs_needed, exclude=used)
        batch.extend(rs_samples)

        if len(batch) < self.batchsize:
            batch.extend(self._random_indices(self.batchsize - len(batch)))

        self.last_stats["anchor"] += 1
        self.last_stats["gds"] += len(gds_samples)
        self.last_stats["fss"] += len(fss_samples)
        self.last_stats["rs"] += len(batch) - 1 - len(gds_samples) - len(fss_samples)
        return batch

    def __iter__(self):
        total = self.data_len * self.sample_num
        num_batches = total // self.batchsize
        self.last_stats = {"epoch": self.epoch, "batches": num_batches, "anchor": 0, "gds": 0, "fss": 0, "rs": 0}
        anchors = np.arange(self.data_len)
        np.random.shuffle(anchors)
        if num_batches > len(anchors):
            anchors = np.resize(anchors, num_batches)
        indices = []
        for anchor in anchors[:num_batches]:
            indices.extend(self.sample_batch(anchor=anchor))
        return iter(indices)

    def __len__(self):
        return (self.data_len * self.sample_num // self.batchsize) * self.batchsize


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]


if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list = {"satellite": transforms.Compose(transform_train_list),
                            "train": transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",
                                     transforms=transform_train_list, names=['satellite', 'drone'])
    samper = Sampler_University(datasets, 8)
    dataloader = DataLoader(datasets, batch_size=8, num_workers=0,
                            sampler=samper, collate_fn=train_collate_fn)
    for data_s, data_d in dataloader:
        print()
