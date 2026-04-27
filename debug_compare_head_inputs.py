import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import timm

from models.taskflow import make_model


def parse_args():
    parser = argparse.ArgumentParser(description="Compare backbone outputs before head")
    parser.add_argument(
        "--exp-a",
        default="checkpoints/baseline",
        type=str,
        help="First experiment directory containing opts.yaml and checkpoint",
    )
    parser.add_argument(
        "--exp-b",
        default="checkpoints/vmamba_tiny_singlebranchcnn",
        type=str,
        help="Second experiment directory containing opts.yaml and checkpoint",
    )
    parser.add_argument("--checkpoint", default="net_119.pth", type=str)
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--h", default=224, type=int)
    parser.add_argument("--w", default=224, type=int)
    parser.add_argument("--custom-b-backbone", default="", type=str, help="Optional custom backbone name for experiment B")
    parser.add_argument("--custom-b-head", default="", type=str, help="Optional custom head name for experiment B")
    parser.add_argument("--custom-b-backbone-weight", default="", type=str, help="Optional custom backbone weight for experiment B")
    parser.add_argument("--custom-b-head-pool", default="avg", type=str, help="Optional custom head_pool for experiment B")
    parser.add_argument("--custom-b-num-bottleneck", default=512, type=int, help="Optional custom num_bottleneck for experiment B")
    return parser.parse_args()


def load_opt(exp_dir, checkpoint_name):
    exp_path = Path(exp_dir).resolve()
    opts_path = exp_path / "opts.yaml"
    with open(opts_path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)
    data["load_from"] = str(exp_path / checkpoint_name)
    data["use_gpu"] = torch.cuda.is_available()
    data.setdefault("nclasses", 2256)
    return SimpleNamespace(**data)


def make_custom_opt(backbone, head, backbone_weight, head_pool, num_bottleneck, h, w):
    return SimpleNamespace(
        backbone=backbone,
        backbone_weight=backbone_weight,
        head=head,
        head_pool=head_pool,
        num_bottleneck=num_bottleneck,
        load_from="",
        nclasses=2256,
        droprate=0.5,
        h=h,
        w=w,
        use_gpu=torch.cuda.is_available(),
    )


def tensor_summary(tensor):
    flat = tensor.detach().float()
    return {
        "shape": tuple(flat.shape),
        "ndim": flat.ndim,
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def print_summary(tag, summary, opt):
    print("\n[{}]".format(tag))
    print("backbone:", opt.backbone)
    print("head:", opt.head)
    print("checkpoint:", opt.load_from)
    print("shape:", summary["shape"])
    print("ndim:", summary["ndim"])
    print("dtype:", summary["dtype"])
    print("mean:", summary["mean"])
    print("std:", summary["std"])
    print("min:", summary["min"])
    print("max:", summary["max"])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_a = load_opt(args.exp_a, args.checkpoint)
    if args.custom_b_backbone:
        opt_b = make_custom_opt(
            args.custom_b_backbone,
            args.custom_b_head or "SingleBranch",
            args.custom_b_backbone_weight,
            args.custom_b_head_pool,
            args.custom_b_num_bottleneck,
            args.h,
            args.w,
        )
    else:
        opt_b = load_opt(args.exp_b, args.checkpoint)

    original_create_model = timm.create_model

    def offline_create_model(*model_args, **model_kwargs):
        model_kwargs["pretrained"] = False
        return original_create_model(*model_args, **model_kwargs)

    timm.create_model = offline_create_model
    try:
        model_a = make_model(opt_a).to(device).eval()
        model_b = make_model(opt_b).to(device).eval()
    finally:
        timm.create_model = original_create_model

    x = torch.randn(args.batchsize, 3, args.h, args.w, device=device)

    with torch.no_grad():
        feat_a = model_a.backbone(x)
        feat_b = model_b.backbone(x)

    summary_a = tensor_summary(feat_a)
    summary_b = tensor_summary(feat_b)

    print("Input shape:", tuple(x.shape))
    print_summary("Experiment A", summary_a, opt_a)
    print_summary("Experiment B", summary_b, opt_b)


if __name__ == "__main__":
    main()
