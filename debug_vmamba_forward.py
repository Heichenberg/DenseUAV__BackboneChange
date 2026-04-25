import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from models.taskflow import make_model


DEFAULT_WEIGHTS = {
    "VMamba-Tiny": "pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    "VMamba-Small": "pretrained/backbones/vmamba/small/vssm_small_0229_ckpt_epoch_222.pth",
    "VMamba-Base": "pretrained/backbones/vmamba/base/vssm_base_0229_ckpt_epoch_237.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal VMamba forward debug for DenseUAV")
    parser.add_argument("--backbone", default="VMamba-Tiny", choices=list(DEFAULT_WEIGHTS.keys()))
    parser.add_argument("--backbone_weight", default="", type=str, help="Relative to DenseUAV root or absolute path")
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--h", default=224, type=int)
    parser.add_argument("--w", default=224, type=int)
    parser.add_argument("--nclasses", default=10, type=int)
    parser.add_argument("--num_bottleneck", default=512, type=int)
    return parser.parse_args()


def build_opt(args):
    weight = args.backbone_weight or DEFAULT_WEIGHTS[args.backbone]
    return SimpleNamespace(
        h=args.h,
        w=args.w,
        backbone=args.backbone,
        backbone_weight=weight,
        head="SingleBranchCNN",
        nclasses=args.nclasses,
        droprate=0.5,
        num_bottleneck=args.num_bottleneck,
        head_pool="avg",
        load_from="no",
    )


def main():
    args = parse_args()
    opt = build_opt(args)

    print("DenseUAV root:", Path(__file__).resolve().parent)
    print("backbone:", opt.backbone)
    print("backbone_weight:", opt.backbone_weight)
    print("head:", opt.head)

    model = make_model(opt).eval()
    x = torch.randn(args.batchsize, 3, args.h, args.w)

    with torch.no_grad():
        backbone_feature = model.backbone(x)
        drone_res, satellite_res = model(x, None)

    print("\n[Build]")
    print("model_created:", type(model).__name__)
    print("backbone_output_channel:", model.backbone.output_channel)

    print("\n[Backbone Forward]")
    print("input_shape:", tuple(x.shape))
    print("backbone_feature_shape:", tuple(backbone_feature.shape))
    print("backbone_feature_dim_ok:", backbone_feature.ndim == 4)

    print("\n[DenseUAV Forward]")
    print("drone_res_type:", type(drone_res).__name__)
    print("drone_res_len:", len(drone_res) if isinstance(drone_res, (list, tuple)) else "N/A")
    print("drone_cls_shape:", tuple(drone_res[0].shape))
    print("drone_embedding_shape:", tuple(drone_res[1].shape))
    print("satellite_res:", satellite_res)

    format_ok = (
        backbone_feature.ndim == 4
        and isinstance(drone_res, (list, tuple))
        and len(drone_res) == 2
        and drone_res[1].ndim == 2
    )

    print("\n[Check]")
    print("DenseUAV_expected_backbone_feature = [B, C, H, W]")
    print("DenseUAV_expected_head_output = [cls, embedding]")
    print("format_ok:", format_ok)

    if not format_ok:
        raise SystemExit("VMamba forward format check failed.")


if __name__ == "__main__":
    main()
