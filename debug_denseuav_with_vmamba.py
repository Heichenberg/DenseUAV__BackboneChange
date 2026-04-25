import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.taskflow import make_model
from losses.TripletLoss import TripletLoss


DEFAULT_WEIGHTS = {
    "VMamba-Tiny": "pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth",
    "VMamba-Small": "pretrained/backbones/vmamba/small/vssm_small_0229_ckpt_epoch_222.pth",
    "VMamba-Base": "pretrained/backbones/vmamba/base/vssm_base_0229_ckpt_epoch_237.pth",
}


class MinimalDenseUAVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3, normalize_feature=True)

    def forward(self, outputs1, outputs2, labels1, labels2):
        cls1, feat1 = outputs1
        cls2, feat2 = outputs2
        cls_loss = self.cls_loss(cls1, labels1) + self.cls_loss(cls2, labels2)
        feat_concat = torch.cat((feat1, feat2), dim=0)
        label_concat = torch.cat((labels1, labels2), dim=0)
        triplet_loss = self.triplet_loss(feat_concat, label_concat)
        total_loss = cls_loss + triplet_loss
        zero = total_loss.new_tensor(0.0)
        return total_loss, cls_loss, triplet_loss, zero


def parse_args():
    parser = argparse.ArgumentParser(description="DenseUAV full-model VMamba debug")
    parser.add_argument("--backbone", default="VMamba-Tiny", choices=list(DEFAULT_WEIGHTS.keys()))
    parser.add_argument("--backbone_weight", default="", type=str, help="Relative to DenseUAV root or absolute path")
    parser.add_argument("--batchsize", default=2, type=int)
    parser.add_argument("--h", default=224, type=int)
    parser.add_argument("--w", default=224, type=int)
    parser.add_argument("--nclasses", default=10, type=int)
    parser.add_argument("--num_bottleneck", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
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
        batchsize=args.batchsize,
        sample_num=1,
        cls_loss="CELoss",
        feature_loss="TripletLoss",
        kl_loss="no",
    )


def build_loss(opt):
    try:
        from losses.loss import Loss

        criterion = Loss(opt)
        mode = "repository Loss"
    except ModuleNotFoundError as exc:
        criterion = MinimalDenseUAVLoss()
        mode = f"minimal debug loss fallback ({exc})"
    return criterion, mode


def grad_stats(model):
    total = 0
    has_grad = 0
    grad_norm = 0.0
    for param in model.parameters():
        total += 1
        if param.grad is not None:
            has_grad += 1
            grad_norm += float(param.grad.detach().abs().mean())
    return total, has_grad, grad_norm


def main():
    args = parse_args()
    opt = build_opt(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("DenseUAV root:", Path(__file__).resolve().parent)
    print("device:", device)
    print("backbone:", opt.backbone)
    print("backbone_weight:", opt.backbone_weight)
    print("head:", opt.head)

    model = make_model(opt).to(device).train()
    criterion, loss_mode = build_loss(opt)
    criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    drone = torch.randn(args.batchsize, 3, args.h, args.w, device=device)
    satellite = torch.randn(args.batchsize, 3, args.h, args.w, device=device)
    labels = torch.arange(args.batchsize, device=device) % args.nclasses
    labels2 = labels.clone()

    print("\n[Step 1] Single batch forward")
    outputs1, outputs2 = model(drone, satellite)
    print("drone_cls_shape:", tuple(outputs1[0].shape))
    print("drone_embedding_shape:", tuple(outputs1[1].shape))
    print("satellite_cls_shape:", tuple(outputs2[0].shape))
    print("satellite_embedding_shape:", tuple(outputs2[1].shape))

    print("\n[Step 2] Single batch forward + loss")
    loss, cls_loss, triplet_loss, kl_loss = criterion(outputs1, outputs2, labels, labels2)
    print("loss_mode:", loss_mode)
    print("total_loss:", float(loss.detach()))
    print("cls_loss:", float(cls_loss.detach()))
    print("triplet_loss:", float(triplet_loss.detach()))
    print("kl_loss:", float(kl_loss.detach()))

    print("\n[Step 3] Single batch backward")
    optimizer.zero_grad()
    loss.backward()
    total, has_grad, grad_norm = grad_stats(model)
    optimizer.step()
    print("params_total:", total)
    print("params_with_grad:", has_grad)
    print("mean_abs_grad_sum:", grad_norm)

    if has_grad == 0:
        raise SystemExit("Backward failed: no gradients found.")

    print("\n[Check]")
    print("full_model_forward_ok:", True)
    print("loss_ok:", torch.isfinite(loss).item())
    print("backward_ok:", has_grad > 0)


if __name__ == "__main__":
    main()
