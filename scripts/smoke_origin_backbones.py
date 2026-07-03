import argparse
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.make_dataloader import make_dataset
from losses.loss import Loss
from models.taskflow import make_model


BACKBONES = ("RKNet", "DeitS-224", "SwinB-224", "EfficientNet-B2")


def make_opt(backbone, args):
    return SimpleNamespace(
        gpu_ids="-1",
        name="smoke_{}".format(backbone.replace("/", "_")),
        data_dir=args.data_dir,
        num_worker=args.num_worker,
        batchsize=args.batchsize,
        pad=0,
        h=args.h,
        w=args.w,
        rr="",
        ra="",
        re="",
        cj="",
        disable_hflip=True,
        erasing_p=0.3,
        warm_epoch=0,
        lr=0.01,
        backbone_lr=0.01,
        head_lr=0.01,
        DA=False,
        droprate=0.5,
        autocast=False,
        block=2,
        cls_loss="CELoss",
        feature_loss="no",
        kl_loss="no",
        sample_num=1,
        train_strategy="origin",
        dss_gps_file="",
        dss_start_epoch=0,
        dss_gds_topk=64,
        dss_gds_ratio=0.5,
        dss_fss_ratio=0.0,
        dss_fss_topk=64,
        dss_fss_start_epoch=10,
        dss_fss_samples_per_id=1,
        dss_rs_ratio=0.5,
        dss_fss_update_interval=10,
        dss_cache_dir="",
        dss_stage_mode="fixed",
        dss_ce_threshold=2.0,
        dss_plateau_delta=0.05,
        dss_plateau_patience=3,
        dss_ema_momentum=0.9,
        num_epochs=1,
        num_bottleneck=512,
        load_from="",
        backbone=backbone,
        backbone_weight="",
        head="SingleBranchSwin",
        head_pool="avg",
        max_train_batches=1,
        max_total_batches=1,
        max_ids=args.max_ids,
        id_subset_file="",
        seed=666,
        resume="",
        eval_interval=1,
        save_latest=False,
        save_best=False,
        best_metric="satellite_acc",
        use_gpu=torch.cuda.is_available() and not args.cpu,
        nclasses=0,
    )


def smoke_one(backbone, args):
    opt = make_opt(backbone, args)
    dataloader, class_names, _ = make_dataset(opt)
    opt.nclasses = len(class_names)
    model = make_model(opt)
    if opt.use_gpu:
        model = model.cuda()
    model.train(True)
    nnloss = Loss(opt)
    data, data3 = next(iter(dataloader))
    inputs, labels = data
    inputs3, labels3 = data3
    if opt.use_gpu:
        inputs = inputs.cuda()
        inputs3 = inputs3.cuda()
        labels = labels.cuda()
        labels3 = labels3.cuda()
    outputs, outputs2 = model(inputs, inputs3)
    loss, cls_loss, triplet_loss, kl_loss = nnloss(outputs, outputs2, labels, labels3)
    if not torch.isfinite(loss):
        raise RuntimeError("{} produced non-finite loss: {}".format(backbone, loss.item()))
    loss.backward()
    print(
        "{} ok: loss={:.4f}, cls={:.4f}, triplet={:.4f}, kl={:.4f}".format(
            backbone, loss.item(), cls_loss.item(), triplet_loss.item(), kl_loss.item()
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke-test origin training forward/backward for selected backbones.")
    parser.add_argument("--data_dir", default="/home/cjr/GIT_REPO/Dataset/DenseUAV/train")
    parser.add_argument("--backbones", nargs="*", default=list(BACKBONES), choices=BACKBONES)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--max_ids", type=int, default=8)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--h", type=int, default=224)
    parser.add_argument("--w", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for backbone in args.backbones:
        smoke_one(backbone, args)


if __name__ == "__main__":
    main()
