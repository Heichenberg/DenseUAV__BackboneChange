#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

level="C" # A B C
name="VMamba-Tiny-Vector_singlebranch"
root_dir="/home/cjr/GIT_REPO/Compare_Trial/Data/DenseUAV"
data_dir="$root_dir/train"
gpu_ids=0
num_worker=""
lr=0.01
batchsize=""
sample_num=1
block=1
num_bottleneck=512
backbone="VMamba-Tiny-Vector" # VMamba-Tiny VMamba-Small VMamba-Base
backbone_weight=""
head="SingleBranch"
head_pool="avg" # global avg max avg+max
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
kl_loss="KLLoss" # KLLoss
h=224
w=224
load_from="no"
ra="satellite"
re="satellite"
cj="no"
rr="uav"
erasing_p=0.3
num_epochs=""
max_train_batches=0
max_total_batches=0
max_ids=0
id_subset_file=""
seed=666
disable_autocast=0
disable_hflip=0
resume=""
eval_interval=1
best_metric="satellite_acc"
level_b_ids=8 # 8 16 32
backbone_default_name="${backbone}_${head}"
backbone_default_batchsize="${batchsize:-8}"

case "$level" in
    "A")
        [ -n "$name" ] || name="${backbone_default_name}_levelA_smoke"
        [ -n "$batchsize" ] || batchsize=2
        [ -n "$num_worker" ] || num_worker=2
        [ -n "$num_epochs" ] || num_epochs=1
        [ "$max_train_batches" -gt 0 ] || max_train_batches=20
        disable_autocast=1
        ;;
    "B")
        [ -n "$name" ] || name="${backbone_default_name}_levelB_overfit"
        [ -n "$batchsize" ] || batchsize=$level_b_ids
        [ -n "$num_worker" ] || num_worker=0
        [ -n "$num_epochs" ] || num_epochs=50
        [ "$max_ids" -gt 0 ] || max_ids=$level_b_ids
        name="${name}_ids${max_ids}"
        lr=0.001
        ra=""
        re=""
        cj="no"
        rr=""
        erasing_p=0
        feature_loss="no"
        kl_loss="no"
        disable_autocast=1
        disable_hflip=1
        ;;
    "C")
        [ -n "$name" ] || name="${backbone_default_name}_levelC_short"
        [ -n "$batchsize" ] || batchsize=$backbone_default_batchsize
        [ -n "$num_worker" ] || num_worker=0
        [ -n "$num_epochs" ] || num_epochs=5
        lr=0.001
        ra=""
        re=""
        cj="no"
        rr=""
        erasing_p=0
        feature_loss="no"
        kl_loss="no"
        disable_autocast=1
        disable_hflip=1
        ;;
    *)
        echo "Unsupported level: $level"
        exit 1
        ;;
esac

cmd=(
    python train.py
    --name "$name"
    --data_dir "$data_dir"
    --gpu_ids "$gpu_ids"
    --sample_num "$sample_num"
    --block "$block"
    --lr "$lr"
    --num_worker "$num_worker"
    --head "$head"
    --head_pool "$head_pool"
    --num_bottleneck "$num_bottleneck"
    --backbone "$backbone"
    --backbone_weight "$backbone_weight"
    --h "$h"
    --w "$w"
    --batchsize "$batchsize"
    --load_from "$load_from"
    --ra "$ra"
    --re "$re"
    --cj "$cj"
    --rr "$rr"
    --erasing_p "$erasing_p"
    --cls_loss "$cls_loss"
    --feature_loss "$feature_loss"
    --kl_loss "$kl_loss"
    --num_epochs "$num_epochs"
    --max_train_batches "$max_train_batches"
    --max_total_batches "$max_total_batches"
    --max_ids "$max_ids"
    --id_subset_file "$id_subset_file"
    --seed "$seed"
    --resume "$resume"
    --eval_interval "$eval_interval"
    --best_metric "$best_metric"
)

if [ "$disable_autocast" -eq 1 ]; then
    cmd+=(--disable_autocast)
fi
if [ "$disable_hflip" -eq 1 ]; then
    cmd+=(--disable_hflip)
fi

cd "$project_root"
printf 'Running level %s with command:\n' "$level"
printf '  %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
