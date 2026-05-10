set -e

name="VMamba-Tiny_GeoTokenHeadV1-CELOSS+HardMiningTripletLoss+klloss--GC5R_D192-fss45"
root_dir="/home/cjr/GIT_REPO/Compare_Trial/Dataset/DenseUAV"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.005
backbone_lr=${BACKBONE_LR:-0.005}
head_lr=${HEAD_LR:-0.01}
batchsize=16
sample_num=2
train_strategy=${TRAIN_STRATEGY:-dss}  # origin dss
dss_start_epoch=${DSS_START_EPOCH:-5}
dss_gds_topk=${DSS_GDS_TOPK:-32}
dss_gds_ratio=${DSS_GDS_RATIO:-0.4}
dss_fss_ratio=${DSS_FSS_RATIO:-0.2}
dss_rs_ratio=${DSS_RS_RATIO:-0.4}
dss_fss_topk=${DSS_FSS_TOPK:-32}

dss_stage_mode=${DSS_STAGE_MODE:-loss_adaptive}#fixed loss_adaptive
dss_fss_start_epoch=${DSS_FSS_START_EPOCH:-45}
dss_fss_update_interval=${DSS_FSS_UPDATE_INTERVAL:-10}
dss_fss_samples_per_id=${DSS_FSS_SAMPLES_PER_ID:-1}


dss_gps_file=${DSS_GPS_FILE:-"$root_dir/Dense_GPS_train.txt"}
dss_cache_dir=${DSS_CACHE_DIR:-"$root_dir/dss_cache"}

dss_ce_threshold=${DSS_CE_THRESHOLD:-2.0}
dss_plateau_delta=${DSS_PLATEAU_DELTA:-0.05}
dss_plateau_patience=${DSS_PLATEAU_PATIENCE:-3}
dss_ema_momentum=${DSS_EMA_MOMENTUM:-0.9}
block=1
num_bottleneck=512
backbone="VMamba-Tiny" # VMamba-Tiny VMamba-Small VMamba-Base
backbone_weight="pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth" #默认为空，如果填写了按照填写的读取
head="GeoTokenHeadV1"
head_pool="avg" # global avg max avg+max
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="HardMiningTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
kl_loss="KLLoss" # KLLoss
h=224
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate
num_epochs=120

# 短训参数
short_train=${SHORT_TRAIN:-false}
short_train_epochs=${SHORT_EPOCHS:-60}



#训练token mode
token_mode=${TOKEN_MODE:-GC5R_D192}
case "$token_mode" in
    C)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_C"
        ;;
    C5)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_C5"
        ;;
    G)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_G"
        ;;
    GC)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GC"
        ;;
    GC5)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GC5"
        ;;
    GR)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GR"
        ;;
    GS)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GS"
        ;;
    GCR)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GCR"
        ;;
    GCRS)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GCRS"
        ;;
    GCR5_D384)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GCR5_D384"
        ;;
    GCR5_D192)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192"
        ;;
    GCR5_D192_GATE)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GCR5_D192_GATE"
        ;;
    GC5R_D192)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GC5R_D192"
        ;;
    GC5R_D192_GATE)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GC5R_D192_GATE"
        ;;
    *)
        echo "Unsupported TOKEN_MODE: $token_mode"
        echo "Supported: C, C5, G, GC, GC5, GR, GS, GCR, GCRS, GCR5_D384, GCR5_D192, GCR5_D192_GATE, GC5R_D192, GC5R_D192_GATE"
        exit 1
        ;;
esac

if [ "$short_train" = "true" ] || [ "$short_train" = "1" ]; then
    num_epochs=$short_train_epochs
fi

[ -n "$name" ] || name="${backbone}_${head}"
if [ "$token_mode" != "GCRS" ] && [[ "$name" != *"$token_mode"* ]]; then
    name="${name}_${token_mode}"
fi
if [ "$short_train" = "true" ] || [ "$short_train" = "1" ]; then
    name="${name}_short${short_train_epochs}"
fi
if [ "$train_strategy" = "dss" ]; then
    name="${name}_dssv1"
fi
if [ "$backbone_lr" != "0" ] || [ "$head_lr" != "0" ]; then
    name="${name}_blr${backbone_lr}_hlr${head_lr}"
fi

train_cmd="python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --train_strategy $train_strategy --dss_gps_file $dss_gps_file --dss_start_epoch $dss_start_epoch --dss_gds_topk $dss_gds_topk --dss_gds_ratio $dss_gds_ratio --dss_fss_ratio $dss_fss_ratio --dss_fss_topk $dss_fss_topk --dss_fss_start_epoch $dss_fss_start_epoch --dss_fss_samples_per_id $dss_fss_samples_per_id --dss_rs_ratio $dss_rs_ratio --dss_fss_update_interval $dss_fss_update_interval --dss_cache_dir $dss_cache_dir \
                --dss_stage_mode $dss_stage_mode --dss_ce_threshold $dss_ce_threshold --dss_plateau_delta $dss_plateau_delta --dss_plateau_patience $dss_plateau_patience --dss_ema_momentum $dss_ema_momentum \
                --block $block --lr $lr --backbone_lr $backbone_lr --head_lr $head_lr --num_worker $num_worker --head $head --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --backbone_weight $backbone_weight --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$train_cmd"
    exit 0
fi

$train_cmd

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
python plot_ema_loss.py
cd ../../
