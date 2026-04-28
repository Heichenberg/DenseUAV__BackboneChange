set -e

name="VMamba-Tiny_GeoTokenHeadV1-CELOSS+TripletLoss+klloss"
root_dir="/home/cjr/GIT_REPO/Compare_Trial/Data/DenseUAV"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.01
batchsize=8
sample_num=1
block=1
num_bottleneck=512
backbone="VMamba-Tiny" # VMamba-Tiny VMamba-Small VMamba-Base
backbone_weight="pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth" #默认为空，如果填写了按照填写的读取
head="GeoTokenHeadV1"
head_pool="avg" # global avg max avg+max
cls_loss="CELoss" # CELoss FocalLoss
feature_loss="WeightedSoftTripletLoss" # TripletLoss HardMiningTripletLoss WeightedSoftTripletLoss ContrastiveLoss
kl_loss="KLLoss" # KLLoss
h=224
w=224
load_from="no"
ra="satellite"  # random affine
re="satellite"  # random erasing
cj="no"  # color jitter
rr="uav"  # random rotate
num_epochs=120

# 新增参数：尽量只保留这三个
short_train=${SHORT_TRAIN:-false}
short_train_epochs=${SHORT_EPOCHS:-60}
token_mode=${TOKEN_MODE:-GCRS}

case "$token_mode" in
    G)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_G"
        ;;
    GC)
        backbone="VMamba-Tiny-_GeoTokenHeadV1_GC"
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
    *)
        echo "Unsupported TOKEN_MODE: $token_mode"
        echo "Supported: G, GC, GR, GS, GCR, GCRS"
        exit 1
        ;;
esac

if [ "$short_train" = "true" ]; then
    num_epochs=$short_train_epochs
fi

[ -n "$name" ] || name="${backbone}_${head}"
if [ "$token_mode" != "GCRS" ]; then
    name="${name}_${token_mode}"
fi
if [ "$short_train" = "true" ]; then
    name="${name}_short${short_train_epochs}"
fi

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --backbone_weight $backbone_weight --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../
