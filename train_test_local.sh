set -e

name="VMamba-Tiny-_MixerHeadV1"
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
head="MixerHeadV1"
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
[ -n "$name" ] || name="${backbone}_${head}"

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --backbone_weight $backbone_weight --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../
