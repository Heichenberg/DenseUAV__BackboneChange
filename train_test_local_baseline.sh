set -e
set -o pipefail

name="VMamba-Tiny+GC5"
root_dir="/home/cjr/GIT_REPO/Dataset/DenseUAV"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.01
batchsize=8
sample_num=1
block=1
num_bottleneck=512
train_num_epoch=10
train_strategy="origin"

# ================= Backbone / Head 对应关系 =================
# 训练时主要改下面三个变量：name / backbone / head。
# 本脚本显式传 --train_strategy origin，只跑普通 origin 训练模式。
#
# CNN feature map 输出，形状通常是 [B, C, H, W]：
#   backbone: resnet50, senet, RKNet
#   推荐 head: SingleBranchCNN
#   推荐 head_pool: global
#
# Transformer token 输出，形状通常是 [B, N, C] 或 [B, C]：
#   backbone: ViTS-224, ViTS-384, ViTB-224, DeitS-224, DeitB-224
#   推荐 head: SingleBranchSwin
#   推荐 head_pool: avg
#
# Swin / 兼容 NHWC 或 token 输出：
#   backbone: SwinB-224, Swinv2S-256, Swinv2T-256, VMamba-Tiny, VMamba-Small, VMamba-Base
#   推荐 head: SingleBranchSwin
#   推荐 head_pool: avg
#
# 其他 timm CNN / 特征图 backbone：
#   backbone: EfficientNet-B2, EfficientNet-B3, EfficientNet-B5, EfficientNet-B6, Convnext-T, Pvtv2b2, vgg16, cvt13
#   推荐 head: SingleBranchSwin
#   推荐 head_pool: avg
#
# 你未来要跑的四个例子：
#   RKNet:           backbone="RKNet"           head="SingleBranchCNN"  head_pool="global"
#   DeitS-224:       backbone="DeitS-224"       head="SingleBranchSwin" head_pool="avg"
#   SwinB-224:       backbone="SwinB-224"       head="SingleBranchSwin" head_pool="avg"
#   EfficientNet-B2: backbone="EfficientNet-B2" head="SingleBranchSwin" head_pool="avg"
#
# 如果 backbone.py 里为某个 backbone 配了本地预训练权重，直接写 backbone 名称即可；
# 如果想手动指定权重，需要在 python train.py 命令里额外加 --backbone_weight /path/to/xxx.pth。
backbone="VMamba-Tiny"
head="SingleBranchSwin"
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

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --train_strategy $train_strategy --num_epochs $train_num_epoch

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../
