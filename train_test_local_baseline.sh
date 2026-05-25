name="EfficientNet-B2"
root_dir="/home/cjr/GIT_REPO/Compare_Trial/Dataset/DenseUAV"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8
lr=0.003
backbone_lr=0.0009
head_lr=0.003
batchsize=8
sample_num=1
block=1
num_bottleneck=512
backbone="EfficientNet-B2" # SingleBranchCNN ：resnet50  senet   # timm =1.0.27 Convnext-T  Pvtv2b2
# SingleBranch ： ViTS-224  SwinB-224 
# SingleBranchSwin ： 
# EfficientNet-B2 EfficientNet-B3 EfficientNet-B5 EfficientNet-B6 vgg16 cvt13
head="SingleBranch" # SingleBranch / SingleBranchCNN / SingleBranchSwin
head_pool="global" # global avg max avg+max
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

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
                --block $block --lr $lr --backbone_lr $backbone_lr --head_lr $head_lr --num_worker $num_worker --head $head  --head_pool $head_pool \
                --num_bottleneck $num_bottleneck --backbone $backbone --h $h --w $w --batchsize $batchsize --load_from $load_from \
                --ra $ra --re $re --cj $cj --rr $rr --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
                --num_epochs $num_epochs

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../