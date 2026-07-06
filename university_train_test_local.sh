set -e

# University-1652 training/evaluation entry for DenseUAV__BackboneChange.
# Override any variable from the shell, for example:
#   NAME=my_exp BATCHSIZE=8 NUM_EPOCHS=120 bash university_train_test_local.sh

root_dir=${ROOT_DIR:-"/home/cjr/GIT_REPO/Dataset/University-Release"}
data_dir=${DATA_DIR:-"$root_dir/train"}
test_dir=${TEST_DIR:-"$root_dir/test"}

raw_name=${NAME:-"VMamba-MSGE+MGTRF"}
case "$raw_name" in
    UNIV_*) name="$raw_name" ;;
    *) name="UNIV_${raw_name}" ;;
esac

gpu_ids=${GPU_IDS:-0}
num_worker=${NUM_WORKER:-8}
lr=${LR:-0.001}
backbone_lr=${BACKBONE_LR:-0.0006}
head_lr=${HEAD_LR:-0.001}
batchsize=${BATCHSIZE:-8}
sample_num=${SAMPLE_NUM:-1}
num_epochs=${NUM_EPOCHS:-120}

backbone=${BACKBONE:-"VMamba-MSGE"}
backbone_weight=${BACKBONE_WEIGHT:-"pretrained/backbones/vmamba/tiny/vssm1_tiny_0230s_ckpt_epoch_264.pth"}
head=${HEAD:-"MGTRF"}
head_pool=${HEAD_POOL:-"global"}
mgtrf_num_queries=${MGTRF_NUM_QUERIES:-8}
mgtrf_proj_dim=${MGTRF_PROJ_DIM:-384}
mgtrf_drop_rate=${MGTRF_DROP_RATE:-0.0}
cls_loss=${CLS_LOSS:-"CELoss"}
feature_loss=${FEATURE_LOSS:-"WeightedSoftTripletLoss"}
kl_loss=${KL_LOSS:-"no"}

h=${H:-224}
w=${W:-224}
block=${BLOCK:-1}
num_bottleneck=${NUM_BOTTLENECK:-512}
load_from=${LOAD_FROM:-"no"}

# Keep University-1652 training simple and comparable by default.
# DSS is DenseUAV-GPS-oriented and should only be enabled with a suitable GPS file.
train_strategy=${TRAIN_STRATEGY:-origin}

ra=${RA:-"satellite"}
re=${RE:-"satellite"}
cj=${CJ:-"no"}
rr=${RR:-"uav"}
erasing_p=${ERASING_P:-0.3}

token_mode=${TOKEN_MODE:-GC3}
if [ "$backbone" = "VMamba-Tiny" ]; then
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
        GC3)
            backbone="VMamba-Tiny-_GeoTokenHeadV1_GC3"
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
        "")
            ;;
        *)
            echo "Unsupported TOKEN_MODE: $token_mode"
            echo "Supported: C, C5, G, GC, GC3, GC5, GR, GS, GCR, GCRS, GCR5_D384, GCR5_D192, GCR5_D192_GATE, GC5R_D192, GC5R_D192_GATE"
            exit 1
            ;;
    esac
else
    token_mode=""
fi

if [ "${SHORT_TRAIN:-false}" = "true" ] || [ "${SHORT_TRAIN:-0}" = "1" ]; then
    num_epochs=${SHORT_EPOCHS:-5}
    name="${name}_short${num_epochs}"
fi

if [ "$train_strategy" != "origin" ]; then
    name="${name}_${train_strategy}"
fi
if [ -n "$token_mode" ] && [[ "$name" != *"$token_mode"* ]]; then
    name="${name}_${token_mode}"
fi

backbone_weight_arg=""
if [ -n "$backbone_weight" ]; then
    backbone_weight_arg="--backbone_weight $backbone_weight"
fi

train_cmd="python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --sample_num $sample_num \
    --train_strategy $train_strategy \
    --block $block --lr $lr --backbone_lr $backbone_lr --head_lr $head_lr \
    --num_worker $num_worker --head $head --head_pool $head_pool \
    --mgtrf_num_queries $mgtrf_num_queries --mgtrf_proj_dim $mgtrf_proj_dim --mgtrf_drop_rate $mgtrf_drop_rate \
    --num_bottleneck $num_bottleneck --backbone $backbone $backbone_weight_arg \
    --h $h --w $w --batchsize $batchsize --load_from $load_from \
    --ra $ra --re $re --cj $cj --rr $rr --erasing_p $erasing_p \
    --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss \
    --num_epochs $num_epochs"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$train_cmd"
    echo "After training, evaluate with:"
    echo "  cd checkpoints/$name"
    echo "  python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth --mode 1"
    echo "  RESULT_MAT=pytorch_result_1.mat python evaluate_gpu.py"
    echo "  python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth --mode 2"
    echo "  RESULT_MAT=pytorch_result_2.mat python evaluate_gpu.py"
    exit 0
fi

$train_cmd

cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth --mode 1
RESULT_MAT=pytorch_result_1.mat python evaluate_gpu.py
mv pytorch_result_1.mat university_drone_to_satellite_result.mat
mv results.txt university_drone_to_satellite_results.txt

python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --batchsize 128 --checkpoint latest_checkpoint.pth --mode 2
RESULT_MAT=pytorch_result_2.mat python evaluate_gpu.py
mv pytorch_result_2.mat university_satellite_to_drone_result.mat
mv results.txt university_satellite_to_drone_results.txt

cd ../../
