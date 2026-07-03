#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-denseuav_vmamba}"
BASE_ENV="${BASE_ENV:-denseuav}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/cjr/GIT_REPO/DenseUAV__BackboneChange}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists; reusing it."
else
    echo "Cloning '$BASE_ENV' -> '$ENV_NAME' to reduce downloads..."
    conda create --clone "$BASE_ENV" -n "$ENV_NAME" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

# Remove cloned cu113 PyTorch packages before installing cu118 wheels.
conda remove -y pytorch torchvision torchaudio cudatoolkit || true
python -m pip uninstall -y torch torchvision torchaudio || true

python -m pip install \
    -r "$PROJECT_ROOT/conda_env/vmamba/requirements_vmamba_cu118.txt" \
    --default-timeout 1000 \
    --retries 20

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT/third_party/vmamba/kernels/selective_scan"
rm -rf build selective_scan.egg-info
python -m pip install .

cd "$PROJECT_ROOT"
python - <<'PY'
import torch
import selective_scan_cuda_oflex

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("selective_scan_cuda_oflex: ok")
PY

echo "Environment '$ENV_NAME' is ready."
