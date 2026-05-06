#!/usr/bin/env bash
set -e

LOG=env_check/env_diagnose.log

{
echo "================ BASIC ================"
date
hostname
pwd
whoami
uname -a
cat /etc/os-release 2>/dev/null || true

echo
echo "================ CONDA ================"
which conda || true
conda --version || true
conda info || true
conda config --show channels || true
conda config --show channel_priority || true
conda env list || true

echo
echo "================ PYTHON / PIP ================"
which python || true
python -V || true
which pip || true
pip -V || true
python -m pip -V || true

echo
echo "================ NVIDIA / CUDA TOOLCHAIN ================"
nvidia-smi || true
which nvcc || true
nvcc --version || true
which gcc || true
gcc --version || true
which g++ || true
g++ --version || true
ldd --version || true

echo
echo "================ PYTORCH CHECK ================"
python - <<'PY'
import os, sys
print("python executable:", sys.executable)
print("python version:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch file:", torch.__file__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cudnn version:", torch.backends.cudnn.version())
    print("cxx11 abi:", getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        print("torch CUDA_HOME:", CUDA_HOME)
    except Exception as e:
        print("CUDA_HOME check failed:", repr(e))
    if torch.cuda.is_available():
        print("gpu count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch import failed:", repr(e))
PY

echo
echo "================ IMPORTANT PACKAGES ================"
python -m pip list | grep -Ei "torch|cuda|cudnn|triton|selective|scan|mamba|causal|einops|timm|numpy|scipy|opencv|pillow|mmcv|detectron|ninja|packaging|wheel|setuptools|cython|pybind" || true

echo
echo "================ SELECTIVE / MAMBA IMPORT CHECK ================"
python - <<'PY'
import importlib
import importlib.metadata as md

mods = [
    "selective_scan",
    "selective_scan_cuda",
    "selective_scan_cuda_core",
    "selective_scan_cuda_oflex",
    "selective_scan_cuda_oflex_rh",
    "mamba_ssm",
    "causal_conv1d",
]

for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"[OK] import {m}")
        print("     file:", getattr(mod, "__file__", None))
    except Exception as e:
        print(f"[FAIL] import {m}: {type(e).__name__}: {e}")

print()
print("Installed distributions containing selective/scan/mamba/causal:")
for dist in md.distributions():
    name = dist.metadata.get("Name", "")
    low = name.lower()
    if any(k in low for k in ["selective", "scan", "mamba", "causal"]):
        print("-", name, dist.version, "at", dist.locate_file(""))
PY

echo
echo "================ CONDA LIST IMPORTANT ================"
conda list | grep -Ei "python|pip|torch|cuda|cudnn|triton|numpy|scipy|opencv|pillow|gcc|gxx|cmake|ninja|setuptools|wheel|packaging|selective|mamba|causal|timm|einops" || true

echo
echo "================ FULL CONDA LIST ================"
conda list || true

echo
echo "================ FULL PIP FREEZE ================"
python -m pip freeze || true

echo
echo "================ GIT INFO ================"
git rev-parse --show-toplevel 2>/dev/null || true
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true

} 2>&1 | tee "$LOG"

echo
echo "Saved to $LOG"
