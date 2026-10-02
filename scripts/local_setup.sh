#!/usr/bin/env bash
# One-time setup to train (and run) Virgo on your own NVIDIA GPU machine (Ubuntu + conda).
#
#   curl -fsSL https://raw.githubusercontent.com/nangmrr752/virgo-models/main/scripts/local_setup.sh | bash
#   (or: bash scripts/local_setup.sh)
#
# Everything goes in VIRGO_HOME (default /opt/virgo): the code, the conda environment "virgo", model
# downloads (hf-cache), trained models (out) and logs. See LOCAL.md.
set -euo pipefail
ROOT="${VIRGO_HOME:-/opt/virgo}"
mkdir -p "$ROOT"/{out,logs,hf-cache}
cd "$ROOT"
command -v nvidia-smi >/dev/null || { echo "❌ No NVIDIA driver (nvidia-smi). Install one first: sudo ubuntu-drivers autoinstall"; exit 1; }
command -v conda >/dev/null || { echo "❌ conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/"; exit 1; }
[ -d virgo-models/.git ] || git clone -q https://github.com/nangmrr752/virgo-models
git -C virgo-models pull -q

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^virgo " || conda create -y -q -n virgo python=3.11
conda activate virgo
export HF_HOME="$ROOT/hf-cache"
echo "⏳ Installing PyTorch (CUDA 12.1) and Virgo's training packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q -r virgo-models/requirements-train.txt
pip uninstall -y -q torchao 2>/dev/null || true   # conflicts with newer peft

# Settings every Virgo command here uses: source /opt/virgo/env.sh
cat > "$ROOT/env.sh" <<ENV
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate virgo
export VIRGO_HOME="$ROOT"
export HF_HOME="$ROOT/hf-cache"
export WORKSPACE="$ROOT"
cd "$ROOT/virgo-models"
ENV

python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('✅ GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9), 'GB')"
python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1 || python -c "from huggingface_hub import login; login()"
echo "✅ Ready. Train with:  bash $ROOT/virgo-models/scripts/train_local.sh bayon 12b"
