#!/usr/bin/env bash
# One-time setup to train (and run) Virgo on your own NVIDIA GPU machine (Ubuntu + conda).
#
#   cd /opt && git clone https://github.com/nangmrr752/virgo-models
#   bash /opt/virgo-models/scripts/local_setup.sh
#
# The code is wherever you cloned it (here: /opt/virgo-models). Trained models, logs and model
# downloads go in VIRGO_HOME (default: a "virgo-data" folder next to the code, e.g. /opt/virgo-data).
# Creates the conda environment "virgo". See LOCAL.md.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${VIRGO_HOME:-$(dirname "$CODE")/virgo-data}"
mkdir -p "$ROOT"/{out,logs,hf-cache}
command -v nvidia-smi >/dev/null || { echo "❌ No NVIDIA driver (nvidia-smi). Install one first: sudo ubuntu-drivers autoinstall"; exit 1; }
command -v conda >/dev/null || { echo "❌ conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/"; exit 1; }
git -C "$CODE" pull -q || true
cd "$CODE"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^virgo " || conda create -y -q -n virgo python=3.11
conda activate virgo
export HF_HOME="$ROOT/hf-cache"
echo "⏳ Installing PyTorch (CUDA 12.1) and Virgo's training packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements-train.txt
pip uninstall -y -q torchao 2>/dev/null || true   # conflicts with newer peft

# Settings every Virgo command here uses: source $CODE/env.sh (git ignores it)
cat > "$CODE/env.sh" <<ENV
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate virgo
export VIRGO_HOME="$ROOT"
export HF_HOME="$ROOT/hf-cache"
export WORKSPACE="$ROOT"
cd "$CODE"
ENV

python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('✅ GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9), 'GB')"
python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1 || python -c "from huggingface_hub import login; login()"
echo "✅ Ready. Train with:  bash $CODE/scripts/train_local.sh bayon 12b   (models and logs go in $ROOT)"
