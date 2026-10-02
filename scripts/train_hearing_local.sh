#!/usr/bin/env bash
# Trains Virgo's hearing (Whisper, Khmer speech to text) on this machine's GPU and uploads it to your
# private Virgo-1.0-Angkor-Hearing repo, only if it scores better than the current one.
#
#   bash scripts/train_local.sh hearing                         # all of Whisper, 6000 steps (~2-4 h on a 4090)
#   bash scripts/train_local.sh hearing --extra <folder>        # plus your own recordings (.wav + metadata.csv)
#   VIRGO_HEARING_STEPS=3000 bash scripts/train_local.sh hearing
#
# Starts from your current Virgo-1.0-Angkor-Hearing when there is one (else Whisper large-v3-turbo),
# prints the Khmer character error rate (CER) before and after. Stop the Virgo server first: both
# need the GPU.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"   # your server (env.sh), Colab or Kaggle
ROOT="$VIRGO_HOME"
cd "$CODE"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
REPO="$ME/Virgo-1.0-Angkor-Hearing"
BASE=$(python -c "from huggingface_hub import HfApi; print('$REPO' if HfApi().file_exists('$REPO', 'config.json') else 'openai/whisper-large-v3-turbo')" 2>/dev/null || echo openai/whisper-large-v3-turbo)
LOG="$ROOT/logs/Virgo-1.0-Angkor-Hearing-$(date +%Y%m%d-%H%M).log"
mkdir -p "$ROOT/logs"
{
  echo "== Training Virgo's hearing from $BASE ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # Audio libraries the hearing training needs (the chat training environment doesn't have them).
  python -c "import soundfile, librosa" 2>/dev/null || python -m pip install -q soundfile librosa
  # All of Whisper (not a small add-on) on a 20 GB+ GPU: it learns Khmer much better. On a 16 GB T4
  # (Colab/Kaggle) a LoRA add-on instead, so it fits.
  if [ "${GPU_GB:-0}" -ge 20 ]; then HOW=(--full --lr 1e-5); else HOW=(--lr 1e-4); echo "16 GB GPU: training a LoRA add-on"; fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u speech/finetune_stt.py \
    --base "$BASE" "${HOW[@]}" --steps "${VIRGO_HEARING_STEPS:-6000}" --batch 8 \
    --work "$ROOT/hearing-train" --out "$ROOT/out/Virgo-1.0-Angkor-Hearing" --push "$REPO" "$@"
  echo "== Done: restart the Virgo server to use it (docker compose up -d) =="
} 2>&1 | tee "$LOG"
