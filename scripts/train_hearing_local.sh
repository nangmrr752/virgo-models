#!/usr/bin/env bash
# Trains Virgo's hearing (Whisper, Khmer speech to text) on this machine's GPU and uploads it to your
# private Angkor-1.0-STT repo (standard names: scripts/names.py), only if it scores better than the current one.
#
#   bash scripts/train_local.sh hearing                         # all of Whisper, 6000 steps (~2-4 h on a 4090)
#   bash scripts/train_local.sh hearing --extra <folder>        # plus your own recordings (.wav + metadata.csv)
#   VIRGO_HEARING_STEPS=3000 bash scripts/train_local.sh hearing
#   bash scripts/train_local.sh hearing --multilingual          # ONE model for Khmer + English + others (no second Whisper)
#   bash scripts/train_local.sh hearing-1.1                     # Angkor-1.1-STT: Angkor-1.0-STT trained further on more
#                                                               #   Khmer audio (the most used Khmer speech datasets on
#                                                               #   Hugging Face, found automatically); uploaded only if it
#                                                               #   beats Angkor-1.0-STT
#   VIRGO_HEARING_BASE=metythorn/whisper-large-v3-turbo bash scripts/train_local.sh hearing
#                                                               # start from another Whisper (e.g. a Khmer fine-tune)
#
# Starts from your current Angkor-1.0-STT when there is one (else Whisper large-v3-turbo),
# prints the Khmer character error rate (CER) before and after. Stop the Virgo server first: both
# need the GPU.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"   # your server (env.sh), Colab or Kaggle
ROOT="$VIRGO_HOME"
cd "$CODE"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
REPO=$(python scripts/names.py resolve Angkor-1.0-STT 2>/dev/null || echo "$ME/Virgo-1.0-Angkor-Hearing")  # old name until renamed
NAME=Angkor-1.0-STT
EXTRA=()
if [ "${1:-}" = "--v1.1" ]; then  # Angkor-1.1-STT: continues from Angkor-1.0-STT, must beat it
  shift
  NAME=Angkor-1.1-STT
  CURRENT=$REPO
  REPO=$(python scripts/names.py resolve Angkor-1.1-STT 2>/dev/null || echo "$ME/Angkor-1.1-STT")
  [ -n "${VIRGO_HEARING_BASE:-}" ] || VIRGO_HEARING_BASE=$CURRENT
  EXTRA=(--current "$CURRENT" --more-khmer "${VIRGO_HEARING_DATASETS:-8}" --hf-max "${VIRGO_HEARING_MAX:-15000}")
fi
BASE=${VIRGO_HEARING_BASE:-}
# One model for every language (--multilingual): start from OpenAI's Whisper, which hasn't forgotten any.
[ -n "$BASE" ] || BASE=$(python -c "from huggingface_hub import HfApi; print('$REPO' if HfApi().file_exists('$REPO', 'config.json') else 'openai/whisper-large-v3-turbo')" 2>/dev/null || echo openai/whisper-large-v3-turbo)
# (a Khmer-only hearing has forgotten the other languages; a multilingual one continues)
if [[ " $* " == *" --multilingual "* ]] && [ -z "${VIRGO_HEARING_BASE:-}" ] && \
   ! python -c "import sys; sys.path.insert(0, 'speech'); from virgo_speech_marker import multilingual; sys.exit(0 if multilingual('$BASE') else 1)"; then
  BASE=openai/whisper-large-v3-turbo
fi
LOG="$ROOT/logs/$NAME-$(date +%Y%m%d-%H%M).log"
mkdir -p "$ROOT/logs"
{
  echo "== Training Virgo's hearing from $BASE ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # Audio libraries the hearing training needs (the chat training environment doesn't have them).
  python -c "import soundfile, librosa" 2>/dev/null || python -m pip install -q soundfile librosa
  # All of Whisper (not a small add-on) on a 20 GB+ GPU: it learns Khmer much better. On a 16 GB T4
  # (Colab/Kaggle) a LoRA add-on instead, so it fits.
  if [ "${GPU_GB:-0}" -ge 20 ]; then HOW=(--full --lr 1e-5); else HOW=(--lr 1e-4); echo "16 GB GPU: training a LoRA add-on"; fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u speech/finetune_stt.py \
    --base "$BASE" "${HOW[@]}" --steps "${VIRGO_HEARING_STEPS:-$([ "$NAME" = Angkor-1.1-STT ] && echo 10000 || echo 6000)}" --batch 8 \
    --work "$ROOT/hearing-train" --out "$ROOT/out/$NAME" --push "$REPO" ${EXTRA[@]+"${EXTRA[@]}"} "$@"
  echo "== Done: restart the Virgo server to use it (docker compose up -d) =="
} 2>&1 | tee "$LOG"
