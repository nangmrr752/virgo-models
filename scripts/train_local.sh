#!/usr/bin/env bash
# Trains a Virgo chat model on this machine's GPU and uploads it (private) to Hugging Face.
#
#   bash scripts/train_local.sh bayon 27b     # Virgo-1.0-Bayon on Gemma 3 27B, smarter than Angkor (default)
#                                            #   (also: bayon 12b → Virgo-1.0-Bayon-12B, bayon 4b → Virgo-1.0-Bayon-4B)
#   bash scripts/train_local.sh hearing       # Virgo's hearing (Whisper, Khmer), see train_hearing_local.sh
#   bash scripts/train_local.sh voice         # Virgo's live voice (VoxCPM2), see train_voice_local.sh
#   bash scripts/train_local.sh angkor 12b    # Virgo-1.0-Angkor-12B (also: angkor 4b → Virgo-1.0-Angkor)
#
# Needs scripts/local_setup.sh first. Output in <data>/out/<model>, log in <data>/logs (data: /opt/virgo-data
# when the code is in /opt/virgo-models).
# Extra options go to chat/train.py, e.g. --epochs 2 or --max-len 768 (if it runs out of memory).
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"   # your server (env.sh), Colab or Kaggle
ROOT="$VIRGO_HOME"
[ "${1:-}" = voice ] && { shift; exec bash "$CODE/scripts/train_voice_local.sh" "$@"; }  # Virgo's live voice
[ "${1:-}" = hearing ] && { shift; exec bash "$CODE/scripts/train_hearing_local.sh" "$@"; }  # Virgo's hearing (Whisper)
WHICH="${1:-bayon}"; DEFAULT_SIZE=12b; [ "$WHICH" = bayon ] && DEFAULT_SIZE=27b
# Smaller GPUs (Colab/Kaggle T4, 16 GB): 27B doesn't fit, so Bayon defaults to 12B there.
[ "$DEFAULT_SIZE" = 27b ] && [ "${GPU_GB:-0}" -lt 22 ] && DEFAULT_SIZE=12b
SIZE="${2:-$DEFAULT_SIZE}"; shift $(( $# > 2 ? 2 : $# ))
if [ "$SIZE" = 27b ] && [ "${GPU_GB:-0}" -lt 22 ]; then
  echo "❌ 27B needs a 24 GB+ GPU (this one has ${GPU_GB} GB). Use: bash scripts/train_local.sh bayon 12b"; exit 1
fi
case "$SIZE" in 27b) BASE=google/gemma-3-27b-it ;; 12b) BASE=google/gemma-3-12b-it ;; 4b) BASE=google/gemma-3-4b-it ;; *) echo "Size: 27b, 12b or 4b"; exit 1 ;; esac
EXTRA=""; [ "$SIZE" = 27b ] && EXTRA="--max-len 512"  # 27B in 4 bits fills most of a 24 GB GPU
[ "$SIZE" = 12b ] && [ "${GPU_GB:-0}" -lt 22 ] && EXTRA="--max-len 512"  # 12B on a 16 GB T4
case "$WHICH-$SIZE" in
  bayon-27b) NAME=Virgo-1.0-Bayon ;; bayon-12b) NAME=Virgo-1.0-Bayon-12B ;; bayon-4b) NAME=Virgo-1.0-Bayon-4B ;;
  angkor-12b) NAME=Virgo-1.0-Angkor-12B ;; angkor-4b) NAME=Virgo-1.0-Angkor ;;
  *) echo "Model: bayon or angkor"; exit 1 ;;
esac
OUT="$ROOT/out/$NAME"
LOG="$ROOT/logs/$NAME-$(date +%Y%m%d-%H%M).log"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
git pull -q

{
  echo "== Training $NAME on $BASE ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # Extra examples from Gemma 3 27B (chat/distill_vertex.ipynb), when you've made them.
  python -c "from huggingface_hub import hf_hub_download as d; d('$ME/Virgo-1.0-Angkor-Data', 'distilled_gemma27b.jsonl', repo_type='dataset', local_dir='chat/data')" >/dev/null 2>&1 \
    && echo "➕ Using the Gemma 27B examples" || echo "No Gemma 27B examples yet"
  DATA="chat/data/*.jsonl"
  if [ "$WHICH" = bayon ]; then  # the same data, but Virgo calls itself Virgo-1.0-Bayon
    rm -rf chat/data_bayon && mkdir -p chat/data_bayon
    for f in chat/data/*.jsonl; do sed 's/Virgo-1.0-Angkor/Virgo-1.0-Bayon/g' "$f" > "chat/data_bayon/$(basename "$f")"; done
    DATA="chat/data_bayon/*.jsonl"
  fi
  python scripts/check_data.py "$DATA"
  python -u chat/train.py --base "$BASE" --data "$DATA" --out "$OUT" --epochs 3 $EXTRA "$@"
  echo "== Score =="
  python chat/evaluate.py --adapter "$OUT" || echo "(scoring failed; the model is still saved)"
  echo "== Upload =="
  python - "$ME/$NAME" "$OUT" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
HfApi().create_repo(repo, private=True, exist_ok=True)
HfApi().upload_folder(folder_path=folder, repo_id=repo, commit_message=repo.split("/")[1], ignore_patterns=["checkpoint-*"])
PY
  echo "✅ $NAME saved: https://huggingface.co/$ME/$NAME"
} 2>&1 | tee "$LOG"
