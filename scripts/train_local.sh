#!/usr/bin/env bash
# Trains a Virgo chat model on this machine's GPU and uploads it (private) to Hugging Face.
#
#   bash scripts/train_local.sh bayon 12b     # Virgo-1.0-Bayon      (also: bayon 4b → Virgo-1.0-Bayon-4B)
#   bash scripts/train_local.sh angkor 12b    # Virgo-1.0-Angkor-12B (also: angkor 4b → Virgo-1.0-Angkor)
#
# Needs scripts/local_setup.sh first. Output in $VIRGO_HOME/out/<model>, log in $VIRGO_HOME/logs.
# Extra options go to chat/train.py, e.g. --epochs 2 or --max-len 768 (if it runs out of memory).
set -euo pipefail
ROOT="${VIRGO_HOME:-/opt/virgo}"
source "$ROOT/env.sh"
WHICH="${1:-bayon}"; SIZE="${2:-12b}"; shift $(( $# > 2 ? 2 : $# ))
case "$SIZE" in 12b) BASE=google/gemma-3-12b-it ;; 4b) BASE=google/gemma-3-4b-it ;; *) echo "Size: 12b or 4b"; exit 1 ;; esac
case "$WHICH-$SIZE" in
  bayon-12b) NAME=Virgo-1.0-Bayon ;; bayon-4b) NAME=Virgo-1.0-Bayon-4B ;;
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
  python -u chat/train.py --base "$BASE" --data "$DATA" --out "$OUT" --epochs 3 "$@"
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
