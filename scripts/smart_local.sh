#!/usr/bin/env bash
# Makes Virgo's chat models smarter on this machine's GPU (stop the Virgo server first: all of this
# needs the whole GPU). Called by train_local.sh:
#
#   bash scripts/train_local.sh distill [N]        # Bayon-1.0-27B answers N checked problems (default 12000, ~1 day)
#   bash scripts/train_local.sh dpo angkor 12b     # DPO on the good/bad pairs; uploads only if the hard test improves
#   bash scripts/train_local.sh all                # distill → Angkor 12B → DPO → Bayon 27B → DPO → hearing
#
# Every step saves to Hugging Face as it goes and continues where it stopped, so after a power cut or
# Ctrl+C just run the same command again. Logs: <data>/logs.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"
ROOT="$VIRGO_HOME"
cd "$CODE"
STEP="${1:-}"; shift || true
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
DATA_REPO=$(python scripts/names.py resolve Data-1.0 2>/dev/null || echo "$ME/Virgo-1.0-Angkor-Data")
mkdir -p "$ROOT/logs" chat/dpo

fetch() {  # fetch <file> <folder>: a file from Virgo's dataset, if it's there
  python -c "from huggingface_hub import hf_hub_download as d; d('$DATA_REPO', '$1', repo_type='dataset', local_dir='$2')" >/dev/null 2>&1 || true
}

need_gpu_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-0}" -gt 4000 ]; then
    echo "❌ The GPU is busy (${used} MB used): stop the Virgo server first: docker compose stop virgo"; exit 1
  fi
}

distill() {
  local target="${1:-12000}" teacher_dir="$ROOT/out/teacher-Bayon-1.0-27B"
  need_gpu_free
  fetch distilled_smart.jsonl chat/data
  fetch smart_pairs.jsonl chat/dpo
  # The teacher: Bayon-1.0-27B (Virgo's smartest), else plain Gemma 3 27B.
  local bayon adapter=""
  bayon=$(python scripts/names.py resolve Bayon-1.0-27B 2>/dev/null || true)
  if [ -n "$bayon" ] && python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$bayon', 'adapter_config.json') else 1)" 2>/dev/null; then
    python -c "from huggingface_hub import snapshot_download as s; s('$bayon', local_dir='$teacher_dir')" >/dev/null
    adapter="$teacher_dir"
  fi
  echo "== Distilling $target checked examples (teacher: ${bayon:-google/gemma-3-27b-it}) =="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/distill_smart.py --teacher google/gemma-3-27b-it \
    --adapter "$adapter" --target "$target" --hf-repo "$DATA_REPO"
}

name_for() {  # name_for angkor 12b → Angkor-1.0-12B
  echo "${1^}-1.0-${2^^}"
}

dpo() {
  local which="${1:-angkor}" size="${2:-12b}" name out hard_before hard_after repo
  name=$(name_for "$which" "$size")
  out="$ROOT/out/$name"
  need_gpu_free
  python -c "import trl" 2>/dev/null || python -m pip install -q "trl>=0.12"
  fetch smart_pairs.jsonl chat/dpo
  fetch smart_pairs_kaggle.jsonl chat/dpo  # made on Kaggle (chat/distill_kaggle.ipynb)
  cat chat/dpo/*.jsonl > /dev/null 2>&1 && [ -n "$(cat chat/dpo/*.jsonl 2>/dev/null | head -1)" ] || { echo "❌ No DPO pairs yet: run bash scripts/train_local.sh distill first"; exit 1; }
  repo=$(python scripts/names.py resolve "$name" 2>/dev/null || echo "$ME/$name")
  # The trained (SFT) model: this machine's copy, else the one on Hugging Face.
  [ -f "$out/adapter_config.json" ] || python -c "from huggingface_hub import snapshot_download as s; s('$repo', local_dir='$out')" >/dev/null
  local extra=""; [ "$size" = 27b ] && extra="--max-len 512"  # 27B in 4 bits fills most of a 24 GB GPU
  echo "== DPO for $name on $(cat chat/dpo/*.jsonl | wc -l) pairs =="
  python -u chat/evaluate.py --questions chat/eval/hard.jsonl --adapter "$out" --report "$ROOT/logs/$name-hard-before.json" | tail -15
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u chat/train_dpo.py --adapter "$out" --out "$out-dpo" $extra
  python -u chat/evaluate.py --questions chat/eval/hard.jsonl --adapter "$out-dpo" --report "$ROOT/logs/$name-hard-after.json" \
    --compare "$ROOT/logs/$name-hard-before.json" | tail -20
  hard_before=$(python -c "import json; print(json.load(open('$ROOT/logs/$name-hard-before.json'))['overall'])")
  hard_after=$(python -c "import json; print(json.load(open('$ROOT/logs/$name-hard-after.json'))['overall'])")
  # Better than this model before DPO, and than the best upload so far (a model train_local.sh didn't upload).
  local best="$hard_before"
  [ -f "$ROOT/logs/$name-hard-best.json" ] && best=$(python -c "import json; print(max($hard_before, json.load(open('$ROOT/logs/$name-hard-best.json'))['overall']))")
  if python -c "import sys; sys.exit(0 if $hard_after >= $best else 1)"; then
    python - "$repo" "$out-dpo" "$hard_before" "$hard_after" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder, before, after = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
HfApi().upload_folder(folder_path=folder, repo_id=repo, ignore_patterns=["checkpoint-*"],
                      commit_message=f"DPO: hard test {before:.1%} → {after:.1%}")
PY
    rm -rf "$out" && mv "$out-dpo" "$out"
    cp "$ROOT/logs/$name-hard-after.json" "$ROOT/logs/$name-hard-best.json"
    echo "✅ $name improved by DPO ($hard_before → $hard_after) and is uploaded to $repo"
  else
    echo "⚠️ DPO didn't beat the best $name on the hard test ($hard_after vs $best): the model on Hugging Face stays as it was"
  fi
}

run_logged() {  # run_logged <name> <command...>
  local log="$ROOT/logs/$1-$(date +%Y%m%d-%H%M).log"; shift
  "$@" 2>&1 | tee "$log"
}

case "$STEP" in
  distill) run_logged distill distill "$@" ;;
  dpo) run_logged "dpo-${1:-angkor}-${2:-12b}" dpo "$@" ;;
  all)
    # Each step continues where it left off, so running "all" again skips work already done.
    run_logged distill distill "${1:-12000}"
    run_logged Angkor-1.0-12B bash scripts/train_local.sh angkor 12b
    run_logged dpo-angkor-12b dpo angkor 12b
    run_logged Bayon-1.0-27B bash scripts/train_local.sh bayon 27b
    run_logged dpo-bayon-27b dpo bayon 27b
    run_logged hearing bash scripts/train_local.sh hearing
    echo "✅ All done. Start Virgo again (docker compose up -d --build) and score it: bash scripts/train_local.sh score"
    ;;
  *) echo "Use: distill [N] | dpo angkor 12b | all"; exit 1 ;;
esac
