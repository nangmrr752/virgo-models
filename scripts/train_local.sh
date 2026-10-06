#!/usr/bin/env bash
# Trains a Virgo chat model on this machine's GPU and uploads it (private) to Hugging Face.
#
#   bash scripts/train_local.sh bayon 27b     # Bayon-1.0-27B on Gemma 3 27B, smarter than Angkor (default)
#                                            #   (also: bayon 12b → Bayon-1.0-12B, bayon 4b → Bayon-1.0-4B)
#   bash scripts/train_local.sh score         # score the running Virgo server on the test set (chat/eval)
#   bash scripts/train_local.sh hearing       # Virgo's hearing (Whisper, Khmer), see train_hearing_local.sh
#   bash scripts/train_local.sh stttest       # hearing test: Khmer CER of the hearing models, see stt_test.sh
#   bash scripts/train_local.sh voice         # Virgo's live voice (VoxCPM2), see train_voice_local.sh
#   bash scripts/train_local.sh angkor 12b    # Angkor-1.0-12B (also: angkor 4b → Angkor-1.0-4B)
#   bash scripts/train_local.sh distill [N]   # Bayon-1.0-27B answers N checked problems (default 12000), see distill_smart.py
#   bash scripts/train_local.sh dpo angkor 12b  # DPO on the pairs distill made; uploads only if the hard test improves
#   bash scripts/train_local.sh all           # everything in order: distill → Angkor 12B → DPO → Bayon 27B → DPO → hearing
# Names: scripts/names.py — {Name}-{Version}-{Size} in Virgo's Hugging Face organization (VIRGO_HF_ORG);
# uploads keep going to an old repo until you move it with: python scripts/names.py rename --yes
#
# Needs scripts/local_setup.sh first. Output in <data>/out/<model>, log in <data>/logs (data: /opt/virgo-data
# when the code is in /opt/virgo-models).
# Extra options go to chat/train.py, e.g. --epochs 2 or --max-len 768 (if it runs out of memory).
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"   # your server (env.sh), Colab or Kaggle
ROOT="$VIRGO_HOME"
[ "${1:-}" = voice ] && { shift; exec bash "$CODE/scripts/train_voice_local.sh" "$@"; }  # Virgo's live voice
[ "${1:-}" = score ] && { shift; exec bash "$CODE/scripts/score_server.sh" "$@"; }  # score the running server
[ "${1:-}" = stttest ] && { shift; exec bash "$CODE/scripts/stt_test.sh" "$@"; }  # hearing test (Khmer CER)
[ "${1:-}" = hearing-1.1 ] && { shift; exec bash "$CODE/scripts/train_hearing_local.sh" --v1.1 "$@"; }  # Angkor-1.1-STT
[ "${1:-}" = hearing ] && { shift; exec bash "$CODE/scripts/train_hearing_local.sh" "$@"; }  # Virgo's hearing (Whisper)
[ "${1:-}" = distill ] && { shift; exec bash "$CODE/scripts/smart_local.sh" distill "$@"; }  # checked examples + DPO pairs
[ "${1:-}" = basetest-gguf ] && { shift; exec bash "$CODE/scripts/smart_local.sh" basetest-gguf "$@"; }  # a GGUF base in llama.cpp
[ "${1:-}" = basetest ] && { shift; exec bash "$CODE/scripts/smart_local.sh" basetest "$@"; }  # an untrained base model on the big hard test
[ "${1:-}" = baseline ] && { shift; exec bash "$CODE/scripts/smart_local.sh" baseline "$@"; }  # big hard test of the model in use
[ "${1:-}" = mistakes ] && { shift; exec bash "$CODE/scripts/smart_local.sh" mistakes "$@"; }  # DPO pairs from Angkor's own mistakes
[ "${1:-}" = dpo ] && { shift; exec bash "$CODE/scripts/smart_local.sh" dpo "$@"; }
[ "${1:-}" = all ] && { shift; exec bash "$CODE/scripts/smart_local.sh" all "$@"; }
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
  bayon-27b) NAME=Bayon-1.0-27B ;; bayon-12b) NAME=Bayon-1.0-12B ;; bayon-4b) NAME=Bayon-1.0-4B ;;
  angkor-12b) NAME=Angkor-1.0-12B ;; angkor-4b) NAME=Angkor-1.0-4B ;;
  *) echo "Model: bayon or angkor"; exit 1 ;;
esac
OUT="$ROOT/out/$NAME"
LOG="$ROOT/logs/$NAME-$(date +%Y%m%d-%H%M).log"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
git pull -q

{
  echo "== Training $NAME on $BASE ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # Extra examples from Gemma 3 27B (chat/distill_vertex.ipynb), when you've made them.
  DATA_REPO=$(python scripts/names.py resolve Data-1.0 2>/dev/null || echo "$ME/Virgo-1.0-Angkor-Data")
  python -c "from huggingface_hub import hf_hub_download as d; d('$DATA_REPO', 'distilled_gemma27b.jsonl', repo_type='dataset', local_dir='chat/data')" >/dev/null 2>&1 \
    && echo "➕ Using the Gemma 27B examples" || echo "No Gemma 27B examples yet"
  # Checked examples from scripts/distill_smart.py (bash scripts/train_local.sh distill), when made.
  # (distilled_smart_kaggle.jsonl: the same, made on Kaggle with chat/distill_kaggle.ipynb.)
  for f in distilled_smart.jsonl distilled_smart_kaggle.jsonl; do
    python -c "from huggingface_hub import hf_hub_download as d; d('$DATA_REPO', '$f', repo_type='dataset', local_dir='chat/data')" >/dev/null 2>&1 \
      && echo "➕ Using $(wc -l < chat/data/$f) checked examples ($f)" || true
  done
  DATA="chat/data/*.jsonl"
  # The model's own copy of the data, where Virgo calls itself by this model's standard name
  # (Bayon-1.0 / Angkor-1.0).
  SELF="${WHICH^}-1.0"
  rm -rf "chat/data_$WHICH" && mkdir -p "chat/data_$WHICH"
  for f in chat/data/*.jsonl; do sed "s/Virgo-1\.0-Angkor/$SELF/g; s/Virgo-1\.0-Bayon/$SELF/g; s/(model: Angkor-1\.0)/(model: $SELF)/g" "$f" > "chat/data_$WHICH/$(basename "$f")"; done
  DATA="chat/data_$WHICH/*.jsonl"
  python scripts/check_data.py "$DATA"
  python -u chat/train.py --base "$BASE" --data "$DATA" --out "$OUT" --epochs 3 $EXTRA "$@"
  echo "== Score =="
  python chat/evaluate.py --adapter "$OUT" || echo "(scoring failed; the model is still saved)"
  # The hard test decides: the best score so far for this model is kept in <data>/logs/<model>-hardbig-best.json
  # (DPO updates it too), and a model that scores lower isn't uploaded.
  # (The big hard test: chat/eval/hard.jsonl + hard2.jsonl, 326 questions.)
  BEST="$ROOT/logs/$NAME-hardbig-best.json"; NEW="$ROOT/logs/$NAME-hardbig-$(date +%Y%m%d-%H%M).json"
  # No best score yet but a model already on Hugging Face: score that one first, so a worse model can't replace it.
  REPO=$(python scripts/names.py resolve "$NAME" 2>/dev/null || echo "$ME/$NAME")
  if [ ! -f "$BEST" ] && python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$REPO', 'adapter_config.json') else 1)" 2>/dev/null; then
    bash scripts/smart_local.sh baseline "$WHICH" "${NAME##*-}" || true
  fi
  COMPARE=(); [ -f "$BEST" ] && COMPARE=(--compare "$BEST")
  python chat/evaluate.py --questions chat/eval/hard.jsonl,chat/eval/hard2.jsonl --adapter "$OUT" --report "$NEW" "${COMPARE[@]}" | tail -20 \
    || echo "(hard test failed to run)"
  if [ -f "$BEST" ] && [ -f "$NEW" ] && ! python -c "import json,sys; sys.exit(0 if json.load(open('$NEW'))['overall'] >= json.load(open('$BEST'))['overall'] else 1)"; then
    echo "⚠️ $NAME scored lower on the hard test than the best one so far: not uploaded (kept in $OUT)"
    exit 0
  fi
  echo "== Upload =="
  REPO=$(python scripts/names.py resolve "$NAME" 2>/dev/null || echo "$ME/$NAME")  # an old name until renamed
  python - "$REPO" "$OUT" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
HfApi().create_repo(repo, private=True, exist_ok=True)
HfApi().upload_folder(folder_path=folder, repo_id=repo, commit_message=repo.split("/")[1], ignore_patterns=["checkpoint-*"])
PY
  [ -f "$NEW" ] && cp "$NEW" "$BEST"
  echo "✅ $NAME saved: https://huggingface.co/$REPO"
} 2>&1 | tee "$LOG"
