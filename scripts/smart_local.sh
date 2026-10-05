#!/usr/bin/env bash
# Makes Virgo's chat models smarter on this machine's GPU (stop the Virgo server first: all of this
# needs the whole GPU). Called by train_local.sh:
#
#   bash scripts/train_local.sh distill [N]        # Bayon-1.0-27B answers N checked problems (default 12000, ~1 day)
#   bash scripts/train_local.sh mistakes [N]       # Angkor tries N problems (default 6000); Bayon answers the ones it got wrong
#   bash scripts/train_local.sh basetest google/gemma-4-31b-it  # an untrained base model on the big hard test
#   bash scripts/train_local.sh basetest-gguf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL  # a GGUF in llama.cpp (MoE models)
#   bash scripts/train_local.sh baseline angkor 12b  # score the model on Hugging Face on the big hard test (once)
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
# The big hard test (326 questions) decides every upload; its best scores are in <data>/logs/<model>-hardbig-best.json.
HARD="chat/eval/hard.jsonl,chat/eval/hard2.jsonl"

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

mistakes() {  # DPO pairs from Angkor's own mistakes (scripts/student_pairs.py)
  local tasks="${1:-6000}" student="$ROOT/out/Angkor-1.0-12B" teacher_dir="$ROOT/out/teacher-Bayon-1.0-27B" repo bayon adapter=""
  need_gpu_free
  fetch student_pairs.jsonl chat/dpo
  repo=$(python scripts/names.py resolve Angkor-1.0-12B 2>/dev/null || echo "$ME/Angkor-1.0-12B")
  [ -f "$student/adapter_config.json" ] || python -c "from huggingface_hub import snapshot_download as s; s('$repo', local_dir='$student')" >/dev/null
  echo "== Angkor tries $tasks problems =="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/student_pairs.py --stage student --adapter "$student" --tasks "$tasks"
  bayon=$(python scripts/names.py resolve Bayon-1.0-27B 2>/dev/null || true)
  if [ -n "$bayon" ] && python -c "from huggingface_hub import HfApi; import sys; sys.exit(0 if HfApi().file_exists('$bayon', 'adapter_config.json') else 1)" 2>/dev/null; then
    [ -f "$teacher_dir/adapter_config.json" ] || python -c "from huggingface_hub import snapshot_download as s; s('$bayon', local_dir='$teacher_dir')" >/dev/null
    adapter="$teacher_dir"
  fi
  echo "== The teacher answers what Angkor got wrong (teacher: ${bayon:-google/gemma-3-27b-it}) =="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/student_pairs.py --stage teacher --adapter "$adapter" --hf-repo "$DATA_REPO"
}

baseline() {  # baseline angkor 12b: the big hard test score of the model on Hugging Face (what later runs must beat)
  local which="${1:-angkor}" size="${2:-12b}" name repo dir
  name=$(name_for "$which" "$size")
  repo=$(python scripts/names.py resolve "$name" 2>/dev/null || echo "$ME/$name")
  dir="$ROOT/out/$name-current"
  need_gpu_free
  rm -rf "$dir" && python -c "from huggingface_hub import snapshot_download as s; s('$repo', local_dir='$dir')" >/dev/null
  echo "== Big hard test of $repo (the model in use) =="
  python -u chat/evaluate.py --questions "$HARD" --adapter "$dir" --report "$ROOT/logs/$name-hardbig-best.json" | tail -12
  rm -rf "$dir"
  echo "✅ Baseline for $name saved: $ROOT/logs/$name-hardbig-best.json"
}

basetest() {  # basetest google/gemma-4-31b-it: an untrained base model on the big hard test (nothing uploaded)
  local base="${1:?which base model, e.g. google/gemma-4-31b-it}" tag
  tag=$(echo "$base" | tr '/' '_')
  need_gpu_free
  python -c "import transformers; print('transformers', transformers.__version__)"
  echo "== Big hard test of $base (no training) =="
  python -u chat/evaluate.py --questions "$HARD" --base "$base" --adapter none --report "$ROOT/logs/base-$tag-hardbig.json" \
    $([ -f "$ROOT/logs/Angkor-1.0-12B-hardbig-best.json" ] && echo --compare "$ROOT/logs/Angkor-1.0-12B-hardbig-best.json") | tail -14
  echo "Report: $ROOT/logs/base-$tag-hardbig.json (compared with Angkor-1.0-12B, the model in use)"
}

basetest_gguf() {  # basetest-gguf <repo>[:quant]: a GGUF base in llama.cpp (GPU) on the big hard test, nothing uploaded
  # For models our 4-bit loader can't fit, e.g. the Gemma 4 26B A4B MoE in Google's QAT 4-bit: about 15 GB.
  # Text only (--no-mmproj: Gemma 4's image part isn't loaded); llama.cpp fits the model into free GPU memory.
  local spec="${1:?a GGUF on Hugging Face, e.g. unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL}" tag port=8089 image
  image="${VIRGO_LLAMA_IMAGE:-}"
  if [ -z "$image" ]; then
    # The official image needs CUDA 12.8 (driver 570+); for an older driver, llama.cpp is built for its CUDA once.
    local cuda
    cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    if python -c "import sys; sys.exit(0 if tuple(map(int, '${cuda:-0.0}'.split('.'))) >= (12, 8) else 1)"; then
      image=ghcr.io/ggml-org/llama.cpp:server-cuda
    else
      local base=12.2.2; python -c "import sys; sys.exit(0 if tuple(map(int, '${cuda:-0.0}'.split('.'))) >= (12, 4) else 1)" && base=12.4.1
      image="virgo-llama:cuda${base%.*}"
      if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "== Building llama.cpp for this driver's CUDA $cuda (once, ~10-20 min) =="
        docker build -f "$CODE/docker/llama-cuda.Dockerfile" --build-arg CUDA="$base" -t "$image" "$CODE/docker"
      fi
    fi
  fi
  tag=$(echo "$spec" | tr '/:' '__')
  need_gpu_free
  mkdir -p "$ROOT/llama-cache"
  docker rm -f virgo-llama >/dev/null 2>&1 || true
  echo "== llama.cpp: $spec (the first run downloads it into $ROOT/llama-cache) =="
  docker run -d --name virgo-llama --gpus all -p 127.0.0.1:$port:8080 -v "$ROOT/llama-cache:/root/.cache/llama.cpp" -v "$HF_HOME:/root/.cache/huggingface" \
    -e HF_TOKEN="$(python -c 'from huggingface_hub import get_token; print(get_token() or "")')" \
    "$image" -hf "$spec" --no-mmproj -c 8192 --jinja --host 0.0.0.0 --port 8080 ${VIRGO_LLAMA_TEST_ARGS:-} >/dev/null
  echo -n "Waiting for the model to download and load"
  for _ in $(seq 1 720); do  # up to 2 hours (a first download is ~15 GB)
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    docker inspect -f '{{.State.Running}}' virgo-llama 2>/dev/null | grep -q true || { echo; docker logs --tail 30 virgo-llama; docker rm -f virgo-llama >/dev/null; exit 1; }
    echo -n "."; sleep 10
  done
  echo
  curl -sf "http://127.0.0.1:$port/health" >/dev/null || { docker logs --tail 30 virgo-llama; docker rm -f virgo-llama >/dev/null; echo "❌ llama.cpp didn't start"; exit 1; }
  echo "== Big hard test of $spec (no training) =="
  python -u chat/evaluate.py --questions "$HARD" --openai "http://127.0.0.1:$port" --report "$ROOT/logs/base-$tag-hardbig.json" \
    $([ -f "$ROOT/logs/Angkor-1.0-12B-hardbig-best.json" ] && echo --compare "$ROOT/logs/Angkor-1.0-12B-hardbig-best.json") | tail -14 || true
  docker rm -f virgo-llama >/dev/null
  echo "Report: $ROOT/logs/base-$tag-hardbig.json (compared with Angkor-1.0-12B, the model in use)"
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
  fetch student_pairs.jsonl chat/dpo  # from Angkor's own mistakes (train_local.sh mistakes)
  cat chat/dpo/*.jsonl > /dev/null 2>&1 && [ -n "$(cat chat/dpo/*.jsonl 2>/dev/null | head -1)" ] || { echo "❌ No DPO pairs yet: run bash scripts/train_local.sh distill first"; exit 1; }
  repo=$(python scripts/names.py resolve "$name" 2>/dev/null || echo "$ME/$name")
  # The trained (SFT) model: this machine's copy, else the one on Hugging Face.
  [ -f "$out/adapter_config.json" ] || python -c "from huggingface_hub import snapshot_download as s; s('$repo', local_dir='$out')" >/dev/null
  local extra=""; [ "$size" = 27b ] && extra="--max-len 512"  # 27B in 4 bits fills most of a 24 GB GPU
  echo "== DPO for $name on $(cat chat/dpo/*.jsonl | grep -c '"chosen"') pairs =="
  python -u chat/evaluate.py --questions $HARD --adapter "$out" --report "$ROOT/logs/$name-hardbig-before.json" | tail -15
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u chat/train_dpo.py --adapter "$out" --out "$out-dpo" $extra
  python -u chat/evaluate.py --questions $HARD --adapter "$out-dpo" --report "$ROOT/logs/$name-hardbig-after.json" \
    --compare "$ROOT/logs/$name-hardbig-before.json" | tail -20
  hard_before=$(python -c "import json; print(json.load(open('$ROOT/logs/$name-hardbig-before.json'))['overall'])")
  hard_after=$(python -c "import json; print(json.load(open('$ROOT/logs/$name-hardbig-after.json'))['overall'])")
  # Better than this model before DPO, and than the best upload so far (a model train_local.sh didn't upload).
  local best="$hard_before"
  [ -f "$ROOT/logs/$name-hardbig-best.json" ] && best=$(python -c "import json; print(max($hard_before, json.load(open('$ROOT/logs/$name-hardbig-best.json'))['overall']))")
  if python -c "import sys; sys.exit(0 if $hard_after >= $best else 1)"; then
    python - "$repo" "$out-dpo" "$hard_before" "$hard_after" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder, before, after = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
HfApi().upload_folder(folder_path=folder, repo_id=repo, ignore_patterns=["checkpoint-*"],
                      commit_message=f"DPO: hard test {before:.1%} → {after:.1%}")
PY
    rm -rf "$out" && mv "$out-dpo" "$out"
    cp "$ROOT/logs/$name-hardbig-after.json" "$ROOT/logs/$name-hardbig-best.json"
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
  mistakes) run_logged mistakes mistakes "$@" ;;
  basetest) run_logged "basetest" basetest "$@" ;;
  basetest-gguf) run_logged "basetest-gguf" basetest_gguf "$@" ;;
  baseline) run_logged "baseline-${1:-angkor}-${2:-12b}" baseline "$@" ;;
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
  *) echo "Use: distill [N] | mistakes [N] | basetest <hf model> | basetest-gguf <repo:quant> | baseline angkor 12b | dpo angkor 12b | all"; exit 1 ;;
esac
