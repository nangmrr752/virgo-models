#!/usr/bin/env bash
# Trains Virgo's live voice (a VoxCPM2 LoRA) on this machine's GPU and uploads it to your private
# Virgo-1.0-Angkor-Voice repo, next to voice.wav. The Virgo server uses it on its next start.
#
#   bash scripts/train_local.sh voice                       # steadier Virgo voice (clips made in Virgo's own voice)
#   bash scripts/train_local.sh voice --recordings <folder> # your own recordings (.wav + metadata.csv)
#
# Run it when nothing else is training: it needs most of a 24 GB GPU. About 1-2 hours.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/env.sh"
ROOT="$VIRGO_HOME"
ITERS="${VIRGO_VOICE_STEPS:-1500}"
cd "$CODE"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
REPO="$ME/Virgo-1.0-Angkor-Voice"
WORK="$ROOT/voice-train"; ENVDIR="$ROOT/vox-env"; PY="$ENVDIR/bin/python"
LOG="$ROOT/logs/Virgo-1.0-Angkor-Voice-$(date +%Y%m%d-%H%M).log"
mkdir -p "$WORK" "$ROOT/logs"

{
  echo "== Training Virgo's voice ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # VoxCPM2 needs newer libraries than the rest of Virgo: its own environment (made once).
  [ -d "$ROOT/VoxCPM" ] || git clone -q --depth 1 https://github.com/OpenBMB/VoxCPM "$ROOT/VoxCPM"
  git -C "$ROOT/VoxCPM" pull -q || true
  if [ ! -x "$PY" ]; then
    python -m pip install -q uv
    uv venv -q --python 3.11 "$ENVDIR"
  fi
  uv pip install -q --python "$PY" -e "$ROOT/VoxCPM" argbind tensorboardX soundfile huggingface_hub
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))")

  echo "== Clips =="
  "$PY" -u speech/make_voice_data.py --repo "$REPO" --out "$WORK" "$@"

  echo "== Training ($ITERS steps) =="
  rm -rf "$WORK/run"
  cat > "$WORK/lora.yaml" <<YAML
pretrained_path: $BASE
train_manifest: $WORK/train.jsonl
val_manifest: ""
sample_rate: 16000
out_sample_rate: 48000
batch_size: 2
grad_accum_steps: 8
num_workers: 2
preprocessing_num_workers: 2
num_iters: $ITERS
log_interval: 10
valid_interval: 100000
save_interval: 500
learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 100
max_steps: $ITERS
max_batch_tokens: 8192
max_grad_norm: 1.0
save_path: $WORK/run
tensorboard: $WORK/run/logs
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
hf_model_id: "openbmb/VoxCPM2"
distribute: true
YAML
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$ROOT/VoxCPM/scripts/train_voxcpm_finetune.py" --config_path "$WORK/lora.yaml"

  echo "== Upload =="
  LAST=$(ls -d "$WORK"/run/step_* | sort | tail -1)
  "$PY" - "$REPO" "$LAST" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
for name in ("lora_weights.safetensors", "lora_config.json"):
    api.upload_file(path_or_fileobj=os.path.join(folder, name), path_in_repo=name, repo_id=repo, commit_message="Virgo voice LoRA")
PY
  echo "✅ Virgo's voice saved to https://huggingface.co/$REPO (restart the Virgo server to use it)"
  echo "   To undo: delete lora_weights.safetensors and lora_config.json from that repo."
} 2>&1 | tee "$LOG"
