#!/usr/bin/env bash
# Trains Virgo's live voice (a VoxCPM2 LoRA) on this machine's GPU and uploads it to your private
# Virgo-1.0-Angkor-Voice repo, next to voice.wav. The Virgo server uses it on its next start.
#
#   bash scripts/train_local.sh voice                       # steadier Virgo voice (clips made in Virgo's own voice)
#   bash scripts/train_local.sh voice --recordings <folder> # your own recordings (.wav + metadata.csv)
#   bash scripts/train_local.sh voice bayon                 # Bayon's own voice (Virgo-1.0-Bayon-Voice), new from a description
#   bash scripts/train_local.sh voice bakong                # Virgo-Bakong-2.0-Voice: the live voice's own voice
#   VIRGO_VOICE_DESCRIBE="A woman, ..." bash scripts/train_local.sh voice bayon   # choose how it sounds
#
# Run it when nothing else is training: it needs most of a 24 GB GPU. About 1-2 hours.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"   # your server (env.sh), Colab or Kaggle
ROOT="$VIRGO_HOME"
ITERS="${VIRGO_VOICE_STEPS:-1500}"
cd "$CODE"
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
# `voice bayon`: Bayon's own voice (Virgo-1.0-Bayon-Voice). `voice bakong`: the live voice's own voice,
# Virgo-Bakong-{VIRGO_LIVE_VERSION}-Voice (2.0), used for all of Virgo's speech once it exists. Both are
# made from a description the first time (VIRGO_VOICE_DESCRIBE to choose). Otherwise Virgo-1.0-Angkor-Voice.
WHO=angkor; DESIGN=()
case "${1:-}" in
  bayon) shift; WHO=bayon
    DESIGN=(--design "${VIRGO_VOICE_DESCRIBE:-A young man, calm and confident, warm clear voice, natural friendly pace}") ;;
  bakong) shift; WHO=bakong
    DESIGN=(--design "${VIRGO_VOICE_DESCRIBE:-A young woman, bright and lively, warm clear voice, natural conversational pace}") ;;
esac
if [ "$WHO" = bakong ]; then NAME="Virgo-Bakong-${VIRGO_LIVE_VERSION:-2.0}-Voice"; else NAME="Virgo-1.0-${WHO^}-Voice"; fi
REPO="$ME/$NAME"
WORK="$ROOT/voice-train"; [ "$WHO" != angkor ] && WORK="$ROOT/voice-train-$WHO"
# Virgo-Bakong: the cleanest voice. Twice the clips, made with more VoxCPM2 steps, each one checked by
# Whisper (Virgo's hearing) and dropped if it was said wrong; 8 tries for the voice sample; twice the
# training. VIRGO_VOICE_STEPS changes the training length.
QUALITY=()
if [ "$WHO" = bakong ]; then
  HEARING=$(python -c "from huggingface_hub import HfApi; r='$ME/Virgo-1.0-Angkor-Hearing'; print(r if HfApi().file_exists(r, 'config.json') else 'openai/whisper-large-v3-turbo')" 2>/dev/null || echo openai/whisper-large-v3-turbo)
  QUALITY=(--count 1200 --steps 16 --design-tries 8 --check "$HEARING")
  ITERS="${VIRGO_VOICE_STEPS:-3000}"
fi
ENVDIR="$ROOT/vox-env"; PY="$ENVDIR/bin/python"
LOG="$ROOT/logs/$NAME-$(date +%Y%m%d-%H%M).log"
mkdir -p "$WORK" "$ROOT/logs"

{
  echo "== Training $NAME ($(nvidia-smi --query-gpu=name --format=csv,noheader)) =="
  # VoxCPM2 needs newer libraries than the rest of Virgo: its own environment (made once).
  [ -d "$ROOT/VoxCPM" ] || git clone -q --depth 1 https://github.com/OpenBMB/VoxCPM "$ROOT/VoxCPM"
  git -C "$ROOT/VoxCPM" pull -q || true
  if [ ! -x "$PY" ]; then
    python -m pip install -q uv
    uv venv -q --python 3.11 "$ENVDIR"
  fi
  uv pip install -q --python "$PY" -e "$ROOT/VoxCPM" argbind tensorboardX soundfile huggingface_hub
  # The newest PyTorch needs a newer NVIDIA driver than many machines have (e.g. 535 = CUDA 12.2) and
  # then quietly runs on the CPU (days instead of hours). Use a CUDA 12.4 build when the GPU isn't seen.
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch can't see the GPU with this driver: installing a CUDA 12.4 build"
    uv pip install -q --python "$PY" "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124
    uv pip install -q --python "$PY" "torchcodec==0.2.*"
  fi
  "$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print('GPU for the voice:', torch.cuda.get_device_name(0) if ok else 'none'); sys.exit(0 if ok else 1)" \
    || { echo "❌ The voice environment can't use the GPU (see nvidia-smi). Stopping instead of running for days on the CPU."; exit 1; }
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))")

  echo "== Clips =="
  "$PY" -u speech/make_voice_data.py --repo "$REPO" --out "$WORK" "${DESIGN[@]}" "${QUALITY[@]}" "$@"

  echo "== Training ($ITERS steps) =="
  rm -rf "$WORK/run"
  cat > "$WORK/lora.yaml" <<YAML
pretrained_path: $BASE
train_manifest: $WORK/train.jsonl
val_manifest: ""
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 16
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
max_batch_tokens: ${VIRGO_VOICE_TOKENS:-4096}
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
