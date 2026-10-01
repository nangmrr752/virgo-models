#!/usr/bin/env bash
# Starts the Virgo server on a rented GPU (RunPod or any Linux machine with an NVIDIA GPU), at your own
# address (Cloudflare Tunnel, e.g. virgo.camksn.com). See RUNPOD.md.
#
# Needs these environment variables (RunPod: Edit Pod → Environment Variables, or secrets):
#   HF_TOKEN          Hugging Face token (read access to your Virgo models)
#   CF_TUNNEL_TOKEN   your Cloudflare Tunnel token (the tunnel that serves virgo.camksn.com)
#   VIRGO_API_KEY     the password the website uses (same as now)
# Optional: HF_MODEL (which Virgo chat model), SERVE_BAYON=0, USE_VIRGO_VOICE=0.
#
# Everything big (models, Python environments) is kept in $WORKSPACE (default /workspace, RunPod's
# volume), so a restarted pod starts in a few minutes instead of downloading again.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$WORKSPACE/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/pip-cache}"
mkdir -p "$WORKSPACE" "$HF_HOME"

echo "⏳ Installing the Virgo server's Python packages..."
python -m pip install -q -r requirements-serve.txt
python -m pip uninstall -y -q torchao 2>/dev/null || true  # conflicts with newer peft
command -v espeak-ng >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y espeak-ng ffmpeg >/dev/null 2>&1) || true

exec python -u scripts/serve_rented_gpu.py
