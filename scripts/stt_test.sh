#!/usr/bin/env bash
# Virgo's hearing test in one command: Khmer CER (lower is better) for the hearing models, on the FLEURS
# Khmer test set (or your own clips). See speech/eval_stt.py.
#
#   bash scripts/train_local.sh stttest                    # all candidates on FLEURS (200 clips)
#   bash scripts/train_local.sh stttest --limit 50         # quicker
#   bash scripts/train_local.sh stttest --data /opt/virgo/stt-test   # your own clips (audio + metadata.csv)
#   STT_MODELS="whisper:virgoai/Angkor-1.0-STT qwen3asr:seanghay/Qwen3-ASR-0.6B-Khmer" bash scripts/train_local.sh stttest
#
# Qwen3-ASR and Meta Omnilingual get their own small Python environments (<data>/venv-stt-*), so their
# packages never change the training environment. GPU: stop the chat model first so the bigger ones fit:
#   docker compose stop llama    # and afterwards: docker compose start llama
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"
cd "$CODE"
export VIRGO_HOME
MODELS=${STT_MODELS:-"whisper:virgoai/Angkor-1.0-STT whisper:openai/whisper-large-v3-turbo qwen3asr:seanghay/Qwen3-ASR-0.6B-Khmer omni:omniASR_LLM_1B"}
STAMP=$(date +%Y%m%d-%H%M)
REPORTS=()

# A separate Python environment for one engine's package (made once, reused).
engine_python() {
  local name=$1 package=$2 dir="$VIRGO_HOME/venv-stt-$1"
  if [ ! -x "$dir/bin/python" ]; then
    echo "📦 Setting up $name ($package) in $dir" >&2
    python -m venv "$dir" >&2
    "$dir/bin/pip" install -q -U pip >&2
    "$dir/bin/pip" install -q "$package" datasets soundfile librosa >&2
  fi
  echo "$dir/bin/python"
}

for spec in $MODELS; do
  kind=${spec%%:*}
  case "$kind" in
    qwen3asr) py=$(engine_python qwen3asr qwen-asr) ;;
    omni) py=$(engine_python omni omnilingual-asr) ;;
    *) py=python ;;
  esac
  report="$VIRGO_HOME/logs/stt-$STAMP-$(echo "$spec" | tr -c 'A-Za-z0-9.-' _).json"
  mkdir -p "$VIRGO_HOME/logs"
  "$py" -u speech/eval_stt.py --models "$spec" --report "$report" "$@" || echo "⚠️ $spec didn't finish (see above)"
  [ -f "$report" ] && REPORTS+=("$report")
done

[ ${#REPORTS[@]} -gt 0 ] || { echo "No model finished."; exit 1; }
echo
echo "== Khmer hearing, all models: CER (lower is better) =="
python - "${REPORTS[@]}" <<'PY'
import json, sys
rows = []
for path in sys.argv[1:]:
    for spec, r in json.load(open(path, encoding="utf-8"))["results"].items():
        rows.append((r["cer"], r["seconds_per_clip"], r["clips"], spec))
for cer, sec, clips, spec in sorted(rows):
    print(f"  {cer:6.1%}   {sec:5.2f} s/clip   {clips} clips   {spec}")
PY
echo "Reports: $VIRGO_HOME/logs/stt-$STAMP-*.json"
