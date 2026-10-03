#!/usr/bin/env bash
# Scores the running Virgo server on Virgo's test set (chat/eval/questions.jsonl: 184 questions in
# Khmer, English and 5 more languages: identity, math, reasoning, facts, translation, coding, honesty,
# safety, style, support). No second copy of the model: it asks the server you already run.
#
#   bash scripts/train_local.sh score                     # the server's first model (e.g. Bayon)
#   bash scripts/train_local.sh score virgo-1.0-angkor    # one model by name
#
# Reports go to <data>/reports/<model>-<date>.json, each compared with the one before it.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env_auto.sh"
ROOT="$VIRGO_HOME"
cd "$CODE"
[ -f .env ] && set -a && source <(grep -E '^(VIRGO_API_KEY|VIRGO_PORT)=' .env) && set +a
SERVER="${VIRGO_SERVER:-http://127.0.0.1:${VIRGO_PORT:-8088}}"
MODEL="${1:-}"
NAME="${MODEL:-server}"
mkdir -p "$ROOT/reports"
LAST=$(ls -t "$ROOT/reports/$NAME"-*.json 2>/dev/null | head -1 || true)
REPORT="$ROOT/reports/$NAME-$(date +%Y%m%d-%H%M).json"
echo "== Scoring $SERVER ${MODEL:+($MODEL)} =="
python -u chat/evaluate.py --server "$SERVER" --key "${VIRGO_API_KEY:-}" ${MODEL:+--model "$MODEL"} \
  --report "$REPORT" ${LAST:+--compare "$LAST"}
echo "Saved: $REPORT"
