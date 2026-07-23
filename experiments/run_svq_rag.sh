#!/usr/bin/env bash
# =============================================================================
# experiments/run_svq_rag.sh
#
# End-to-end RAG eval on SVQ: CLASP retrieval + LLM generation (default Qwen3-8B),
# scored with EM / F1 (+ retrieval Recall@k). Intended for a CUDA box (e.g. 4090).
#
# USAGE:
#   bash experiments/run_svq_rag.sh <MODEL> [LOCALE] [MAX_SAMPLES]
#
# ENV OVERRIDES:
#   PYTHON       default .venv/bin/python
#   GENERATOR    HF model id (default: Qwen/Qwen3-8B)
#   TOP_K        passages fed to the LLM (default: 5)
#   CONFIG       SVQ reasoning config (default: span_reasoning_in_lang)
#   DRY_RUN      "1" to stub the generator (retrieval + scoring only, no LLM)
# =============================================================================
set -euo pipefail

MODEL="${1:-}"
[[ -z "$MODEL" ]] && { echo "USAGE: bash experiments/run_svq_rag.sh <MODEL> [LOCALE] [MAX_SAMPLES]"; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: checkpoint not found: $MODEL"; exit 1; }
LOCALE="${2:-en_us}"
MAX_SAMPLES="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

GENERATOR="${GENERATOR:-Qwen/Qwen3-8B}"
TOP_K="${TOP_K:-5}"
CONFIG="${CONFIG:-span_reasoning_in_lang}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="artifacts_user/svq_rag_${TS}"
mkdir -p "$OUT_DIR"

FLAGS=(--model-path "$MODEL" --config "$CONFIG" --locale "$LOCALE"
       --top-k "$TOP_K" --output-json "${OUT_DIR}/svq_rag_${LOCALE}.json")
[[ -n "$MAX_SAMPLES" ]] && FLAGS+=(--max-samples "$MAX_SAMPLES")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    FLAGS+=(--dry-run-generator)
else
    FLAGS+=(--generator "$GENERATOR")
fi

echo ">>> SVQ RAG  (locale=$LOCALE, top_k=$TOP_K, generator=${DRY_RUN:+dry-run}${GENERATOR})"
"$PYTHON" scripts/run_svq_rag_eval.py "${FLAGS[@]}" 2>&1 | tee "${OUT_DIR}/rag.log"

echo ""
echo "DONE — results: ${OUT_DIR}"
