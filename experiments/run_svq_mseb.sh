#!/usr/bin/env bash
# =============================================================================
# experiments/run_svq_mseb.sh
#
# Evaluate a CLASP checkpoint on MSEB SVQ tasks (acoustic-hypothesis reranking
# and, on Linux/x86, audio->text retrieval) via scripts/run_mseb_task.py.
#
# USAGE:
#   bash experiments/run_svq_mseb.sh <MODEL> [TASK ...]
#
# Defaults to one English reranking task if no TASK is given. Reranking tasks
# import cleanly on macOS; retrieval tasks need `scann` (Linux/x86 or Docker).
#
# ENV OVERRIDES:
#   PYTHON        python interpreter (default: .venv-mseb/bin/python)
#   DEVICE        torch device passed to the encoder (default: auto)
#   OUT_DIR       results dir (default: artifacts_user/svq_mseb_<ts>)
#
# NOTE: MSEB requires Python >= 3.12. See docs/MSEB.md for env setup.
# =============================================================================
set -euo pipefail

MODEL="${1:-}"
[[ -z "$MODEL" ]] && { echo "USAGE: bash experiments/run_svq_mseb.sh <MODEL> [TASK ...]"; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: checkpoint not found: $MODEL"; exit 1; }
shift || true

TASKS=("$@")
[[ ${#TASKS[@]} -eq 0 ]] && TASKS=("SVQEnUsQueryReranking")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

PYTHON="${PYTHON:-${ROOT}/.venv-mseb/bin/python}"
[[ -x "$PYTHON" ]] || { echo "ERROR: python not found at $PYTHON (create the 3.12 mseb env — see docs/MSEB.md)"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts_user/svq_mseb_${TS}}"
LOG_DIR="${ROOT}/logs/svq_mseb_${TS}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

DEVICE_FLAG=()
[[ -n "${DEVICE:-}" ]] && DEVICE_FLAG=(--device "$DEVICE")

echo "============================================================"
echo "  MSEB SVQ evaluation"
echo "  Model    : ${MODEL}"
echo "  Tasks    : ${TASKS[*]}"
echo "  Python   : ${PYTHON}"
echo "  Out dir  : ${OUT_DIR}"
echo "============================================================"

for task in "${TASKS[@]}"; do
    echo ">>> ${task}"
    "$PYTHON" scripts/run_mseb_task.py \
        --task "$task" \
        --model-path "$MODEL" \
        --results-jsonl "${OUT_DIR}/${task}.jsonl" \
        "${DEVICE_FLAG[@]}" \
        2>&1 | tee "${LOG_DIR}/${task}.log"
done

echo ""
echo "DONE — results: ${OUT_DIR}"
echo "       logs   : ${LOG_DIR}"
