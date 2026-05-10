#!/usr/bin/env bash
# End-to-end SPIRAL retrieval evaluation (shared metrics/plots with matrix mode).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/venv/bin/activate"
fi

SPIRAL_JSONL="${SPIRAL_JSONL:-${REPO_ROOT}/spiral_dataset/data.jsonl}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt}"
OUTPUT_DIR="${SPIRAL_OUTPUT_DIR:-${REPO_ROOT}/results/spiral}"

ARGS=(
  --mode spiral
  --dataset-path "${SPIRAL_JSONL}"
  --model-path "${MODEL_PATH}"
  --spiral-output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SPIRAL_AUDIO_BASE:-}" ]]; then
  ARGS+=(--spiral-audio-base "${SPIRAL_AUDIO_BASE}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${HUBERT_MODEL:-}" ]]; then
  ARGS+=(--hubert-model "${HUBERT_MODEL}")
fi
if [[ -n "${SENTENCE_TRANSFORMER:-}" ]]; then
  ARGS+=(--sentence-transformer "${SENTENCE_TRANSFORMER}")
fi

exec python "${REPO_ROOT}/scripts/run_retrieval_eval.py" "${ARGS[@]}" "$@"
