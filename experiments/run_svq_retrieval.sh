#!/usr/bin/env bash
# =============================================================================
# experiments/run_svq_retrieval.sh
#
# Baseline CLASP retrieval on SVQ, using the SAME pipeline as the other datasets:
# build a total_dataset PKL (audio <-> transcript) then run run_retrieval_eval.py.
# Produces Hits@K / MRR directly comparable to VoxPopuli / Spoken-SQuAD.
#
# USAGE:
#   bash experiments/run_svq_retrieval.sh <MODEL> [CONFIG] [MAX_SAMPLES]
#
# ARGS:
#   MODEL         CLASP checkpoint (.pt)
#   CONFIG        SVQ HF config (default: audio_en_us_clean)
#   MAX_SAMPLES   cap rows (default: all)
#
# ENV OVERRIDES: PYTHON (default .venv/bin/python), DEVICE, NUM_CANDIDATES, LOCALE
# =============================================================================
set -euo pipefail

MODEL="${1:-}"
[[ -z "$MODEL" ]] && { echo "USAGE: bash experiments/run_svq_retrieval.sh <MODEL> [CONFIG] [MAX_SAMPLES]"; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: checkpoint not found: $MODEL"; exit 1; }
CONFIG="${2:-audio_en_us_clean}"
MAX_SAMPLES="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
PKL="data/datasets/total_dataset_svq_${CONFIG}.pkl"
OUT_DIR="artifacts_user/svq_retrieval_${TS}"
mkdir -p "$OUT_DIR"

BUILD_FLAGS=(--config "$CONFIG" --output "$PKL" --replicate-for-train)
[[ -n "$MAX_SAMPLES" ]] && BUILD_FLAGS+=(--max-samples "$MAX_SAMPLES")
[[ -n "${LOCALE:-}" ]] && BUILD_FLAGS+=(--locale "$LOCALE")
[[ -n "${DEVICE:-}" ]] && BUILD_FLAGS+=(--device "$DEVICE")

echo ">>> Building SVQ PKL ($CONFIG) -> $PKL"
"$PYTHON" scripts/build_svq_pkl.py "${BUILD_FLAGS[@]}"

N=$("$PYTHON" -c "import pickle;d=pickle.load(open('$PKL','rb'));print(len(d['test']['text']))")
NUM_CANDIDATES="${NUM_CANDIDATES:-$N}"

echo ">>> Retrieval eval (num_candidates=$NUM_CANDIDATES over $N rows)"
"$PYTHON" scripts/run_retrieval_eval.py \
    --dataset-path "$PKL" --mode candidate \
    --model-path "$MODEL" \
    --audio-key hubert-emb --text-key text \
    --num-candidates "$NUM_CANDIDATES" \
    2>&1 | tee "${OUT_DIR}/retrieval.log"

echo ""
echo "DONE — PKL: $PKL   log: ${OUT_DIR}/retrieval.log"
