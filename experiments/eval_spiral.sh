#!/usr/bin/env bash
# =============================================================================
# experiments/eval_spiral.sh
#
# Avaliação de retrieval no SPIRAL (on-the-fly, sem treino).
# Opcionalmente roda noise robustness se um PKL de suporte for fornecido.
#
# USO:
#   bash experiments/eval_spiral.sh <MODEL> [PKL_PARA_NOISE]
#
# EXEMPLOS:
#   # só SPIRAL retrieval:
#   bash experiments/eval_spiral.sh models/checkpoints/clasp_voxpopuli.pt
#
#   # SPIRAL retrieval + noise eval (requer PKL com audio_path):
#   bash experiments/eval_spiral.sh \
#     models/checkpoints/clasp_voxpopuli.pt \
#     data/datasets/total_dataset_voxpopuli.pkl
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   SPIRAL_JSONL       — JSONL de avaliação (default: spiral_dataset/data.jsonl)
#   SPIRAL_WAV_BASE    — base dir dos WAVs (default: spiral_dataset/wavs/)
#   SNR_LEVELS         — níveis de SNR para noise eval (default: "20,15,10,5")
#   MAX_SPIRAL_SAMPLES — limita amostras no eval SPIRAL (default: sem limite)
# =============================================================================

set -euo pipefail

MODEL="${1:-}"
NOISE_PKL="${2:-}"

if [[ -z "$MODEL" ]]; then
    echo "USO: bash experiments/eval_spiral.sh <MODEL> [PKL_PARA_NOISE]"
    echo "  Ex: bash experiments/eval_spiral.sh models/checkpoints/clasp_voxpopuli.pt"
    exit 1
fi

[[ -f "$MODEL" ]] || { echo "ERRO: Model não encontrado: $MODEL"; exit 1; }

SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"
MAX_SPIRAL_SAMPLES="${MAX_SPIRAL_SAMPLES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

SPIRAL_JSONL="${SPIRAL_JSONL:-${ROOT}/spiral_dataset/data.jsonl}"
SPIRAL_WAV_BASE="${SPIRAL_WAV_BASE:-${ROOT}/spiral_dataset/wavs}"

[[ -f "$SPIRAL_JSONL" ]] || { echo "ERRO: SPIRAL JSONL não encontrado: ${SPIRAL_JSONL}"; exit 1; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/eval_spiral_${TIMESTAMP}"
ARTIFACTS="${ROOT}/artifacts_user/eval_spiral_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$ARTIFACTS"

echo "============================================================"
echo "  CLASP — SPIRAL eval-only"
echo "  Model        : ${MODEL}"
echo "  SPIRAL JSONL : ${SPIRAL_JSONL}"
echo "  Logs         : ${LOG_DIR}"
echo "  Artefatos    : ${ARTIFACTS}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# 1 — SPIRAL retrieval eval (on-the-fly)                                      #
# --------------------------------------------------------------------------- #
echo "[1/$( [[ -n "$NOISE_PKL" ]] && echo 2 || echo 1 )] SPIRAL retrieval evaluation …"

spiral_samples_flag=""
[[ -n "$MAX_SPIRAL_SAMPLES" ]] && spiral_samples_flag="--max-samples ${MAX_SPIRAL_SAMPLES}"

# shellcheck disable=SC2086
python scripts/run_retrieval_eval.py \
    --mode spiral \
    --dataset-path "$SPIRAL_JSONL" \
    --model-path "$MODEL" \
    --spiral-audio-base "$SPIRAL_WAV_BASE" \
    --spiral-output-dir "${ARTIFACTS}/spiral_eval" \
    --hits-k 1,5,10 \
    --spiral-audio-pooling mean \
    $spiral_samples_flag \
    2>&1 | tee "${LOG_DIR}/eval_spiral.log"
echo "[1] SPIRAL eval concluído. Resultados: ${ARTIFACTS}/spiral_eval/"

# --------------------------------------------------------------------------- #
# 2 — Noise robustness eval (opcional, requer PKL com audio_path)             #
# --------------------------------------------------------------------------- #
if [[ -n "$NOISE_PKL" ]]; then
    [[ -f "$NOISE_PKL" ]] || { echo "ERRO: PKL para noise não encontrado: $NOISE_PKL"; exit 1; }

    echo ""
    echo "[2/2] Noise robustness evaluation …"

    NUM_CANDIDATES=$(python -c "
import pickle
d = pickle.load(open('$NOISE_PKL', 'rb'))
key = 'test' if 'test' in d else 'validation'
print(len(d[key].get('text', d[key].get('hubert-emb', []))))
")
    echo "  Num candidates (split inteiro): ${NUM_CANDIDATES}"

    python scripts/run_noise_robustness_eval.py \
        --dataset-path "$NOISE_PKL" \
        --model-path "$MODEL" \
        --audio-paths-from-pickle \
        --num-candidates "$NUM_CANDIDATES" \
        --snr-levels "$SNR_LEVELS" \
        --output-csv "${ARTIFACTS}/noise_results.csv" \
        2>&1 | tee "${LOG_DIR}/eval_noise.log"
    echo "[2/2] Noise eval concluído. CSV: ${ARTIFACTS}/noise_results.csv"
fi

echo ""
echo "============================================================"
echo "  Eval SPIRAL concluído!"
echo "  Artefatos: ${ARTIFACTS}/"
echo "  Logs     : ${LOG_DIR}/"
echo "============================================================"
