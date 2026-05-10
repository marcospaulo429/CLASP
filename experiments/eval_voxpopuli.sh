#!/usr/bin/env bash
# =============================================================================
# experiments/eval_voxpopuli.sh
#
# Avaliação de retrieval + noise robustness no VoxPopuli (sem treino).
#
# USO:
#   bash experiments/eval_voxpopuli.sh <PKL> <MODEL>
#
# EXEMPLOS:
#   bash experiments/eval_voxpopuli.sh \
#     data/datasets/total_dataset_voxpopuli.pkl \
#     models/checkpoints/clasp_voxpopuli.pt
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   SNR_LEVELS   — níveis de SNR para noise eval (default: "20,15,10,5")
#   WHAM_DIR     — dir. com ruído ambiente WHAM (default: desabilitado)
# =============================================================================

set -euo pipefail

PKL="${1:-}"
MODEL="${2:-}"

if [[ -z "$PKL" || -z "$MODEL" ]]; then
    echo "USO: bash experiments/eval_voxpopuli.sh <PKL> <MODEL>"
    echo "  Ex: bash experiments/eval_voxpopuli.sh data/datasets/total_dataset_voxpopuli.pkl models/checkpoints/clasp_voxpopuli.pt"
    exit 1
fi

[[ -f "$PKL"   ]] || { echo "ERRO: PKL não encontrado: $PKL";   exit 1; }
[[ -f "$MODEL" ]] || { echo "ERRO: Model não encontrado: $MODEL"; exit 1; }

SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"
WHAM_DIR="${WHAM_DIR:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/eval_voxpopuli_${TIMESTAMP}"
ARTIFACTS="$LOG_DIR"
mkdir -p "$LOG_DIR"

# Tamanho do split de avaliação (dataset inteiro)
NUM_CANDIDATES=$(python -c "
import pickle
d = pickle.load(open('$PKL', 'rb'))
key = 'test' if 'test' in d else 'validation'
print(len(d[key].get('text', d[key].get('hubert-emb', []))))
")

echo "============================================================"
echo "  CLASP — VoxPopuli eval-only"
echo "  PKL            : ${PKL}"
echo "  Model          : ${MODEL}"
echo "  Num candidates : ${NUM_CANDIDATES} (split inteiro)"
echo "  Logs           : ${LOG_DIR}"
echo "  Artefatos      : ${ARTIFACTS}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# 1 — Retrieval eval (candidate mode)                                         #
# --------------------------------------------------------------------------- #
echo "[1/2] Retrieval evaluation (candidate mode) …"
python scripts/run_retrieval_eval.py \
    --dataset-path "$PKL" \
    --mode candidate \
    --model-path "$MODEL" \
    --audio-key hubert-emb \
    --text-key text \
    --threshold 0.5 \
    --num-candidates "$NUM_CANDIDATES" \
    --hits-k 1,5,10,50 \
    --plot-out "${ARTIFACTS}/retrieval_candidate.png" \
    2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
echo "[1/2] Retrieval eval concluído."
echo ""

# --------------------------------------------------------------------------- #
# 2 — Noise robustness eval                                                   #
# --------------------------------------------------------------------------- #
echo "[2/2] Noise robustness evaluation …"

wham_flag=""
[[ -n "$WHAM_DIR" ]] && wham_flag="--wham-dir ${WHAM_DIR}"

# shellcheck disable=SC2086
python scripts/run_noise_robustness_eval.py \
    --dataset-path "$PKL" \
    --model-path "$MODEL" \
    --audio-paths-from-pickle \
    --num-candidates "$NUM_CANDIDATES" \
    --snr-levels "$SNR_LEVELS" \
    --output-csv "${ARTIFACTS}/noise_results.csv" \
    $wham_flag \
    2>&1 | tee "${LOG_DIR}/eval_noise.log"
echo "[2/2] Noise eval concluído. CSV: ${ARTIFACTS}/noise_results.csv"

echo ""
echo "============================================================"
echo "  Eval VoxPopuli concluído!"
echo "  Artefatos: ${ARTIFACTS}/"
echo "  Logs     : ${LOG_DIR}/"
echo "============================================================"
