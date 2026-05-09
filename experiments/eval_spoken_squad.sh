#!/usr/bin/env bash
# =============================================================================
# experiments/eval_spoken_squad.sh
#
# Eval-only Spoken-SQuAD: retrieval + noise (sem treino), agnóstico ao modo.
#
# USO:
#   bash experiments/eval_spoken_squad.sh <PKL> <MODEL>
#
# Detecta automaticamente se o PKL é mean ou chunked via _meta.pooling_mode.
#
# Flags via ambiente:
#   SNR_LEVELS    default: "20,15,10,5"
#   NOISE_TYPES   default: "white,reverb"
#   SKIP_NOISE    "1" para pular noise eval
# =============================================================================
set -euo pipefail

PKL="${1:-}"; MODEL="${2:-}"
[[ -z "$PKL" || -z "$MODEL" ]] && { echo "USO: bash experiments/eval_spoken_squad.sh <PKL> <MODEL>"; exit 1; }
[[ -f "$PKL"   ]] || { echo "ERRO: PKL não encontrado: $PKL"; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERRO: MODEL não encontrado: $MODEL"; exit 1; }

SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"
NOISE_TYPES="${NOISE_TYPES:-white,reverb}"
SKIP_NOISE="${SKIP_NOISE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

POOLING=$(python - <<PY
import pickle
d=pickle.load(open("$PKL","rb"))
m=(d.get("_meta") or {}).get("pooling_mode","")
print(m)
PY
)
[[ -z "$POOLING" ]] && { echo "ERRO: PKL sem _meta.pooling_mode (rebuild com novo build_spoken_squad_pkl.py)"; exit 1; }

if [[ "$POOLING" == "chunked" ]]; then
    RETRIEVAL_MODE="paragraph_grouped"
else
    RETRIEVAL_MODE="candidate"
fi

TS="$(date +%Y%m%d_%H%M%S)_${POOLING}"
LOG_DIR="${ROOT}/logs/eval_spoken_squad_${TS}"
ARTIFACTS="${ROOT}/artifacts_user/eval_spoken_squad_${TS}"
mkdir -p "$LOG_DIR" "$ARTIFACTS"

echo "============================================================"
echo "  Eval Spoken-SQuAD"
echo "  PKL pooling-mode  : ${POOLING}"
echo "  Retrieval mode    : ${RETRIEVAL_MODE}"
echo "  Logs              : ${LOG_DIR}"
echo "============================================================"

# ----- retrieval -----
echo "[1/2] Retrieval (${RETRIEVAL_MODE}) …"
if [[ "$RETRIEVAL_MODE" == "paragraph_grouped" ]]; then
    python scripts/run_retrieval_eval.py \
        --dataset-path "$PKL" --mode paragraph_grouped \
        --model-path "$MODEL" \
        --audio-key hubert-emb --text-key text \
        --hits-k 1,5,10,50 \
        2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
else
    NUM_CAND=$(python -c "
import pickle; d=pickle.load(open('$PKL','rb'))
key='test' if 'test' in d else 'validation'
print(len(d[key]['text']))")
    python scripts/run_retrieval_eval.py \
        --dataset-path "$PKL" --mode candidate \
        --model-path "$MODEL" \
        --audio-key hubert-emb --text-key text \
        --threshold 0.5 --num-candidates "$NUM_CAND" --hits-k 1,5,10,50 \
        2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
fi

# ----- noise -----
if [[ "$SKIP_NOISE" == "1" ]]; then
    echo "[2/2] Noise eval skipped."
else
    echo "[2/2] Noise eval …"
    python scripts/run_noise_robustness_eval.py \
        --dataset-path "$PKL" --model-path "$MODEL" \
        --audio-key hubert-emb --text-key text \
        --snr-levels "$SNR_LEVELS" --noise-types "$NOISE_TYPES" \
        --hits-k 1,5,10,50 \
        --output-csv "${ARTIFACTS}/noise_results.csv" \
        2>&1 | tee "${LOG_DIR}/eval_noise.log"
fi

echo "DONE — logs: ${LOG_DIR}/  artefatos: ${ARTIFACTS}/"
