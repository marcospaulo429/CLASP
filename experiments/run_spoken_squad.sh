#!/usr/bin/env bash
# =============================================================================
# experiments/run_spoken_squad.sh
#
# Pipeline completo Spoken-SQuAD: build PKL → train → retrieval eval → noise eval
# Roda para 1 ou 2 modos de pooling em sequência.
#
# USO:
#   bash experiments/run_spoken_squad.sh [MAX_TRAIN] [MAX_VAL] [NUM_EPOCHS]
#
# Flags via ambiente:
#   POOLING_MODES   "mean,chunked" (default) | "mean" | "chunked"
#   TRAIN_JSON      default: spoken_train-v1.1.json
#   TRAIN_WAV_DIR   default: train_wav/
#   VAL_JSON        default: spoken_test-v1.1.json
#   VAL_WAV_DIR     default: dev_wav/
#   SNR_LEVELS      default: "20,15,10,5"
#   NOISE_TYPES     default: "white,reverb"
#   WANDB_API_KEY   se setada, ativa W&B logging
#   WANDB_PROJECT   default: clasp-spoken-squad
#   SKIP_NOISE      "1" para pular noise eval
#   INIT_CKPT       (default: vazio — sem warm-start; arquitetura/distribui\u00e7\u00e3o difere)
# =============================================================================
set -euo pipefail

MAX_TRAIN="${1:-all}"
MAX_VAL="${2:-all}"
NUM_EPOCHS="${3:-50}"

POOLING_MODES="${POOLING_MODES:-mean,chunked}"
SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"
NOISE_TYPES="${NOISE_TYPES:-white,reverb}"
SKIP_NOISE="${SKIP_NOISE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-clasp-spoken-squad}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
INIT_CKPT="${INIT_CKPT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

TRAIN_JSON="${TRAIN_JSON:-${ROOT}/spoken_train-v1.1.json}"
TRAIN_WAV_DIR="${TRAIN_WAV_DIR:-${ROOT}/train_wav}"
VAL_JSON="${VAL_JSON:-${ROOT}/spoken_test-v1.1.json}"
VAL_WAV_DIR="${VAL_WAV_DIR:-${ROOT}/dev_wav}"

for f in "$TRAIN_JSON" "$VAL_JSON"; do
    [[ -f "$f" ]] || { echo "ERRO: JSON não encontrado: $f"; exit 1; }
done
for d in "$TRAIN_WAV_DIR" "$VAL_WAV_DIR"; do
    [[ -d "$d" ]] || { echo "ERRO: WAV dir não encontrado: $d"; exit 1; }
done

train_para_flag=""; [[ "$MAX_TRAIN" != "all" ]] && train_para_flag="--max-train-paragraphs ${MAX_TRAIN}"
val_para_flag="";   [[ "$MAX_VAL"   != "all" ]] && val_para_flag="--max-val-paragraphs ${MAX_VAL}"
init_flag="";       [[ -n "$INIT_CKPT" && -f "$INIT_CKPT" ]] && init_flag="--init-checkpoint ${INIT_CKPT}"

run_one_mode() {
    local MODE="$1"
    local TS
    TS="$(date +%Y%m%d_%H%M%S)_${MODE}"
    local LOG_DIR="${ROOT}/logs/spoken_squad_${TS}"
    local ARTIFACTS="${ROOT}/artifacts/spoken_squad_${TS}"
    mkdir -p "$LOG_DIR"
    if ! mkdir -p "$ARTIFACTS" 2>/dev/null; then
        ARTIFACTS="${ROOT}/artifacts_user/spoken_squad_${TS}"
        mkdir -p "$ARTIFACTS"
    fi
    local PKL="${ROOT}/data/datasets/total_dataset_spoken_squad_${TS}.pkl"
    local MODEL="${ROOT}/models/checkpoints/clasp_spoken_squad_${TS}.pt"

    local wandb_flag=""
    [[ -n "$WANDB_API_KEY" ]] && wandb_flag="--wandb-project ${WANDB_PROJECT} --wandb-run-name spoken_squad_${TS}"

    local retrieval_mode
    if [[ "$MODE" == "chunked" ]]; then
        retrieval_mode="paragraph_grouped"
    else
        retrieval_mode="candidate"
    fi

    echo "############################################################"
    echo "  POOLING MODE       : ${MODE}"
    echo "  RETRIEVAL EVAL MODE: ${retrieval_mode}"
    echo "  PKL                : ${PKL}"
    echo "  MODEL              : ${MODEL}"
    echo "  Logs               : ${LOG_DIR}"
    echo "  Artefatos          : ${ARTIFACTS}"
    echo "############################################################"

    # ----------------------- 1) build PKL ------------------------------
    echo "[1/4] Build PKL (${MODE}) …"
    # shellcheck disable=SC2086
    python scripts/build_spoken_squad_pkl.py \
        --train-json "$TRAIN_JSON" --train-wav-dir "$TRAIN_WAV_DIR" \
        --val-json   "$VAL_JSON"   --val-wav-dir   "$VAL_WAV_DIR" \
        --output     "$PKL" \
        --pooling-mode "$MODE" \
        $train_para_flag $val_para_flag \
        2>&1 | tee "${LOG_DIR}/build.log"

    # ----------------------- 2) train ----------------------------------
    echo "[2/4] Train …"
    # shellcheck disable=SC2086
    python scripts/train.py \
        --dataset-path "$PKL" --save-path "$MODEL" \
        --mode joint --num-epochs "$NUM_EPOCHS" --patience 10 \
        --batch-size-train 32 --batch-size-val 16 \
        --learning-rate 5e-6 \
        $init_flag $wandb_flag \
        2>&1 | tee "${LOG_DIR}/train.log"

    # ----------------------- 3) retrieval eval -------------------------
    echo "[3/4] Retrieval eval (${retrieval_mode}) …"
    if [[ "$retrieval_mode" == "paragraph_grouped" ]]; then
        python scripts/run_retrieval_eval.py \
            --dataset-path "$PKL" --mode paragraph_grouped \
            --model-path "$MODEL" \
            --audio-key hubert-emb --text-key text \
            --hits-k 1,5,10,50 \
            2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
    else
        local NUM_CAND
        NUM_CAND=$(python -c "
import pickle
d=pickle.load(open('$PKL','rb'))
key='test' if 'test' in d else 'validation'
print(len(d[key]['text']))")
        echo "  num-candidates = ${NUM_CAND}"
        python scripts/run_retrieval_eval.py \
            --dataset-path "$PKL" --mode candidate \
            --model-path "$MODEL" \
            --audio-key hubert-emb --text-key text \
            --threshold 0.5 --num-candidates "$NUM_CAND" \
            --hits-k 1,5,10,50 \
            2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
    fi

    # ----------------------- 4) noise eval -----------------------------
    if [[ "$SKIP_NOISE" == "1" ]]; then
        echo "[4/4] Noise eval skipped (SKIP_NOISE=1)."
    else
        echo "[4/4] Noise robustness eval …"
        python scripts/run_noise_robustness_eval.py \
            --dataset-path "$PKL" \
            --model-path "$MODEL" \
            --audio-key hubert-emb --text-key text \
            --snr-levels "$SNR_LEVELS" \
            --noise-types "$NOISE_TYPES" \
            --hits-k 1,5,10,50 \
            --output-csv "${ARTIFACTS}/noise_results.csv" \
            2>&1 | tee "${LOG_DIR}/eval_noise.log"
        echo "  Noise CSV: ${ARTIFACTS}/noise_results.csv"
    fi

    echo "==> [${MODE}] DONE. Logs: ${LOG_DIR}/  Artefatos: ${ARTIFACTS}/"
    echo ""
}

IFS=',' read -r -a MODES_ARR <<< "$POOLING_MODES"
for m in "${MODES_ARR[@]}"; do
    m="$(echo "$m" | xargs)"  # trim
    case "$m" in
        mean|chunked)  run_one_mode "$m" ;;
        "")            ;;
        *)             echo "ERRO: pooling mode inválido: ${m}"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Pipeline Spoken-SQuAD concluído para modos: ${POOLING_MODES}"
echo "============================================================"
