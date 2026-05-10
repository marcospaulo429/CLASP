#!/usr/bin/env bash
# =============================================================================
# experiments/run_voxpopuli.sh
#
# Pipeline completo: build PKL → treino → eval retrieval → noise robustness
# Dataset: VoxPopuli EN (HuggingFace facebook/voxpopuli)
#
# USO:
#   bash experiments/run_voxpopuli.sh [MAX_TRAIN] [MAX_VAL] [NUM_EPOCHS]
#
# EXEMPLOS:
#   bash experiments/run_voxpopuli.sh                  # defaults
#   bash experiments/run_voxpopuli.sh 5000 500 100     # 5k treino, 500 val, 100 epochs
#   bash experiments/run_voxpopuli.sh all all 50       # dataset completo
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   WANDB_API_KEY    — habilita logging no W&B
#   WANDB_PROJECT    — nome do projeto W&B (default: clasp-voxpopuli)
#   SNR_LEVELS       — níveis de SNR para noise eval (default: "20,15,10,5")
# =============================================================================
#
# Nota: --num-candidates é definido automaticamente como o tamanho do split
# de teste do PKL (dataset inteiro, sem limite).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Parâmetros                                                                  #
# --------------------------------------------------------------------------- #
MAX_TRAIN="${1:-all}"
MAX_VAL="${2:-all}"
NUM_EPOCHS="${3:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-clasp-voxpopuli}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    source "${ROOT}/venv/bin/activate"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/voxpopuli_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

PKL="${PKL:-${ROOT}/data/datasets/total_dataset_voxpopuli_${TIMESTAMP}.pkl}"
MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_voxpopuli_${TIMESTAMP}.pt}"
INIT_CKPT="${ROOT}/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"
TRAIN_WAV_DIR="${ROOT}/data/datasets/voxpopuli_en_train_wav"
VAL_WAV_DIR="${ROOT}/data/datasets/voxpopuli_en_validation_wav"
ARTIFACTS="$LOG_DIR"
# artifacts/ may be owned by root; use LOG_DIR as fallback

# flags condicionais
train_flag=""; [[ "$MAX_TRAIN" != "all" ]] && train_flag="--max-samples ${MAX_TRAIN}"
val_flag="";   [[ "$MAX_VAL"   != "all" ]] && val_flag="--max-val-samples ${MAX_VAL}"
init_flag="";  [[ -f "$INIT_CKPT" ]] && init_flag="--init-checkpoint ${INIT_CKPT}"
wandb_flag=""; [[ -n "$WANDB_API_KEY" ]] && wandb_flag="--wandb-project ${WANDB_PROJECT} --wandb-run-name voxpopuli_${TIMESTAMP}"

# --------------------------------------------------------------------------- #
echo "============================================================"
echo "  CLASP — VoxPopuli full pipeline"
echo "  Timestamp : ${TIMESTAMP}"
echo "  PKL       : ${PKL}"
echo "  Model     : ${MODEL}"
echo "  Logs      : ${LOG_DIR}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# Phase 1 — Build PKL                                                         #
# --------------------------------------------------------------------------- #
if [[ -f "$PKL" ]]; then
    echo "[1/4] PKL já existe, pulando build: ${PKL}"
else
    echo "[1/4] Building VoxPopuli dataset PKL …"
    # shellcheck disable=SC2086
    python scripts/build_voxpopuli_pkl.py \
        --hf-split train \
        --output "$PKL" \
        --train-audio-cache-dir "$TRAIN_WAV_DIR" \
        --audio-cache-dir "$VAL_WAV_DIR" \
        --require-gold-only \
        $train_flag \
        $val_flag \
        2>&1 | tee "${LOG_DIR}/build.log"
    echo "[1/4] PKL pronto: ${PKL}"
fi
echo ""

# --------------------------------------------------------------------------- #
# Phase 2 — Treino                                                            #
# --------------------------------------------------------------------------- #
if [[ -f "$MODEL" ]]; then
    echo "[2/4] Modelo já existe, pulando treino: ${MODEL}"
else
    echo "[2/4] Training …"
    # shellcheck disable=SC2086
    python scripts/train.py \
        --dataset-path "$PKL" \
        --save-path "$MODEL" \
        --mode joint \
        --num-epochs "$NUM_EPOCHS" \
        --patience 10 \
        --batch-size-train 32 \
        --batch-size-val 16 \
        --learning-rate 5e-6 \
        $init_flag \
        $wandb_flag \
        2>&1 | tee "${LOG_DIR}/train.log"
    echo "[2/4] Modelo salvo: ${MODEL}"
fi
echo ""

# --------------------------------------------------------------------------- #
# Phase 3 — Retrieval eval (candidate mode)                                   #
# --------------------------------------------------------------------------- #
echo "[3/4] Retrieval evaluation (candidate mode) …"

# Usa o split de test inteiro como pool de candidatos
NUM_CANDIDATES=$(python -c "
import pickle
d = pickle.load(open('$PKL', 'rb'))
key = 'test' if 'test' in d else 'validation'
print(len(d[key].get('text', d[key].get('hubert-emb', []))))
")
echo "  Num candidates (test split inteiro): ${NUM_CANDIDATES}"

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
echo "[3/4] Retrieval eval concluído."
echo ""

# --------------------------------------------------------------------------- #
# Phase 4 — Noise robustness eval                                             #
# --------------------------------------------------------------------------- #
echo "[4/4] Noise robustness evaluation …"
python scripts/run_noise_robustness_eval.py \
    --dataset-path "$PKL" \
    --model-path "$MODEL" \
    --audio-paths-from-pickle \
    --num-candidates "$NUM_CANDIDATES" \
    --snr-levels "$SNR_LEVELS" \
    --output-csv "${ARTIFACTS}/noise_results.csv" \
    2>&1 | tee "${LOG_DIR}/eval_noise.log"
echo "[4/4] Noise eval concluído. CSV: ${ARTIFACTS}/noise_results.csv"

echo ""
echo "============================================================"
echo "  Pipeline VoxPopuli concluído!"
echo "  Modelo   : ${MODEL}"
echo "  Artefatos: ${ARTIFACTS}/"
echo "  Logs     : ${LOG_DIR}/"
echo "============================================================"
