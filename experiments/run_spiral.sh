#!/usr/bin/env bash
# =============================================================================
# experiments/run_spiral.sh
#
# Pipeline completo para SPIRAL:
#   treino com VoxPopuli (ou PKL existente) → eval retrieval SPIRAL → noise eval
#
# O SPIRAL não tem script de build de PKL próprio — usa avaliação on-the-fly
# a partir do JSONL. O treino é feito sobre um PKL já existente (VoxPopuli por
# padrão). Se quiser usar outro PKL, defina TRAIN_PKL antes de chamar o script.
#
# USO:
#   bash experiments/run_spiral.sh [NUM_EPOCHS]
#
# EXEMPLOS:
#   bash experiments/run_spiral.sh           # default: 50 epochs
#   bash experiments/run_spiral.sh 100       # 100 epochs
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   TRAIN_PKL          — PKL de treino (default: treina com VoxPopuli primeiro)
#   SPIRAL_JSONL       — JSONL de avaliação (default: spiral_dataset/data.jsonl)
#   SPIRAL_WAV_BASE    — base dir dos WAVs do SPIRAL (default: spiral_dataset/wavs/)
#   WANDB_API_KEY      — habilita logging no W&B
#   WANDB_PROJECT      — nome do projeto W&B (default: clasp-spiral)
#   SNR_LEVELS         — níveis de SNR para noise eval (default: "20,15,10,5")
#   MAX_SPIRAL_SAMPLES — limita amostras no eval SPIRAL (default: sem limite)
# =============================================================================
#
# Nota: --num-candidates é definido automaticamente como o tamanho do split
# de teste/validação do PKL de treino (dataset inteiro, sem limite).

set -euo pipefail

# --------------------------------------------------------------------------- #
# Parâmetros                                                                  #
# --------------------------------------------------------------------------- #
NUM_EPOCHS="${1:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-clasp-spiral}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
SNR_LEVELS="${SNR_LEVELS:-20,15,10,5}"
MAX_SPIRAL_SAMPLES="${MAX_SPIRAL_SAMPLES:-}"

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
LOG_DIR="${ROOT}/logs/spiral_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SPIRAL_JSONL="${SPIRAL_JSONL:-${ROOT}/spiral_dataset/data.jsonl}"
SPIRAL_WAV_BASE="${SPIRAL_WAV_BASE:-${ROOT}/spiral_dataset/wavs}"
INIT_CKPT="${ROOT}/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"
ARTIFACTS="${ROOT}/artifacts/spiral_${TIMESTAMP}"
mkdir -p "$ARTIFACTS"

# Verifica JSONL do SPIRAL
[[ -f "$SPIRAL_JSONL" ]] || { echo "ERRO: SPIRAL JSONL não encontrado: ${SPIRAL_JSONL}"; exit 1; }

# --------------------------------------------------------------------------- #
# Phase 1 — Treino (com PKL existente ou VoxPopuli)                          #
# --------------------------------------------------------------------------- #

# Usa PKL existente se TRAIN_PKL foi definido, caso contrário cria um VoxPopuli rápido
if [[ -n "${TRAIN_PKL:-}" ]]; then
    PKL="$TRAIN_PKL"
    echo "[1/3] Usando PKL de treino existente: ${PKL}"
    [[ -f "$PKL" ]] || { echo "ERRO: TRAIN_PKL não encontrado: ${PKL}"; exit 1; }
    skip_build=true
else
    PKL="${ROOT}/data/datasets/total_dataset_spiral_voxpopuli_${TIMESTAMP}.pkl"
    skip_build=false
fi

MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_spiral_${TIMESTAMP}.pt}"

init_flag="";  [[ -f "$INIT_CKPT" ]] && init_flag="--init-checkpoint ${INIT_CKPT}"
wandb_flag=""; [[ -n "$WANDB_API_KEY" ]] && wandb_flag="--wandb-project ${WANDB_PROJECT} --wandb-run-name spiral_${TIMESTAMP}"

echo "============================================================"
echo "  CLASP — SPIRAL full pipeline"
echo "  Timestamp    : ${TIMESTAMP}"
echo "  SPIRAL JSONL : ${SPIRAL_JSONL}"
echo "  Model        : ${MODEL}"
echo "  Logs         : ${LOG_DIR}"
echo "============================================================"
echo ""

if [[ "$skip_build" == "false" ]]; then
    if [[ -f "$PKL" ]]; then
        echo "[1a/3] PKL já existe, pulando build: ${PKL}"
    else
        echo "[1a/3] Building VoxPopuli PKL para treino SPIRAL …"
        python scripts/build_voxpopuli_pkl.py \
            --output "$PKL" \
            --audio-cache-dir "${ROOT}/data/datasets/voxpopuli_en_validation_wav" \
            --replicate-for-train \
            --max-samples 2000 \
            2>&1 | tee "${LOG_DIR}/build.log"
        echo "[1a/3] PKL pronto: ${PKL}"
        echo ""
    fi
fi

if [[ -f "$MODEL" ]]; then
    echo "[1/3] Modelo já existe, pulando treino: ${MODEL}"
else
    echo "[1/3] Training …"
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
    echo "[1/3] Modelo salvo: ${MODEL}"
fi
echo ""

# --------------------------------------------------------------------------- #
# Phase 2 — SPIRAL retrieval eval (on-the-fly embeddings)                    #
# --------------------------------------------------------------------------- #
echo "[2/3] SPIRAL retrieval evaluation …"

# Número de candidatos = tamanho do split de avaliação do PKL de treino
NUM_CANDIDATES=$(python -c "
import pickle
d = pickle.load(open('$PKL', 'rb'))
key = 'test' if 'test' in d else 'validation'
print(len(d[key].get('text', d[key].get('hubert-emb', []))))
")
echo "  Num candidates (split inteiro): ${NUM_CANDIDATES}"

spiral_samples_flag=""
[[ -n "$MAX_SPIRAL_SAMPLES" ]] && spiral_samples_flag="--max-samples ${MAX_SPIRAL_SAMPLES}"

# shellcheck disable=SC2086
python scripts/run_retrieval_eval.py \
    --mode spiral \
    --dataset-path "$SPIRAL_JSONL" \
    --model-path "$MODEL" \
    --spiral-audio-base "$SPIRAL_WAV_BASE" \
    --spiral-output-dir "${ARTIFACTS}/spiral_eval" \
    --num-candidates "$NUM_CANDIDATES" \
    --hits-k 1,5,10 \
    --spiral-audio-pooling mean \
    $spiral_samples_flag \
    2>&1 | tee "${LOG_DIR}/eval_spiral.log"
echo "[2/3] SPIRAL eval concluído. Resultados: ${ARTIFACTS}/spiral_eval/"
echo ""

# --------------------------------------------------------------------------- #
# Phase 3 — Noise robustness eval (sobre WAVs do SPIRAL)                     #
# --------------------------------------------------------------------------- #
echo "[3/3] Noise robustness evaluation (SPIRAL WAVs) …"

# Para noise eval precisamos de um PKL com split test; reutilizamos o PKL de treino
python scripts/run_noise_robustness_eval.py \
    --dataset-path "$PKL" \
    --model-path "$MODEL" \
    --audio-paths-from-pickle \
    --num-candidates "$NUM_CANDIDATES" \
    --snr-levels "$SNR_LEVELS" \
    --output-csv "${ARTIFACTS}/noise_results.csv" \
    2>&1 | tee "${LOG_DIR}/eval_noise.log"
echo "[3/3] Noise eval concluído. CSV: ${ARTIFACTS}/noise_results.csv"

echo ""
echo "============================================================"
echo "  Pipeline SPIRAL concluído!"
echo "  Modelo   : ${MODEL}"
echo "  Artefatos: ${ARTIFACTS}/"
echo "  Logs     : ${LOG_DIR}/"
echo "============================================================"
