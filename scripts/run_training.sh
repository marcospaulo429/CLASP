#!/usr/bin/env bash
# =============================================================================
# run_training.sh — Treino completo do CLASP com VoxPopuli EN (train split)
#
# USO:
#   bash scripts/run_training.sh [MAX_TRAIN_SAMPLES] [MAX_VAL_SAMPLES] [NUM_EPOCHS]
#
# EXEMPLOS:
#   bash scripts/run_training.sh              # defaults: 10000 train / todos val / 50 epochs
#   bash scripts/run_training.sh 5000         # 5000 amostras de treino
#   bash scripts/run_training.sh 50000 1000   # 50k train, 1000 validation
#   bash scripts/run_training.sh 182000 all 100  # dataset completo, 100 epochs
#
# ARGUMENTOS:
#   MAX_TRAIN_SAMPLES  — nº máximo de amostras do train split a processar
#                        ("all" ou vazio = sem limite)
#   MAX_VAL_SAMPLES    — nº máximo de amostras do validation split
#                        ("all" ou vazio = sem limite, usa as ~1752 disponíveis)
#   NUM_EPOCHS         — número de epochs de treino (default 50)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Parâmetros                                                               #
# --------------------------------------------------------------------------- #
MAX_TRAIN="${1:-10000}"
MAX_VAL="${2:-all}"
NUM_EPOCHS="${3:-50}"

# W&B — defina antes de chamar o script ou exporte no ambiente:
#   export WANDB_API_KEY=sua_chave
#   export WANDB_PROJECT=meu-projeto   (default: clasp-voxpopuli)
WANDB_PROJECT="${WANDB_PROJECT:-clasp-voxpopuli}"
WANDB_API_KEY="${WANDB_API_KEY:-}"  # deve vir do ambiente

# Mapeia "all" ou string vazia para sem-limite (sem flag ao script Python)
train_samples_flag=""
if [[ "$MAX_TRAIN" != "all" && "$MAX_TRAIN" != "" ]]; then
    train_samples_flag="--max-samples ${MAX_TRAIN}"
fi

val_samples_flag=""
if [[ "$MAX_VAL" != "all" && "$MAX_VAL" != "" ]]; then
    val_samples_flag="--max-val-samples ${MAX_VAL}"
fi

# --------------------------------------------------------------------------- #
# 1. Paths (relativos à raiz do projeto)                                      #
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

PKL_OUTPUT="${ROOT}/data/datasets/total_dataset_voxpopuli_en_train.pkl"
TRAIN_WAV_DIR="${ROOT}/data/datasets/voxpopuli_en_train_wav"
VAL_WAV_DIR="${ROOT}/data/datasets/voxpopuli_en_validation_wav"
INIT_CHECKPOINT="${ROOT}/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"
SAVE_MODEL="${ROOT}/models/checkpoints/clasp_voxpopuli_finetuned.pt"

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BUILD_LOG="${LOG_DIR}/build_${TIMESTAMP}.log"
TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
WANDB_RUN_NAME="voxpopuli_train_${TIMESTAMP}"

# --------------------------------------------------------------------------- #
# 2. Resumo                                                                   #
# --------------------------------------------------------------------------- #
echo "============================================================"
echo "  CLASP VoxPopuli Training Pipeline"
echo "============================================================"
echo "  Max train samples : ${MAX_TRAIN}"
echo "  Max val samples   : ${MAX_VAL}"
echo "  Epochs            : ${NUM_EPOCHS}"
echo "  PKL output        : ${PKL_OUTPUT}"
echo "  Model output      : ${SAVE_MODEL}"
echo "  W&B project       : ${WANDB_PROJECT}"
echo "  W&B run name      : ${WANDB_RUN_NAME}"
echo "  Build log         : ${BUILD_LOG}"
echo "  Train log         : ${TRAIN_LOG}"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
# 3. Phase 1 — Build do dataset PKL                                           #
# --------------------------------------------------------------------------- #
echo "[Phase 1/2] Building dataset pickle …"
echo "  (este passo pode demorar bastante com muitas amostras)"
echo ""

cd "$ROOT"

# shellcheck disable=SC2086
python scripts/build_voxpopuli_pkl.py \
    --hf-split train \
    --output "$PKL_OUTPUT" \
    --train-audio-cache-dir "$TRAIN_WAV_DIR" \
    --audio-cache-dir "$VAL_WAV_DIR" \
    --require-gold-only \
    $train_samples_flag \
    $val_samples_flag \
    2>&1 | tee "$BUILD_LOG"

echo ""
echo "[Phase 1/2] Dataset pronto: ${PKL_OUTPUT}"

# Verificação rápida do pkl
python - <<'PY'
import pickle, sys
pkl_path = sys.argv[1] if len(sys.argv) > 1 else None
import os
pkl_path = os.environ.get("_PKL_PATH")
if pkl_path:
    d = pickle.load(open(pkl_path, "rb"))
    for split, data in d.items():
        n = len(data.get("hubert-emb", []))
        shapes = {k: (list(data[k][0].shape) if data[k] else "empty") for k in data if k != "audio_path"}
        print(f"  [{split}] {n} samples — embedding shapes: {shapes}")
PY
export _PKL_PATH="$PKL_OUTPUT"
python - <<'PY'
import pickle, os
pkl_path = os.environ["_PKL_PATH"]
d = pickle.load(open(pkl_path, "rb"))
print("\nPKL verification:")
for split, data in d.items():
    n = len(data.get("hubert-emb", []))
    shapes = {}
    for k in ("hubert-emb", "text", "image"):
        if data.get(k):
            shapes[k] = list(data[k][0].shape)
    print(f"  [{split:12s}] {n:>6d} samples  embedding shapes: {shapes}")
PY

echo ""

# --------------------------------------------------------------------------- #
# 4. Phase 2 — Treino                                                         #
# --------------------------------------------------------------------------- #
echo "[Phase 2/2] Starting training …"
echo ""

init_flag=""
if [[ -f "$INIT_CHECKPOINT" ]]; then
    echo "  Using init checkpoint: ${INIT_CHECKPOINT}"
    init_flag="--init-checkpoint ${INIT_CHECKPOINT}"
else
    echo "  WARNING: checkpoint não encontrado (${INIT_CHECKPOINT}), treinando do zero."
fi

wandb_flag=""
if [[ -n "$WANDB_API_KEY" ]]; then
    wandb_flag="--wandb-project ${WANDB_PROJECT} --wandb-run-name ${WANDB_RUN_NAME}"
    echo "  W&B logging ENABLED (project=${WANDB_PROJECT})"
else
    echo "  W&B logging DISABLED (WANDB_API_KEY não definida)"
fi

# shellcheck disable=SC2086
python scripts/train.py \
    --dataset-path "$PKL_OUTPUT" \
    --save-path "$SAVE_MODEL" \
    --mode joint \
    --num-epochs "$NUM_EPOCHS" \
    --patience 10 \
    --batch-size-train 32 \
    --batch-size-val 16 \
    --learning-rate 5e-6 \
    $init_flag \
    $wandb_flag \
    2>&1 | tee "$TRAIN_LOG"

echo ""
echo "============================================================"
echo "  Treinamento concluído!"
echo "  Modelo salvo em: ${SAVE_MODEL}"
echo "  Logs em:         ${LOG_DIR}/"
echo "============================================================"
