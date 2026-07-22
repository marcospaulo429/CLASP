#!/usr/bin/env bash
# =============================================================================
# experiments/run_emotions_all_voices.sh
#
# Build PKL + eval para cada combinação emoção/intensidade do spoken_squad_emotions.
# Estrutura esperada: data/datasets/spoken_squad_emotions/{emotion}/{intensity}/
#
# USO:
#   bash experiments/run_emotions_all_voices.sh
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   COMBOS       — lista "emotion/intensity" separadas por espaço (default: todas)
#   POOLING_MODE — mean ou chunked (default: mean)
#   MODEL        — caminho do modelo (default: CLASP_Concat_Final_Fusion_Encoder.pt)
#   SKIP_NOISE   — "1" para pular noise eval (default: 1)
#   MAX_PARALLEL — workers paralelos (default: 3; reduza para 2 se der OOM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

COMBOS="${COMBOS:-angry/normal angry/strong happy/normal happy/strong neutral/normal sad/normal sad/strong}"
POOLING_MODE="${POOLING_MODE:-mean}"
MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_spoken_squad%2Bvoxpopuli.pt}"
SKIP_NOISE="${SKIP_NOISE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

TRAIN_JSON="${ROOT}/data/datasets/spoken_squad/spoken_train-v1.1.json"
VAL_JSON="${ROOT}/data/datasets/spoken_squad/spoken_test-v1.1.json"
EMOTIONS_DIR="${ROOT}/data/datasets/spoken_squad_emotions"
mkdir -p "${ROOT}/logs"

[[ -f "$MODEL" ]]      || { echo "ERRO: modelo não encontrado: $MODEL"; exit 1; }
[[ -f "$TRAIN_JSON" ]] || { echo "ERRO: JSON não encontrado: $TRAIN_JSON"; exit 1; }
[[ -f "$VAL_JSON" ]]   || { echo "ERRO: JSON não encontrado: $VAL_JSON"; exit 1; }

echo "============================================================"
echo "  emotions all-voices pipeline (paralelo)"
echo "  Combos       : ${COMBOS}"
echo "  Pooling      : ${POOLING_MODE}"
echo "  Modelo eval  : ${MODEL}"
echo "  Max parallel : ${MAX_PARALLEL}"
echo "  Skip noise   : ${SKIP_NOISE}"
echo "============================================================"
echo ""

pids=()
running=0

for COMBO in $COMBOS; do
    EMOTION="${COMBO%%/*}"
    INTENSITY="${COMBO##*/}"
    WAV_DIR="${EMOTIONS_DIR}/${EMOTION}/${INTENSITY}"
    TAG="${EMOTION}_${INTENSITY}"
    PKL="${ROOT}/data/datasets/total_dataset_emotions_${TAG}_${POOLING_MODE}.pkl"
    LOG="${ROOT}/logs/emotions_${TAG}.log"

    if [[ ! -d "$WAV_DIR" ]]; then
        echo "[${TAG}] AVISO: diretório não encontrado, pulando."
        continue
    fi

    echo "[${TAG}] Iniciando — log: ${LOG}"

    (
        set -euo pipefail
        if [[ -f "$PKL" ]]; then
            echo "[${TAG}] PKL já existe, pulando build."
        else
            echo "[${TAG}] Build PKL ..."
            python scripts/build_spoken_squad_pkl.py \
                --train-json    "$TRAIN_JSON" \
                --train-wav-dir "$WAV_DIR" \
                --val-json      "$VAL_JSON" \
                --val-wav-dir   "$WAV_DIR" \
                --output        "$PKL" \
                --pooling-mode  "$POOLING_MODE"
            echo "[${TAG}] PKL pronto."
        fi

        echo "[${TAG}] Eval ..."
        SKIP_NOISE="$SKIP_NOISE" bash experiments/eval_spoken_squad.sh "$PKL" "$MODEL"
        echo "[${TAG}] CONCLUÍDO."
    ) > "$LOG" 2>&1 &

    pids+=($!)
    running=$(( running + 1 ))

    if [[ $running -ge $MAX_PARALLEL ]]; then
        wait -n
        running=$(( running - 1 ))
    fi
done

wait "${pids[@]}"

echo ""
echo "============================================================"
echo "  Todas as combinações concluídas!"
echo "  Logs em: ${ROOT}/logs/emotions_<emotion>_<intensity>.log"
echo "============================================================"
