#!/usr/bin/env bash
# =============================================================================
# experiments/run_seedvc_all_voices.sh
#
# Build PKL + eval para cada voz do spoken_squad_seed-vc, em paralelo.
#
# USO:
#   bash experiments/run_seedvc_all_voices.sh
#
# VARIÁVEIS DE AMBIENTE OPCIONAIS:
#   VOICES       — lista de vozes separadas por espaço (default: todas)
#   POOLING_MODE — mean ou chunked (default: mean)
#   MODEL        — caminho do modelo (default: CLASP_Concat_Final_Fusion_Encoder.pt)
#   SKIP_NOISE   — "1" para pular noise eval (default: 0)
#   MAX_PARALLEL — workers paralelos (default: 3; reduza para 2 se der OOM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

VOICES="${VOICES:-1089-134686-0000 2803-154320-0012 3081-166546-0023 6319-275224-0006 azuma trump}"
POOLING_MODE="${POOLING_MODE:-mean}"
MODEL="${MODEL:-${ROOT}/models/checkpoints/clasp_spoken_squad%2Bvoxpopuli.pt}"
SKIP_NOISE="${SKIP_NOISE:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

TRAIN_JSON="${ROOT}/data/datasets/spoken_squad/spoken_train-v1.1.json"
VAL_JSON="${ROOT}/data/datasets/spoken_squad/spoken_test-v1.1.json"
SEEDVC_DIR="${ROOT}/data/datasets/spoken_squad_seed-vc"
mkdir -p "${ROOT}/logs"

[[ -f "$MODEL" ]]      || { echo "ERRO: modelo não encontrado: $MODEL"; exit 1; }
[[ -f "$TRAIN_JSON" ]] || { echo "ERRO: JSON não encontrado: $TRAIN_JSON"; exit 1; }
[[ -f "$VAL_JSON" ]]   || { echo "ERRO: JSON não encontrado: $VAL_JSON"; exit 1; }

echo "============================================================"
echo "  seed-vc all-voices pipeline (paralelo)"
echo "  Vozes        : ${VOICES}"
echo "  Pooling      : ${POOLING_MODE}"
echo "  Modelo eval  : ${MODEL}"
echo "  Max parallel : ${MAX_PARALLEL}"
echo "  Skip noise   : ${SKIP_NOISE}"
echo "============================================================"
echo ""

pids=()
running=0

for VOZ in $VOICES; do
    WAV_DIR="${SEEDVC_DIR}/${VOZ}"
    PKL="${ROOT}/data/datasets/total_dataset_seedvc_${VOZ}_${POOLING_MODE}.pkl"
    LOG="${ROOT}/logs/seedvc_${VOZ}.log"

    if [[ ! -d "$WAV_DIR" ]]; then
        echo "[${VOZ}] AVISO: diretório não encontrado, pulando."
        continue
    fi

    echo "[${VOZ}] Iniciando — log: ${LOG}"

    (
        set -euo pipefail
        if [[ -f "$PKL" ]]; then
            echo "[${VOZ}] PKL já existe, pulando build."
        else
            echo "[${VOZ}] Build PKL ..."
            python scripts/build_spoken_squad_pkl.py \
                --train-json    "$TRAIN_JSON" \
                --train-wav-dir "$WAV_DIR" \
                --val-json      "$VAL_JSON" \
                --val-wav-dir   "$WAV_DIR" \
                --output        "$PKL" \
                --pooling-mode  "$POOLING_MODE"
            echo "[${VOZ}] PKL pronto."
        fi

        echo "[${VOZ}] Eval ..."
        SKIP_NOISE="$SKIP_NOISE" bash experiments/eval_spoken_squad.sh "$PKL" "$MODEL"
        echo "[${VOZ}] CONCLUÍDO."
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
echo "  Todas as vozes concluídas!"
echo "  Logs em: ${ROOT}/logs/seedvc_<voz>.log"
echo "============================================================"
