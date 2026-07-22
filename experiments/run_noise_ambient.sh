#!/usr/bin/env bash
# =============================================================================
# experiments/run_noise_ambient.sh
#
# Spoken-SQuAD noise robustness eval with ambient (ESC-50) + white + reverb,
# at SNR ∈ {5, 10, 20} dB.
#
# USAGE:
#   bash experiments/run_noise_ambient.sh <PKL> <MODEL>
#
# ENV OVERRIDES:
#   SNR_LEVELS    default: "20,10,5"
#   NOISE_TYPES   default: "white,reverb,ambient"
#   ESC50_DIR     default: data/noise_sources/esc50 (auto-detected if present)
#   WHAM_DIR      optional alternative ambient source (used if set)
#   AMBIENT_N     default: 64  (ambient WAVs pre-loaded into memory pool)
#   CHUNK_BATCH   default: 4   (chunk batch size for HuBERT/EfficientNet)
#   HITS_K        default: "1,5,10,50"
# =============================================================================
set -euo pipefail

PKL="${1:-}"; MODEL="${2:-}"
[[ -z "$PKL" || -z "$MODEL" ]] && {
    echo "USO: bash experiments/run_noise_ambient.sh <PKL> <MODEL>"
    exit 1
}
[[ -f "$PKL"   ]] || { echo "ERRO: PKL não encontrado: $PKL"; exit 1; }
[[ -f "$MODEL" ]] || { echo "ERRO: MODEL não encontrado: $MODEL"; exit 1; }

SNR_LEVELS="${SNR_LEVELS:-20,10,5}"
NOISE_TYPES="${NOISE_TYPES:-white,reverb,ambient}"
AMBIENT_N="${AMBIENT_N:-64}"
CHUNK_BATCH="${CHUNK_BATCH:-4}"
HITS_K="${HITS_K:-1,5,10,50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate"

# Auto-detect ESC-50 unless caller overrode it
ESC50_DIR="${ESC50_DIR:-${ROOT}/data/noise_sources/esc50}"
WHAM_DIR="${WHAM_DIR:-}"

# Build the ambient-source flag(s)
AMBIENT_FLAGS=()
if [[ ",$NOISE_TYPES," == *",ambient,"* ]]; then
    if [[ -n "$WHAM_DIR" && -d "$WHAM_DIR/noise" ]]; then
        AMBIENT_FLAGS+=(--wham-dir "$WHAM_DIR")
    elif [[ -d "$ESC50_DIR/audio" ]]; then
        AMBIENT_FLAGS+=(--esc50-dir "$ESC50_DIR")
    else
        echo "ERRO: ambient noise pedido mas nenhuma fonte encontrada."
        echo "  Tente: bash scripts/download_esc50.sh"
        echo "  Ou defina ESC50_DIR / WHAM_DIR."
        exit 1
    fi
fi

POOLING=$(python - <<PY
import pickle
d = pickle.load(open("$PKL", "rb"))
print((d.get("_meta") or {}).get("pooling_mode", ""))
PY
)
[[ -z "$POOLING" ]] && {
    echo "ERRO: PKL sem _meta.pooling_mode (rebuild com o novo build_spoken_squad_pkl.py)"
    exit 1
}

TS="$(date +%Y%m%d_%H%M%S)_${POOLING}"
LOG_DIR="${ROOT}/logs/noise_ambient_${TS}"
ARTIFACTS="${ROOT}/artifacts_user/noise_ambient_${TS}"
mkdir -p "$LOG_DIR" "$ARTIFACTS"

echo "============================================================"
echo "  Noise robustness (incl. ESC-50 ambient)"
echo "  PKL              : ${PKL}"
echo "  Model            : ${MODEL}"
echo "  Pooling mode     : ${POOLING}"
echo "  Noise types      : ${NOISE_TYPES}"
echo "  SNR levels (dB)  : ${SNR_LEVELS}"
echo "  Ambient flags    : ${AMBIENT_FLAGS[*]:-<none>}"
echo "  Ambient pool     : ${AMBIENT_N} WAVs"
echo "  Chunk batch size : ${CHUNK_BATCH}"
echo "  Logs             : ${LOG_DIR}"
echo "  Artifacts        : ${ARTIFACTS}"
echo "============================================================"

python scripts/run_noise_robustness_eval.py \
    --dataset-path "$PKL" \
    --model-path "$MODEL" \
    --audio-key hubert-emb --text-key text \
    --snr-levels "$SNR_LEVELS" --noise-types "$NOISE_TYPES" \
    --ambient-num-samples "$AMBIENT_N" \
    --chunk-batch-size "$CHUNK_BATCH" \
    --hits-k "$HITS_K" \
    --output-csv "${ARTIFACTS}/noise_results.csv" \
    "${AMBIENT_FLAGS[@]}" \
    2>&1 | tee "${LOG_DIR}/eval_noise.log"

echo ""
echo "DONE — CSV: ${ARTIFACTS}/noise_results.csv"
echo "       Log: ${LOG_DIR}/eval_noise.log"
