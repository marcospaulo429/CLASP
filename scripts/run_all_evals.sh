#!/usr/bin/env bash
# run_all_evals.sh — reproduce all noise robustness evaluations on the 4090.
#
# Fill in the PATHS section below, then run:
#   chmod +x scripts/run_all_evals.sh
#   bash scripts/run_all_evals.sh
#
# Results land in results/YYYY-MM-DD/
# W&B logging is optional — leave WANDB_PROJECT empty to disable.

set -euo pipefail

# =============================================================================
# PATHS — edit these to match the 4090 layout
# =============================================================================

CLASP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (auto-detected)

DATASET_PKL="data/datasets/total_dataset_spoken_squad.pkl"
TRAIN_JSON="data/datasets/spoken_squad/spoken_train-v1.1.json"
WAV_DIR="data/datasets/spoken_squad/train_wav"
ESC50_DIR="data/datasets/ESC-50"                 # set to "" to skip ambient

# Model checkpoints to evaluate — add/remove as needed
declare -A MODELS=(
    ["CLASP_Concat_Final"]="models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"
    ["CLASP_noise"]="models/checkpoints/CLASP_noise.pt"
    ["CLASP_noise_augmented"]="models/checkpoints/CLASP_noise_augmented.pt"
    ["CLASP_clean"]="models/checkpoints/CLASP_clean.pt"
    ["CLASP1"]="models/checkpoints/CLASP1.pt"
    ["CLASP2"]="models/checkpoints/CLASP2.pt"
)

# Evaluation parameters (matching paper's level grids)
SNR_LEVELS="30,20,10,5,0,-5,-10"
DRR_LEVELS="10,5,0,-5,-10,-15,-20"
NUM_CANDIDATES=100

# W&B — leave WANDB_PROJECT empty ("") to disable
WANDB_PROJECT="clasp-paper"
WANDB_ENTITY=""   # your W&B username or team, or leave empty

# =============================================================================
# Setup
# =============================================================================

cd "$CLASP_ROOT"

RESULTS_DIR="results/$(date +%Y-%m-%d)"
mkdir -p "$RESULTS_DIR"

PYTHON=".venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $CLASP_ROOT/.venv — run 'uv sync' first."
    exit 1
fi

# Build ambient flags
AMBIENT_FLAGS=()
if [[ -n "$ESC50_DIR" && -d "$ESC50_DIR" ]]; then
    AMBIENT_FLAGS=(--ambient-dir "$ESC50_DIR" --ambient-source esc50)
    echo "Ambient noise: ESC-50 from $ESC50_DIR"
else
    echo "Ambient noise: SKIPPED (ESC50_DIR not set or not found)"
fi

# =============================================================================
# Run evaluations
# =============================================================================

FAILED=()

for NAME in "${!MODELS[@]}"; do
    PT="${MODELS[$NAME]}"

    if [[ ! -f "$PT" ]]; then
        echo ""
        echo ">>> SKIPPING $NAME — checkpoint not found: $PT"
        FAILED+=("$NAME (missing: $PT)")
        continue
    fi

    echo ""
    echo "================================================================="
    echo ">>> Evaluating: $NAME"
    echo "    Checkpoint : $PT"
    echo "================================================================="

    CSV="$RESULTS_DIR/${NAME}.csv"

    WANDB_FLAGS=()
    if [[ -n "$WANDB_PROJECT" ]]; then
        WANDB_FLAGS=(
            --wandb-project "$WANDB_PROJECT"
            --wandb-run-name "$NAME"
        )
        [[ -n "$WANDB_ENTITY" ]] && WANDB_FLAGS+=(--wandb-entity "$WANDB_ENTITY")
    fi

    "$PYTHON" scripts/run_noise_robustness_eval.py \
        --dataset-path  "$DATASET_PKL" \
        --model-path    "$PT" \
        --train-json    "$TRAIN_JSON" \
        --wav-dir       "$WAV_DIR" \
        --snr-levels    "$SNR_LEVELS" \
        --drr-levels    "$DRR_LEVELS" \
        --num-candidates "$NUM_CANDIDATES" \
        --output-csv    "$CSV" \
        "${AMBIENT_FLAGS[@]}" \
        "${WANDB_FLAGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/${NAME}.log"

    echo ">>> Done: $CSV"
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "================================================================="
echo "ALL EVALUATIONS COMPLETE"
echo "Results saved to: $RESULTS_DIR"
echo "================================================================="

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "SKIPPED (checkpoint not found):"
    for item in "${FAILED[@]}"; do
        echo "  - $item"
    done
    echo ""
    echo "Rebuild missing checkpoints with scripts/train.py, then re-run."
fi

# Print a quick Hits@1 summary across all CSVs
echo ""
echo "Hits@1 summary:"
printf "  %-35s %s\n" "Model" "Clean → Min_noisy"
for CSV in "$RESULTS_DIR"/*.csv; do
    NAME="$(basename "$CSV" .csv)"
    CLEAN=$(awk -F, 'NR==2{print $2}' "$CSV" 2>/dev/null || echo "?")
    MIN=$(awk -F, 'NR>2{print $2}' "$CSV" 2>/dev/null | sort -n | head -1 || echo "?")
    printf "  %-35s %s → %s\n" "$NAME" "$CLEAN" "$MIN"
done
