#!/bin/bash

# eval_spoken_squad_parallel.sh — Run retrieval + noise evals in parallel
# Usage:
#   bash experiments/eval_spoken_squad_parallel.sh <pkl> <model>

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: bash experiments/eval_spoken_squad_parallel.sh <pkl> <model>"
    echo ""
    echo "Example:"
    echo "  bash experiments/eval_spoken_squad_parallel.sh \\"
    echo "    data/datasets/total_dataset_spoken_squad_20260506_174210.pkl \\"
    echo "    models/checkpoints/clasp_spoken_squad_warmstart_20260508_134151.pt"
    exit 1
fi

PKL="$1"
MODEL="$2"

if [[ ! -f "$PKL" ]]; then
    echo "Error: PKL not found: $PKL"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate venv if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    source "${ROOT}/venv/bin/activate"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${ROOT}/logs/eval_spoken_squad_parallel_${TIMESTAMP}"
ARTIFACTS="${ROOT}/artifacts_user/eval_spoken_squad_parallel_${TIMESTAMP}"

mkdir -p "$LOG_DIR" "$ARTIFACTS"

# Get number of candidates for full split
N=$(python -c "import pickle; d=pickle.load(open('$PKL','rb')); k='test' if 'test' in d else 'validation'; print(len(d[k]['text']))")

echo "============================================================"
echo "  CLASP — Spoken SQuAD eval (retrieval + noise in parallel)"
echo "  PKL            : $PKL"
echo "  Model          : $MODEL"
echo "  Val JSON       : ${ROOT}/spoken_test-v1.1.json"
echo "  Num candidates : $N (split inteiro)"
echo "  Logs           : $LOG_DIR"
echo "  Artefatos      : $ARTIFACTS"
echo "============================================================"
echo ""

# Function to run retrieval eval
run_retrieval() {
    echo "[retrieval] Starting candidate mode evaluation..."
    python "${ROOT}/scripts/run_retrieval_eval.py" \
        --dataset-path "$PKL" \
        --mode candidate \
        --model-path "$MODEL" \
        --audio-key hubert-emb \
        --text-key text \
        --threshold 0.5 \
        --num-candidates "$N" \
        --hits-k 1,5,10,50 \
        2>&1 | tee "${LOG_DIR}/eval_retrieval.log"
    echo "[retrieval] ✓ Completed"
}

# Function to run noise robustness eval (4 SNR levels in parallel)
run_noise() {
    echo "[noise] Starting noise robustness evaluation (4 SNR levels in parallel)..."
    
    SNR_LEVELS=(20 15 10 5)
    PIDS=()
    
    for snr in "${SNR_LEVELS[@]}"; do
        echo "[noise-$snr] Starting SNR $snr dB..."
        python "${ROOT}/scripts/run_noise_robustness_eval.py" \
            --dataset-path "$PKL" \
            --model-path "$MODEL" \
            --train-json "${ROOT}/spoken_test-v1.1.json" \
            --wav-dir "${ROOT}/dev_wav/" \
            --num-candidates "$N" \
            --snr-levels "$snr" \
            --output-csv "${ARTIFACTS}/noise_results_snr${snr}.csv" \
            2>&1 | tee "${LOG_DIR}/eval_noise_snr${snr}.log" &
        PIDS+=($!)
    done
    
    # Wait for all SNR levels
    NOISE_FAILED=0
    for i in "${!PIDS[@]}"; do
        snr=${SNR_LEVELS[$i]}
        pid=${PIDS[$i]}
        if wait $pid 2>/dev/null; then
            echo "[noise-$snr] ✓ SNR $snr completed"
        else
            echo "[noise-$snr] ✗ SNR $snr failed"
            NOISE_FAILED=1
        fi
    done
    
    # Merge all CSV results
    if [[ $NOISE_FAILED -eq 0 ]]; then
        echo "[noise] Merging SNR results into single CSV..."
        {
            head -1 "${ARTIFACTS}/noise_results_snr20.csv"
            for snr in "${SNR_LEVELS[@]}"; do
                tail -n +2 "${ARTIFACTS}/noise_results_snr${snr}.csv"
            done
        } > "${ARTIFACTS}/noise_results.csv"
        echo "[noise] ✓ Results merged to: ${ARTIFACTS}/noise_results.csv"
    else
        echo "[noise] ✗ Some SNR levels failed, skipping merge"
        return 1
    fi
}

echo "Starting parallel evals..."
echo ""

# Start both in background
run_retrieval &
RETRIEVAL_PID=$!

run_noise &
NOISE_PID=$!

echo "[main] Retrieval PID: $RETRIEVAL_PID"
echo "[main] Noise PID: $NOISE_PID"
echo ""

# Wait for both
RETRIEVAL_EXIT=0
NOISE_EXIT=0

wait $RETRIEVAL_PID 2>/dev/null || RETRIEVAL_EXIT=$?
wait $NOISE_PID 2>/dev/null || NOISE_EXIT=$?

echo ""
echo "============================================================"
echo "  Results Summary"
echo "============================================================"

if [[ $RETRIEVAL_EXIT -eq 0 ]]; then
    echo "✓ Retrieval eval: SUCCESS"
    echo "  Log: ${LOG_DIR}/eval_retrieval.log"
else
    echo "✗ Retrieval eval: FAILED (exit code $RETRIEVAL_EXIT)"
fi

if [[ $NOISE_EXIT -eq 0 ]]; then
    echo "✓ Noise robustness eval: SUCCESS"
    echo "  Log: ${LOG_DIR}/eval_noise.log"
    echo "  Results: ${ARTIFACTS}/noise_results.csv"
else
    echo "✗ Noise robustness eval: FAILED (exit code $NOISE_EXIT)"
fi

echo ""

# Exit with failure if either job failed
if [[ $RETRIEVAL_EXIT -ne 0 ]] || [[ $NOISE_EXIT -ne 0 ]]; then
    exit 1
fi
