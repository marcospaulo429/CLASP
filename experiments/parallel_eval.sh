#!/bin/bash

# parallel_eval.sh — Run multiple dataset evals in parallel
# Usage:
#   bash experiments/parallel_eval.sh -j 2 experiments/eval_queue.txt
#   bash experiments/parallel_eval.sh --config experiments/eval_queue.txt  (default -j 1)
#   bash experiments/parallel_eval.sh -j 3 voxpopuli <pkl> <model> spoken_squad <pkl> <model> spiral <model> [<pkl_noise>]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
MAX_PARALLEL=1
CONFIG_FILE=""
MANUAL_EVALS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash experiments/parallel_eval.sh [-j N] [--config FILE | EVAL1 EVAL2 ...]"
            echo ""
            echo "Options:"
            echo "  -j, --jobs N      Number of parallel jobs (default: 1)"
            echo "  --config FILE     Read eval jobs from config file (one per line)"
            echo "                    Format: <dataset_name> <pkl_or_model> [<model>] [<pkl_noise>]"
            echo ""
            echo "Manual format (without --config):"
            echo "  bash parallel_eval.sh -j 2 \\"
            echo "    voxpopuli <pkl> <model> \\"
            echo "    spoken_squad <pkl> <model>"
            echo ""
            echo "Config file example (experiments/eval_queue.txt):"
            echo "  voxpopuli data/datasets/total_dataset_voxpopuli_en.pkl models/checkpoints/clasp_vox.pt"
            echo "  spoken_squad data/datasets/total_dataset_spoken_squad_20260506_174210.pkl models/checkpoints/clasp_spoken_squad_warmstart_20260508_134151.pt"
            echo "  spiral models/checkpoints/clasp_vox.pt  # optional: spiral_dataset/data.jsonl"
            exit 0
            ;;
        *)
            MANUAL_EVALS+=("$1")
            shift
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS=()
RESULTS=()

run_eval() {
    local dataset="$1"
    local arg2="$2"
    local arg3="$3"
    local arg4="$4"
    local start_time=$(date +%s)
    
    echo -e "${YELLOW}[parallel_eval] Starting: $dataset${NC}"
    
    case "$dataset" in
        voxpopuli)
            bash "${ROOT}/experiments/eval_voxpopuli.sh" "$arg2" "$arg3" 2>&1 | sed "s/^/[voxpopuli] /"
            ;;
        spoken_squad)
            bash "${ROOT}/experiments/eval_spoken_squad.sh" "$arg2" "$arg3" 2>&1 | sed "s/^/[spoken_squad] /"
            ;;
        spiral)
            # arg2 = model, arg3 = optional pkl for noise
            if [[ -n "$arg3" ]]; then
                bash "${ROOT}/experiments/eval_spiral.sh" "$arg2" "$arg3" 2>&1 | sed "s/^/[spiral] /"
            else
                bash "${ROOT}/experiments/eval_spiral.sh" "$arg2" 2>&1 | sed "s/^/[spiral] /"
            fi
            ;;
        *)
            echo -e "${RED}[parallel_eval] Unknown dataset: $dataset${NC}"
            return 1
            ;;
    esac
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}[parallel_eval] ✓ $dataset completed in ${duration}s${NC}"
        RESULTS+=("$dataset: ✓ (${duration}s)")
    else
        echo -e "${RED}[parallel_eval] ✗ $dataset failed (exit code $exit_code)${NC}"
        RESULTS+=("$dataset: ✗ (exit code $exit_code)")
    fi
}

# Parse jobs from config file or manual args
jobs=()

if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo -e "${RED}Error: config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        jobs+=("$line")
    done < "$CONFIG_FILE"
elif [[ ${#MANUAL_EVALS[@]} -gt 0 ]]; then
    # Parse manual eval format: dataset1 arg1 arg2 [arg3] dataset2 ...
    local i=0
    while [[ $i -lt ${#MANUAL_EVALS[@]} ]]; do
        local dataset="${MANUAL_EVALS[$i]}"
        ((i++))
        local arg1="${MANUAL_EVALS[$i]:-}"
        ((i++))
        local arg2="${MANUAL_EVALS[$i]:-}"
        ((i++))
        local arg3="${MANUAL_EVALS[$i]:-}"
        
        # Check if arg3 is next dataset or argument
        if [[ "$arg3" =~ ^(voxpopuli|spoken_squad|spiral)$ ]]; then
            ((i--))
            jobs+=("$dataset $arg1 $arg2")
        else
            if [[ -n "$arg3" ]]; then
                jobs+=("$dataset $arg1 $arg2 $arg3")
                ((i++))
            else
                jobs+=("$dataset $arg1 $arg2")
            fi
        fi
    done
else
    echo -e "${RED}Error: no jobs specified. Use --config FILE or manual args${NC}"
    echo "Try: bash experiments/parallel_eval.sh -h"
    exit 1
fi

echo -e "${YELLOW}========================================${NC}"
echo "Parallel Eval Queue"
echo -e "${YELLOW}========================================${NC}"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Total jobs: ${#jobs[@]}"
echo ""

for job in "${jobs[@]}"; do
    echo "  • $job"
done
echo ""

# Run jobs with max parallelism
active_jobs=0
for job in "${jobs[@]}"; do
    # Wait if we have too many active jobs
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL ]]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset 'PIDS[$i]'
            fi
        done
        PIDS=("${PIDS[@]}")
        sleep 5
    done
    
    # Launch job in background
    run_eval $job &
    PIDS+=($!)
done

# Wait for all jobs
echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Print summary
echo ""
echo -e "${YELLOW}========================================${NC}"
echo "Summary"
echo -e "${YELLOW}========================================${NC}"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "Logs and artifacts saved in:"
echo "  Logs:      ${ROOT}/logs/"
echo "  Artifacts: ${ROOT}/artifacts_user/"
