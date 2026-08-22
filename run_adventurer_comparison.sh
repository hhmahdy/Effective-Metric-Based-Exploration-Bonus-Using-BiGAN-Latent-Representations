#!/usr/bin/env bash
# Run two Adventurer methods over multiple random seeds and generate comparisons.
#
# Methods:
#   1. Adventurer: original state novelty, alpha=0.9
#   2. Adventurer + Transition Novelty

set -euo pipefail

ENVIRONMENT_ID="${ENVIRONMENT_ID:-FetchPickAndPlace-v3}"
SEEDS="${SEEDS:-0 1 2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-96}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
TOTAL_ENVIRONMENT_STEPS="${TOTAL_ENVIRONMENT_STEPS:-12288000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"
BIGAN_BATCH_SIZE="${BIGAN_BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-10}"
ROOT_RUN="${ROOT_RUN:-runs/adventurer-comparison}"

if (( TOTAL_ENVIRONMENT_STEPS % (NUM_PARALLEL_ENVS * ROLLOUT_STEPS) != 0 )); then
    echo "ERROR: TOTAL_ENVIRONMENT_STEPS must be divisible by NUM_PARALLEL_ENVS * ROLLOUT_STEPS" >&2
    echo "       ${TOTAL_ENVIRONMENT_STEPS} % (${NUM_PARALLEL_ENVS} * ${ROLLOUT_STEPS}) != 0" >&2
    exit 1
fi

run_method() {
    local seed="$1"
    local novelty_type="$2"
    local output_directory="$3"
    shift 3
    local extra_args=("$@")

    echo "------------------------------------------------------------"
    echo "Seed: ${seed}"
    echo "Novelty: ${novelty_type}"
    echo "Output: ${output_directory}"
    echo "------------------------------------------------------------"

    python -u main.py \
        --environment-id "${ENVIRONMENT_ID}" \
        --novelty-type "${novelty_type}" \
        --num-parallel-envs "${NUM_PARALLEL_ENVS}" \
        --seed "${seed}" \
        --total-environment-steps "${TOTAL_ENVIRONMENT_STEPS}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --minibatch-size "${MINIBATCH_SIZE}" \
        --bigan-batch-size "${BIGAN_BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output-directory "${output_directory}" \
        --plot-directory "${output_directory}/plt" \
        --tensorboard-directory "${output_directory}/tensorboard" \
        "${extra_args[@]}"
}

for seed in ${SEEDS}; do
    SEED_ROOT="${ROOT_RUN}/seed_${seed}"
    BASELINE_RUN="${SEED_ROOT}/adventurer"
    TRANSITION_RUN="${SEED_ROOT}/adventurer-transition"
    COMPARISON_RUN="${SEED_ROOT}/comparison"

    echo "============================================================"
    echo "Starting two-method comparison for seed ${seed}"
    echo "============================================================"

    run_method "${seed}" "state" "${BASELINE_RUN}" --state-alpha 0.9
    run_method "${seed}" "transition" "${TRANSITION_RUN}"

    python - <<'PY' "${BASELINE_RUN}" "${TRANSITION_RUN}" "${COMPARISON_RUN}" "${SMOOTHING_WINDOW}"
import sys
from pathlib import Path

from evaluation.comparison import save_multi_game_score_comparison

baseline_run = Path(sys.argv[1])
transition_run = Path(sys.argv[2])
comparison_run = Path(sys.argv[3])
smoothing_window = int(sys.argv[4])

runs = {
    "Adventurer": baseline_run,
    "Adventurer + Transition Novelty": transition_run,
}

csv_path, png_path = save_multi_game_score_comparison(
    runs,
    comparison_run,
    smoothing_window=smoothing_window,
    dpi=300,
    metric_name="reward/total",
    y_label="Total Reward",
    file_stem="total_reward_comparison",
)

print("CSV:", csv_path)
print("PNG:", png_path)
PY

done

echo "============================================================"
echo "All seed comparisons completed"
echo "Root output: ${ROOT_RUN}"
echo "============================================================"
