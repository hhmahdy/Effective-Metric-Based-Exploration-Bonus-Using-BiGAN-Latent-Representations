#!/usr/bin/env bash
# Run the three metric-based exploration bonus variants and compare them.
#
# Variants (Contribution 2):
#   V1 Baseline    Adventurer reconstruction novelty B(s)
#   V2 Latent only b_t = ||E(s_t) - E(s_{t+1})||_2          (zeta == 1)
#   V3 EME exactly b_t = ||E(s_t) - E(s_{t+1})||_2 * min(max(zeta(r), 1), M)
#   V4 Ours        b_t = ||E(s_t) - E(s_{t+1})||_2 * min(zeta(r) / E[zeta], M)
#
# V3 reproduces published EME. On sparse-reward Atari its lower clamp binds
# almost everywhere, so V4 normalises zeta by its own running mean instead and
# stays informative at any reward magnitude.
#
# Example:
#   ENVIRONMENT_ID=ALE/MontezumaRevenge-v5 SEEDS="0 1 2" ./run_metric_eme_comparison.sh

set -euo pipefail

ENVIRONMENT_ID="${ENVIRONMENT_ID:-ALE/MontezumaRevenge-v5}"
SEEDS="${SEEDS:-0 1 2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-96}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
TOTAL_ENVIRONMENT_STEPS="${TOTAL_ENVIRONMENT_STEPS:-12288000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"
BIGAN_BATCH_SIZE="${BIGAN_BATCH_SIZE:-64}"
ENSEMBLE_K="${ENSEMBLE_K:-5}"
MAX_REWARD_SCALING="${MAX_REWARD_SCALING:-5.0}"
LATENT_NORM="${LATENT_NORM:-L2}"
FREEZE_ENCODER_AFTER_UPDATES="${FREEZE_ENCODER_AFTER_UPDATES:-}"
DEVICE="${DEVICE:-cuda}"
SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-10}"
ROOT_RUN="${ROOT_RUN:-runs/metric-eme-comparison}"

if (( TOTAL_ENVIRONMENT_STEPS % (NUM_PARALLEL_ENVS * ROLLOUT_STEPS) != 0 )); then
    echo "ERROR: TOTAL_ENVIRONMENT_STEPS must be divisible by NUM_PARALLEL_ENVS * ROLLOUT_STEPS" >&2
    exit 1
fi

run_variant() {
    local seed="$1"
    local label="$2"
    local output_directory="$3"
    shift 3
    local extra_args=("$@")

    echo "------------------------------------------------------------"
    echo "Seed: ${seed}   Variant: ${label}"
    echo "Output: ${output_directory}"
    echo "------------------------------------------------------------"

    local freeze_args=()
    if [[ -n "${FREEZE_ENCODER_AFTER_UPDATES}" ]]; then
        freeze_args=(--freeze-encoder-after-updates "${FREEZE_ENCODER_AFTER_UPDATES}")
    fi

    python -u main.py \
        --env "${ENVIRONMENT_ID}" \
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
        "${freeze_args[@]}" \
        "${extra_args[@]}"
}

for seed in ${SEEDS}; do
    SEED_ROOT="${ROOT_RUN}/seed_${seed}"
    V1_RUN="${SEED_ROOT}/v1-adventurer-bigan"
    V2_RUN="${SEED_ROOT}/v2-latent-discrepancy"
    V3_RUN="${SEED_ROOT}/v3-latent-discrepancy-eme"
    V4_RUN="${SEED_ROOT}/v4-latent-discrepancy-eme-normalised"
    COMPARISON_RUN="${SEED_ROOT}/comparison"

    echo "============================================================"
    echo "Metric-based exploration bonus comparison, seed ${seed}"
    echo "============================================================"

    run_variant "${seed}" "V1 Adventurer (BiGAN novelty)" "${V1_RUN}" \
        --novelty bigan --state-alpha 0.9
    run_variant "${seed}" "V2 latent discrepancy only" "${V2_RUN}" \
        --novelty latent_discrepancy --latent-norm "${LATENT_NORM}"
    run_variant "${seed}" "V3 latent discrepancy + clamped EME scaling" "${V3_RUN}" \
        --novelty latent_discrepancy --eme True --eme-mode clamped \
        --ensemble_K "${ENSEMBLE_K}" \
        --max-reward-scaling "${MAX_REWARD_SCALING}" \
        --latent-norm "${LATENT_NORM}"
    run_variant "${seed}" "V4 latent discrepancy + normalised EME scaling" "${V4_RUN}" \
        --novelty latent_discrepancy --eme True --eme-mode normalised \
        --ensemble_K "${ENSEMBLE_K}" \
        --max-reward-scaling "${MAX_REWARD_SCALING}" \
        --latent-norm "${LATENT_NORM}"

    python - <<'PY' "${V1_RUN}" "${V2_RUN}" "${V3_RUN}" "${V4_RUN}" "${COMPARISON_RUN}" "${SMOOTHING_WINDOW}"
import sys
from pathlib import Path

from evaluation.comparison import save_multi_game_score_comparison

v1_run, v2_run, v3_run, v4_run, comparison_run, smoothing_window = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    Path(sys.argv[4]),
    Path(sys.argv[5]),
    int(sys.argv[6]),
)

runs = {
    "V1 Adventurer (BiGAN novelty)": v1_run,
    "V2 Latent discrepancy": v2_run,
    "V3 Latent + clamped EME": v3_run,
    "V4 Latent + normalised EME": v4_run,
}

figures = [
    ("episode/game_score", "Game Score", "game_score_comparison"),
    ("reward/intrinsic", "Intrinsic Reward", "intrinsic_reward_comparison"),
    ("reward/total", "Total Reward", "total_reward_comparison"),
]
for metric_name, y_label, file_stem in figures:
    try:
        csv_path, png_path = save_multi_game_score_comparison(
            runs,
            comparison_run,
            smoothing_window=smoothing_window,
            dpi=300,
            metric_name=metric_name,
            y_label=y_label,
            file_stem=file_stem,
        )
    except ValueError as error:
        print(f"skipping {metric_name}: {error}")
        continue
    print("CSV:", csv_path)
    print("PNG:", png_path)

# Diagnostics that only exist for the metric-based variants.
metric_runs = {
    "V2 Latent discrepancy": v2_run,
    "V3 Latent + clamped EME": v3_run,
    "V4 Latent + normalised EME": v4_run,
}
diagnostics = [
    ("intrinsic/latent_distance", "||E(s_t) - E(s_t+1)||", "latent_distance"),
    ("intrinsic/ensemble_variance", "Ensemble Variance", "ensemble_variance"),
    ("intrinsic/bonus_scale", "Bonus Scale", "bonus_scale"),
    ("intrinsic/bonus", "Raw Bonus b_t", "raw_bonus"),
]
for metric_name, y_label, file_stem in diagnostics:
    try:
        csv_path, png_path = save_multi_game_score_comparison(
            metric_runs,
            comparison_run,
            smoothing_window=smoothing_window,
            dpi=300,
            metric_name=metric_name,
            y_label=y_label,
            file_stem=file_stem,
        )
    except ValueError as error:
        print(f"skipping {metric_name}: {error}")
        continue
    print("CSV:", csv_path)
    print("PNG:", png_path)
PY

done

echo "============================================================"
echo "All variant comparisons completed"
echo "Root output: ${ROOT_RUN}"
echo "============================================================"
