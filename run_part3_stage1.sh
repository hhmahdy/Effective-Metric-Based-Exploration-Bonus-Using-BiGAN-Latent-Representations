#!/usr/bin/env bash
#
# Part-3 Stage-1 launcher on CorridorTV.
#
# Stage 1 runs the controlled representation-intervention experiment
# (BiGAN + N_T / IDF + N_T / RND + N_T) on the CorridorTV environment for the
# exact Stage-1 budget:
#
#   "approximately 2M steps; exact budget of 1990656 = 162 updates * 12288
#    constrained by rollout divisibility"
#
# (2,000,000 is not divisible by 96*128 = 12,288; the 96x128 configuration is
# intentionally not changed.)
#
# This script only *launches* runs and collects per-arm exit codes. It does NOT
# analyse results: computing figures, first-discovery, IQM, and diagnostics from
# each run's metrics.jsonl is a separate step. The trainer writes its own
# periodic snapshots into each run directory; this script leaves them untouched
# and provides no resume logic.

set -euo pipefail

# Change to the repository root (the directory containing this script).
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration (environment-variable overrides with defaults)
# ---------------------------------------------------------------------------
STAGE_NUM="${STAGE_NUM:-1}"
ENVIRONMENT_ID="${ENVIRONMENT_ID:-CorridorTV-v0}"
SEEDS="${SEEDS:-0 1 2}"
REPRESENTATIONS="${REPRESENTATIONS:-bigan idf rnd}"
NUM_PARALLEL_ENVS="${NUM_PARALLEL_ENVS:-96}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"
TOTAL_ENVIRONMENT_STEPS="${TOTAL_ENVIRONMENT_STEPS:-1990656}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/part3}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

echo "[stage1] repository root: $(pwd)"
echo "[stage1] environment id:  ${ENVIRONMENT_ID}"

# 1. main.py imports cleanly.
if ! "${PYTHON_BIN}" -c "import main" >/dev/null 2>&1; then
  echo "ERROR: '${PYTHON_BIN}' cannot import main; check PYTHON_BIN and the Python environment." >&2
  exit 1
fi
echo "[stage1] preflight: '${PYTHON_BIN}' import main -> OK"

# 2. Budget divisibility: TOTAL_ENVIRONMENT_STEPS % (NUM_PARALLEL_ENVS * ROLLOUT_STEPS) == 0.
SAMPLES_PER_UPDATE=$(( NUM_PARALLEL_ENVS * ROLLOUT_STEPS ))
REMAINDER=$(( TOTAL_ENVIRONMENT_STEPS % SAMPLES_PER_UPDATE ))
if [ "${REMAINDER}" -ne 0 ]; then
  echo "ERROR: TOTAL_ENVIRONMENT_STEPS=${TOTAL_ENVIRONMENT_STEPS} is not divisible by NUM_PARALLEL_ENVS*ROLLOUT_STEPS=${SAMPLES_PER_UPDATE}; remainder=${REMAINDER}." >&2
  echo "       The Stage-1 exact budget must satisfy this divisibility (2,000,000 is not divisible by 12288; 1990656 is)." >&2
  exit 1
fi
echo "[stage1] preflight: budget divisibility -> OK (${TOTAL_ENVIRONMENT_STEPS} / ${SAMPLES_PER_UPDATE} = $(( TOTAL_ENVIRONMENT_STEPS / SAMPLES_PER_UPDATE )) updates)"

# 3. Environment registration (CorridorTV).
REGISTRATION_FAILURE_MSG="CorridorTV environment '${ENVIRONMENT_ID}' is not registered; Stage 1 cannot be launched until environments/corridortv.py is importable and registers the spec (see Part-3 protocol: do NOT invent a substitute environment)."
if ! "${PYTHON_BIN}" -c "import gymnasium, environments.corridortv, gymnasium as g; g.spec('${ENVIRONMENT_ID}')" >/dev/null 2>&1; then
  echo "ERROR: ${REGISTRATION_FAILURE_MSG}" >&2
  exit 1
fi
echo "[stage1] preflight: '${ENVIRONMENT_ID}' spec registration -> OK"

# 4. Output directory tree.
mkdir -p "${OUTPUT_ROOT}"/{raw,processed,figures,checkpoints,logs} \
         "${OUTPUT_ROOT}/logs/stage${STAGE_NUM}"
echo "[stage1] preflight: output tree under ${OUTPUT_ROOT} -> OK"

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

CONSOLE_LOG_DIR="${OUTPUT_ROOT}/logs/stage${STAGE_NUM}"
FAILED=0
STATUS_ENTRIES=""

for seed in ${SEEDS}; do
  for rep in ${REPRESENTATIONS}; do
    key="${seed}/${rep}"
    run_dir="${OUTPUT_ROOT}/raw/stage${STAGE_NUM}/seed_${seed}/${rep}-nt"
    console_log="${CONSOLE_LOG_DIR}/${seed}_${rep}.console.log"

    # Skip if a completed metrics.jsonl already exists (append-mode data would
    # be contaminated by rerunning in place).
    if [ -f "${run_dir}/metrics.jsonl" ] && [ "${FORCE_RERUN}" != "1" ]; then
      echo "[stage1] skip seed=${seed} rep=${rep}: run dir already has metrics.jsonl (set FORCE_RERUN=1 to rerun) -> ${run_dir}"
      STATUS_ENTRIES="${STATUS_ENTRIES}${key}:SKIPPED:${run_dir}\n"
      continue
    fi

    if [ "${FORCE_RERUN}" = "1" ] && [ -d "${run_dir}" ]; then
      backup="${run_dir}.prev.$(date +%s)"
      echo "[stage1] FORCE_RERUN=1: moving existing run dir to ${backup}"
      mv "${run_dir}" "${backup}"
    fi

    mkdir -p "${run_dir}"
    echo "[stage1] launching seed=${seed} rep=${rep} -> ${run_dir}"

    # Piping through tee preserves the exit code of main.py (PIPESTATUS[0]).
    set +e
    "${PYTHON_BIN}" -u main.py \
      --part3 --representation "${rep}" \
      --env "${ENVIRONMENT_ID}" \
      --seed "${seed}" \
      --num-parallel-envs "${NUM_PARALLEL_ENVS}" \
      --rollout-steps "${ROLLOUT_STEPS}" \
      --minibatch-size "${MINIBATCH_SIZE}" \
      --total-environment-steps "${TOTAL_ENVIRONMENT_STEPS}" \
      --device "${DEVICE}" \
      --output-directory "${run_dir}" 2>&1 | tee "${console_log}"
    exit_code="${PIPESTATUS[0]}"
    set -e

    if [ "${exit_code}" -ne 0 ]; then
      echo "[stage1] FAILED seed=${seed} rep=${rep} exit_code=${exit_code} run_dir=${run_dir}"
      FAILED=1
      STATUS_ENTRIES="${STATUS_ENTRIES}${key}:FAILED:${run_dir}\n"
    else
      echo "[stage1] OK seed=${seed} rep=${rep} exit_code=0 run_dir=${run_dir}"
      STATUS_ENTRIES="${STATUS_ENTRIES}${key}:OK:${run_dir}\n"
    fi
  done
done

# ---------------------------------------------------------------------------
# Final manifest
# ---------------------------------------------------------------------------

echo ""
echo "===================================================================="
echo "Stage 1 summary (seed/rep -> status)"
echo "===================================================================="
printf '%b' "${STATUS_ENTRIES}" | while IFS=: read -r key status run_dir; do
  [ -z "${key}" ] && continue
  printf '%-7s seed/rep=%s -> %s\n' "${status}" "${key}" "${run_dir}"
done

echo ""
echo "NOTE: Analysis of each run's metrics.jsonl (figures, first-discovery, IQM,"
echo "diagnostics) is a SEPARATE step and is NOT part of this script. The trainer"
echo "writes periodic snapshots into each run directory; this launcher leaves them"
echo "untouched (no resume logic)."

if [ "${FAILED}" -ne 0 ]; then
  echo ""
  echo "ERROR: one or more Stage 1 arms failed."
  exit 1
fi

echo ""
echo "[stage1] all arms completed (see manifest above)."
