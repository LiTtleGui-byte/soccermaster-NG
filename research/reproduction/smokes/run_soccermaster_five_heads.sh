#!/usr/bin/env bash
set -uo pipefail

readonly REPO=/home/tianlin/SoccerMaster
readonly RUN_DIR="$REPO/runs/path_smoke_20260819/soccermaster"
readonly PYTHON="$REPO/.envs/SoccerMaster-repro/bin/python"
readonly SOURCE_ROOT="$REPO/research/src"
readonly OPS_BUILD="$SOURCE_ROOT/soccermaster/models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
readonly CALIBRATION_SOURCE="$REPO/baseline/code/sn_calibration/src"
readonly SCRIPT="$REPO/research/reproduction/smokes/soccermaster_five_heads.py"

if [[ -e "$RUN_DIR/run.log" || -L "$RUN_DIR/run.log" ]]; then
  echo "Refusing to overwrite $RUN_DIR/run.log" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Select exactly one physical GPU}"
export PYTHONPATH="$SOURCE_ROOT:$OPS_BUILD:$CALIBRATION_SOURCE"
export LD_LIBRARY_PATH="$REPO/.envs/SoccerMaster-repro/lib/python3.10/site-packages/torch/lib:$REPO/.envs/SoccerMaster-repro/lib"
export MPLCONFIGDIR="$REPO/.runtime/path_smoke_20260819/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

timeout --signal=TERM --kill-after=60s 600s /usr/bin/time -v "$PYTHON" "$SCRIPT" 2>&1 | tee "$RUN_DIR/run.log"
exit_code=${PIPESTATUS[0]}
echo "[PIPELINE_EXIT_CODE] $exit_code" | tee -a "$RUN_DIR/run.log"
exit "$exit_code"
