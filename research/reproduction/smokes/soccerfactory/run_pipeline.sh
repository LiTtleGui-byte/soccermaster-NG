#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/home/tianlin/SoccerMaster
readonly ROOT="$REPO/runs/path_smoke_20260819/soccerfactory"
readonly PYTHON="$REPO/.envs/SoccerMaster-repro/bin/python"
readonly SMOKE="$REPO/research/reproduction/smokes/soccerfactory"
readonly SOURCE_ROOT="$REPO/research/src"
readonly TRACKLAB="$REPO/baseline/code/soccerfactory/tracklab"
readonly GAMESTATE="$REPO/baseline/code/soccerfactory/sn-gamestate"
readonly CALIBRATION="$REPO/baseline/code/sn_calibration/src"

if [[ -e "$ROOT/pipeline.log" || -L "$ROOT/pipeline.log" ]]; then
  echo "Refusing to overwrite $ROOT/pipeline.log" >&2
  exit 2
fi
mkdir -p "$ROOT"
exec > >(tee "$ROOT/pipeline.log") 2>&1

heartbeat() {
  local started=$SECONDS
  while sleep 30; do
    echo "[PIPELINE_HEARTBEAT] elapsed_seconds=$((SECONDS-started))"
  done
}
heartbeat &
readonly HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Select exactly one physical GPU}"
export G10_LOCAL_STEP1_GPU_APPROVED=YES
export G10_LOCAL_ENRICHMENT_GPU_APPROVED=YES
export G10_LOCAL_REFINER_GPU_APPROVED=YES
export PYTHONPATH="$SOURCE_ROOT:$TRACKLAB:$GAMESTATE:$CALIBRATION"
export LD_LIBRARY_PATH="$REPO/.envs/SoccerMaster-repro/lib/python3.10/site-packages/torch/lib:$REPO/.envs/SoccerMaster-repro/lib"
export PYTHONDONTWRITEBYTECODE=1

echo "[PIPELINE_PHASE] step1_gpu"
"$PYTHON" "$SMOKE/step1.py" --run
echo "[PIPELINE_PHASE] enrichment_gpu"
"$PYTHON" "$SMOKE/enrichment.py" --run
echo "[PIPELINE_PHASE] refiner_gpu"
"$PYTHON" "$SMOKE/refiner.py" --run

export CUDA_VISIBLE_DEVICES=""
echo "[PIPELINE_PHASE] step3_cpu"
timeout --signal=TERM --kill-after=60s 600s "$PYTHON" "$SMOKE/step3.py"
echo "[PIPELINE_PHASE] conversion_cpu"
timeout --signal=TERM --kill-after=60s 600s "$PYTHON" "$SMOKE/convert.py"
echo "[PIPELINE_PHASE] dataloader_cpu"
timeout --signal=TERM --kill-after=60s 600s "$PYTHON" "$SMOKE/dataloader.py"
echo "[PIPELINE_RESULT] passed"
