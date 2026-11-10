#!/usr/bin/env bash
# Runs the exp18 model-design ablations (full attention, no RoPE, discrete tokenizer) one after another,
# continuing past failures and summarizing exit codes at the end. Logs go to temp/.
# Usage: bash run.sh  (set FOUNDATIONIMU_PYTHON to choose the interpreter)
cd "$(dirname "$0")"
PY="${FOUNDATIONIMU_PYTHON:-python}"
mkdir -p temp

EXPS=(exp18_arch_fullattn exp18_arch_norope exp18_arch_discrete)
declare -A RC
for exp in "${EXPS[@]}"; do
    echo "=== $(date '+%F %T') launching ${exp} ==="
    "$PY" -m src.train "experiment=${exp}" 2>&1 | tee "temp/${exp}.log"
    RC[$exp]=${PIPESTATUS[0]}
    echo "=== $(date '+%F %T') ${exp} exited with code ${RC[$exp]} ==="
    sleep 2   # keep Hydra's second-resolution run dirs collision-free
done

echo "=== summary ==="
for exp in "${EXPS[@]}"; do echo "${exp}: exit ${RC[$exp]}"; done
