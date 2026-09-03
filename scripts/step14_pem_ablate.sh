#!/usr/bin/env bash
# PEM channel ablation after step12 D. IDEA 7.2 singles + pairs (111 is F_scratch).
set -euo pipefail
cd /home/finnwe/project/paper/MPFADet
exec 9>logs/step14_pipeline.lock
flock -n 9 || { echo "step14 already running"; exit 0; }
PY=/home/finnwe/miniconda3/envs/mpfadet/bin/python
export PYTHONUNBUFFERED=1
LOG=logs/step14_pipeline.log

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

wait_py() {
  local needle=$1
  while pgrep -f "${PY} ${needle}" >/dev/null; do sleep 20; done
}

log "wait step12 PIPELINE_DONE"
while ! grep -q PIPELINE_DONE logs/step12_pipeline.log 2>/dev/null; do
  sleep 30
done

run_mask() {
  local mask=$1 name=$2
  local out="outputs/train_F_${name}"
  if [[ -f "$out/best.pt" && -f "logs/step14_eval_${name}_val.json" ]]; then
    log "skip ${name} (already evaled)"
    return
  fi
  log "train F ${name} mask=${mask}"
  "$PY" scripts/train.py --dual --phase-mask "$mask" --epochs 30 --batch 8 --imgsz 512 --out "$out" | tee "logs/step14_train_${name}.log"
  "$PY" scripts/eval_ckpt.py --ckpt "$out/best.pt" --split val --report "logs/step14_eval_${name}_val.json"
  "$PY" scripts/eval_ckpt.py --ckpt "$out/best.pt" --split test --report "logs/step14_eval_${name}_test.json"
}

run_mask 100 IF
run_mask 010 Coh
run_mask 001 Residual
run_mask 110 IFCoh
run_mask 101 IFRes
run_mask 011 CohRes

log "STEP14_DONE"
