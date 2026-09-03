#!/usr/bin/env bash
# Chain remaining IDEA table runs after current E concat. Match conda python only.
set -euo pipefail
cd /home/finnwe/project/paper/MPFADet
exec 9>logs/step12_pipeline.lock
flock -n 9 || { echo "pipeline already running"; exit 0; }
PY=/home/finnwe/miniconda3/envs/mpfadet/bin/python
export PYTHONUNBUFFERED=1
LOG=logs/step12_pipeline.log

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

wait_py() {
  local needle=$1
  while pgrep -f "${PY} ${needle}" >/dev/null; do sleep 20; done
}

log "pipeline restart, wait E concat pid-pattern"
wait_py "scripts/train.py --dual --fusion concat"

if [[ -f outputs/train_E_concat/best.pt ]]; then
  log "eval E concat"
  "$PY" scripts/eval_ckpt.py --ckpt outputs/train_E_concat/best.pt --split val --report logs/step12_eval_Econcat_val.json
  "$PY" scripts/eval_ckpt.py --ckpt outputs/train_E_concat/best.pt --split test --report logs/step12_eval_Econcat_test.json
fi

if [[ $(ls data/processed/pspec/train/*.png 2>/dev/null | wc -l) -lt 3200 ]]; then
  log "link pspec splits"
  "$PY" scripts/step11_link_pspec.py
fi

log "train E add"
"$PY" scripts/train.py --dual --fusion add --epochs 30 --batch 8 --imgsz 512 --out outputs/train_E_add | tee logs/step12_train_E_add.log
"$PY" scripts/eval_ckpt.py --ckpt outputs/train_E_add/best.pt --split val --report logs/step12_eval_Eadd_val.json
"$PY" scripts/eval_ckpt.py --ckpt outputs/train_E_add/best.pt --split test --report logs/step12_eval_Eadd_test.json

log "train D mag+pspec"
"$PY" scripts/train.py --dual --phase-subdir pspec --epochs 30 --batch 8 --imgsz 512 --out outputs/train_D | tee logs/step13_train_D.log
"$PY" scripts/eval_ckpt.py --ckpt outputs/train_D/best.pt --split val --report logs/step13_eval_D_val.json
"$PY" scripts/eval_ckpt.py --ckpt outputs/train_D/best.pt --split test --report logs/step13_eval_D_test.json

log "PIPELINE_DONE"
