#!/usr/bin/env bash
# Train RetiJev: LoRA + pointer head on a frozen Qwen3.5 base, with soft-label distillation (KD_W) and paraphrase
# consistency (CONS_W) on top of cross-entropy and the ordinal term. Run from the repository root.
#
#   bash scripts/train.sh                                                  # 0.8B on data/sample
#   BASE=Qwen/Qwen3.5-9B-Base OUT=runs/retijev-9b bash scripts/train.sh
#   DATA=/path/to/data TEACHER=/path/to/teacher_train.jsonl PARA_GROUPS=/path/to/groups.json bash scripts/train.sh
#   NPROC=4 bash scripts/train.sh                                          # data parallel on 4 GPUs
#
# A missing TEACHER file turns distillation off; a missing PARA_GROUPS file turns the consistency term off.
# Extra arguments are passed to medjev.train, e.g. `bash scripts/train.sh --max_records 50 --epochs 1`.
set -euo pipefail
BASE=${BASE:-Qwen/Qwen3.5-0.8B-Base}
DATA=${DATA:-data/sample}
TEACHER=${TEACHER:-$DATA/teacher_train.jsonl}
PARA_GROUPS=${PARA_GROUPS:-$DATA/groups.json}
OUT=${OUT:-runs/retijev-0.8b}
KD_W=${KD_W:-0.5}
CONS_W=${CONS_W:-0.5}
NPROC=${NPROC:-1}

KD=(--kd_w "$KD_W" --kd_T 2 --cons_w "$CONS_W")
[ -f "$TEACHER" ] && KD+=(--teacher "$TEACHER")
[ -f "$PARA_GROUPS" ] && KD+=(--groups "$PARA_GROUPS")
if [ "$NPROC" -gt 1 ]; then
  LAUNCH=(torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" -m retijev.train)
else
  LAUNCH=(python -m retijev.train)
fi
"${LAUNCH[@]}" "${KD[@]}" -- \
  --out "$OUT" --base "$BASE" --data "$DATA/train.jsonl" --val_data "$DATA/development.jsonl" \
  --epochs 3 --batch 4 --accum 1 --dtype bf16 --checkpointing 1 --lora 64 --lr 3e-5 --head_lr 0 --ord_w 0.3 \
  --log_every 50 --val_every 0 --save_every 0 --wandb 0 "$@"
