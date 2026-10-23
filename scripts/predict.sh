#!/usr/bin/env bash
# Per-question predictions and micro accuracy of a trained checkpoint on one split. Run from the repository root.
#
#   RUN=runs/retijev-0.8b bash scripts/predict.sh
#   RUN=<hf-org>/<model> DATA=/path/to/data SPLIT=development bash scripts/predict.sh
set -euo pipefail
RUN=${RUN:?set RUN to a checkpoint directory or a Hugging Face repo id}
DATA=${DATA:-data/sample}
SPLIT=${SPLIT:-development}
OUT=${OUT:-runs/preds/$(basename "$RUN")_${SPLIT}.jsonl}
python -m retijev.predict --run "$RUN" --data "$DATA" --split "$SPLIT" --out "$OUT" "$@"
