#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_pretrained small "$@"
vh_train \
    detection \
    --arch small --schedule 3x \
    --data-root "$COCO_ROOT" \
    "${PRETRAINED_ARGS[@]}" \
    --output "$OUTPUT_ROOT/train/visionhope_small_coco_3x" \
    --batch-size 4 --grad-accum-steps 1 \
    --precision fp16 --drop-path-rate 0.4 \
    --no-activation-checkpointing --share-direction-skip \
    --collect-device gpu --save-top-k 5 --seed 42 \
    "$@"
