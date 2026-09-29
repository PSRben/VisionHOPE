#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_pretrained small "$@"
vh_train \
    segmentation \
    --arch small \
    --data-root "$ADE20K_ROOT" \
    "${PRETRAINED_ARGS[@]}" \
    --output "$OUTPUT_ROOT/train/visionhope_small_ade20k" \
    --batch-size 2 --grad-accum-steps 1 \
    --precision fp32 --drop-path-rate 0.4 \
    --no-activation-checkpointing --share-direction-skip \
    --collect-device gpu --save-top-k 5 --seed 42 \
    "$@"
