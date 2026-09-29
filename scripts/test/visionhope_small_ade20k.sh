#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_test \
    segmentation \
    --model visionhope_small --id visionhope_small_ade20k \
    --data-root "$ADE20K_ROOT" \
    --checkpoint "$WEIGHTS_ROOT/visionhope_small_ade20k.pth" \
    --output "$OUTPUT_ROOT/test/visionhope_small_ade20k" \
    --precision fp32 --batch-size 1 --workers 2 --collect-device gpu \
    --mode fused --share-direction-skip \
    "$@"
