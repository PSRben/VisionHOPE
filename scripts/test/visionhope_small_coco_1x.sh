#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_test \
    detection \
    --model visionhope_small --id visionhope_small_coco_1x --schedule 1x \
    --data-root "$COCO_ROOT" \
    --checkpoint "$WEIGHTS_ROOT/visionhope_small_coco_1x.pth" \
    --output "$OUTPUT_ROOT/test/visionhope_small_coco_1x" \
    --precision fp32 --batch-size 1 --workers 2 --collect-device gpu \
    --mode fused --share-direction-skip \
    "$@"
