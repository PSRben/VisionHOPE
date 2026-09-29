#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_test \
    detection \
    --model visionhope_tiny --id visionhope_tiny_coco_3x --schedule 3x \
    --data-root "$COCO_ROOT" \
    --checkpoint "$WEIGHTS_ROOT/visionhope_tiny_coco_3x.pth" \
    --output "$OUTPUT_ROOT/test/visionhope_tiny_coco_3x" \
    --precision fp32 --batch-size 1 --workers 2 --collect-device gpu \
    --mode fused --share-direction-skip \
    "$@"
