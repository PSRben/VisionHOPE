#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_test \
    classification \
    --model visionhope_small --id visionhope_small \
    --data-root "$IMAGENET_ROOT" \
    --checkpoint "$WEIGHTS_ROOT/visionhope_small.pth" \
    --output "$OUTPUT_ROOT/test/visionhope_small" \
    --precision fp32 --crop-pct 1.0 \
    --no-channels-last --input-size 3 224 224 --interpolation bicubic \
    --batch-size 32 --workers 4 --mode fused --share-direction-skip \
    "$@"
