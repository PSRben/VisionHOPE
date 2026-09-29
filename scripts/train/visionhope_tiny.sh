#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"

vh_train \
    classification \
    --model visionhope_tiny \
    --data-root "$IMAGENET_ROOT" \
    --output "$OUTPUT_ROOT/train/visionhope_tiny" \
    --batch-size 256 --validation-batch-size 256 \
    --epochs 300 --cooldown-epochs 10 --warmup-epochs 20 \
    --warmup-lr 1e-6 --min-lr 1e-5 \
    --weight-decay 0.05 --memory-weight-decay 0.01 \
    --drop-path-rate 0.2 --clip-grad 5.0 \
    --precision bf16 --no-channels-last \
    --ema --ema-decay 0.9998 \
    --mesa --mesa-start-epoch 75 --mesa-end-epoch -1 --mesa-loss softce --mesa-weight 1.0 \
    --mixup 0.8 --cutmix 1.0 --mixup-off-epoch 0 --smoothing 0.1 \
    --prefetcher --train-interpolation random \
    --crop-pct 0.875 --image-size 224 \
    --grad-accum-steps 1 --no-activation-checkpointing --share-direction-skip \
    --workers 8 --seed 42 --save-top-k 20 --checkpoint-metric auto \
    "$@"
