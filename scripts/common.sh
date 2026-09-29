#!/usr/bin/env bash
set -euo pipefail

VISIONHOPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGENET_ROOT="${IMAGENET_ROOT:-$VISIONHOPE_ROOT/data/imagenet}"
COCO_ROOT="${COCO_ROOT:-$VISIONHOPE_ROOT/data/coco}"
ADE20K_ROOT="${ADE20K_ROOT:-$VISIONHOPE_ROOT/data/ade20k}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$VISIONHOPE_ROOT/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VISIONHOPE_ROOT/outputs}"
export PYTHON="${PYTHON:-python}"

vh_run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        exec "$@"
    fi
}

vh_train() {
    local processes="${NPROC_PER_NODE:-8}"
    if [[ ! "$processes" =~ ^[1-9][0-9]*$ ]]; then
        printf 'NPROC_PER_NODE must be a positive integer\n' >&2
        exit 2
    fi
    vh_run env "PYTHON=$PYTHON" "NPROC_PER_NODE=$processes" \
        bash "$VISIONHOPE_ROOT/run_tasks.sh" "$@"
}

vh_test() {
    vh_run env "PYTHON=$PYTHON" \
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}" \
        bash "$VISIONHOPE_ROOT/run_inference.sh" "$@"
}

vh_pretrained() {
    local arch="$1"
    shift
    PRETRAINED_ARGS=(--pretrained "$WEIGHTS_ROOT/visionhope_${arch}.pth")
    local argument
    for argument in "$@"; do
        case "$argument" in
            --resume|--resume=*|--pretrained|--pretrained=*)
                PRETRAINED_ARGS=()
                break
                ;;
        esac
    done
}
