#!/usr/bin/env bash
set -euo pipefail
if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
    exec "${PYTHON:-python}" -m torch.distributed.run --standalone \
        --nproc_per_node "${NPROC_PER_NODE}" -m visionhope.tasks.cli train "$@"
fi
exec "${PYTHON:-python}" -m visionhope.tasks.cli train "$@"
