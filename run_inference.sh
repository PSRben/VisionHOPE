#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" -m visionhope.tasks.cli inference "$@"
