#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON_BIN:-python}" "$(dirname "$0")/evaluate.py" --preset concurrency "$@"
