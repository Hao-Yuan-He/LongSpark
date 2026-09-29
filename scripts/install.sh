#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
"$python_bin" -c 'import platform, sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 required"'
"$python_bin" -m pip install --no-compile torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install --no-compile -r "$repo_dir/requirements.txt"
"$python_bin" -m pip install --no-compile --no-deps -e "$repo_dir"
"$python_bin" -m pip check
