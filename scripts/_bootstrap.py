"""Make `longspark` and the vendored SGLang importable when the repo is not pip-installed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT / 'src', ROOT / 'vendor/sglang/python', ROOT / 'vendor')
                if str(p) not in sys.path]
