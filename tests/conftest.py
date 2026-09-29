import sys
from pathlib import Path

# Make src/ and the vendored SGLang importable without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT / 'src', ROOT / 'vendor/sglang/python', ROOT / 'vendor')
                if str(p) not in sys.path]
