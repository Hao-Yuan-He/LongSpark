"""Repository locations used by the launcher and its worker processes."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
# The vendored SGLang runtime and SpecForge model definitions are imported
# from the source tree rather than installed as packages.
VENDOR_PATHS = (REPO_ROOT / 'vendor/sglang/python', REPO_ROOT / 'vendor')
