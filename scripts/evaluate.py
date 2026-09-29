#!/usr/bin/env python3
"""Repository-local entry: never imports an external experiment directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor/sglang/python'), str(ROOT/'vendor')]

if __name__ == '__main__':
    from longspark.cli import main
    main()
