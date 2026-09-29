"""Small file helpers shared by the launcher and workers."""
import hashlib
import json
from pathlib import Path


def write_json(path, obj):
    """Write `obj` as indented JSON, refusing to overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
