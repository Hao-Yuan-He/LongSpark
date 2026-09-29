"""Integrity checks for evaluation inputs and local checkpoints."""
import json
from pathlib import Path

from .paths import REPO_ROOT
from .utils import sha256_file


def validate_fixtures():
    """Check data/ inputs against data/manifest.json, when it is present."""
    manifest = REPO_ROOT / 'data/manifest.json'
    if not manifest.exists():
        return dict(fixtures=0, note='data/ absent; fixture integrity not checked')
    fixtures = json.loads(manifest.read_text())
    for name, item in fixtures.items():
        if sha256_file(REPO_ROOT / item['path']) != item['sha256']:
            raise RuntimeError(f'Input fixture modified: {name}')
    return dict(fixtures=len(fixtures))


def validate_models(job):
    """Check that every checkpoint the job needs has a config and all weight shards."""
    roles = ['target'] + ([] if job['method'] == 'vanilla' else [job['method']])
    for role in roles:
        root = Path(job['models'][role])
        if not (root / 'config.json').is_file():
            raise FileNotFoundError(f'Missing {role} config.json: {root}')
        if not any(root.glob('*.safetensors')):
            raise FileNotFoundError(f'Missing {role} safetensors weights: {root}')
        index = root / 'model.safetensors.index.json'
        if index.is_file():
            for filename in set(json.loads(index.read_text())['weight_map'].values()):
                if not (root / filename).is_file():
                    raise FileNotFoundError(f'Missing checkpoint shard: {root / filename}')
