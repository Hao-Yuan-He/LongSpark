#!/usr/bin/env python3
"""Compare fully drained greedy output from separate AR and LongSpark processes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import _bootstrap  # noqa: F401
from longspark.gpus import require_idle_gpus, worker_environment
from longspark.paths import REPO_ROOT as ROOT
from longspark.utils import read_jsonl, sha256_file, write_json
from longspark.validate import validate_fixtures, validate_models

PACKAGES = ('torch', 'transformers', 'triton', 'sglang-kernel', 'flashinfer-python', 'huggingface-hub')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=('4B', '8B', '14B'), required=True)
    p.add_argument('--dataset', choices=('native', 'longspec_32k', 'code_64k', 'longswe_128k'), required=True)
    p.add_argument('--models', type=Path, default=ROOT / 'configs/models.local.json')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--seed', type=int, default=980426)
    a = p.parse_args()
    if not 1 <= a.max_new_tokens <= 2048 or not 1 <= a.concurrency <= 4:
        p.error('Output cap must be 1..2048 and concurrency 1..4')
    gpus = [int(x) for x in a.gpus.split(',')]
    tp = 1 if a.dataset == 'native' else 4
    if len(set(gpus)) != len(gpus) or min(gpus) < 0 or len(gpus) < tp + 1:
        p.error(f'Distinct GPU IDs required: {tp} target devices and one drafter')
    validate_fixtures()
    models = {k: str((ROOT / v).resolve()) for k, v in json.loads(a.models.read_text())[a.size].items()}
    validate_models(dict(method='longspark', models=models))
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True)
    source_files = [p for folder in ('src', 'scripts') for p in (ROOT / folder).rglob('*.py')]
    job = dict(size=a.size, dataset=a.dataset, models=models, target_tp=tp,
               max_new_tokens=a.max_new_tokens, concurrency=a.concurrency, seed=a.seed,
               temperature=0., deterministic_inference=True, physical_gpus=gpus[:tp+1],
               python=sys.version, executable=sys.executable,
               packages={name: importlib.metadata.version(name) for name in PACKAGES},
               deterministic_mm='triton', batch_variant_mm_fallback=False,
               source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(source_files)})
    write_json(out / 'job.json', job)
    require_idle_gpus(gpus[:tp + 1])
    env = worker_environment(gpus[:tp + 1])
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Keep the deterministic Triton implementation. The bundled DeepGEMM
    # path is incompatible with this environment's storage-less dispatch.
    env['SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM'] = '0'
    env['SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT'] = '0'
    for method in ('vanilla', 'longspark'):
        print(f'RUN {a.size} {a.dataset} {method}', flush=True)
        with (out / f'{method}.log').open('x') as log:
            subprocess.run([sys.executable, '-m', 'longspark.correctness', str(out / 'job.json'), method],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def read(method):
        return {r['source_id']: r for r in read_jsonl(out / method / 'result.requests.jsonl')}

    ar, draft = read('vanilla'), read('longspark')
    checks = []
    for key in sorted(ar.keys() | draft.keys()):
        left, right = ar.get(key), draft.get(key)
        x, y = (left or {}).get('output_ids', []), (right or {}).get('output_ids', [])
        exact = bool(left and right and x == y and left['input_sha256'] == right['input_sha256'])
        mismatch = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v),
                        min(len(x), len(y)) if len(x) != len(y) else None)
        checks.append(dict(source_id=key, passed=exact, reference_tokens=len(x), speculative_tokens=len(y),
                           first_mismatch=mismatch))
    result_files = [out / m / f for m in ('vanilla', 'longspark')
                    for f in ('result.json', 'result.requests.jsonl', 'execution.json')]
    receipt = dict(passed=bool(checks) and all(x['passed'] for x in checks), comparison='exact-token-ids',
                   size=a.size, dataset=a.dataset, temperature=0., deterministic_inference=True,
                   checked_requests=len(checks), checked_reference_tokens=sum(x['reference_tokens'] for x in checks),
                   checks=checks, files_sha256={str(p.relative_to(out)): sha256_file(p) for p in result_files})
    write_json(out / 'correctness.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
