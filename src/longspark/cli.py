"""Inspect, launch, and resume LongSpark evaluation with isolated GPU workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .gpus import require_idle_gpus, worker_environment
from .paths import REPO_ROOT
from .plan import PRESETS, build_jobs, job_key
from .utils import sha256_file, write_json
from .validate import validate_fixtures, validate_models

FILTERS = ('size', 'method', 'dataset', 'seed', 'concurrency')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=PRESETS, required=True)
    for name in FILTERS:
        p.add_argument('--' + name, help='Comma-separated filter; omitted means all values in the preset')
    p.add_argument('--temperature', type=float, choices=(0., 1.), default=1.)
    p.add_argument('--graph-max-batch', type=int,
                   help='Override graph capacity (at least concurrency, at most 128)')
    p.add_argument('--models', type=Path, default=REPO_ROOT / 'configs/models.local.json')
    p.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    p.add_argument('--gpus', default='0,1,2,3,4',
                   help='Physical GPU IDs; first TP GPUs are target, next is draft')
    p.add_argument('--execute', action='store_true', help='Default is a CPU-only plan preview')
    p.add_argument('--smoke', action='store_true',
                   help='Short functional check, excluded from metric summaries')
    p.add_argument('--resume', action='store_true',
                   help='Skip only completed, audited, configuration-matching cells')
    return p, p.parse_args()


def select_jobs(p, args):
    jobs = build_jobs(args.preset)
    for key in FILTERS:
        value = getattr(args, key)
        if value is None:
            continue
        requested = set(value.split(','))
        available = {str(j[key]) for j in jobs}
        if not requested <= available:
            p.error(f'Unknown {key}: {requested - available}')
        jobs = [j for j in jobs if str(j[key]) in requested]
    if not jobs:
        p.error('Empty selection')
    for j in jobs:
        j['temperature'] = args.temperature
        if args.graph_max_batch is not None:
            if not j['concurrency'] <= args.graph_max_batch <= 128:
                p.error('--graph-max-batch must cover concurrency and be at most 128')
            j['graph_max_batch'] = args.graph_max_batch
    if args.smoke:
        for j in jobs:
            j.update(smoke=True, concurrency=2, max_new_tokens=32, warmup_seconds=0., measurement_seconds=2.)
        # A concurrency sweep collapses to a single C2 smoke per model/seed/dataset/method.
        jobs = list({job_key(j): j for j in jobs}.values())
    return jobs


def is_completed(dest, job):
    """True if `dest` holds an audited result for exactly this job configuration."""
    jobfile, audit = dest / 'job.json', dest / 'audit.json'
    if json.loads(jobfile.read_text()) != job or not audit.exists():
        return False
    receipt = json.loads(audit.read_text())
    if not receipt.get('passed'):
        return False
    result = dest / 'result.json'
    if not result.exists() or sha256_file(result) != receipt.get('result_sha256'):
        raise RuntimeError(f'Missing or modified audited result: {dest}')
    return True


def run_worker(jobfile, dest, env):
    with (dest / 'worker.log').open('x') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'longspark.worker', str(jobfile)],
            env=env, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            raise


def main():
    p, args = parse_args()
    jobs = select_jobs(p, args)
    if not args.execute:
        print(json.dumps(dict(cells=len(jobs), jobs=jobs), indent=2))
        return
    validate_fixtures()
    models = json.loads(args.models.read_text())
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != len(set(gpus)) or min(gpus) < 0:
        p.error('Distinct nonnegative GPU IDs required')
    out = args.output.resolve() / ('smoke' if args.smoke else 'runs') / args.preset
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for j in jobs:
            count = j['target_tp'] + int(j['method'] != 'vanilla')
            if len(gpus) < count:
                p.error(f'{j["method"]} TP{j["target_tp"]} needs {count} GPUs')
            selected = gpus[:count]
            j['models'] = {k: str((REPO_ROOT / Path(v)).resolve()) for k, v in models[j['size']].items()}
            j['physical_gpus'] = selected
            dest = out / job_key(j)
            jobfile, audit = dest / 'job.json', dest / 'audit.json'
            if jobfile.exists():
                if args.resume and is_completed(dest, j):
                    print('SKIP', job_key(j), flush=True)
                    continue
                raise FileExistsError(f'{dest} exists; preserve it or select a new output directory')
            # Check model identities before reserving output or using GPUs.
            validate_models(j)
            require_idle_gpus(selected)
            env = worker_environment(selected)
            write_json(jobfile, j)
            print('RUN', job_key(j), flush=True)
            if run_worker(jobfile, dest, env) or not audit.exists():
                raise RuntimeError(f'Worker failed; inspect {dest}/worker.log')
            if not json.loads(audit.read_text()).get('passed'):
                raise RuntimeError(f'Audit failed: {dest}')
            time.sleep(2)
    print(f'Completed {len(jobs)} selected cells. Results: {out}')
