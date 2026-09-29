#!/usr/bin/env python3
"""Arithmetic seed means and throughput speedup; smoke runs are excluded."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

import _bootstrap  # noqa: F401
from longspark.plan import LONG_SEEDS, SEEDS, SHORT_DATASETS
from longspark.utils import sha256_file, write_json

CELL = ('preset', 'temperature', 'size', 'dataset', 'concurrency', 'method')
AVERAGE = ('preset', 'temperature', 'size', 'concurrency', 'method')


def load_results(root):
    """Group audited, non-smoke results by cell; returns {cell key: [(seed, metrics)]}."""
    groups, seen = defaultdict(list), set()
    for jobpath in sorted(root.rglob('job.json')):
        job = json.loads(jobpath.read_text())
        folder = jobpath.parent
        if job.get('smoke') or not (folder / 'audit.json').exists():
            continue
        audit = json.loads((folder / 'audit.json').read_text())
        if not audit['passed'] or sha256_file(folder / 'result.json') != audit['result_sha256']:
            raise RuntimeError(f'Invalid result receipt: {folder}')
        key = tuple(job[k] for k in CELL)
        ident = (*key, job['seed'])
        if ident in seen:
            raise RuntimeError(f'Duplicate cell: {ident}')
        seen.add(ident)
        r = json.loads((folder / 'result.json').read_text())
        if job['mode'] == 'finite':
            values = dict(seconds=r['makespan_seconds'])
        else:
            m = r['measured']
            values = dict(tps=m['output_tokens_per_second'], tau=m['actual_advance_per_verify'],
                          tpot=m['window_decode_ms_per_token'])
        groups[key].append((job['seed'], values))
    return groups


def summarize(root):
    rows = []
    for key, items in load_results(root).items():
        row = dict(zip(CELL, key))
        row['seeds'] = sorted(s for s, _ in items)
        expected = LONG_SEEDS if row['preset'] == 'long_context' else SEEDS
        row['complete'] = row['seeds'] == sorted(expected)
        for metric in items[0][1]:
            values = [v[metric] for _, v in items if v[metric] is not None]
            row[metric] = mean(values) if values else None
            row[metric + '_sd'] = stdev(values) if len(values) > 1 else None
        rows.append(row)
    lookup = {tuple(r[k] for k in CELL): r for r in rows}
    for r in rows:
        baseline = lookup.get((*(r[k] for k in CELL[:-1]), 'vanilla'))
        if 'tps' in r and baseline and baseline['seeds'] == r['seeds']:
            r['speedup'] = r['tps'] / baseline['tps']
    averages = defaultdict(list)
    for r in rows:
        if r['preset'] in ('short_context', 'concurrency'):
            averages[tuple(r[k] for k in AVERAGE)].append(r)
    for key, items in averages.items():
        if {r['dataset'] for r in items} != set(SHORT_DATASETS) or not all(r['complete'] for r in items):
            continue
        avg = dict(zip(AVERAGE, key), dataset='avg.', seeds=sorted(SEEDS), complete=True)
        # First average across seeds per dataset, then equally across datasets.
        for metric in ('speedup', 'tau'):
            if all(r.get(metric) is not None for r in items):
                avg[metric] = mean(r[metric] for r in items)
        rows.append(avg)
    return sorted(rows, key=lambda r: tuple(r[k] for k in CELL))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = summarize(a.results)
    write_json(a.output, rows)
    print(f'Wrote {len(rows)} aggregate rows; seeds are listed explicitly. Missing cells are not imputed.')
