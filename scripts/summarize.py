#!/usr/bin/env python3
"""Arithmetic seed means and throughput speedup; smoke runs are excluded."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from longspark.plan import EIGHT, LONG_SEEDS, SEEDS

def summarize(root):
    groups=defaultdict(list);seen=set()
    for jobpath in sorted(root.rglob('job.json')):
        job=json.loads(jobpath.read_text());folder=jobpath.parent
        if job.get('smoke'):continue
        if not (folder/'audit.json').exists():continue
        audit=json.loads((folder/'audit.json').read_text());raw=(folder/'result.json').read_bytes()
        if not audit['passed'] or hashlib.sha256(raw).hexdigest()!=audit['result_sha256']:
            raise RuntimeError(f'Invalid result receipt: {folder}')
        key=tuple(job[k] for k in ('preset','temperature','size','dataset','concurrency','method'))
        ident=(*key,job['seed'])
        if ident in seen:raise RuntimeError(f'Duplicate cell: {ident}')
        seen.add(ident);r=json.loads(raw)
        if job['mode']=='finite':values=dict(seconds=r['makespan_seconds'])
        else:
            m=r['measured'];values=dict(tps=m['output_tokens_per_second'],tau=m['actual_advance_per_verify'],
                tpot=m['window_decode_ms_per_token'])
        groups[key].append((job['seed'],values))
    rows=[]
    for key,items in groups.items():
        row=dict(zip(('preset','temperature','size','dataset','concurrency','method'),key));row['seeds']=sorted(s for s,_ in items)
        expected=LONG_SEEDS if row['preset']=='long_context' else SEEDS
        row['complete']=row['seeds']==sorted(expected)
        for metric in items[0][1]:
            values=[v[metric] for _,v in items if v[metric] is not None]
            row[metric]=mean(values) if values else None
            row[metric+'_sd']=stdev(values) if len(values)>1 else None
        rows.append(row)
    lookup={(r['preset'],r['temperature'],r['size'],r['dataset'],r['concurrency'],r['method']):r for r in rows}
    for r in rows:
        baseline=lookup.get((r['preset'],r['temperature'],r['size'],r['dataset'],r['concurrency'],'vanilla'))
        if 'tps' in r and baseline and baseline['seeds']==r['seeds']:
            r['speedup']=r['tps']/baseline['tps']
    averages=defaultdict(list)
    for r in rows:
        if r['preset'] in ('short_context','concurrency'):
            averages[(r['preset'],r['temperature'],r['size'],r['concurrency'],r['method'])].append(r)
    for key,items in averages.items():
        if {r['dataset'] for r in items}!=set(EIGHT) or not all(r['complete'] for r in items):
            continue
        avg=dict(zip(('preset','temperature','size','concurrency','method'),key),dataset='avg.',
                 seeds=sorted(SEEDS),complete=True)
        # First average across seeds per dataset, then equally across datasets.
        for metric in ('speedup','tau'):
            if all(r.get(metric) is not None for r in items):
                avg[metric]=mean(r[metric] for r in items)
        rows.append(avg)
    return sorted(rows,key=lambda r:(r['preset'],r['temperature'],r['size'],r['dataset'],r['concurrency'],r['method']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=summarize(a.results);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
    print(f'Wrote {len(rows)} aggregate rows; seeds are listed explicitly. Missing cells are not imputed.')
