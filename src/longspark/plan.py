"""Evaluation presets expanded into one job dict per cell (CPU only)."""
from itertools import product

METHODS = ('vanilla', 'longspark')
SIZES = ('4B', '8B', '14B')
SEEDS = (980426, 980427, 2026)
LONG_SEEDS = (*SEEDS, *range(980428, 980435))
SHORT_DATASETS = ('gsm8k', 'math500', 'aime25', 'humaneval', 'mbpp', 'livecodebench', 'mt-bench', 'alpaca')
LONG_DATASETS = ('longspec_32k', 'code_64k', 'longswe_128k')
PRESETS = ('short_context', 'long_context', 'concurrency', 'fixed_requests')


def build_jobs(preset):
    if preset not in PRESETS:
        raise ValueError(preset)
    long = preset in ('long_context', 'fixed_requests')
    finite = preset == 'fixed_requests'
    seeds = LONG_SEEDS if preset == 'long_context' else SEEDS
    datasets = LONG_DATASETS if long else SHORT_DATASETS
    if preset == 'concurrency':
        concurrencies = (8, 16, 32, 64, 128)
    else:
        concurrencies = (16,) if long else (32,)
    # Input files live under data/<group>/<dataset>.jsonl.
    group = 'fixed32' if finite else ('long_context' if long else 'eight')
    jobs = []
    for size, seed, concurrency, dataset, method in product(SIZES, seeds, concurrencies, datasets, METHODS):
        if long:
            graph_max_batch = 32
            target_kv_gib = 48 if dataset == 'longspec_32k' else 96
            draft_state_gib = 24 if dataset == 'longspec_32k' else 56
        else:
            graph_max_batch = 128 if preset == 'concurrency' and concurrency != 32 else 64
            target_kv_gib = 64
            draft_state_gib = 8
        jobs.append(dict(
            preset=preset, size=size, method=method, seed=seed,
            dataset=dataset, fixture=f'data/{group}/{dataset}.jsonl', concurrency=concurrency,
            temperature=1., max_new_tokens=8192 if long else 2048,
            target_tp=4 if long else 1, target_kv_gib=target_kv_gib,
            draft_state_gib=0 if method == 'vanilla' else draft_state_gib,
            graph_max_batch=graph_max_batch, position_variant='fixed_ntk' if long else 'native',
            context_length=140288 if long else 40968, eos_token_ids=[151645, 151643] if long else [151645],
            prefill_chunk_size=8192, prefill_mode='chunked' if long else 'batched',
            capacity_policy='full_reservation' if long else 'runtime_admission',
            warmup_seconds=30., measurement_seconds=120., fuse_kv_export=False,
            mode='finite' if finite else 'steady'))
    return jobs


def job_key(job):
    """Relative result directory for one cell, e.g. t1/8B/math500/c32/s980426/longspark."""
    return (f"t{job['temperature']:g}/{job['size']}/{job['dataset']}"
            f"/c{job['concurrency']}/s{job['seed']}/{job['method']}")
