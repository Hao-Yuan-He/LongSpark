"""A correctness-only finite run; timings are not benchmark measurements."""
import json
import os
from pathlib import Path
import sys
from .plan import ROOT, read_rows
from .cli import write_new


def select_inputs(dataset):
    if dataset == 'native':
        rows = []
        for name in ('math500', 'humaneval', 'gsm8k'):
            data = read_rows(ROOT / f'data/eight/{name}.jsonl')
            rows.extend([data[0], max(data[1:], key=lambda x: len(x['input_ids']))])
        return rows
    data = read_rows(ROOT / f'data/long_context/{dataset}.jsonl')
    return [data[0], data[-1]]


def main(jobfile, method):
    jobfile = Path(jobfile)
    job = json.loads(jobfile.read_text())
    out = jobfile.parent / method
    out.mkdir()
    os.environ['LONGSPARK_GQA5'] = '1' if (job['size'], method) == ('14B', 'longspark') else '0'
    import torch
    from transformers import AutoTokenizer
    from split_methods import benchmark as bench, lifecycle, target as tm
    from split_methods.vanilla import VanillaSlots
    from .runtime import install, target_process
    from .finite_workload import run_finite
    install()
    tm._target_process = target_process
    long = job['dataset'] != 'native'
    config = bench.SplitConfig(target_tp=job['target_tp'], target_devices=tuple(range(job['target_tp'])),
        draft_device=None if method == 'vanilla' else job['target_tp'],
        target_kv_gib_per_rank=24 if long else 8, draft_state_gib=8,
        max_running_requests=4, cuda_graph=True, deterministic_inference=True,
        position_variant='fixed_ntk' if long else 'native', context_length=140288 if long else 40968,
        eos_token_ids=(151645,151643) if long else (151645,))
    source = select_inputs(job['dataset'])
    tokenizer = AutoTokenizer.from_pretrained(job['models']['target'], local_files_only=True)
    target = draft = None
    try:
        target = bench.TargetClient(config, model_path=job['models']['target'], method=method,
            draft_path=None if method == 'vanilla' else job['models'][method], seed=job['seed'])
        if not all(x['deterministic_inference'] and x['attention_num_splits'] == 1 for x in target.ready):
            raise RuntimeError('Target did not activate deterministic FA3 inference')
        draft = VanillaSlots(config, target_ready=target.ready) if method == 'vanilla' else bench.LongSparkDraft(
            config, model_path=job['models']['target'], draft_path=job['models'][method], target_ready=target.ready)
        if method != 'vanilla':
            target.prefill_consumer = lambda states: draft.update(states, prefill=True)
        bench.warmup(target, draft, config, source[0]['input_ids'], 0.)
        result = run_finite(target, draft, config, tokenizer, source,
            concurrency=job['concurrency'], seed=job['seed'], output=out/'result.json',
            max_new_tokens=job['max_new_tokens'], temperature=0., total_limit=131072 if long else 40960)
        graphs = result['graphs']
        if any(x.get('fallbacks') or not x.get('replays') for x in graphs['target']):
            raise RuntimeError('Target graph replay audit failed')
        if method != 'vanilla':
            if graphs['draft'].get('fallbacks') or not graphs['draft'].get('replays'):
                raise RuntimeError('Draft graph replay audit failed')
            if not result['transport'].get('registered_ipc') or result['transport'].get('host_bytes_received'):
                raise RuntimeError('Device transport audit failed')
        write_new(out/'execution.json', dict(method=method, target_ranks=target.ready,
            purpose='correctness-only', torch_version=torch.__version__,
            prompt_lengths=[len(r['input_ids']) for r in source],
            kernel_policy='deterministic FA3 with num_splits=1; CUDA graphs enabled',
            source_file=str(Path(tm.__file__).relative_to(ROOT))))
    finally:
        graceful = sys.exc_info()[0] is None
        try:
            lifecycle.release_draft_graphs(draft)
        finally:
            try:
                if target is not None:
                    target.close(graceful=graceful)
            finally:
                if torch.distributed.is_initialized():
                    lifecycle.cleanup_parallel_runtime()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
