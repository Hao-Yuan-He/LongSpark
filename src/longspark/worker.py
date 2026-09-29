"""One cell per process, using only the unified optimized LongSpark implementation."""
import hashlib
import json
import os
from pathlib import Path
import sys
from .plan import ROOT, read_rows
from .cli import write_new

def audit_requests(output, source, job):
    rows=read_rows(output.with_suffix('.requests.jsonl'))
    if not rows:raise RuntimeError('No requests recorded')
    for row in rows:
        index=row['source_index'];original=source[index]
        digest=hashlib.sha256(json.dumps(original['input_ids'],separators=(',',':')).encode()).hexdigest()
        if row['source_id']!=original['source_id'] or row['input_sha256']!=digest:
            raise RuntimeError('Input identity/order mismatch')
        if row['seed']!=job['seed'] or row['output_tokens']>job['max_new_tokens'] or row.get('preemptions',0):
            raise RuntimeError('Seed/output-cap/preemption audit failed')
        if job['target_tp']==4 and row['prompt_tokens']+row['output_tokens']>131072:
            raise RuntimeError('128K public context limit exceeded')
    if job['mode']=='finite' and len(rows)!=len(source):raise RuntimeError('Finite workload did not drain')
    return len(rows)

def main(jobfile):
    jobfile=Path(jobfile);job=json.loads(jobfile.read_text());out=jobfile.parent
    os.environ['LONGSPARK_GQA5']='1' if (job['size'],job['method'])==('14B','longspark') else '0'
    import torch
    from transformers import AutoTokenizer
    from split_methods import benchmark as bench, lifecycle, target as tm
    from split_methods.steady_benchmark import run_steady
    from split_methods.vanilla import VanillaSlots
    from .runtime import install, target_process
    from .latency_metrics import install_metrics
    from .finite_workload import run_finite
    if Path(tm.__file__).resolve()!=ROOT/'src/split_methods/target.py':
        raise RuntimeError('Unexpected external LongSpark runtime')
    install_metrics();install();tm._target_process=target_process
    class BatchedPrefillClient(bench.TargetClient):
        def prefill(self, ids, blocks, temperature, *, preserved_anchors=None):
            self._send(dict(op='prefill',ids=ids,blocks=blocks,temperature=temperature,
                preserved_anchors=preserved_anchors,final_chunk=True))
            if self.channel is None:return self._responses()[0]
            payload=self.channel.receive();self._responses();return payload
    method=job['method'];tp=job['target_tp']
    config=bench.SplitConfig(target_tp=tp,target_devices=tuple(range(tp)),
        draft_device=None if method=='vanilla' else tp,
        target_kv_gib_per_rank=job['target_kv_gib'],draft_state_gib=job['draft_state_gib'],
        max_running_requests=job['graph_max_batch'],cuda_graph=True,fuse_kv_export=False,
        position_variant=job['position_variant'],context_length=job['context_length'],
        prefill_chunk_size=job['prefill_chunk_size'],eos_token_ids=tuple(job['eos_token_ids']))
    if job['fuse_kv_export']:raise RuntimeError('KV export fusion is disabled in these settings')
    source=read_rows(ROOT/job['fixture'])
    if job.get('smoke'):source=source[:3]
    tokenizer=AutoTokenizer.from_pretrained(job['models']['target'],local_files_only=True)
    client=BatchedPrefillClient if job['prefill_mode']=='batched' else bench.TargetClient
    target=draft=None
    try:
        target=client(config,model_path=job['models']['target'],method=method,
            draft_path=None if method=='vanilla' else job['models'][method],seed=job['seed'])
        draft=VanillaSlots(config,target_ready=target.ready) if method=='vanilla' else bench.LongSparkDraft(
            config,model_path=job['models']['target'],draft_path=job['models'][method],target_ready=target.ready)
        if method!='vanilla':target.prefill_consumer=lambda states:draft.update(states,prefill=True)
        if tp==4:
            for rank in target.ready:
                a=rank['position_audit']
                if a['position_variant']!='fixed_ntk' or a['alpha']!=4. or a['max_cache_error']!=0.:
                    raise RuntimeError('Fixed NTK audit failed')
        warm=read_rows(ROOT/'data/eight/math500.jsonl')[0]['input_ids']
        bench.warmup(target,draft,config,warm,job['temperature'])
        output=out/'result.json'
        if job['mode']=='finite':
            report=run_finite(target,draft,config,tokenizer,source,concurrency=job['concurrency'],
                seed=job['seed'],output=output,max_new_tokens=job['max_new_tokens'],temperature=job['temperature'])
        else:
            report=run_steady(target,draft,config,tokenizer,source,concurrency=job['concurrency'],
                max_new_tokens=job['max_new_tokens'],temperature=job['temperature'],seed=job['seed'],output=output,
                warmup_seconds=job['warmup_seconds'],measurement_seconds=job['measurement_seconds'],
                ignore_eos=False,capacity_policy=job['capacity_policy'])
        if report['status']!='completed' or report['config']['fuse_kv_export']:
            raise RuntimeError('Completion/configuration mismatch')
        if any(g.get('fallbacks',0) or not g.get('replays',0) for g in report['graphs']['target']):
            raise RuntimeError('Target CUDA graph validation failed')
        if method!='vanilla':
            graph=report['graphs']['draft']
            if graph.get('fallbacks',0) or not graph.get('replays',0):raise RuntimeError('Draft graph validation failed')
            if any(isinstance(v,dict) and v.get('fallbacks',0) for v in graph.values()):
                raise RuntimeError('Auxiliary graph fallback')
            transport=target.channel.report()
            if not transport.get('registered_ipc') or transport.get('host_bytes_received'):
                raise RuntimeError('Registered P2P transport validation failed')
        n=audit_requests(output,source,job)
    finally:
        graceful=sys.exc_info()[0] is None
        try:lifecycle.release_draft_graphs(draft)
        finally:
            try:
                if target is not None:target.close(graceful=graceful)
            finally:
                if torch.distributed.is_initialized():lifecycle.cleanup_parallel_runtime()
    write_new(out/'audit.json',dict(passed=True,requests=n,smoke=job.get('smoke',False),
        result_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),fuse_kv_export=False))

if __name__=='__main__':main(sys.argv[1])
