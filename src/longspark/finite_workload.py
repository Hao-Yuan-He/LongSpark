"""Finite closed-loop client: all N requests arrive at zero, fill C, then drain.

Only orchestration/reporting is new. Proposal, target verification, rejection
sampling, KV state, Graph and transport use the existing validated V2 methods.
"""
from collections import Counter, deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from split_methods import benchmark as bench


@torch.inference_mode()
def run_finite(target, draft, config, tokenizer, source, *, concurrency, seed,
               output, max_new_tokens=8192, temperature=1., total_limit=131072):
    if not source or not 1 <= concurrency <= min(config.max_running_requests, draft.capacity):
        raise ValueError('Empty workload or invalid concurrency')
    if len({r['source_id'] for r in source}) != len(source):
        raise ValueError('Finite workload must not repeat source prompts')
    output = Path(output)
    paths = [output, output.with_suffix('.requests.jsonl'), output.with_suffix('.events.jsonl')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refuse to overwrite finite results')
    output.parent.mkdir(parents=True, exist_ok=True)
    target_cap = min(x['max_tokens'] for x in target.ready)
    required = min(concurrency, len(source)) * max(len(r['input_ids']) + max_new_tokens + 8 for r in source)
    if required > target_cap:
        raise ValueError('Full output reservation cannot fit')
    rows = []
    for index, original in enumerate(source):
        ids = original['input_ids']
        if len(ids) + max_new_tokens > total_limit or len(ids) + max_new_tokens + 8 > config.context_length:
            raise ValueError('Public or implementation context exceeded')
        rows.append(dict(id=original['source_id'], source_id=original['source_id'],
            source_index=index, instance_id=original.get('instance_id'), input_ids=ids,
            input_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            generated=[], rounds=[], submitted_seconds=0., first_dispatch_seconds=None,
            first_token_seconds=None, completion_seconds=None, finish_reason=None))
    eos = set(config.eos_token_ids)
    target.reset_measurement(seed)
    bench.reset_draft_stats(draft)
    torch.manual_seed(seed)
    bench.reset_peak_memory(draft)
    pending, active = deque(rows), {}
    events, stages, occupancy = [], [], []
    histogram = Counter()
    max_active = 0
    started = time.perf_counter()

    def now():
        return time.perf_counter() - started

    def timed(name, function, *args, synchronize=False, **kwargs):
        begin = now()
        result = function(*args, **kwargs)
        if synchronize:
            bench.sync_device(draft)
        stages.append(dict(type='stage', name=name, begin=begin, end=now()))
        return result

    def record_occupancy():
        occupancy.append(dict(type='occupancy', time=now(), resident=len(active), pending=len(pending)))

    def finish(ids, timestamp):
        for rid in ids:
            row = active.pop(rid)
            row['completion_seconds'] = timestamp
            row['finish_reason'] = 'eos' if row['generated'][-1] in eos else 'length'
        timed('release', target.release, ids)
        timed('draft_release', draft.release, ids)
        record_occupancy()

    record_occupancy()
    while pending or active:
        # Same fill-before-decode policy as the existing steady client. Unlike
        # that client, this queue is finite and never fabricates replacements.
        while pending and len(active) < concurrency:
            row = pending.popleft()
            rid = row['id']
            row['first_dispatch_seconds'] = now()
            active[rid] = row
            record_occupancy()
            payload = timed('prefill_rpc', target.prefill, [rid], [row['input_ids']], temperature,
                            preserved_anchors=[None])
            timed('prefill_state_update', draft.update, payload['states'], prefill=True, synchronize=True)
            timestamp = now()
            token = payload['bonus'][0]
            row['generated'].append(token)
            row['anchor'] = token
            row['first_token_seconds'] = timestamp
            events.append(dict(time=timestamp, phase='prefill', emissions=[dict(id=rid, source_id=rid, tokens=1)]))
            if token in eos or len(row['generated']) >= max_new_tokens:
                finish([rid], timestamp)
        if not active:
            continue
        max_active = max(max_active, len(active))
        running = list(active)
        anchors = [active[rid]['anchor'] for rid in running]
        histogram[len(running)] += 1
        if getattr(draft, 'is_vanilla', False):
            proposals = [[] for _ in running]
            payload = timed('target_decode_rpc', target.decode, running, anchors, temperature)
        else:
            proposed, logits, q = timed('proposal', draft.propose, running, anchors, temperature, synchronize=True)
            proposals = proposed.tolist()
            blocks = [[a] + p for a, p in zip(anchors, proposals, strict=True)]
            payload = timed('verify_rpc', target.verify, running, blocks, temperature,
                            draft_logits=logits, sampled_q=q)
        timed('decode_state_update', draft.update, payload['states'], synchronize=True)
        timestamp = now()
        emissions, finished = [], []
        for i, rid in enumerate(running):
            row = active[rid]
            accepted, bonus = payload['commit_lengths'][i] - 1, payload['bonus'][i]
            emitted = accepted_emitted = 0
            for j, token in enumerate(proposals[i][:accepted] + [bonus]):
                if len(row['generated']) >= max_new_tokens:
                    break
                row['generated'].append(token)
                emitted += 1
                accepted_emitted += int(j < accepted)
                if token in eos:
                    break
            row['rounds'].append((accepted, accepted_emitted, emitted))
            row['anchor'] = bonus
            emissions.append(dict(id=rid, source_id=rid, tokens=emitted,
                                  accepted_raw=accepted, accepted_emitted=accepted_emitted))
            if row['generated'][-1] in eos or len(row['generated']) >= max_new_tokens:
                finished.append(rid)
        events.append(dict(time=timestamp, phase='decode' if getattr(draft,'is_vanilla',False) else 'verify', emissions=emissions))
        if finished:
            finish(finished, timestamp)
    loop_elapsed = now()
    # Primary makespan ends on the last delivered token, excluding final
    # cleanup, detokenization, metric calculation and all filesystem writes.
    makespan = max(r['completion_seconds'] for r in rows)
    graphs = dict(target=target.graph_stats(), draft=draft.graph_stats())
    if any(x.get('fallbacks', 0) for x in graphs['target']) or graphs['draft'].get('fallbacks',0):
        raise RuntimeError('Finite workload had Graph fallback')
    reports = []
    vanilla = getattr(draft, 'is_vanilla', False)
    for original in rows:
        row = dict(original)
        ids, generated, rounds = row.pop('input_ids'), row.pop('generated'), row.pop('rounds')
        row.pop('anchor', None)
        row.update(prompt_tokens=len(ids), output_tokens=len(generated), output_ids=generated,
            text=tokenizer.decode(generated, skip_special_tokens=True), seed=seed,
            verify_rounds=0 if vanilla else len(rounds), target_decode_steps=len(rounds) if vanilla else 0,
            accepted_raw=sum(r[0] for r in rounds), accepted_emitted=sum(r[1] for r in rounds),
            decode_emitted_tokens=sum(r[2] for r in rounds),
            tpot_ms=1000 * (row['completion_seconds']-row['first_token_seconds']) / (len(generated)-1) if len(generated)>1 else None,
            ttft_ms=1000 * row['first_token_seconds'], e2e_ms=1000 * row['completion_seconds'])
        reports.append(row)
    from .latency_metrics import distribution
    tokens = sum(r['output_tokens'] for r in reports)
    rounds = sum(r['verify_rounds'] for r in reports)
    summary = dict(status='completed', method=target.method, config=asdict(config), seed=seed,
        temperature=temperature, position_variant=config.position_variant, concurrency=concurrency,
        requests=len(reports), completed_requests=len(reports), pending_requests=0,
        makespan_seconds=makespan, elapsed_including_final_release_seconds=loop_elapsed,
        output_tokens=tokens, tokens_per_second=tokens/makespan, mean_output_tokens=tokens/len(reports),
        eos_requests=sum(r['finish_reason']=='eos' for r in reports), max_new_tokens=max_new_tokens,
        public_total_token_limit=total_limit, verify_rounds=rounds,
        actual_advance_per_verify=sum(r['decode_emitted_tokens'] for r in reports)/rounds if rounds else None,
        draft_acceptance_rate=sum(r['accepted_emitted'] for r in reports)/(7*rounds) if rounds else None,
        request_tpot_ms=distribution([r['tpot_ms'] for r in reports if r['tpot_ms'] is not None]),
        request_ttft_ms=distribution([r['ttft_ms'] for r in reports]),
        preemptions=0, fully_reserved_kv_rows=required, max_active_requests=max_active,
        actual_batch_histogram=dict(sorted(histogram.items())), graphs=graphs, transport=bench.transport_report(target),
        stage_seconds={name:sum(s['end']-s['begin'] for s in stages if s['name']==name) for name in {s['name'] for s in stages}},
        timing='All N arrive at zero; initial and refill serial chunked prefill included, fill up to C before decode, full drain included. Model loading/Graph warmup/detokenization/log writes excluded.',
        sampling='Unmodified standard online rejection/residual T1; same seed does not lock target trajectories.')
    for path, values in ((paths[1], reports), (paths[2], sorted(events+stages+occupancy, key=lambda r:r.get('time',r.get('begin',0))))):
        with path.open('x') as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    with output.open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return summary
