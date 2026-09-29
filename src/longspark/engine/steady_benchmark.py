"""Steady-state benchmark: a closed loop at fixed concurrency, measured over a fixed time window.

Prefill is eager and serial. This measures the split target/draft engine
directly; it is not an HTTP server or an overlapping production scheduler.
"""

from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .bench_utils import (charged_tokens, prefix_tokens, reset_draft_stats,
                          sync_device, reset_peak_memory, peak_memory, transport_report)
from .steady_metrics import summarize_window, request_tpot


def check_resident_capacity(required, capacity, policy):
    if policy not in ('full_reservation', 'runtime_admission'):
        raise ValueError('unknown KV admission policy')
    if policy == 'full_reservation' and required > capacity:
        raise ValueError(f'fully resident workload needs {required} KV rows; only {capacity} available')


@torch.inference_mode()
def run_steady(target, draft, config, tokenizer, source, *, concurrency,
               max_new_tokens, temperature, seed, output, warmup_seconds=30.,
               measurement_seconds=120., prefill_token_budget=8192, reference_requests=0,
               ignore_eos=False, capacity_policy='full_reservation',
               initial_fill_max_dispatches=None, initial_fill_timeout_seconds=600.):
    if not 1 <= concurrency <= min(config.max_running_requests, draft.capacity):
        raise ValueError("concurrency exceeds allocated slots")
    if warmup_seconds < 0 or measurement_seconds <= 0:
        raise ValueError("invalid time window")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [output, output.with_suffix(".requests.jsonl"), output.with_suffix(".events.jsonl")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"refuse to overwrite: {output}")
    if not source:
        raise ValueError("empty fixture")
    if initial_fill_max_dispatches is None:
        initial_fill_max_dispatches = max(4 * concurrency, len(source) + concurrency)
    if not isinstance(initial_fill_max_dispatches, int) or initial_fill_max_dispatches < concurrency:
        raise ValueError("initial fill dispatch limit must be an integer at least concurrency")
    if not math.isfinite(initial_fill_timeout_seconds) or initial_fill_timeout_seconds <= 0:
        raise ValueError("initial fill timeout must be finite and positive")
    target_cap = min(x["max_tokens"] for x in target.ready)
    capacity = target_cap
    required = concurrency * max(len(row["input_ids"]) + max_new_tokens + 8 for row in source)
    # Full-suite runs admit actual live histories; 32K protocols reserve
    # complete output headroom. Formal results still require no preemptions
    # and the declared actual decode batch under either policy (checked below).
    check_resident_capacity(required, capacity, capacity_policy)
    for row in source:
        if len(row["input_ids"]) + max_new_tokens + 8 > config.context_length:
            raise ValueError("prepare explicit prompt truncation/output headroom before testing")
        if len(row["input_ids"]) + max_new_tokens + 8 > capacity:
            raise ValueError("even one complete request cannot fit the configured pool")
    identities = [row.get("source_id", str(i)) for i, row in enumerate(source)]
    input_hashes = [hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()
                    for row in source]
    eos = set(config.eos_token_ids)
    if ignore_eos:
        eos = set()
    target.reset_measurement(seed)
    reset_draft_stats(draft)
    torch.manual_seed(seed)
    reset_peak_memory(draft)
    active, pending, rows = {}, deque(), []
    events, stages, occupancy = [], [], []
    started = time.perf_counter()
    window_begin = window_end = None
    initial_fill = None
    preemptions = first_dispatches = 0
    max_used = max_active = 0

    def now():
        return time.perf_counter() - started

    def outside_window():
        return window_end is not None and now() >= window_end

    def initial_fill_state():
        return dict(requested_concurrency=concurrency, resident_requests=len(active),
            pending_requests=len(pending), dispatches=first_dispatches,
            elapsed_seconds=now(), max_dispatches=initial_fill_max_dispatches,
            timeout_seconds=initial_fill_timeout_seconds,
            completed_requests=[dict(id=r['id'], source_id=r['source_id'],
                finish_reason=r['finish_reason'], output_ids=list(r['generated']))
                for r in rows if r['completion_seconds'] is not None])

    def fail_initial_fill(reason):
        detail = dict(status='failed', reason=reason, initial_fill=initial_fill_state(),
            target_token_capacity=target_cap,
            draft_token_capacity=None)
        with output.with_suffix('.initial_fill_failure.json').open('x') as stream:
            json.dump(detail, stream, ensure_ascii=False, indent=2)
        print(json.dumps(dict(phase='initial_fill_failed', **detail)), flush=True)
        raise RuntimeError(f'Initial fill cannot reach resident C={concurrency}: {reason}')

    def submit(when):
        i = len(rows)
        index = i % len(source)
        row = dict(id=f"{identities[index]}:occ-{i // len(source)}", source_id=identities[index],
                   source_index=index, occurrence=i // len(source), input_sha256=input_hashes[index],
                   input_ids=source[index]["input_ids"], generated=[], rounds=[], preemptions=0,
                   submitted_seconds=when, first_dispatch_seconds=None,
                   first_token_seconds=None, completion_seconds=None, finish_reason=None)
        rows.append(row)
        pending.append(row)

    def resident_transition():
        occupancy.append(dict(time=now(), resident=len(active), pending=len(pending),
                              pending_prefirst=sum(r["first_token_seconds"] is None for r in pending)))

    def timed(name, function, *args, synchronize=False, **kwargs):
        begin = now()
        value = function(*args, **kwargs)
        if synchronize:
            sync_device(draft)
        stages.append(dict(name=name, begin=begin, end=now()))
        return value

    def release(ids):
        if ids:
            begin = now()
            target.release(ids)
            draft.release(ids)
            stages.append(dict(name="release", begin=begin, end=now()))

    def finish(ids, timestamp):
        for rid in ids:
            row = active.pop(rid)
            row["completion_seconds"] = timestamp
            row["finish_reason"] = "eos" if row["generated"][-1] in eos else "length"
            # Constant CLOSED-LOOP outstanding requests: replacement is
            # submitted at completion, before releasing the old GPU state.
            if window_end is None or timestamp < window_end:
                submit(timestamp)
        release(ids)
        resident_transition()

    for _ in range(concurrency):
        submit(0.)
    resident_transition()
    while not outside_window():
        used = sum(charged_tokens(r) for r in active.values())
        while active and used + 8 * len(active) > capacity:
            if len(active) == 1:
                raise RuntimeError("single-request KV exhaustion")
            rid, victim = active.popitem()
            release([rid])
            used -= charged_tokens(victim)
            victim["preemptions"] += 1
            victim.setdefault("preemption_points", []).append(len(victim["generated"]))
            preemptions += 1
            pending.appendleft(victim)
            resident_transition()
        # Fill all currently feasible slots before the next decode batch.
        # Each long prompt is prefetched in actual bounded target chunks,
        # with incremental LongSpark state updates and a final long-context summary.
        while pending and len(active) < concurrency and not outside_window():
            if window_begin is None:
                if first_dispatches >= initial_fill_max_dispatches:
                    fail_initial_fill('dispatch limit reached; immediate EOS/length completions may prevent residency')
                if now() >= initial_fill_timeout_seconds:
                    fail_initial_fill('initial fill timeout reached')
            admitted, prefill_tokens = [], 0
            while pending and len(active) + len(admitted) < concurrency:
                if window_begin is None and first_dispatches + len(admitted) >= initial_fill_max_dispatches:
                    break
                row = pending[0]
                needed = charged_tokens(row)
                guard = 64 * (len(active) + len(admitted) + 1)
                if used + needed + guard > capacity and (active or admitted):
                    break
                if admitted and prefill_tokens + needed > prefill_token_budget:
                    break
                admitted.append(pending.popleft())
                used += needed
                prefill_tokens += needed
            if not admitted:
                break
            ids = [r["id"] for r in admitted]
            preserved = [r["generated"][-1] if r["generated"] else None for r in admitted]
            for row in admitted:
                if row["first_dispatch_seconds"] is None:
                    row["first_dispatch_seconds"] = now()
                    first_dispatches += 1
                active[row["id"]] = row
            resident_transition()
            payload = timed("prefill_rpc", target.prefill, ids,
                            [prefix_tokens(r) for r in admitted], temperature,
                            preserved_anchors=preserved)
            timed("prefill_state_update", draft.update, payload["states"], prefill=True, synchronize=True)
            timestamp = now()
            emissions, finished = [], []
            for row, token in zip(admitted, payload["bonus"], strict=True):
                if not row["generated"]:
                    row["generated"].append(token)
                    row["first_token_seconds"] = timestamp
                    emissions.append(dict(id=row["id"], source_id=row["source_id"], tokens=1))
                row["anchor"] = token
                if row["generated"][-1] in eos or len(row["generated"]) >= max_new_tokens:
                    finished.append(row["id"])
            if emissions:
                events.append(dict(time=timestamp, phase="prefill", emissions=emissions))
            if finished:
                finish(finished, timestamp)
            # Initial EOS requests are replaced BEFORE the first decode batch.
            # Bounded dispatch/time guards above prevent endless immediate-EOS
            # refills without admitting an underfilled batch into the timer.
        if window_begin is None:
            if len(active) != concurrency:
                fail_initial_fill('KV admission cannot fill all requested slots')
            initial_fill = initial_fill_state()
            window_begin = now() + warmup_seconds
            window_end = window_begin + measurement_seconds
            print(json.dumps(dict(phase="window_scheduled", initial_fill_seconds=now(),
                begin_seconds=window_begin, end_seconds=window_end,
                resident_requests=len(active), pending_requests=len(pending),
                initial_dispatches=first_dispatches,
                initial_completed_requests=initial_fill['completed_requests'])), flush=True)
        if outside_window():
            break
        if not active:
            if not pending:
                raise RuntimeError("closed-loop scheduler lost outstanding requests")
            continue
        max_active = max(max_active, len(active))
        max_used = max(max_used, sum(charged_tokens(r) for r in active.values()))
        running = list(active)
        anchors = [active[rid]["anchor"] for rid in running]
        if getattr(draft, 'is_vanilla', False):
            proposed_cpu = [[] for _ in running]
            payload = timed('target_decode_rpc', target.decode, running, anchors, temperature)
        else:
            proposed, logits, q = timed("proposal", draft.propose, running, anchors,
                                        temperature, synchronize=True)
            proposed_cpu = proposed.tolist()
            blocks = [[a] + tokens for a, tokens in zip(anchors, proposed_cpu, strict=True)]
            payload = timed("verify_rpc", target.verify, running, blocks, temperature,
                            draft_logits=logits, sampled_q=q)
        timed("decode_state_update", draft.update, payload["states"], synchronize=True)
        timestamp = now()
        emissions, finished = [], []
        for i, rid in enumerate(running):
            row = active[rid]
            accepted, bonus = payload["commit_lengths"][i] - 1, payload["bonus"][i]
            actual = accepted_emitted = 0
            for j, token in enumerate(proposed_cpu[i][:accepted] + [bonus]):
                if len(row["generated"]) >= max_new_tokens:
                    break
                row["generated"].append(token)
                actual += 1
                accepted_emitted += int(j < accepted)
                if token in eos:
                    break
            entry = dict(id=rid, source_id=row["source_id"], tokens=actual,
                         accepted_raw=accepted, accepted_emitted=accepted_emitted)
            emissions.append(entry)
            row["rounds"].append((timestamp, accepted, accepted_emitted, actual))
            row["anchor"] = bonus
            if row["generated"][-1] in eos or len(row["generated"]) >= max_new_tokens:
                finished.append(rid)
        events.append(dict(time=timestamp, phase="decode" if getattr(draft, 'is_vanilla', False) else "verify", emissions=emissions))
        if finished:
            finish(finished, timestamp)
        if preemptions > max(1000, len(rows) * 8):
            raise RuntimeError("excessive KV thrashing")
    stopped = now()
    # End without draining. Active/queued rows are retained, not thrown away
    # just because they did not finish in the observation window.
    release(list(active))
    graph_stats = dict(target=target.graph_stats(), draft=draft.graph_stats())
    if config.cuda_graph:
        if any(s["fallbacks"] for s in graph_stats["target"]) or graph_stats["draft"]["fallbacks"]:
            raise RuntimeError("Graph workload had eager fallback")
    reports = []
    for row in rows:
        rounds = row.pop("rounds")
        input_ids = row.pop("input_ids")
        generated = row.pop("generated")
        row.update(prompt_tokens=len(input_ids), output_tokens=len(generated), output_ids=generated,
                   verify_rounds=len(rounds), accepted_raw=sum(r[1] for r in rounds),
                   accepted_emitted=sum(r[2] for r in rounds),
                   text=tokenizer.decode(generated, skip_special_tokens=True), seed=seed,
                   in_flight_at_window_end=(row["submitted_seconds"] < window_end and
                       (row["completion_seconds"] is None or row["completion_seconds"] >= window_end)))
        row["tpot_ms"] = 1000 * request_tpot(row) if request_tpot(row) is not None else None
        if getattr(draft, 'is_vanilla', False):
            row['target_decode_steps'] = row['verify_rounds']
            row['verify_rounds'] = 0
        reports.append(row)
    reference_checks = []
    if reference_requests:
        if temperature != 0:
            raise ValueError("reference checks require greedy T=0")
        selected = [r for r in reports if r["output_tokens"]][:reference_requests]
        if not selected:
            raise RuntimeError("reference smoke produced no output")
        for row in selected:
            n = min(row["output_tokens"], 64)
            reference = target.greedy_reference([row["id"]],
                [source[row["source_index"]]["input_ids"]], n, eos)[0]
            reference_checks.append(dict(id=row["id"], checked_tokens=n,
                exact=reference == row["output_ids"][:n], reference_output_ids=reference))
        with output.with_suffix(".reference.json").open("x") as stream:
            json.dump(reference_checks, stream, indent=2)
        if not all(r["exact"] for r in reference_checks):
            raise RuntimeError("steady scheduler diverged from target greedy reference")
    # Reporting/tokenization/NFS writes are deliberately outside the timer.
    metrics = summarize_window(reports, events, stages, occupancy,
        begin=window_begin, end=window_end, concurrency=concurrency)
    if preemptions or metrics["mean_verify_batch"] != concurrency:
        raise RuntimeError("condition did not sustain the declared resident concurrency")
    subwindows = []
    for i in range(4):
        left = window_begin + i * measurement_seconds / 4
        right = window_begin + (i + 1) * measurement_seconds / 4
        subwindows.append(summarize_window(reports, events, stages, occupancy,
            begin=left, end=right, concurrency=concurrency))
    with paths[1].open("x") as stream:
        for row in reports:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with paths[2].open("x") as stream:
        for event in sorted(events + [dict(type="stage", **s) for s in stages]
                            + [dict(type="occupancy", **o) for o in occupancy],
                            key=lambda item: item.get("time", item.get("begin", 0))):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    report = dict(status="completed", method=type(draft).__name__, config=asdict(config),
        temperature=temperature, seed=seed, concurrency=concurrency,
        capacity_policy=capacity_policy, fully_reserved_kv_rows=required,
        generation_context_limit=config.generation_context_limit,
        ignore_eos=ignore_eos, position_variant=config.position_variant,
        draft_position_audit=draft.position_audit,
        unique_source_prompts=len(set(input_hashes)), max_new_tokens=max_new_tokens,
        initial_fill=initial_fill,
        warmup_seconds=warmup_seconds, measurement_seconds=measurement_seconds,
        measured=metrics, subwindows=subwindows, elapsed_including_fill_warmup_seconds=stopped,
        boundary_operation_overrun_seconds=stopped-window_end,
        submitted_requests=len(reports), completed_requests=sum(r["completion_seconds"] is not None for r in reports),
        preemptions=preemptions, max_active_requests=max_active, max_live_history_tokens=max_used,
        target_pools=target.ready, target_token_capacity=target_cap,
        draft_token_capacity=None,
        draft_state_allocated_bytes=draft.allocated_state_bytes,
        draft_peak_allocated_bytes=peak_memory(draft),
        graphs=graph_stats, transport=transport_report(target),
        reference_checks=reference_checks,
        protocol="fixed client concurrency; replace at completion; initial feasible fill then timed warmup; no final drain",
        timing="coordinator emission after target response and draft state commit; no HTTP/detokenization; chunked serial prefill; prefill_rpc includes streamed draft-state updates",
        metric="window_decode_ms_per_token = 1000 * sum(clipped first-to-last request durations) / postfirst tokens in window; includes censored waits",
        sampling="all 7 draft positions use temperature; standard online rejection sampling; no trajectory locking")
    report['method'] = target.method
    if getattr(draft, 'is_vanilla', False):
        for m in [metrics, *subwindows]:
            for old, new in [('actual_verify_batch_histogram', 'actual_decode_batch_histogram'),
                             ('mean_verify_batch', 'mean_decode_batch'),
                             ('verify_request_rounds', 'target_decode_request_steps'),
                             ('actual_advance_per_verify', 'emitted_tokens_per_decode_step')]:
                m[new] = m.pop(old)
            m['raw_advance_per_verify'] = m['accepted_emitted_rate'] = None
        report['sampling'] = 'target-only autoregressive sampling; no draft or rejection sampling'
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report
