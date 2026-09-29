"""Shared warm-up, decoding, and accounting helpers for LongSpark evaluation."""
import time

import torch


def prefix_tokens(row):
    return row["input_ids"] + row.get("generated", [])[:-1]


def charged_tokens(row):
    return len(row["input_ids"]) + max(0, len(row.get("generated", [])) - 1)


def sync_device(draft):
    if draft.device.type == 'cuda':
        torch.cuda.synchronize(draft.device)


def peak_memory(draft):
    return torch.cuda.max_memory_allocated(draft.device) if draft.device.type == 'cuda' else 0


def reset_peak_memory(draft):
    if draft.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(draft.device)


def transport_report(target):
    return target.channel.report() if target.channel is not None else dict(mode='none', draft_gpu_count=0)


def decode_step(target, draft, ids, anchors, temperature):
    if getattr(draft, 'is_vanilla', False):
        return [[] for _ in ids], target.decode(ids, anchors, temperature)
    proposed, logits, q = draft.propose(ids, anchors, temperature)
    sync_device(draft)
    proposed_cpu = proposed.tolist()
    blocks = [[a] + tokens for a, tokens in zip(anchors, proposed_cpu, strict=True)]
    return proposed_cpu, target.verify(ids, blocks, temperature, draft_logits=logits, sampled_q=q)




def reset_draft_stats(draft):
    if getattr(draft, "graph", None):
        draft.graph.reset_stats()
    if hasattr(draft, "graph_replays"):
        draft.graph_replays = draft.graph_fallbacks = 0
    for name in ('logits_graph', 'hidden_projection_graph'):
        if getattr(draft, name, None):
            getattr(draft, name).replays = 0


@torch.inference_mode()
def warmup(target, draft, config, input_ids, temperature):
    tic = time.perf_counter()
    sizes = sorted(set(config.graph_batch_sizes + [draft.capacity]), reverse=True)
    capacity = min(x["max_tokens"] for x in target.ready)
    prompt = input_ids[:32]
    for size in sizes:
        if size > draft.capacity or size * (len(prompt) + 32) > capacity:
            continue
        ids = [f"__warmup_{size}_{i}" for i in range(size)]
        payload = target.prefill(ids, [prompt] * size, temperature)
        draft.update(payload["states"], prefill=True)
        anchors = payload["bonus"]
        for _ in range(2):
            _, payload = decode_step(target, draft, ids, anchors, temperature)
            draft.update(payload["states"])
            anchors = payload["bonus"]
        target.release(ids)
        draft.release(ids)
    sync_device(draft)
    return time.perf_counter() - tic
