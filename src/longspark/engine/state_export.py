"""Rank-zero KV collection and batch-shaped LongSpark state messages.

Only layout and transport change. Token order, accepted prefixes, tensor dtype,
and the draft's incremental global-summary arithmetic remain unchanged.
"""

import torch
import torch.distributed as dist


BATCH_FORMAT = "longspark_state_batch_v1"


def gather_selected_kv(pool, layer_ids, locations, *, tp, rank, group, workspace):
    """Collect K and V together, returning full heads on rank zero only.

    Head-major packing lets one gather concatenate TP shards in their original
    head order. Non-root ranks do not allocate full-head result buffers.
    The cached receive storage is private to the target and is never sent over
    CUDA IPC directly; TensorChannel still owns its ACK-protected wire buffer.
    """
    if not layer_ids or locations.ndim != 1 or locations.numel() == 0:
        raise ValueError("KV export requires selected layers and nonempty 1D locations")
    parts = []
    for layer in layer_ids:
        key, value = pool.get_kv_buffer(layer)
        parts.extend((key.index_select(0, locations), value.index_select(0, locations)))
    n, heads, dim = parts[0].shape
    local = torch.stack(parts).view(len(layer_ids), 2, n, heads, dim)
    if tp == 1:
        result = local.permute(1, 0, 2, 3, 4).contiguous()
        return result[0], result[1]
    local = local.permute(3, 1, 0, 2, 4).contiguous()
    outputs = None
    if rank == 0:
        shape = (tp, *local.shape)
        result = workspace.get("gather_output")
        if result is None or result.shape != shape or result.dtype != local.dtype or result.device != local.device:
            result = torch.empty(shape, dtype=local.dtype, device=local.device)
            workspace["gather_output"] = result
            workspace["gather_views"] = list(result.unbind(0))
        outputs = workspace["gather_views"]
    dist.gather(local, gather_list=outputs, dst=0, group=group)
    if rank != 0:
        return None, None
    result = workspace["gather_output"].flatten(0, 1).permute(1, 2, 3, 0, 4).contiguous()
    return result[0], result[1]


def batch_longspark_states(packets, counts, keys, values, *, global_output=None, global_lse=None):
    """Send three decode tensors per batch instead of three per request."""
    if len(packets) != len(counts) or sum(counts) != keys.shape[1] or keys.shape != values.shape:
        raise ValueError("inconsistent batched KV export")
    payload = dict(
        format=BATCH_FORMAT,
        ids=[p["id"] for p in packets],
        lengths=[p["length"] for p in packets],
        commit_lens=[p["commit_len"] for p in packets],
        export_counts=list(counts),
        keys=keys,
        values=values,
        last_hidden=torch.stack([p["last_hidden"] for p in packets]),
    )
    if global_output is not None:
        payload.update(global_output=global_output, global_lse=global_lse)
    return payload


def state_metadata(payload):
    if payload.get("format") != BATCH_FORMAT:
        raise ValueError("unknown batched LongSpark state format")
    return [dict(id=rid, length=length, commit_len=count) for rid, length, count in
            zip(payload["ids"], payload["lengths"], payload["commit_lens"], strict=True)]


def unpack_longspark_states(payload):
    """Compatibility views for prefill/CPU tests; decode consumes batches directly."""
    if not isinstance(payload, dict):
        return payload
    packets = state_metadata(payload)
    if len(packets) != len(payload["export_counts"]):
        raise ValueError("batched export counts disagree with request metadata")
    offset = 0
    for i, (packet, count) in enumerate(zip(packets, payload["export_counts"], strict=True)):
        packet.update(keys=payload["keys"][:, offset:offset + count],
                      values=payload["values"][:, offset:offset + count],
                      last_hidden=payload["last_hidden"][i])
        if "global_output" in payload:
            packet.update(global_output=payload["global_output"][i], global_lse=payload["global_lse"][i])
        offset += count
    if offset != payload["keys"].shape[1] or payload["keys"].shape != payload["values"].shape:
        raise ValueError("batched tensor extent disagrees with export counts")
    return packets
