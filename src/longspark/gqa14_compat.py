"""Process-local GQA5 adapter; the frozen shared GQA4 implementation is untouched.

The score kernel extends the original 64-row tile to a masked 128-row tile
for 5*16 query rows. The existing merge kernel already maps arbitrary GQA
groups correctly; only its Python wrapper had a GQA4-only assertion.
"""
import torch
import triton
import triton.language as tl
from sglang.srt.speculative import global16_raw256_incremental as original

_original_scores = original.cache_position_scores
_original_merge = original.merge_accepted_positions_


@triton.jit
def _scores5(query, keys, output,
             qb: tl.constexpr, qh: tl.constexpr, qg: tl.constexpr, qd: tl.constexpr,
             kb: tl.constexpr, kw: tl.constexpr, kh: tl.constexpr, kd: tl.constexpr,
             ob: tl.constexpr, oh: tl.constexpr, og: tl.constexpr, ow: tl.constexpr,
             KV_HEADS: tl.constexpr, SCALE: tl.constexpr):
    program = tl.program_id(0)
    request = program // KV_HEADS
    kv = program % KV_HEADS
    rows = tl.arange(0, 128)
    dims = tl.arange(0, 128)
    positions = tl.arange(0, 16)
    heads = kv * 5 + rows // 16
    global_rows = rows % 16
    q = tl.load(query + request * qb + heads[:, None] * qh
                + global_rows[:, None] * qg + dims[None, :] * qd,
                mask=rows[:, None] < 80, other=0.)
    k = tl.load(keys + request * kb + positions[None, :] * kw
                + kv * kh + dims[:, None] * kd,
                mask=positions[None, :] < 8, other=0.)
    scores = tl.dot(q, k) * SCALE
    tl.store(output + request * ob + heads[:, None] * oh
             + global_rows[:, None] * og + positions[None, :] * ow,
             scores, mask=(rows[:, None] < 80) & (positions[None, :] < 8))


def cache_position_scores(*, layer_id, global_query, new_keys, output,
                          softmax_scale, softcap=0.):
    if global_query.shape[1] != new_keys.shape[2] * 5:
        return _original_scores(layer_id=layer_id, global_query=global_query,
            new_keys=new_keys, output=output, softmax_scale=softmax_scale, softcap=softcap)
    batch, heads, rows, dim = global_query.shape
    if rows != 16 or dim != 128 or new_keys.shape != (batch, 8, heads // 5, 128):
        raise ValueError('GQA5 query/key shape mismatch')
    if output.shape != (batch, heads, 16, 8) or output.dtype != torch.float32:
        raise ValueError('GQA5 score workspace mismatch')
    if global_query.dtype not in (torch.bfloat16, torch.float16) or new_keys.dtype != global_query.dtype:
        raise ValueError('GQA5 expects matching BF16/FP16 query and keys')
    if not (global_query.is_cuda and new_keys.is_cuda and output.is_cuda):
        raise ValueError('GQA5 requires CUDA tensors')
    if softcap != 0:
        raise ValueError('This Qwen3-specific adapter is validated only with softcap=0')
    _scores5[(batch * (heads // 5),)](global_query, new_keys, output,
        *global_query.stride(), *new_keys.stride(), *output.stride(),
        KV_HEADS=heads // 5, SCALE=float(softmax_scale), num_warps=4)
    return original.Global16IncrementalEvidence(layer_id=int(layer_id), batch_size=batch,
        verify_rows=8, source_key_ptr=int(new_keys.data_ptr()), score_workspace_ptr=int(output.data_ptr()))


def merge_accepted_positions_(*, state_output, state_lse, position_scores,
                              value_cache, accepted_cache_locs, accept_lens,
                              accept_offsets, state_slots=None):
    if state_output.shape[1] != value_cache.shape[2] * 5:
        return _original_merge(state_output=state_output, state_lse=state_lse,
            position_scores=position_scores, value_cache=value_cache,
            accepted_cache_locs=accepted_cache_locs, accept_lens=accept_lens,
            accept_offsets=accept_offsets, state_slots=state_slots)
    capacity, heads, rows, dim = state_output.shape
    batch = position_scores.shape[0]
    tensors = (state_output, state_lse, position_scores, value_cache, accepted_cache_locs, accept_lens)
    if not all(t.is_cuda and t.device == state_output.device for t in tensors):
        raise ValueError('GQA5 merge tensors must share a CUDA device')
    if any(t.dtype != torch.float32 for t in (state_output, state_lse, position_scores)):
        raise ValueError('GQA5 state/LSE/scores must use FP32')
    if rows != 16 or dim != 128 or state_lse.shape != (capacity, heads, 16):
        raise ValueError('GQA5 state shape mismatch')
    if position_scores.shape != (batch, heads, 16, 8):
        raise ValueError('GQA5 score shape mismatch')
    if value_cache.ndim != 4 or value_cache.shape[1:] != (1, heads // 5, 128):
        raise ValueError('GQA5 V cache shape mismatch')
    if accept_lens.shape != (batch,) or accept_lens.dtype != torch.int32:
        raise ValueError('Invalid acceptance lengths')
    # This split runner supplies fixed [batch,8] locations and persistent slots.
    # Do not silently emulate unsupported compact layouts.
    if accepted_cache_locs.shape != (batch, 8) or accept_offsets is not None:
        raise ValueError('GQA5 adapter expects fixed verify locations')
    if state_slots is None:
        if capacity != batch:
            raise ValueError('Batch-local state requires capacity=batch')
        slots_arg = accept_lens
    else:
        if state_slots.shape != (batch,) or state_slots.dtype != torch.int32 or state_slots.device != state_output.device:
            raise ValueError('Invalid persistent state slots')
        slots_arg = state_slots
    original._merge_accepted_positions_kernel[(batch * heads, 1)](
        state_output, state_lse, position_scores, value_cache, accepted_cache_locs,
        accept_lens, accept_lens, slots_arg, *accepted_cache_locs.stride(),
        *state_output.stride(), *state_lse.stride(), *position_scores.stride(), *value_cache.stride(),
        query_heads=heads, kv_heads=heads // 5, BLOCK_D=128,
        USE_STATE_SLOTS=state_slots is not None, FIXED_CACHE_LOCS=True, num_warps=4)


def install():
    original.cache_position_scores = cache_position_scores
    original.merge_accepted_positions_ = merge_accepted_positions_


@torch.inference_mode()
def self_test():
    """Compare GQA5 FP32 scores/state/LSE against the original torch reference."""
    torch.cuda.set_device(2)
    device = 'cuda:2'
    torch.manual_seed(980426)
    torch.backends.cuda.matmul.allow_tf32 = False
    evidence = []
    for batch in (1, 3, 64):
        query = torch.randn(1, 40, 16, 128, device=device, dtype=torch.bfloat16).expand(batch, -1, -1, -1)
        capacity = batch + 3
        output = torch.randn(capacity, 2, 40, 16, 128, device=device)[:, 1]
        lse = torch.randn(capacity, 2, 40, 16, device=device)[:, 1]
        slots = torch.randperm(capacity, device=device)[:batch].to(torch.int32)
        expected_output, expected_lse = output[slots].clone(), lse[slots].clone()
        scores_error = output_error = lse_error = 0.
        for step in range(16):
            keys = torch.randn(batch, 8, 8, 128, device=device, dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            scores = torch.empty(batch, 40, 16, 8, device=device)
            cache_position_scores(layer_id=0, global_query=query, new_keys=keys,
                output=scores, softmax_scale=128**-.5)
            reference = original.reference_position_scores(global_query=query, new_keys=keys, softmax_scale=128**-.5)
            torch.testing.assert_close(scores, reference, rtol=2e-5, atol=2e-5)
            scores_error = max(scores_error, (scores-reference).abs().max().item())
            lens = ((torch.arange(batch, device=device) + step) % 8 + 1).to(torch.int32)
            locs = torch.arange(batch*8, device=device).reshape(batch, 8)
            expected_output, expected_lse = original.reference_merge_accepted_positions(
                state_output=expected_output, state_lse=expected_lse, position_scores=reference,
                accepted_values=values, accept_lens=lens)
            merge_accepted_positions_(state_output=output, state_lse=lse, position_scores=scores,
                value_cache=values.reshape(batch*8, 1, 8, 128), accepted_cache_locs=locs,
                accept_lens=lens, accept_offsets=None, state_slots=slots)
            torch.testing.assert_close(output[slots], expected_output, rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(lse[slots], expected_lse, rtol=3e-5, atol=3e-5)
            output_error = max(output_error, (output[slots]-expected_output).abs().max().item())
            lse_error = max(lse_error, (lse[slots]-expected_lse).abs().max().item())
        evidence.append(dict(batch=batch, rounds=16, score_max_abs=scores_error,
            state_max_abs=output_error, lse_max_abs=lse_error))
    return dict(passed=True, tests=evidence, physical_gpu=2,
        changes='14B LongSpark incremental state only; shared source and GQA4 path unchanged')
