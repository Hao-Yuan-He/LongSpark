"""Incremental Global16 state updates for Target verify.

The full confirmed prefix is scanned once during prefill.  Every later Target
verify caches only the Global16 scores for the eight newly produced Target K
rows.  After verification chooses a committed prefix, a second kernel reads
only those accepted Target V rows and merges them into the persistent
normalized-output/LSE state.

Both kernels operate on caller-owned workspaces.  They do not allocate a
Target-K/V bank, copy Target K/V, or launch another attention pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only unit-test environments.
    triton = None
    tl = None


GLOBAL_ROWS = 16
VERIFY_ROWS = 8
HEAD_DIM = 128


@dataclass(frozen=True)
class Global16IncrementalEvidence:
    layer_id: int
    batch_size: int
    verify_rows: int
    source_key_ptr: int
    score_workspace_ptr: int
    full_prefix_rows_read: int = 0
    target_kv_copy_bytes: int = 0
    postverify_delta_fa_calls: int = 0


def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap > 0:
        return torch.tanh(scores / softcap) * softcap
    return scores


def reference_position_scores(
    *,
    global_query: torch.Tensor,
    new_keys: torch.Tensor,
    softmax_scale: float,
    softcap: float = 0.0,
) -> torch.Tensor:
    """FP32 reference for the score-cache kernel.

    ``global_query`` is ``[B, Hq, 16, D]`` and ``new_keys`` is
    ``[B, 8, Hkv, D]``.  GQA head mapping is identical to Target attention:
    consecutive groups of query heads share one KV head.
    """

    if global_query.ndim != 4 or new_keys.ndim != 4:
        raise ValueError("Global query and new Target keys must both be rank-4")
    batch, query_heads, global_rows, head_dim = global_query.shape
    if new_keys.shape[0] != batch or new_keys.shape[1] != VERIFY_ROWS:
        raise ValueError("new Target keys must contain exactly eight rows per request")
    kv_heads = int(new_keys.shape[2])
    if global_rows != GLOBAL_ROWS:
        raise ValueError("Global16 requires exactly sixteen query rows")
    if new_keys.shape[-1] != head_dim:
        raise ValueError("Global query and Target key head dimensions differ")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    query = global_query.float().view(
        batch,
        kv_heads,
        groups,
        GLOBAL_ROWS,
        head_dim,
    )
    keys = new_keys.float().permute(0, 2, 1, 3)
    scores = torch.einsum("bhrgd,bhwd->bhrgw", query, keys)
    scores = _apply_softcap(scores * float(softmax_scale), float(softcap))
    return scores.reshape(batch, query_heads, GLOBAL_ROWS, VERIFY_ROWS)


def reference_merge_accepted_positions(
    *,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
    position_scores: torch.Tensor,
    accepted_values: torch.Tensor,
    accept_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference online-softmax merge used by CPU and CUDA tests.

    ``accepted_values`` remains padded as ``[B, 8, Hkv, D]``.  Only rows below
    each request's ``accept_lens`` participate.
    """

    if state_output.ndim != 4 or state_lse.shape != state_output.shape[:-1]:
        raise ValueError("state output/LSE shapes do not match")
    batch, query_heads, global_rows, head_dim = state_output.shape
    if position_scores.shape != (
        batch,
        query_heads,
        global_rows,
        VERIFY_ROWS,
    ):
        raise ValueError("position score workspace has the wrong shape")
    if accepted_values.shape[:2] != (batch, VERIFY_ROWS):
        raise ValueError("accepted Target values must be padded to eight rows")
    kv_heads = int(accepted_values.shape[2])
    if accepted_values.shape[-1] != head_dim or query_heads % kv_heads:
        raise ValueError("accepted Target values do not match GQA state")
    if accept_lens.shape != (batch,):
        raise ValueError("accept_lens must contain one value per request")
    if torch.any(accept_lens < 1) or torch.any(accept_lens > VERIFY_ROWS):
        raise ValueError("commit length must be between one and eight")

    positions = torch.arange(
        VERIFY_ROWS,
        device=position_scores.device,
        dtype=accept_lens.dtype,
    )
    valid = positions[None, :] < accept_lens[:, None]
    masked_scores = position_scores.float().masked_fill(
        ~valid[:, None, None, :],
        float("-inf"),
    )
    merged_lse = torch.logsumexp(
        torch.stack(
            (
                state_lse.float(),
                torch.logsumexp(masked_scores, dim=-1),
            )
        ),
        dim=0,
    )
    prior_weight = torch.exp(state_lse.float() - merged_lse)
    delta_weight = torch.exp(masked_scores - merged_lse[..., None])
    delta_weight = torch.where(
        valid[:, None, None, :],
        delta_weight,
        torch.zeros_like(delta_weight),
    )
    groups = query_heads // kv_heads
    weights = delta_weight.view(
        batch,
        kv_heads,
        groups,
        global_rows,
        VERIFY_ROWS,
    )
    values = accepted_values.float().permute(0, 2, 1, 3)
    delta_output = torch.einsum("bhrgw,bhwd->bhrgd", weights, values).reshape(
        batch,
        query_heads,
        global_rows,
        head_dim,
    )
    output = state_output.float() * prior_weight[..., None] + delta_output
    return output, merged_lse


if triton is not None:

    @triton.jit
    def _cache_position_scores_kernel(
        query,
        keys,
        output,
        query_stride_b: tl.constexpr,
        query_stride_h: tl.constexpr,
        query_stride_g: tl.constexpr,
        query_stride_d: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_w: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_d: tl.constexpr,
        output_stride_b: tl.constexpr,
        output_stride_h: tl.constexpr,
        output_stride_g: tl.constexpr,
        output_stride_w: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        softmax_scale: tl.constexpr,
        softcap: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr,
    ):
        request_kv_head = tl.program_id(axis=0)
        request = request_kv_head // kv_heads
        kv_head = request_kv_head - request * kv_heads
        query_heads_per_kv = query_heads // kv_heads

        rows = tl.arange(0, 64)
        dims = tl.arange(0, 128)
        positions = tl.arange(0, 16)
        query_head_in_group = rows // 16
        global_row = rows - query_head_in_group * 16
        query_head = kv_head * query_heads_per_kv + query_head_in_group

        query_offsets = (
            request * query_stride_b
            + query_head[:, None] * query_stride_h
            + global_row[:, None] * query_stride_g
            + dims[None, :] * query_stride_d
        )
        key_offsets = (
            request * key_stride_b
            + positions[None, :] * key_stride_w
            + kv_head * key_stride_h
            + dims[:, None] * key_stride_d
        )
        query_block = tl.load(query + query_offsets)
        key_block = tl.load(
            keys + key_offsets,
            mask=positions[None, :] < 8,
            other=0.0,
        )
        scores = tl.dot(query_block, key_block) * softmax_scale
        if HAS_SOFTCAP:
            scores = tl.libdevice.tanh(scores / softcap) * softcap

        output_offsets = (
            request * output_stride_b
            + query_head[:, None] * output_stride_h
            + global_row[:, None] * output_stride_g
            + positions[None, :] * output_stride_w
        )
        tl.store(
            output + output_offsets,
            scores,
            mask=positions[None, :] < 8,
        )


    @triton.jit
    def _accept_offsets_kernel(
        accept_lens,
        accept_offsets,
        batch_size: tl.constexpr,
        BATCH_BLOCK: tl.constexpr,
    ):
        request = tl.program_id(axis=0)
        rows = tl.arange(0, BATCH_BLOCK)
        values = tl.load(
            accept_lens + rows,
            mask=rows < request,
            other=0,
        )
        tl.store(accept_offsets + request, tl.sum(values))


    @triton.jit
    def _merge_accepted_positions_kernel(
        state_output,
        state_lse,
        scores,
        value_cache,
        accepted_cache_locs,
        accept_lens,
        accept_offsets,
        state_slots,
        accepted_locs_stride_b: tl.constexpr,
        accepted_locs_stride_w: tl.constexpr,
        state_output_stride_b: tl.constexpr,
        state_output_stride_h: tl.constexpr,
        state_output_stride_g: tl.constexpr,
        state_output_stride_d: tl.constexpr,
        state_lse_stride_b: tl.constexpr,
        state_lse_stride_h: tl.constexpr,
        state_lse_stride_g: tl.constexpr,
        score_stride_b: tl.constexpr,
        score_stride_h: tl.constexpr,
        score_stride_g: tl.constexpr,
        score_stride_w: tl.constexpr,
        value_stride_slot: tl.constexpr,
        value_stride_page: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_d: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_STATE_SLOTS: tl.constexpr,
        FIXED_CACHE_LOCS: tl.constexpr,
    ):
        request_query_head = tl.program_id(axis=0)
        dim_block = tl.program_id(axis=1)
        request = request_query_head // query_heads
        state_request = (
            tl.load(state_slots + request) if USE_STATE_SLOTS else request
        )
        query_head = request_query_head - request * query_heads
        query_heads_per_kv = query_heads // kv_heads
        kv_head = query_head // query_heads_per_kv

        global_rows = tl.arange(0, 16)
        positions = tl.arange(0, 8)
        dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        accept_len = tl.load(accept_lens + request)
        accept_offset = tl.load(accept_offsets + request)

        score_offsets = (
            request * score_stride_b
            + query_head * score_stride_h
            + global_rows[:, None] * score_stride_g
            + positions[None, :] * score_stride_w
        )
        position_mask = positions[None, :] < accept_len
        position_scores = tl.load(
            scores + score_offsets,
            mask=position_mask,
            other=-float("inf"),
        )
        delta_max = tl.max(position_scores, axis=1)
        delta_lse = delta_max + tl.log(
            tl.sum(tl.exp(position_scores - delta_max[:, None]), axis=1)
        )
        prior_lse_offsets = (
            state_request * state_lse_stride_b
            + query_head * state_lse_stride_h
            + global_rows * state_lse_stride_g
        )
        prior_lse = tl.load(state_lse + prior_lse_offsets)
        maximum = tl.maximum(prior_lse, delta_lse)
        merged_lse = maximum + tl.log(
            tl.exp(prior_lse - maximum) + tl.exp(delta_lse - maximum)
        )

        state_offsets = (
            state_request * state_output_stride_b
            + query_head * state_output_stride_h
            + global_rows[:, None] * state_output_stride_g
            + dims[None, :] * state_output_stride_d
        )
        dim_mask = dims[None, :] < 128
        accumulator = (
            tl.load(state_output + state_offsets, mask=dim_mask, other=0.0)
            * tl.exp(prior_lse - merged_lse)[:, None]
        )
        for position in range(8):
            valid = position < accept_len
            position_score = tl.load(
                scores
                + request * score_stride_b
                + query_head * score_stride_h
                + global_rows * score_stride_g
                + position * score_stride_w,
                mask=valid,
                other=-float("inf"),
            )
            if FIXED_CACHE_LOCS:
                cache_loc = tl.load(
                    accepted_cache_locs
                    + request * accepted_locs_stride_b
                    + position * accepted_locs_stride_w,
                    mask=valid,
                    other=0,
                )
            else:
                cache_loc = tl.load(
                    accepted_cache_locs + accept_offset + position,
                    mask=valid,
                    other=0,
                )
            value_offsets = (
                cache_loc * value_stride_slot
                + kv_head * value_stride_h
                + dims * value_stride_d
            )
            value = tl.load(
                value_cache + value_offsets,
                mask=(dims < 128) & valid,
                other=0.0,
            )
            weight = tl.exp(position_score - merged_lse)
            accumulator += weight[:, None] * value[None, :]

        tl.store(
            state_output + state_offsets,
            accumulator,
            mask=dim_mask,
        )
        tl.store(
            state_lse + prior_lse_offsets,
            merged_lse,
            mask=dim_block == 0,
        )


def _require_cuda_triton() -> None:
    if triton is None:
        raise RuntimeError("Global16 incremental kernels require Triton")


def cache_position_scores(
    *,
    layer_id: int,
    global_query: torch.Tensor,
    new_keys: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    softcap: float = 0.0,
) -> Global16IncrementalEvidence:
    """Cache all eight position scores into a preallocated FP32 workspace."""

    _require_cuda_triton()
    if not global_query.is_cuda or not new_keys.is_cuda or not output.is_cuda:
        raise ValueError("Global16 incremental score caching requires CUDA tensors")
    if global_query.ndim != 4 or new_keys.ndim != 4:
        raise ValueError("Global query and new Target keys must be rank-4")
    batch, query_heads, global_rows, head_dim = global_query.shape
    if new_keys.shape[0] != batch or new_keys.shape[1] != VERIFY_ROWS:
        raise ValueError("new Target keys must contain eight rows per request")
    kv_heads = int(new_keys.shape[2])
    if (
        global_rows != GLOBAL_ROWS
        or head_dim != HEAD_DIM
        or new_keys.shape[-1] != HEAD_DIM
        or query_heads // kv_heads != 4
    ):
        raise ValueError("Global16 incremental score shape is incompatible with GQA")
    if output.shape != (batch, query_heads, GLOBAL_ROWS, VERIFY_ROWS):
        raise ValueError("score output workspace has the wrong shape")
    if output.dtype != torch.float32:
        raise ValueError("score output workspace must use FP32")
    if global_query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Global16 queries must use BF16 or FP16")
    if new_keys.dtype != global_query.dtype:
        raise ValueError("Global query and Target keys must use the same dtype")

    _cache_position_scores_kernel[(batch * kv_heads,)](
        global_query,
        new_keys,
        output,
        *global_query.stride(),
        *new_keys.stride(),
        *output.stride(),
        query_heads=query_heads,
        kv_heads=kv_heads,
        softmax_scale=float(softmax_scale),
        softcap=float(softcap),
        HAS_SOFTCAP=bool(softcap > 0),
        num_warps=4,
    )
    return Global16IncrementalEvidence(
        layer_id=int(layer_id),
        batch_size=batch,
        verify_rows=VERIFY_ROWS,
        source_key_ptr=int(new_keys.data_ptr()),
        score_workspace_ptr=int(output.data_ptr()),
    )


def merge_accepted_positions_(
    *,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
    position_scores: torch.Tensor,
    value_cache: torch.Tensor,
    accepted_cache_locs: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_offsets: torch.Tensor | None,
    state_slots: torch.Tensor | None = None,
) -> None:
    """Merge accepted Target V rows into persistent state without allocation.

    ``position_scores`` and acceptance metadata stay in scheduler batch order.
    When ``state_slots`` is provided, output/LSE are updated directly in the
    persistent request-slot pool instead of gathering and scattering a batch.
    """

    _require_cuda_triton()
    state_capacity, query_heads, global_rows, head_dim = state_output.shape
    batch = int(position_scores.shape[0])
    if (
        not state_output.is_cuda
        or not state_lse.is_cuda
        or not position_scores.is_cuda
        or not value_cache.is_cuda
        or not accepted_cache_locs.is_cuda
        or not accept_lens.is_cuda
        or (accept_offsets is not None and not accept_offsets.is_cuda)
        or (state_slots is not None and not state_slots.is_cuda)
    ):
        raise ValueError("Global16 incremental merge requires CUDA tensors")
    if (
        state_output.dtype != torch.float32
        or state_lse.dtype != torch.float32
        or position_scores.dtype != torch.float32
    ):
        raise ValueError("Global16 state, LSE, and cached scores must use FP32")
    if state_lse.shape != (state_capacity, query_heads, global_rows):
        raise ValueError("Global16 state output/LSE shapes do not match")
    if position_scores.shape != (
        batch,
        query_heads,
        GLOBAL_ROWS,
        VERIFY_ROWS,
    ):
        raise ValueError("Global16 cached score shape is invalid")
    if global_rows != GLOBAL_ROWS or head_dim != HEAD_DIM:
        raise ValueError("Global16 state shape does not match the fixed contract")
    if value_cache.ndim != 4 or value_cache.shape[1] != 1:
        raise ValueError("accepted Target V must come from a page-size-one cache")
    kv_heads = int(value_cache.shape[2])
    if value_cache.shape[-1] != HEAD_DIM or query_heads // kv_heads != 4:
        raise ValueError("Target V cache does not match Global16 GQA state")
    fixed_cache_locs = accepted_cache_locs.ndim == 2
    if fixed_cache_locs:
        if accepted_cache_locs.shape != (batch, VERIFY_ROWS):
            raise ValueError(
                "fixed verify cache locations must have shape [batch, 8]"
            )
        accepted_locs_stride_b, accepted_locs_stride_w = (
            accepted_cache_locs.stride()
        )
    elif accepted_cache_locs.ndim == 1:
        accepted_locs_stride_b = accepted_locs_stride_w = 0
    else:
        raise ValueError(
            "accepted cache locations must be compact flat or fixed [batch, 8]"
        )
    if accept_lens.shape != (batch,) or accept_lens.dtype != torch.int32:
        raise ValueError("accept_lens must be one int32 value per request")
    if not fixed_cache_locs and (
        accept_offsets is None
        or accept_offsets.shape[0] < batch
        or accept_offsets.dtype != torch.int32
    ):
        raise ValueError("compact cache locations require int32 accept offsets")
    accept_offsets_arg = accept_lens if accept_offsets is None else accept_offsets
    if state_slots is None:
        if state_capacity != batch:
            raise ValueError(
                "batch-local Global16 state must have one row per request"
            )
        state_slots_arg = accept_offsets_arg
    else:
        if state_slots.shape != (batch,) or state_slots.dtype != torch.int32:
            raise ValueError("state_slots must be one int32 pool index per request")
        if state_slots.device != state_output.device:
            raise ValueError("state_slots must share the Global16 state device")
        state_slots_arg = state_slots

    if not fixed_cache_locs:
        batch_block = triton.next_power_of_2(batch)
        _accept_offsets_kernel[(batch,)](
            accept_lens,
            accept_offsets_arg,
            batch_size=batch,
            BATCH_BLOCK=batch_block,
            num_warps=1,
        )
    # Keep all 128 value dimensions in one program.  Splitting D across two
    # programs creates a race: the D=0 program can publish merged LSE before
    # the D=64 program has loaded the prior LSE, causing the latter to merge
    # the same accepted rows twice at larger batches.
    _merge_accepted_positions_kernel[(batch * query_heads, 1)](
        state_output,
        state_lse,
        position_scores,
        value_cache,
        accepted_cache_locs,
        accept_lens,
        accept_offsets_arg,
        state_slots_arg,
        accepted_locs_stride_b,
        accepted_locs_stride_w,
        *state_output.stride(),
        *state_lse.stride(),
        *position_scores.stride(),
        *value_cache.stride(),
        query_heads=query_heads,
        kv_heads=kv_heads,
        BLOCK_D=HEAD_DIM,
        USE_STATE_SLOTS=state_slots is not None,
        FIXED_CACHE_LOCS=fixed_cache_locs,
        num_warps=4,
    )


__all__ = [
    "GLOBAL_ROWS",
    "HEAD_DIM",
    "VERIFY_ROWS",
    "Global16IncrementalEvidence",
    "cache_position_scores",
    "merge_accepted_positions_",
    "reference_merge_accepted_positions",
    "reference_position_scores",
]
