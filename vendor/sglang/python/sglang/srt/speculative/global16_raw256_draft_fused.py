"""Fused Draft-side merge for Global16 + Raw256 + Local7.

Raw256 is still computed by FA3 directly from the Target KV cache.  This
kernel handles the remaining two tiny attention sources and merges all three
normalized outputs without materializing a gathered Global16 batch, source
outputs, score tensors, or an assembled K/V bank.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only unit-test environments.
    triton = None
    tl = None


GLOBAL_ROWS = 16
LOCAL_ROWS = 7
QUERY_ROWS = 7
HEAD_DIM = 128
_MERGE_NUM_WARPS = int(os.getenv("DFK_GLOBAL16_DRAFT_MERGE_NUM_WARPS", "8"))
if _MERGE_NUM_WARPS not in (4, 8):
    raise ValueError("Global16 Draft merge warps must be 4 or 8")


if triton is not None:

    @triton.jit
    def _build_raw256_page_table_kernel(
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        raw_lengths,
        req_to_token_stride_b: tl.constexpr,
        req_to_token_stride_t: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_t: tl.constexpr,
    ):
        request = tl.program_id(axis=0)
        positions = tl.arange(0, 256)
        request_slot = tl.load(req_pool_indices + request)
        seq_len = tl.load(seq_lens + request)
        raw_len = tl.minimum(seq_len, 256)
        source_position = seq_len - raw_len + positions
        cache_locs = tl.load(
            req_to_token
            + request_slot * req_to_token_stride_b
            + source_position * req_to_token_stride_t,
            mask=positions < raw_len,
            other=0,
        )
        tl.store(
            page_table
            + request * page_table_stride_b
            + positions * page_table_stride_t,
            cache_locs,
        )
        tl.store(raw_lengths + request, raw_len)


    @triton.jit(
        do_not_specialize=["global_key_stride_b"],
        do_not_specialize_on_alignment=["raw_lse_stride_h"],
    )
    def _fused_global_local_raw_merge_kernel(
        query,
        global_keys,
        global_values,
        state_slots,
        local_keys,
        local_values,
        local_mask,
        raw_output,
        raw_lse,
        output,
        query_stride_b: tl.constexpr,
        query_stride_h: tl.constexpr,
        query_stride_r: tl.constexpr,
        query_stride_d: tl.constexpr,
        global_key_stride_b,
        global_key_stride_h: tl.constexpr,
        global_key_stride_g: tl.constexpr,
        global_key_stride_d: tl.constexpr,
        global_value_stride_b: tl.constexpr,
        global_value_stride_h: tl.constexpr,
        global_value_stride_g: tl.constexpr,
        global_value_stride_d: tl.constexpr,
        local_key_stride_b: tl.constexpr,
        local_key_stride_h: tl.constexpr,
        local_key_stride_l: tl.constexpr,
        local_key_stride_d: tl.constexpr,
        local_value_stride_b: tl.constexpr,
        local_value_stride_h: tl.constexpr,
        local_value_stride_l: tl.constexpr,
        local_value_stride_d: tl.constexpr,
        local_mask_stride_b: tl.constexpr,
        local_mask_stride_r: tl.constexpr,
        local_mask_stride_l: tl.constexpr,
        raw_output_stride_t: tl.constexpr,
        raw_output_stride_h: tl.constexpr,
        raw_output_stride_d: tl.constexpr,
        raw_lse_stride_t: tl.constexpr,
        raw_lse_stride_h,
        output_stride_b: tl.constexpr,
        output_stride_h: tl.constexpr,
        output_stride_r: tl.constexpr,
        output_stride_d: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        softmax_scale: tl.constexpr,
        softcap: tl.constexpr,
        HAS_SOFTCAP: tl.constexpr,
    ):
        request_head = tl.program_id(axis=0)
        request = request_head // query_heads
        query_head = request_head - request * query_heads
        kv_head = query_head // (query_heads // kv_heads)
        state_request = tl.load(state_slots + request)

        rows = tl.arange(0, 16)
        sources = tl.arange(0, 16)
        dims = tl.arange(0, 128)
        row_mask = rows < 7
        global_source_mask = sources < 16
        local_source_mask = sources < 7

        query_offsets = (
            request * query_stride_b
            + query_head * query_stride_h
            + rows[:, None] * query_stride_r
            + dims[None, :] * query_stride_d
        )
        query_block = tl.load(
            query + query_offsets,
            mask=row_mask[:, None],
            other=0.0,
        )

        global_key_offsets = (
            request * global_key_stride_b
            + query_head * global_key_stride_h
            + sources[:, None] * global_key_stride_g
            + dims[None, :] * global_key_stride_d
        )
        global_key_block = tl.load(
            global_keys + global_key_offsets,
            mask=global_source_mask[:, None],
            other=0.0,
        )
        global_scores = tl.dot(query_block, tl.trans(global_key_block))
        # The existing PyTorch serving path materializes BF16 einsum scores
        # before promoting them for softmax.
        global_scores = global_scores.to(tl.bfloat16).to(tl.float32)
        global_scores *= softmax_scale
        if HAS_SOFTCAP:
            global_scores = (
                tl.libdevice.tanh(global_scores / softcap) * softcap
            )
        global_scores = tl.where(
            row_mask[:, None] & global_source_mask[None, :],
            global_scores,
            0.0,
        )
        global_max = tl.max(global_scores, axis=1)
        global_weights = tl.exp(global_scores - global_max[:, None])
        global_weights = tl.where(
            global_source_mask[None, :],
            global_weights,
            0.0,
        )
        global_normalizer = tl.sum(global_weights, axis=1)

        global_value_offsets = (
            state_request * global_value_stride_b
            + query_head * global_value_stride_h
            + sources[:, None] * global_value_stride_g
            + dims[None, :] * global_value_stride_d
        )
        global_value_block = tl.load(
            global_values + global_value_offsets,
            mask=global_source_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        global_weighted = tl.dot(
            global_weights.to(tl.bfloat16),
            global_value_block,
        )
        global_weighted = global_weighted.to(tl.bfloat16).to(tl.float32)
        global_output = global_weighted / global_normalizer[:, None]
        global_lse = global_max + tl.log(global_normalizer)

        local_key_offsets = (
            request * local_key_stride_b
            + kv_head * local_key_stride_h
            + sources[:, None] * local_key_stride_l
            + dims[None, :] * local_key_stride_d
        )
        local_key_block = tl.load(
            local_keys + local_key_offsets,
            mask=local_source_mask[:, None],
            other=0.0,
        )
        local_scores = tl.dot(query_block, tl.trans(local_key_block))
        local_scores = local_scores.to(tl.bfloat16).to(tl.float32)
        local_scores *= softmax_scale
        if HAS_SOFTCAP:
            local_scores = (
                tl.libdevice.tanh(local_scores / softcap) * softcap
            )
        local_visible = tl.load(
            local_mask
            + request * local_mask_stride_b
            + rows[:, None] * local_mask_stride_r
            + sources[None, :] * local_mask_stride_l,
            mask=row_mask[:, None] & local_source_mask[None, :],
            other=0,
        ).to(tl.int1)
        local_scores = tl.where(
            local_visible,
            local_scores,
            -float("inf"),
        )
        has_local = tl.sum(local_visible.to(tl.int32), axis=1) > 0
        local_max = tl.max(local_scores, axis=1)
        safe_local_max = tl.where(has_local, local_max, 0.0)
        local_weights = tl.where(
            local_visible,
            tl.exp(local_scores - safe_local_max[:, None]),
            0.0,
        )
        local_normalizer = tl.sum(local_weights, axis=1)
        safe_local_normalizer = tl.where(has_local, local_normalizer, 1.0)

        local_value_offsets = (
            request * local_value_stride_b
            + kv_head * local_value_stride_h
            + sources[:, None] * local_value_stride_l
            + dims[None, :] * local_value_stride_d
        )
        local_value_block = tl.load(
            local_values + local_value_offsets,
            mask=local_source_mask[:, None],
            other=0.0,
        )
        local_weighted = tl.dot(
            local_weights.to(tl.bfloat16),
            local_value_block,
        )
        local_weighted = local_weighted.to(tl.bfloat16).to(tl.float32)
        local_output = local_weighted / safe_local_normalizer[:, None]
        local_lse = tl.where(
            has_local,
            safe_local_max + tl.log(safe_local_normalizer),
            -float("inf"),
        )

        token_rows = request * 7 + rows
        raw_lse_rows = tl.load(
            raw_lse
            + token_rows * raw_lse_stride_t
            + query_head * raw_lse_stride_h,
            mask=row_mask,
            other=0.0,
        )
        raw_output_offsets = (
            token_rows[:, None] * raw_output_stride_t
            + query_head * raw_output_stride_h
            + dims[None, :] * raw_output_stride_d
        )
        raw_output_block = tl.load(
            raw_output + raw_output_offsets,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        merged_max = tl.maximum(
            raw_lse_rows,
            tl.maximum(global_lse, local_lse),
        )
        raw_weight = tl.exp(raw_lse_rows - merged_max)
        global_weight = tl.exp(global_lse - merged_max)
        local_weight = tl.where(
            has_local,
            tl.exp(local_lse - merged_max),
            0.0,
        )
        merged_normalizer = raw_weight + global_weight + local_weight
        merged_output = (
            raw_output_block * raw_weight[:, None]
            + global_output * global_weight[:, None]
            + local_output * local_weight[:, None]
        ) / merged_normalizer[:, None]

        output_offsets = (
            request * output_stride_b
            + query_head * output_stride_h
            + rows[:, None] * output_stride_r
            + dims[None, :] * output_stride_d
        )
        tl.store(
            output + output_offsets,
            merged_output,
            mask=row_mask[:, None],
        )


def can_use_fused_global_local_raw_merge(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    state_slots: torch.Tensor | None,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    local_attention_mask: torch.Tensor,
    raw_output: torch.Tensor,
    raw_lse: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether tensors meet the fixed production kernel contract."""

    return bool(
        triton is not None
        and query.is_cuda
        and query.dtype == torch.bfloat16
        and query.ndim == 4
        and query.shape[2:] == (QUERY_ROWS, HEAD_DIM)
        and global_keys.shape
        == (query.shape[0], query.shape[1], GLOBAL_ROWS, HEAD_DIM)
        and global_keys.dtype == torch.bfloat16
        and global_values.is_cuda
        and global_values.dtype == torch.float32
        and global_values.ndim == 4
        and global_values.shape[1:]
        == (query.shape[1], GLOBAL_ROWS, HEAD_DIM)
        and state_slots is not None
        and state_slots.is_cuda
        and state_slots.dtype == torch.int32
        and state_slots.shape == (query.shape[0],)
        and local_keys.shape == local_values.shape
        and local_keys.ndim == 4
        and local_keys.shape[0] == query.shape[0]
        and local_keys.shape[2:] == (LOCAL_ROWS, HEAD_DIM)
        and query.shape[1] % local_keys.shape[1] == 0
        and local_keys.dtype == query.dtype
        and local_values.dtype == query.dtype
        and local_attention_mask.is_cuda
        and local_attention_mask.dtype == torch.bool
        and local_attention_mask.shape
        == (query.shape[0], QUERY_ROWS, LOCAL_ROWS)
        and raw_output.is_cuda
        and raw_output.dtype == query.dtype
        and raw_output.shape
        == (query.shape[0] * QUERY_ROWS, query.shape[1], HEAD_DIM)
        and raw_lse.is_cuda
        and raw_lse.dtype == torch.float32
        and raw_lse.shape
        == (query.shape[0] * QUERY_ROWS, query.shape[1])
        and output.is_cuda
        and output.dtype == query.dtype
        and output.shape == query.shape
    )


def build_raw256_page_table(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    raw_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only the final Raw256 locations into caller-owned workspaces."""

    if triton is None:
        raise RuntimeError("Raw256 page-table construction requires Triton")
    batch = int(req_pool_indices.shape[0])
    if (
        not req_to_token.is_cuda
        or req_to_token.dtype != torch.int32
        or req_to_token.ndim != 2
        or not req_pool_indices.is_cuda
        or req_pool_indices.dtype not in {torch.int32, torch.int64}
        or req_pool_indices.shape != (batch,)
        or not seq_lens.is_cuda
        or seq_lens.dtype not in {torch.int32, torch.int64}
        or seq_lens.shape != (batch,)
        or not page_table.is_cuda
        or page_table.dtype != torch.int32
        or page_table.shape != (batch, 256)
        or not raw_lengths.is_cuda
        or raw_lengths.dtype != torch.int32
        or raw_lengths.shape != (batch,)
    ):
        raise ValueError("Raw256 page-table tensors violate the production contract")
    _build_raw256_page_table_kernel[(batch,)](
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        raw_lengths,
        *req_to_token.stride(),
        *page_table.stride(),
        num_warps=4,
        num_stages=1,
    )
    return page_table, raw_lengths


def fused_global_local_raw_merge(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    state_slots: torch.Tensor,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    local_attention_mask: torch.Tensor,
    raw_output: torch.Tensor,
    raw_lse: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    softcap: float = 0.0,
    validate_contract: bool = True,
) -> torch.Tensor:
    """Launch the fixed-shape serving kernel into caller-owned output."""

    if validate_contract and not can_use_fused_global_local_raw_merge(
        query=query,
        global_keys=global_keys,
        global_values=global_values,
        state_slots=state_slots,
        local_keys=local_keys,
        local_values=local_values,
        local_attention_mask=local_attention_mask,
        raw_output=raw_output,
        raw_lse=raw_lse,
        output=output,
    ):
        raise ValueError("Global16 fused Draft tensors violate the production contract")
    batch, query_heads = query.shape[:2]
    kv_heads = int(local_keys.shape[1])
    _fused_global_local_raw_merge_kernel[(batch * query_heads,)](
        query,
        global_keys,
        global_values,
        state_slots,
        local_keys,
        local_values,
        local_attention_mask,
        raw_output,
        raw_lse,
        output,
        *query.stride(),
        *global_keys.stride(),
        *global_values.stride(),
        *local_keys.stride(),
        *local_values.stride(),
        *local_attention_mask.stride(),
        *raw_output.stride(),
        *raw_lse.stride(),
        *output.stride(),
        query_heads=query_heads,
        kv_heads=kv_heads,
        softmax_scale=float(softmax_scale),
        softcap=float(softcap),
        HAS_SOFTCAP=float(softcap) > 0,
        num_warps=_MERGE_NUM_WARPS,
        num_stages=2,
    )
    return output


__all__ = [
    "build_raw256_page_table",
    "can_use_fused_global_local_raw_merge",
    "fused_global_local_raw_merge",
]
