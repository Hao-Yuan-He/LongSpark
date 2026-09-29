"""One-kernel Draft attention over Global16, Raw256, and Local7."""

from __future__ import annotations

from functools import lru_cache
import json
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
RAW_ROWS = 256
HEAD_DIM = 128


@lru_cache(maxsize=16)
def _launch_config_table(encoded):
    table = json.loads(encoded)
    if not isinstance(table, dict):
        raise ValueError("Draft attention launch configurations must be a JSON object")
    result = {}
    for batch, config in table.items():
        if batch != "default" and (not batch.isdecimal() or int(batch) < 1):
            raise ValueError("Draft attention batch keys must be positive integers or default")
        if (not isinstance(config, list) or len(config) != 3
                or config[0] not in (32, 64, 128) or config[1] not in (4, 8)
                or config[2] not in (1, 2, 3)):
            raise ValueError("Unsupported Draft attention launch configuration")
        result[batch] = tuple(config)
    return result


def draft_attention_launch_config(batch):
    table = _launch_config_table(os.getenv("DFK_GLOBAL16_DRAFT_ATTENTION_CONFIGS", "{}"))
    # Exact batch entries prevent extrapolating a measured winner to unseen shapes.
    return table.get(str(batch), table.get("default", (64, 4, 2)))


if triton is not None:

    @triton.jit
    def _fused_global_local_raw_attention_kernel(
        query,
        global_keys,
        global_values,
        state_slots,
        local_keys,
        local_values,
        local_mask,
        key_cache,
        value_cache,
        page_table,
        raw_lengths,
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
        key_cache_stride_s: tl.constexpr,
        key_cache_stride_p: tl.constexpr,
        key_cache_stride_h: tl.constexpr,
        key_cache_stride_d: tl.constexpr,
        value_cache_stride_s: tl.constexpr,
        value_cache_stride_p: tl.constexpr,
        value_cache_stride_h: tl.constexpr,
        value_cache_stride_d: tl.constexpr,
        page_table_stride_b: tl.constexpr,
        page_table_stride_t: tl.constexpr,
        output_stride_b: tl.constexpr,
        output_stride_h: tl.constexpr,
        output_stride_r: tl.constexpr,
        output_stride_d: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        softmax_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        request_head = tl.program_id(0)
        request = request_head // query_heads
        query_head = request_head - request * query_heads
        kv_head = query_head // (query_heads // kv_heads)

        rows = tl.arange(0, 16)
        dims = tl.arange(0, 128)
        row_mask = rows < 7
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

        running_max = tl.full((16,), -float("inf"), tl.float32)
        running_sum = tl.zeros((16,), tl.float32)
        accumulator = tl.zeros((16, 128), tl.float32)
        raw_length = tl.load(raw_lengths + request)

        for raw_start in range(0, 256, BLOCK_N):
            raw_columns = raw_start + tl.arange(0, BLOCK_N)
            raw_visible = raw_columns < raw_length
            cache_slots = tl.load(
                page_table
                + request * page_table_stride_b
                + raw_columns * page_table_stride_t,
                mask=raw_visible,
                other=0,
            )
            raw_key_offsets = (
                cache_slots[:, None] * key_cache_stride_s
                + kv_head * key_cache_stride_h
                + dims[None, :] * key_cache_stride_d
            )
            raw_keys = tl.load(
                key_cache + raw_key_offsets,
                mask=raw_visible[:, None],
                other=0.0,
            )
            scores = tl.dot(query_block, tl.trans(raw_keys)) * softmax_scale
            scores = tl.where(
                row_mask[:, None] & raw_visible[None, :],
                scores,
                -float("inf"),
            )
            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, block_max)
            # Empty Raw slots (including graph padding) precede valid Global16.
            # Avoid -inf - -inf poisoning the later nonempty sources with NaNs.
            normalizer_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            old_scale = tl.exp(running_max - normalizer_max)
            probabilities = tl.exp(scores - normalizer_max[:, None])
            raw_value_offsets = (
                cache_slots[:, None] * value_cache_stride_s
                + kv_head * value_cache_stride_h
                + dims[None, :] * value_cache_stride_d
            )
            raw_values = tl.load(
                value_cache + raw_value_offsets,
                mask=raw_visible[:, None],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), raw_values
            )
            running_sum = running_sum * old_scale + tl.sum(
                probabilities, axis=1
            )
            running_max = next_max

        sources = tl.arange(0, 16)
        state_request = tl.load(state_slots + request)
        global_key_offsets = (
            request * global_key_stride_b
            + query_head * global_key_stride_h
            + sources[:, None] * global_key_stride_g
            + dims[None, :] * global_key_stride_d
        )
        global_key_block = tl.load(global_keys + global_key_offsets)
        global_scores = tl.dot(query_block, tl.trans(global_key_block))
        global_scores = global_scores.to(tl.bfloat16).to(tl.float32)
        global_scores *= softmax_scale
        global_scores = tl.where(
            row_mask[:, None], global_scores, -float("inf")
        )
        global_block_max = tl.max(global_scores, axis=1)
        next_max = tl.maximum(running_max, global_block_max)
        old_scale = tl.exp(running_max - next_max)
        global_probabilities = tl.exp(global_scores - next_max[:, None])
        global_value_offsets = (
            state_request * global_value_stride_b
            + query_head * global_value_stride_h
            + sources[:, None] * global_value_stride_g
            + dims[None, :] * global_value_stride_d
        )
        global_value_block = tl.load(
            global_values + global_value_offsets
        ).to(tl.bfloat16)
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            global_probabilities.to(tl.bfloat16), global_value_block
        )
        running_sum = running_sum * old_scale + tl.sum(
            global_probabilities, axis=1
        )
        running_max = next_max

        local_visible_columns = sources < 7
        local_key_offsets = (
            request * local_key_stride_b
            + kv_head * local_key_stride_h
            + sources[:, None] * local_key_stride_l
            + dims[None, :] * local_key_stride_d
        )
        local_key_block = tl.load(
            local_keys + local_key_offsets,
            mask=local_visible_columns[:, None],
            other=0.0,
        )
        local_scores = tl.dot(query_block, tl.trans(local_key_block))
        local_scores = local_scores.to(tl.bfloat16).to(tl.float32)
        local_scores *= softmax_scale
        local_visible = tl.load(
            local_mask
            + request * local_mask_stride_b
            + rows[:, None] * local_mask_stride_r
            + sources[None, :] * local_mask_stride_l,
            mask=row_mask[:, None] & local_visible_columns[None, :],
            other=0,
        ).to(tl.int1)
        local_scores = tl.where(local_visible, local_scores, -float("inf"))
        local_block_max = tl.max(local_scores, axis=1)
        next_max = tl.maximum(running_max, local_block_max)
        old_scale = tl.exp(running_max - next_max)
        local_probabilities = tl.where(
            local_visible,
            tl.exp(local_scores - next_max[:, None]),
            0.0,
        )
        local_value_offsets = (
            request * local_value_stride_b
            + kv_head * local_value_stride_h
            + sources[:, None] * local_value_stride_l
            + dims[None, :] * local_value_stride_d
        )
        local_value_block = tl.load(
            local_values + local_value_offsets,
            mask=local_visible_columns[:, None],
            other=0.0,
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            local_probabilities.to(tl.bfloat16), local_value_block
        )
        running_sum = running_sum * old_scale + tl.sum(
            local_probabilities, axis=1
        )

        merged_output = accumulator / running_sum[:, None]
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


def can_use_fused_global_local_raw_attention(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    state_slots: torch.Tensor | None,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    local_attention_mask: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    raw_lengths: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether tensors meet the fixed production kernel contract."""

    batch = int(query.shape[0]) if query.ndim > 0 else 0
    return bool(
        triton is not None
        and query.is_cuda
        and query.dtype == torch.bfloat16
        and query.ndim == 4
        and query.shape[2:] == (QUERY_ROWS, HEAD_DIM)
        and global_keys.shape
        == (batch, query.shape[1], GLOBAL_ROWS, HEAD_DIM)
        and global_keys.dtype == torch.bfloat16
        and global_values.is_cuda
        and global_values.dtype == torch.float32
        and global_values.ndim == 4
        and global_values.shape[1:]
        == (query.shape[1], GLOBAL_ROWS, HEAD_DIM)
        and state_slots is not None
        and state_slots.is_cuda
        and state_slots.dtype == torch.int32
        and state_slots.shape == (batch,)
        and local_keys.shape == local_values.shape
        and local_keys.ndim == 4
        and local_keys.shape[0] == batch
        and local_keys.shape[2:] == (LOCAL_ROWS, HEAD_DIM)
        and query.shape[1] % local_keys.shape[1] == 0
        and local_keys.dtype == query.dtype
        and local_values.dtype == query.dtype
        and local_attention_mask.is_cuda
        and local_attention_mask.dtype == torch.bool
        and local_attention_mask.shape == (batch, QUERY_ROWS, LOCAL_ROWS)
        and key_cache.is_cuda
        and value_cache.is_cuda
        and key_cache.dtype == query.dtype
        and value_cache.dtype == query.dtype
        and key_cache.shape == value_cache.shape
        and key_cache.ndim == 4
        and key_cache.shape[1:] == (1, local_keys.shape[1], HEAD_DIM)
        and page_table.is_cuda
        and page_table.dtype == torch.int32
        and page_table.shape == (batch, RAW_ROWS)
        and raw_lengths.is_cuda
        and raw_lengths.dtype == torch.int32
        and raw_lengths.shape == (batch,)
        and output.is_cuda
        and output.dtype == query.dtype
        and output.shape == query.shape
    )


def fused_global_local_raw_attention(
    *,
    query: torch.Tensor,
    global_keys: torch.Tensor,
    global_values: torch.Tensor,
    state_slots: torch.Tensor,
    local_keys: torch.Tensor,
    local_values: torch.Tensor,
    local_attention_mask: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    raw_lengths: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Launch one attention kernel directly over all three sources."""

    if float(softcap) != 0.0:
        raise ValueError("fused all-source Draft attention requires softcap=0")
    if not can_use_fused_global_local_raw_attention(
        query=query,
        global_keys=global_keys,
        global_values=global_values,
        state_slots=state_slots,
        local_keys=local_keys,
        local_values=local_values,
        local_attention_mask=local_attention_mask,
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        raw_lengths=raw_lengths,
        output=output,
    ):
        raise ValueError("fused all-source Draft tensors violate the contract")
    batch, query_heads = query.shape[:2]
    kv_heads = int(local_keys.shape[1])
    block_n, num_warps, num_stages = draft_attention_launch_config(batch)
    _fused_global_local_raw_attention_kernel[(batch * query_heads,)](
        query,
        global_keys,
        global_values,
        state_slots,
        local_keys,
        local_values,
        local_attention_mask,
        key_cache,
        value_cache,
        page_table,
        raw_lengths,
        output,
        *query.stride(),
        *global_keys.stride(),
        *global_values.stride(),
        *local_keys.stride(),
        *local_values.stride(),
        *local_attention_mask.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        *page_table.stride(),
        *output.stride(),
        query_heads=query_heads,
        kv_heads=kv_heads,
        softmax_scale=float(softmax_scale),
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


__all__ = [
    "can_use_fused_global_local_raw_attention",
    "fused_global_local_raw_attention",
]
