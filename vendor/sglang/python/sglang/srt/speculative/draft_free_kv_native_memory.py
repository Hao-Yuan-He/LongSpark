from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.draft_free_kv_candidate_memory import (
    HEAD_DIM,
    HYBRID_AGGREGATE_ROWS,
    HYBRID_BANK_ROWS,
    HYBRID_RAW_ROWS,
    HYBRID_RAW_UNION_ROWS,
    HYBRID_SOURCE_CAP,
    REQUIRED_PAGE_SIZE,
    VERIFY_WIDTH,
    DraftFreeKVCandidateMemoryPlan,
    HybridRaw192Agg64PolicyLUT,
)


SUPPORTED_LOCAL_KV_HEADS = (4, 8)
RESIDUAL_RANK = 128
RMS_NORM_EPS = 1e-6
SOFTMAX_SCALE = HEAD_DIM**-0.5


@dataclass(frozen=True)
class DraftFreeKVCommittedMemoryPlan:
    source_end: torch.Tensor
    source_lengths: torch.Tensor
    source_start: torch.Tensor
    aggregate_offsets: torch.Tensor
    aggregate_valid: torch.Tensor
    aggregate_positions: torch.Tensor
    aggregate_locs: torch.Tensor
    raw_offsets: torch.Tensor
    raw_valid: torch.Tensor
    raw_positions: torch.Tensor
    raw_locs: torch.Tensor
    cache_batch_idx: torch.Tensor
    cache_leftpad: torch.Tensor
    cache_seqlens: torch.Tensor


@dataclass(frozen=True)
class DraftFreeKVLocalMemoryBank:
    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DraftFreeKVCandidateLocalMemoryBank:
    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DraftFreeKVCandidateLayerParts:
    """Union-backed pieces shared by full-bank and capture-tail consumers."""

    aggregate_keys: torch.Tensor
    aggregate_values: torch.Tensor
    raw_union_keys: torch.Tensor
    raw_union_values: torch.Tensor


def _validate_rank_one_int32(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be rank 1, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must use int32, got {tensor.dtype}")


def build_draft_free_kv_committed_memory_plan(
    committed_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    policy_lut: HybridRaw192Agg64PolicyLUT,
    source_cap: int = HYBRID_SOURCE_CAP,
    aggregate_rows: int = HYBRID_AGGREGATE_ROWS,
    raw_rows: int = HYBRID_RAW_ROWS,
    page_size: int = REQUIRED_PAGE_SIZE,
) -> DraftFreeKVCommittedMemoryPlan:
    """Plan one post-accept prefix per request without host synchronization."""

    if source_cap != HYBRID_SOURCE_CAP:
        raise ValueError(
            f"native memory requires source_cap={HYBRID_SOURCE_CAP}, got {source_cap}"
        )
    if aggregate_rows != HYBRID_AGGREGATE_ROWS:
        raise ValueError(
            "native memory requires "
            f"aggregate_rows={HYBRID_AGGREGATE_ROWS}, got {aggregate_rows}"
        )
    if raw_rows != HYBRID_RAW_ROWS:
        raise ValueError(
            f"native memory requires raw_rows={HYBRID_RAW_ROWS}, got {raw_rows}"
        )
    if page_size != REQUIRED_PAGE_SIZE:
        raise ValueError(
            f"native memory requires page_size={REQUIRED_PAGE_SIZE}, got {page_size}"
        )
    if policy_lut.source_cap != source_cap:
        raise ValueError("policy LUT source cap does not match native source cap")

    _validate_rank_one_int32(committed_seq_lens, name="committed_seq_lens")
    _validate_rank_one_int32(req_pool_indices, name="req_pool_indices")
    if req_to_token.dim() != 2:
        raise ValueError(
            "req_to_token must be rank 2, " f"got shape {tuple(req_to_token.shape)}"
        )
    if req_to_token.dtype != torch.int32:
        raise ValueError(f"req_to_token must use int32, got {req_to_token.dtype}")
    if committed_seq_lens.numel() == 0:
        raise ValueError("native memory requires at least one committed request")
    if req_pool_indices.numel() != committed_seq_lens.numel():
        raise ValueError("req_pool_indices must contain one row per committed sequence")
    if req_pool_indices.device != committed_seq_lens.device:
        raise ValueError("req_pool_indices and committed_seq_lens must share a device")
    if req_to_token.device != committed_seq_lens.device:
        raise ValueError("req_to_token and committed_seq_lens must share a device")
    if policy_lut.device != committed_seq_lens.device:
        raise ValueError(
            f"policy LUT must be on {committed_seq_lens.device}, "
            f"got {policy_lut.device}"
        )

    # Sequence-length and request-row value ranges are scheduler-owned. Do not
    # inspect CUDA values here: this function is intended for the verify hot path.
    source_end = committed_seq_lens.to(dtype=torch.int64)
    source_lengths = source_end.clamp(max=source_cap)
    source_start = source_end - source_lengths

    aggregate_offsets = policy_lut.aggregate_offsets.index_select(0, source_lengths)
    aggregate_valid = policy_lut.aggregate_valid.index_select(0, source_lengths)
    raw_offsets = policy_lut.raw_offsets.index_select(0, source_lengths)
    raw_valid = policy_lut.raw_valid.index_select(0, source_lengths)

    aggregate_positions = source_start[:, None] + aggregate_offsets
    raw_positions = source_start[:, None] + raw_offsets
    request_rows = req_pool_indices.to(dtype=torch.int64)[:, None]
    aggregate_locs = req_to_token[request_rows, aggregate_positions]
    raw_locs = req_to_token[request_rows, raw_positions]

    return DraftFreeKVCommittedMemoryPlan(
        source_end=source_end,
        source_lengths=source_lengths,
        source_start=source_start,
        aggregate_offsets=aggregate_offsets,
        aggregate_valid=aggregate_valid,
        aggregate_positions=aggregate_positions,
        aggregate_locs=aggregate_locs,
        raw_offsets=raw_offsets,
        raw_valid=raw_valid,
        raw_positions=raw_positions,
        raw_locs=raw_locs,
        cache_batch_idx=req_pool_indices,
        cache_leftpad=source_start.to(dtype=torch.int32),
        cache_seqlens=committed_seq_lens,
    )


def _native_flash_attn_with_kvcache(**kwargs):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    return flash_attn_with_kvcache(**kwargs)


def _validate_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[int, int, torch.device, torch.dtype]:
    if (
        key_cache.dim() != 3
        or key_cache.shape[1] not in SUPPORTED_LOCAL_KV_HEADS
        or key_cache.shape[2] != HEAD_DIM
    ):
        raise ValueError(
            "key_cache must have shape [slots, local_kv_heads, "
            f"{HEAD_DIM}] with local_kv_heads in {SUPPORTED_LOCAL_KV_HEADS}, "
            f"got {tuple(key_cache.shape)}"
        )
    if value_cache.shape != key_cache.shape:
        raise ValueError(
            "value_cache must match key_cache shape, "
            f"got {tuple(value_cache.shape)} and {tuple(key_cache.shape)}"
        )
    if not key_cache.dtype.is_floating_point:
        raise ValueError(f"key_cache must be floating point, got {key_cache.dtype}")
    if value_cache.dtype != key_cache.dtype:
        raise ValueError("key_cache and value_cache must share a dtype")
    if value_cache.device != key_cache.device:
        raise ValueError("key_cache and value_cache must share a device")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("key_cache and value_cache must be contiguous")
    return key_cache.shape[0], key_cache.shape[1], key_cache.device, key_cache.dtype


def _validate_seed_adapter(
    module: nn.Module,
    *,
    name: str,
    expected_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.shape != expected_shape:
        raise ValueError(
            f"{name}.weight must have shape {expected_shape}, "
            f"got {None if weight is None else tuple(weight.shape)}"
        )
    if weight.device != device:
        raise ValueError(f"{name}.weight must be on {device}, got {weight.device}")
    if weight.dtype != dtype:
        raise ValueError(f"{name}.weight must have dtype {dtype}, got {weight.dtype}")
    if getattr(module, "bias", None) is not None:
        raise ValueError(f"{name} must not have a bias")


def _validate_plan(
    plan: DraftFreeKVCommittedMemoryPlan,
    *,
    device: torch.device,
) -> int:
    batch_size = plan.source_end.numel()
    expected = {
        "source_end": ((batch_size,), torch.int64),
        "source_lengths": ((batch_size,), torch.int64),
        "source_start": ((batch_size,), torch.int64),
        "aggregate_offsets": ((batch_size, HYBRID_AGGREGATE_ROWS), torch.int64),
        "aggregate_valid": ((batch_size, HYBRID_AGGREGATE_ROWS), torch.bool),
        "aggregate_positions": (
            (batch_size, HYBRID_AGGREGATE_ROWS),
            torch.int64,
        ),
        "aggregate_locs": ((batch_size, HYBRID_AGGREGATE_ROWS), torch.int32),
        "raw_offsets": ((batch_size, HYBRID_RAW_ROWS), torch.int64),
        "raw_valid": ((batch_size, HYBRID_RAW_ROWS), torch.bool),
        "raw_positions": ((batch_size, HYBRID_RAW_ROWS), torch.int64),
        "raw_locs": ((batch_size, HYBRID_RAW_ROWS), torch.int32),
        "cache_batch_idx": ((batch_size,), torch.int32),
        "cache_leftpad": ((batch_size,), torch.int32),
        "cache_seqlens": ((batch_size,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(plan, name)
        if tensor.shape != shape:
            raise ValueError(
                f"plan.{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise ValueError(f"plan.{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"plan.{name} must be on {device}, got {tensor.device}")
    return batch_size


def _validate_candidate_plan(
    plan: DraftFreeKVCandidateMemoryPlan,
    *,
    device: torch.device,
) -> int:
    source_end = getattr(plan, "source_end", None)
    if not isinstance(source_end, torch.Tensor) or source_end.dim() != 2:
        raise ValueError("candidate plan.source_end must be a rank-2 tensor")
    batch_size = source_end.shape[0]
    if batch_size == 0 or source_end.shape[1] != VERIFY_WIDTH:
        raise ValueError(
            "candidate plan.source_end must have shape "
            f"[B, {VERIFY_WIDTH}], got {tuple(source_end.shape)}"
        )

    aggregate_union_rows = VERIFY_WIDTH * HYBRID_AGGREGATE_ROWS
    expected = {
        "source_end": ((batch_size, VERIFY_WIDTH), torch.int64),
        "aggregate_valid": (
            (batch_size, VERIFY_WIDTH, HYBRID_AGGREGATE_ROWS),
            torch.bool,
        ),
        "aggregate_union_locs": (
            (batch_size, aggregate_union_rows),
            torch.int32,
        ),
        "aggregate_union_inverse": (
            (batch_size, VERIFY_WIDTH, HYBRID_AGGREGATE_ROWS),
            torch.int64,
        ),
        "raw_valid": (
            (batch_size, VERIFY_WIDTH, HYBRID_RAW_ROWS),
            torch.bool,
        ),
        "raw_union_locs": (
            (batch_size, HYBRID_RAW_UNION_ROWS),
            torch.int32,
        ),
        "raw_union_inverse": (
            (batch_size, VERIFY_WIDTH, HYBRID_RAW_ROWS),
            torch.int64,
        ),
        "cache_batch_idx": ((batch_size * VERIFY_WIDTH,), torch.int32),
        "cache_leftpad": ((batch_size * VERIFY_WIDTH,), torch.int32),
        "cache_seqlens": ((batch_size * VERIFY_WIDTH,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(plan, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"candidate plan.{name} must be a tensor")
        if tensor.shape != shape:
            raise ValueError(
                f"candidate plan.{name} must have shape {shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise ValueError(
                f"candidate plan.{name} must have dtype {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise ValueError(
                f"candidate plan.{name} must be on {device}, got {tensor.device}"
            )
    return batch_size


def _reconstruct_candidate_union(
    union_values: torch.Tensor,
    inverse: torch.Tensor,
    *,
    rows: int,
) -> torch.Tensor:
    batch_size, _, heads, head_dim = union_values.shape
    flat_inverse = inverse.reshape(batch_size, VERIFY_WIDTH * rows)
    gather_index = flat_inverse[:, :, None, None].expand(-1, -1, heads, head_dim)
    return union_values.gather(1, gather_index).view(
        batch_size,
        VERIFY_WIDTH,
        rows,
        heads,
        head_dim,
    )


@torch.inference_mode()
def build_draft_free_kv_native_layer_memory(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seed_down: nn.Module,
    seed_up: nn.Module,
    log_alpha: torch.Tensor,
    plan: DraftFreeKVCommittedMemoryPlan,
    req_to_token: torch.Tensor,
    *,
    flash_attn: Callable | None = None,
    rms_norm_eps: float = RMS_NORM_EPS,
    softmax_scale: float = SOFTMAX_SCALE,
    page_size: int = REQUIRED_PAGE_SIZE,
) -> DraftFreeKVLocalMemoryBank:
    """Build one selected layer's local raw-seed memory bank with FA3."""

    if rms_norm_eps != RMS_NORM_EPS:
        raise ValueError(
            f"native memory requires rms_norm_eps={RMS_NORM_EPS}, got {rms_norm_eps}"
        )
    if softmax_scale != SOFTMAX_SCALE:
        raise ValueError(
            f"native memory requires softmax_scale={SOFTMAX_SCALE}, "
            f"got {softmax_scale}"
        )
    if page_size != REQUIRED_PAGE_SIZE:
        raise ValueError(
            f"native memory requires page_size={REQUIRED_PAGE_SIZE}, got {page_size}"
        )

    _, local_kv_heads, device, dtype = _validate_cache(key_cache, value_cache)
    batch_size = _validate_plan(plan, device=device)
    if req_to_token.dim() != 2 or req_to_token.dtype != torch.int32:
        raise ValueError("req_to_token must be a rank-2 int32 page table")
    if req_to_token.device != device:
        raise ValueError(f"req_to_token must be on {device}, got {req_to_token.device}")
    if not req_to_token.is_contiguous():
        raise ValueError("req_to_token page table must be contiguous")

    _validate_seed_adapter(
        seed_down,
        name="seed_down",
        expected_shape=(RESIDUAL_RANK, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    _validate_seed_adapter(
        seed_up,
        name="seed_up",
        expected_shape=(HEAD_DIM, RESIDUAL_RANK),
        device=device,
        dtype=dtype,
    )
    if not isinstance(log_alpha, torch.Tensor) or log_alpha.numel() != 1:
        raise ValueError("log_alpha must be a scalar tensor")
    if log_alpha.device != device:
        raise ValueError(f"log_alpha must be on {device}, got {log_alpha.device}")
    if not log_alpha.dtype.is_floating_point:
        raise ValueError(f"log_alpha must be floating point, got {log_alpha.dtype}")

    aggregate_locs = plan.aggregate_locs.to(dtype=torch.int64).reshape(-1)
    raw_locs = plan.raw_locs.to(dtype=torch.int64).reshape(-1)
    aggregate_seed = key_cache.index_select(0, aggregate_locs).view(
        batch_size,
        HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    raw_keys = key_cache.index_select(0, raw_locs).view(
        batch_size,
        HYBRID_RAW_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    raw_values = value_cache.index_select(0, raw_locs).view_as(raw_keys)

    seed_float = aggregate_seed.float()
    normalized_seed = seed_float * torch.rsqrt(
        seed_float.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    normalized_seed = normalized_seed.to(dtype=dtype)
    delta = seed_up(F.silu(seed_down(normalized_seed)))
    query = log_alpha.exp().to(dtype=dtype) * aggregate_seed + delta
    packed_query = query.reshape(
        batch_size * HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    ).contiguous()
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * HYBRID_AGGREGATE_ROWS,
        HYBRID_AGGREGATE_ROWS,
        dtype=torch.int32,
        device=device,
    )

    kernel = flash_attn or _native_flash_attn_with_kvcache
    aggregate_values = kernel(
        q=packed_query,
        k_cache=key_cache.view(-1, page_size, local_kv_heads, HEAD_DIM),
        v_cache=value_cache.view(-1, page_size, local_kv_heads, HEAD_DIM),
        page_table=req_to_token,
        cache_batch_idx=plan.cache_batch_idx,
        cache_leftpad=plan.cache_leftpad,
        cache_seqlens=plan.cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=HYBRID_AGGREGATE_ROWS,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        num_splits=1,
        return_softmax_lse=False,
        ver=3,
    )
    expected_output_shape = (
        batch_size * HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    if not isinstance(aggregate_values, torch.Tensor):
        raise RuntimeError("FA3 native memory call must return one output tensor")
    if aggregate_values.shape != expected_output_shape:
        raise RuntimeError(
            "FA3 native memory output shape mismatch: "
            f"expected {expected_output_shape}, got {tuple(aggregate_values.shape)}"
        )
    if aggregate_values.dtype != dtype or aggregate_values.device != device:
        raise RuntimeError("FA3 native memory output dtype/device mismatch")
    aggregate_values = aggregate_values.view(
        batch_size,
        HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )

    aggregate_keys = aggregate_seed.permute(0, 2, 1, 3)
    aggregate_values = aggregate_values.permute(0, 2, 1, 3)
    raw_keys = raw_keys.permute(0, 2, 1, 3)
    raw_values = raw_values.permute(0, 2, 1, 3)
    valid_mask = torch.cat((plan.aggregate_valid, plan.raw_valid), dim=1)
    valid_values = valid_mask[:, None, :, None]
    memory_keys = torch.cat((aggregate_keys, raw_keys), dim=2).masked_fill(
        ~valid_values, 0
    )
    memory_values = torch.cat((aggregate_values, raw_values), dim=2).masked_fill(
        ~valid_values, 0
    )
    if memory_keys.shape != (
        batch_size,
        local_kv_heads,
        HYBRID_BANK_ROWS,
        HEAD_DIM,
    ):
        raise RuntimeError("native memory bank shape invariant failed")

    return DraftFreeKVLocalMemoryBank(
        keys=memory_keys,
        values=memory_values,
        valid_mask=valid_mask,
    )


@torch.inference_mode()
def build_draft_free_kv_candidate_native_layer_parts(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seed_down: nn.Module,
    seed_up: nn.Module,
    log_alpha: torch.Tensor,
    plan: DraftFreeKVCandidateMemoryPlan,
    req_to_token: torch.Tensor,
    *,
    flash_attn: Callable | None = None,
    verify_width: int = VERIFY_WIDTH,
    rms_norm_eps: float = RMS_NORM_EPS,
    softmax_scale: float = SOFTMAX_SCALE,
    page_size: int = REQUIRED_PAGE_SIZE,
) -> DraftFreeKVCandidateLayerParts:
    """Build one layer's aggregate candidates and shared raw union once."""

    if verify_width != VERIFY_WIDTH:
        raise ValueError(
            f"candidate native memory requires verify_width={VERIFY_WIDTH}, "
            f"got {verify_width}"
        )
    if rms_norm_eps != RMS_NORM_EPS:
        raise ValueError(
            "candidate native memory requires "
            f"rms_norm_eps={RMS_NORM_EPS}, got {rms_norm_eps}"
        )
    if softmax_scale != SOFTMAX_SCALE:
        raise ValueError(
            "candidate native memory requires "
            f"softmax_scale={SOFTMAX_SCALE}, got {softmax_scale}"
        )
    if page_size != REQUIRED_PAGE_SIZE:
        raise ValueError(
            "candidate native memory requires "
            f"page_size={REQUIRED_PAGE_SIZE}, got {page_size}"
        )

    _, local_kv_heads, device, dtype = _validate_cache(key_cache, value_cache)
    batch_size = _validate_candidate_plan(plan, device=device)
    if req_to_token.dim() != 2 or req_to_token.dtype != torch.int32:
        raise ValueError("req_to_token must be a rank-2 int32 page table")
    if req_to_token.device != device:
        raise ValueError(f"req_to_token must be on {device}, got {req_to_token.device}")
    if not req_to_token.is_contiguous():
        raise ValueError("req_to_token page table must be contiguous")

    _validate_seed_adapter(
        seed_down,
        name="seed_down",
        expected_shape=(RESIDUAL_RANK, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    _validate_seed_adapter(
        seed_up,
        name="seed_up",
        expected_shape=(HEAD_DIM, RESIDUAL_RANK),
        device=device,
        dtype=dtype,
    )
    if not isinstance(log_alpha, torch.Tensor) or log_alpha.numel() != 1:
        raise ValueError("log_alpha must be a scalar tensor")
    if log_alpha.device != device:
        raise ValueError(f"log_alpha must be on {device}, got {log_alpha.device}")
    if not log_alpha.dtype.is_floating_point:
        raise ValueError(f"log_alpha must be floating point, got {log_alpha.dtype}")

    aggregate_union_locs = plan.aggregate_union_locs.to(torch.int64).reshape(-1)
    raw_union_locs = plan.raw_union_locs.to(torch.int64).reshape(-1)
    aggregate_union_seed = key_cache.index_select(0, aggregate_union_locs).view(
        batch_size,
        VERIFY_WIDTH * HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    raw_union_keys = key_cache.index_select(0, raw_union_locs).view(
        batch_size,
        HYBRID_RAW_UNION_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    raw_union_values = value_cache.index_select(0, raw_union_locs).view_as(
        raw_union_keys
    )
    aggregate_seed = _reconstruct_candidate_union(
        aggregate_union_seed,
        plan.aggregate_union_inverse,
        rows=HYBRID_AGGREGATE_ROWS,
    )
    seed_float = aggregate_seed.float()
    normalized_seed = seed_float * torch.rsqrt(
        seed_float.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    normalized_seed = normalized_seed.to(dtype=dtype)
    delta = seed_up(F.silu(seed_down(normalized_seed)))
    query = log_alpha.exp().to(dtype=dtype) * aggregate_seed + delta
    packed_query = query.reshape(
        batch_size * VERIFY_WIDTH * HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    ).contiguous()
    cu_seqlens_q = torch.arange(
        0,
        (batch_size * VERIFY_WIDTH + 1) * HYBRID_AGGREGATE_ROWS,
        HYBRID_AGGREGATE_ROWS,
        dtype=torch.int32,
        device=device,
    )

    kernel = flash_attn or _native_flash_attn_with_kvcache
    aggregate_values = kernel(
        q=packed_query,
        k_cache=key_cache.view(-1, page_size, local_kv_heads, HEAD_DIM),
        v_cache=value_cache.view(-1, page_size, local_kv_heads, HEAD_DIM),
        page_table=req_to_token,
        cache_batch_idx=plan.cache_batch_idx,
        cache_leftpad=plan.cache_leftpad,
        cache_seqlens=plan.cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=HYBRID_AGGREGATE_ROWS,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        num_splits=1,
        return_softmax_lse=False,
        ver=3,
    )
    expected_output_shape = (
        batch_size * VERIFY_WIDTH * HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )
    if not isinstance(aggregate_values, torch.Tensor):
        raise RuntimeError("FA3 candidate native memory call must return one tensor")
    if aggregate_values.shape != expected_output_shape:
        raise RuntimeError(
            "FA3 candidate native memory output shape mismatch: "
            f"expected {expected_output_shape}, got {tuple(aggregate_values.shape)}"
        )
    if aggregate_values.dtype != dtype or aggregate_values.device != device:
        raise RuntimeError("FA3 candidate native memory output dtype/device mismatch")
    aggregate_values = aggregate_values.view(
        batch_size,
        VERIFY_WIDTH,
        HYBRID_AGGREGATE_ROWS,
        local_kv_heads,
        HEAD_DIM,
    )

    aggregate_keys = aggregate_seed.permute(0, 1, 3, 2, 4)
    aggregate_values = aggregate_values.permute(0, 1, 3, 2, 4)
    return DraftFreeKVCandidateLayerParts(
        aggregate_keys=aggregate_keys,
        aggregate_values=aggregate_values,
        raw_union_keys=raw_union_keys,
        raw_union_values=raw_union_values,
    )


@torch.inference_mode()
def build_draft_free_kv_candidate_native_layer_memory(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seed_down: nn.Module,
    seed_up: nn.Module,
    log_alpha: torch.Tensor,
    plan: DraftFreeKVCandidateMemoryPlan,
    req_to_token: torch.Tensor,
    *,
    flash_attn: Callable | None = None,
    verify_width: int = VERIFY_WIDTH,
    rms_norm_eps: float = RMS_NORM_EPS,
    softmax_scale: float = SOFTMAX_SCALE,
    page_size: int = REQUIRED_PAGE_SIZE,
) -> DraftFreeKVCandidateLocalMemoryBank:
    """Build one layer's eight full banks from the shared union pieces."""

    parts = build_draft_free_kv_candidate_native_layer_parts(
        key_cache,
        value_cache,
        seed_down,
        seed_up,
        log_alpha,
        plan,
        req_to_token,
        flash_attn=flash_attn,
        verify_width=verify_width,
        rms_norm_eps=rms_norm_eps,
        softmax_scale=softmax_scale,
        page_size=page_size,
    )
    raw_keys = _reconstruct_candidate_union(
        parts.raw_union_keys,
        plan.raw_union_inverse,
        rows=HYBRID_RAW_ROWS,
    )
    raw_values = _reconstruct_candidate_union(
        parts.raw_union_values,
        plan.raw_union_inverse,
        rows=HYBRID_RAW_ROWS,
    )
    raw_keys = raw_keys.permute(0, 1, 3, 2, 4)
    raw_values = raw_values.permute(0, 1, 3, 2, 4)
    valid_mask = torch.cat((plan.aggregate_valid, plan.raw_valid), dim=2)
    valid_values = valid_mask[:, :, None, :, None]
    memory_keys = torch.cat((parts.aggregate_keys, raw_keys), dim=3).masked_fill(
        ~valid_values, 0
    )
    memory_values = torch.cat((parts.aggregate_values, raw_values), dim=3).masked_fill(
        ~valid_values, 0
    )
    batch_size = plan.source_end.shape[0]
    local_kv_heads = key_cache.shape[1]
    expected_bank_shape = (
        batch_size,
        VERIFY_WIDTH,
        local_kv_heads,
        HYBRID_BANK_ROWS,
        HEAD_DIM,
    )
    if memory_keys.shape != expected_bank_shape:
        raise RuntimeError("candidate native memory bank shape invariant failed")

    return DraftFreeKVCandidateLocalMemoryBank(
        keys=memory_keys,
        values=memory_values,
        valid_mask=valid_mask,
    )
