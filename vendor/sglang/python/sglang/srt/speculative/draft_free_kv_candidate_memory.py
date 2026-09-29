from __future__ import annotations

from dataclasses import dataclass

import torch


HYBRID_SOURCE_CAP = 2048
HYBRID_AGGREGATE_ROWS = 64
HYBRID_RAW_ROWS = 192
HYBRID_BANK_ROWS = HYBRID_AGGREGATE_ROWS + HYBRID_RAW_ROWS
VERIFY_WIDTH = 8
HYBRID_RAW_UNION_ROWS = HYBRID_RAW_ROWS + VERIFY_WIDTH - 1
SELECTED_LAYERS = 5
HEAD_DIM = 128
REQUIRED_PAGE_SIZE = 1

_INDEX_DTYPES = (torch.int32, torch.int64)


def _nearest_unselected(
    target: int,
    historical_count: int,
    selected: set[int],
) -> int | None:
    if len(selected) >= historical_count:
        return None
    if target not in selected:
        return target
    for distance in range(1, historical_count):
        earlier = target - distance
        if earlier >= 0 and earlier not in selected:
            return earlier
        later = target + distance
        if later < historical_count and later not in selected:
            return later
    return None


def _hybrid_aggregate_ranks(total: int) -> list[int]:
    """Mirror the checkpoint's hybrid_raw192_agg64 historical rank policy."""

    historical_count = max(0, total - HYBRID_RAW_ROWS)
    selected: set[int] = set()

    if historical_count:
        minimum_distance = HYBRID_RAW_ROWS + 1
        maximum_distance = total
        if maximum_distance <= minimum_distance:
            distances = [maximum_distance] * 32
        else:
            ratio = maximum_distance / minimum_distance
            distances = [
                round(minimum_distance * ratio ** (slot / 31)) for slot in range(32)
            ]
        for distance in distances:
            target = max(
                0,
                min(historical_count - 1, total - int(distance)),
            )
            chosen = _nearest_unselected(target, historical_count, selected)
            if chosen is not None:
                selected.add(chosen)

    for slot in range(32):
        if len(selected) >= historical_count:
            break
        target = min(
            historical_count - 1,
            int((slot + 0.5) * historical_count / 32),
        )
        chosen = _nearest_unselected(target, historical_count, selected)
        if chosen is not None:
            selected.add(chosen)

    return sorted(selected)


@dataclass(frozen=True)
class HybridRaw192Agg64PolicyLUT:
    """Fixed-width policy rows indexed by visible source length."""

    aggregate_offsets: torch.Tensor
    aggregate_valid: torch.Tensor
    raw_offsets: torch.Tensor
    raw_valid: torch.Tensor
    source_cap: int = HYBRID_SOURCE_CAP

    @property
    def device(self) -> torch.device:
        return self.aggregate_offsets.device

    def to(self, device: torch.device | str) -> "HybridRaw192Agg64PolicyLUT":
        device = torch.device(device)
        if self.device == device:
            return self
        return HybridRaw192Agg64PolicyLUT(
            aggregate_offsets=self.aggregate_offsets.to(device=device),
            aggregate_valid=self.aggregate_valid.to(device=device),
            raw_offsets=self.raw_offsets.to(device=device),
            raw_valid=self.raw_valid.to(device=device),
            source_cap=self.source_cap,
        )


def build_hybrid_raw192_agg64_policy_lut(
    *,
    source_cap: int = HYBRID_SOURCE_CAP,
    aggregate_rows: int = HYBRID_AGGREGATE_ROWS,
    raw_rows: int = HYBRID_RAW_ROWS,
) -> HybridRaw192Agg64PolicyLUT:
    """Build the exact CPU policy table once, before serving hot paths."""

    if source_cap != HYBRID_SOURCE_CAP:
        raise ValueError(
            f"hybrid_raw192_agg64 requires source_cap={HYBRID_SOURCE_CAP}, "
            f"got {source_cap}"
        )
    if aggregate_rows != HYBRID_AGGREGATE_ROWS:
        raise ValueError(
            "hybrid_raw192_agg64 requires "
            f"aggregate_rows={HYBRID_AGGREGATE_ROWS}, got {aggregate_rows}"
        )
    if raw_rows != HYBRID_RAW_ROWS:
        raise ValueError(
            f"hybrid_raw192_agg64 requires raw_rows={HYBRID_RAW_ROWS}, "
            f"got {raw_rows}"
        )

    aggregate_offsets = torch.zeros(
        source_cap + 1,
        aggregate_rows,
        dtype=torch.int64,
    )
    aggregate_valid = torch.zeros(
        source_cap + 1,
        aggregate_rows,
        dtype=torch.bool,
    )
    raw_offsets = torch.zeros(
        source_cap + 1,
        raw_rows,
        dtype=torch.int64,
    )
    raw_valid = torch.zeros(
        source_cap + 1,
        raw_rows,
        dtype=torch.bool,
    )

    for total in range(source_cap + 1):
        aggregate_ranks = _hybrid_aggregate_ranks(total)
        aggregate_start = aggregate_rows - len(aggregate_ranks)
        if aggregate_ranks:
            aggregate_offsets[total, aggregate_start:] = torch.tensor(
                aggregate_ranks,
                dtype=torch.int64,
            )
            aggregate_valid[total, aggregate_start:] = True

        raw_count = min(total, raw_rows)
        raw_start = raw_rows - raw_count
        if raw_count:
            raw_offsets[total, raw_start:] = torch.arange(
                total - raw_count,
                total,
                dtype=torch.int64,
            )
            raw_valid[total, raw_start:] = True

    return HybridRaw192Agg64PolicyLUT(
        aggregate_offsets=aggregate_offsets,
        aggregate_valid=aggregate_valid,
        raw_offsets=raw_offsets,
        raw_valid=raw_valid,
    )


@dataclass(frozen=True)
class DraftFreeKVCandidateMemoryPlan:
    source_end: torch.Tensor
    source_lengths: torch.Tensor
    source_start: torch.Tensor
    aggregate_offsets: torch.Tensor
    aggregate_valid: torch.Tensor
    aggregate_positions: torch.Tensor
    aggregate_union_positions: torch.Tensor
    aggregate_union_valid: torch.Tensor
    aggregate_union_locs: torch.Tensor
    aggregate_union_inverse: torch.Tensor
    aggregate_union_counts: torch.Tensor
    raw_offsets: torch.Tensor
    raw_valid: torch.Tensor
    raw_positions: torch.Tensor
    raw_union_positions: torch.Tensor
    raw_union_valid: torch.Tensor
    raw_union_locs: torch.Tensor
    raw_union_inverse: torch.Tensor
    cache_batch_idx: torch.Tensor
    cache_leftpad: torch.Tensor
    cache_seqlens: torch.Tensor
    append_only_transitions: torch.Tensor


def _validate_index_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dimensions: int,
    device: torch.device | None = None,
) -> None:
    if tensor.dim() != dimensions:
        raise ValueError(
            f"{name} must be rank {dimensions}, got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype not in _INDEX_DTYPES:
        raise ValueError(f"{name} must use int32 or int64, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def build_draft_free_kv_candidate_memory_plan(
    committed_prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    policy_lut: HybridRaw192Agg64PolicyLUT,
    source_cap: int = HYBRID_SOURCE_CAP,
    verify_width: int = VERIFY_WIDTH,
    aggregate_rows: int = HYBRID_AGGREGATE_ROWS,
    raw_rows: int = HYBRID_RAW_ROWS,
    page_size: int = REQUIRED_PAGE_SIZE,
) -> DraftFreeKVCandidateMemoryPlan:
    """Build eight candidate policies and direct request-pool FA3 slices."""

    if source_cap != HYBRID_SOURCE_CAP:
        raise ValueError(
            f"candidate memory requires source_cap={HYBRID_SOURCE_CAP}, "
            f"got {source_cap}"
        )
    if verify_width != VERIFY_WIDTH:
        raise ValueError(
            f"candidate memory requires verify_width={VERIFY_WIDTH}, "
            f"got {verify_width}"
        )
    if aggregate_rows != HYBRID_AGGREGATE_ROWS:
        raise ValueError(
            f"candidate memory requires aggregate_rows={HYBRID_AGGREGATE_ROWS}, "
            f"got {aggregate_rows}"
        )
    if raw_rows != HYBRID_RAW_ROWS:
        raise ValueError(
            f"candidate memory requires raw_rows={HYBRID_RAW_ROWS}, got {raw_rows}"
        )
    if page_size != REQUIRED_PAGE_SIZE:
        raise ValueError(
            f"candidate memory requires page_size={REQUIRED_PAGE_SIZE}, "
            f"got {page_size}"
        )
    if policy_lut.source_cap != source_cap:
        raise ValueError(
            "policy LUT source cap does not match candidate memory source cap"
        )

    _validate_index_tensor(
        committed_prefix_lens,
        name="committed_prefix_lens",
        dimensions=1,
    )
    _validate_index_tensor(
        req_pool_indices,
        name="req_pool_indices",
        dimensions=1,
        device=committed_prefix_lens.device,
    )
    _validate_index_tensor(
        req_to_token,
        name="req_to_token",
        dimensions=2,
        device=committed_prefix_lens.device,
    )
    if committed_prefix_lens.numel() == 0:
        raise ValueError("candidate memory requires at least one request")
    if req_pool_indices.numel() != committed_prefix_lens.numel():
        raise ValueError(
            "req_pool_indices must contain one row per committed prefix length"
        )
    if policy_lut.device != committed_prefix_lens.device:
        raise ValueError(
            f"policy LUT must be on {committed_prefix_lens.device}, "
            f"got {policy_lut.device}"
        )

    prefix_lens = committed_prefix_lens.to(dtype=torch.int64)
    request_rows = req_pool_indices.to(dtype=torch.int64)
    # Value ranges are scheduler-owned serving contracts. Checking CUDA values
    # here would synchronize the host on every verify round; keep this hot path
    # limited to static shape/dtype/device validation above.

    commit_positions = torch.arange(
        1,
        verify_width + 1,
        dtype=torch.int64,
        device=prefix_lens.device,
    )
    source_end = prefix_lens[:, None] + commit_positions[None, :]
    source_lengths = source_end.clamp(max=source_cap)
    source_start = source_end - source_lengths

    flat_lengths = source_lengths.reshape(-1)
    aggregate_offsets = policy_lut.aggregate_offsets.index_select(0, flat_lengths).view(
        -1, verify_width, aggregate_rows
    )
    aggregate_valid = policy_lut.aggregate_valid.index_select(0, flat_lengths).view(
        -1, verify_width, aggregate_rows
    )
    raw_offsets = policy_lut.raw_offsets.index_select(0, flat_lengths).view(
        -1, verify_width, raw_rows
    )
    raw_valid = policy_lut.raw_valid.index_select(0, flat_lengths).view(
        -1, verify_width, raw_rows
    )

    aggregate_positions = source_start[:, :, None] + aggregate_offsets
    raw_positions = source_start[:, :, None] + raw_offsets

    # Sort a fixed 8 * 64 row plan so repeated seed queries can be evaluated
    # once and mapped back without a dynamic-shape torch.unique operation.
    flat_aggregate_positions = aggregate_positions.flatten(1)
    flat_aggregate_valid = aggregate_valid.flatten(1)
    aggregate_sentinel = req_to_token.shape[1]
    aggregate_sort_keys = torch.where(
        flat_aggregate_valid,
        flat_aggregate_positions,
        torch.full_like(flat_aggregate_positions, aggregate_sentinel),
    )
    sorted_aggregate_positions, aggregate_sort_order = aggregate_sort_keys.sort(dim=1)
    sorted_aggregate_valid = sorted_aggregate_positions != aggregate_sentinel
    aggregate_unique_starts = sorted_aggregate_valid.clone()
    aggregate_unique_starts[:, 1:] &= (
        sorted_aggregate_positions[:, 1:] != sorted_aggregate_positions[:, :-1]
    )
    sorted_aggregate_inverse = (
        aggregate_unique_starts.cumsum(dim=1, dtype=torch.int64) - 1
    ).clamp(min=0)
    aggregate_union_counts = aggregate_unique_starts.sum(dim=1, dtype=torch.int64)
    aggregate_union_positions = torch.zeros_like(flat_aggregate_positions)
    aggregate_union_positions.scatter_add_(
        1,
        sorted_aggregate_inverse,
        torch.where(
            aggregate_unique_starts,
            sorted_aggregate_positions,
            torch.zeros_like(sorted_aggregate_positions),
        ),
    )
    aggregate_union_valid = (
        torch.arange(
            verify_width * aggregate_rows,
            dtype=torch.int64,
            device=prefix_lens.device,
        )[None, :]
        < aggregate_union_counts[:, None]
    )
    aggregate_union_locs = req_to_token[
        request_rows[:, None], aggregate_union_positions
    ]
    flat_aggregate_inverse = torch.zeros_like(sorted_aggregate_inverse)
    flat_aggregate_inverse.scatter_(
        1,
        aggregate_sort_order,
        torch.where(
            sorted_aggregate_valid,
            sorted_aggregate_inverse,
            torch.zeros_like(sorted_aggregate_inverse),
        ),
    )
    aggregate_union_inverse = flat_aggregate_inverse.view(
        -1, verify_width, aggregate_rows
    )

    # The eight recent-token windows are adjacent intervals. Gather their
    # values once from a union of at most 192 + 8 - 1 positions, then use this
    # inverse map instead of materializing eight copies of the overlapping K/V.
    raw_union_start = (prefix_lens + 1 - raw_rows).clamp(min=0)
    raw_union_end = prefix_lens + verify_width
    raw_union_lengths = raw_union_end - raw_union_start
    raw_union_offsets = torch.arange(
        HYBRID_RAW_UNION_ROWS,
        dtype=torch.int64,
        device=prefix_lens.device,
    )
    raw_union_valid = raw_union_offsets[None, :] < raw_union_lengths[:, None]
    raw_union_positions = raw_union_start[:, None] + torch.minimum(
        raw_union_offsets[None, :],
        raw_union_lengths[:, None] - 1,
    )
    raw_union_locs = req_to_token[request_rows[:, None], raw_union_positions]
    raw_union_inverse = raw_positions - raw_union_start[:, None, None]
    raw_union_inverse = torch.where(
        raw_valid,
        raw_union_inverse,
        torch.zeros_like(raw_union_inverse),
    )

    # Pass req_to_token itself to FA3 as the page table. These tensors select
    # each physical request row and its exact [source_start:source_end] slice,
    # so no candidate-specific page table or full K/V tensor is materialized.
    cache_batch_idx = (
        request_rows[:, None].expand(-1, verify_width).reshape(-1).to(dtype=torch.int32)
    )
    cache_leftpad = source_start.reshape(-1).to(dtype=torch.int32)
    cache_seqlens = source_end.reshape(-1).to(dtype=torch.int32)
    append_only_transitions = source_start[:, 1:] == source_start[:, :-1]

    return DraftFreeKVCandidateMemoryPlan(
        source_end=source_end,
        source_lengths=source_lengths,
        source_start=source_start,
        aggregate_offsets=aggregate_offsets,
        aggregate_valid=aggregate_valid,
        aggregate_positions=aggregate_positions,
        aggregate_union_positions=aggregate_union_positions,
        aggregate_union_valid=aggregate_union_valid,
        aggregate_union_locs=aggregate_union_locs,
        aggregate_union_inverse=aggregate_union_inverse,
        aggregate_union_counts=aggregate_union_counts,
        raw_offsets=raw_offsets,
        raw_valid=raw_valid,
        raw_positions=raw_positions,
        raw_union_positions=raw_union_positions,
        raw_union_valid=raw_union_valid,
        raw_union_locs=raw_union_locs,
        raw_union_inverse=raw_union_inverse,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        cache_seqlens=cache_seqlens,
        append_only_transitions=append_only_transitions,
    )


class DraftFreeKVCandidateWorkspace:
    """Round-owned storage for eight candidate aggregate values per layer."""

    def __init__(
        self,
        *,
        workspace_rows: int,
        local_kv_heads: int,
        dtype: torch.dtype,
        device: torch.device | str,
        lazy: bool = True,
        verify_width: int = VERIFY_WIDTH,
        layers: int = SELECTED_LAYERS,
        aggregate_rows: int = HYBRID_AGGREGATE_ROWS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        if workspace_rows <= 0:
            raise ValueError(f"workspace_rows must be positive, got {workspace_rows}")
        if local_kv_heads <= 0:
            raise ValueError(f"local_kv_heads must be positive, got {local_kv_heads}")
        if verify_width != VERIFY_WIDTH:
            raise ValueError(
                f"candidate workspace requires verify_width={VERIFY_WIDTH}, "
                f"got {verify_width}"
            )
        if layers != SELECTED_LAYERS:
            raise ValueError(
                f"candidate workspace requires layers={SELECTED_LAYERS}, got {layers}"
            )
        if aggregate_rows != HYBRID_AGGREGATE_ROWS:
            raise ValueError(
                "candidate workspace requires "
                f"aggregate_rows={HYBRID_AGGREGATE_ROWS}, got {aggregate_rows}"
            )
        if head_dim != HEAD_DIM:
            raise ValueError(
                f"candidate workspace requires head_dim={HEAD_DIM}, got {head_dim}"
            )
        if not dtype.is_floating_point:
            raise ValueError(f"workspace dtype must be floating point, got {dtype}")

        self.workspace_rows = int(workspace_rows)
        self.verify_width = verify_width
        self.layers = layers
        self.local_kv_heads = int(local_kv_heads)
        self.aggregate_rows = aggregate_rows
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self._candidate_values: torch.Tensor | None = None
        self.active = torch.zeros(
            self.workspace_rows,
            dtype=torch.bool,
            device=self.device,
        )
        self.generations = torch.full(
            (self.workspace_rows,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.round_ids = torch.full(
            (self.workspace_rows,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.layer_ready = torch.zeros(
            self.workspace_rows,
            self.layers,
            dtype=torch.bool,
            device=self.device,
        )
        if not lazy:
            self.ensure_allocated()

    @property
    def candidate_values(self) -> torch.Tensor | None:
        return self._candidate_values

    def ensure_allocated(self) -> torch.Tensor:
        if self._candidate_values is None:
            self._candidate_values = torch.empty(
                self.workspace_rows,
                self.verify_width,
                self.layers,
                self.local_kv_heads,
                self.aggregate_rows,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        return self._candidate_values

    def _normalize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        _validate_index_tensor(
            rows,
            name="rows",
            dimensions=1,
            device=self.device,
        )
        rows = rows.to(dtype=torch.int64)
        if rows.numel() == 0:
            raise ValueError("rows must not be empty")
        if torch.unique(rows).numel() != rows.numel():
            raise ValueError("rows must not contain duplicates")
        if bool(torch.any(rows < 0)) or bool(torch.any(rows >= self.workspace_rows)):
            raise ValueError(f"rows must be between 0 and {self.workspace_rows - 1}")
        return rows

    def _normalize_identity(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self._normalize_rows(rows)
        for name, values in (
            ("generations", generations),
            ("round_ids", round_ids),
        ):
            _validate_index_tensor(
                values,
                name=name,
                dimensions=1,
                device=self.device,
            )
            if values.numel() != rows.numel():
                raise ValueError(f"{name} must contain one value per workspace row")
        return (
            rows,
            generations.to(dtype=torch.int64),
            round_ids.to(dtype=torch.int64),
        )

    def _require_current_identity(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, generations, round_ids = self._normalize_identity(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
        if not bool(self.active.index_select(0, rows).all()):
            raise RuntimeError("candidate workspace row is not active")
        if not torch.equal(self.generations.index_select(0, rows), generations):
            raise RuntimeError("stale candidate workspace generation")
        if not torch.equal(self.round_ids.index_select(0, rows), round_ids):
            raise RuntimeError("stale candidate workspace round")
        return rows, generations, round_ids

    def reserve(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> None:
        rows, generations, round_ids = self._normalize_identity(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
        if bool(self.active.index_select(0, rows).any()):
            raise RuntimeError(
                "candidate workspace row is already active; release or cancel it first"
            )
        self.ensure_allocated()
        self.generations.index_copy_(0, rows, generations)
        self.round_ids.index_copy_(0, rows, round_ids)
        self.layer_ready.index_fill_(0, rows, False)
        self.active.index_fill_(0, rows, True)

    def write_layer(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
        layer: int,
        values: torch.Tensor,
    ) -> None:
        rows, _, _ = self._require_current_identity(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
        if not 0 <= layer < self.layers:
            raise ValueError(
                f"layer must be between 0 and {self.layers - 1}, got {layer}"
            )
        expected_shape = (
            rows.numel(),
            self.verify_width,
            self.local_kv_heads,
            self.aggregate_rows,
            self.head_dim,
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"values must have shape {expected_shape}, got {tuple(values.shape)}"
            )
        if values.dtype != self.dtype:
            raise ValueError(f"values must have dtype {self.dtype}, got {values.dtype}")
        if values.device != self.device:
            raise ValueError(f"values must be on {self.device}, got {values.device}")

        candidate_values = self.ensure_allocated()
        candidate_values[:, :, layer].index_copy_(0, rows, values)
        self.layer_ready[:, layer].index_fill_(0, rows, True)

    def gather_accepted(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> torch.Tensor:
        rows, _, _ = self._require_current_identity(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
        _validate_index_tensor(
            commit_lens,
            name="commit_lens",
            dimensions=1,
            device=self.device,
        )
        if commit_lens.numel() != rows.numel():
            raise ValueError("commit_lens must contain one value per workspace row")
        commit_lens = commit_lens.to(dtype=torch.int64)
        if bool(torch.any(commit_lens < 1)) or bool(
            torch.any(commit_lens > self.verify_width)
        ):
            raise ValueError(f"commit_lens must be between 1 and {self.verify_width}")
        if not bool(self.layer_ready.index_select(0, rows).all()):
            raise RuntimeError(
                "candidate workspace is not ready for every selected layer"
            )

        candidate_values = self.ensure_allocated()
        return candidate_values[rows, commit_lens - 1]

    def release(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> None:
        rows, _, _ = self._require_current_identity(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
        self.active.index_fill_(0, rows, False)
        self.layer_ready.index_fill_(0, rows, False)

    def cancel(
        self,
        *,
        rows: torch.Tensor,
        generations: torch.Tensor,
        round_ids: torch.Tensor,
    ) -> None:
        self.release(
            rows=rows,
            generations=generations,
            round_ids=round_ids,
        )
