"""Differentiable shared-K/V training adapter for Global16 + Raw256.

Offline target prefill stores one full Target K/V tensor per source sequence.
Many sampled anchors can refer to that source.  This adapter deliberately uses
``expand`` views for those source K/V tensors, rather than ``index_select`` or
an anchor-sized K/V bank, so Raw256 remains a borrowed Target-cache operand in
the training definition too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .global16_raw256 import (
    Global16MemoryState,
    Global16Raw256Reference,
    attention_statistics_from_4d,
    merge_normalized_attention,
)


# The differentiable full-prefix reference keeps attention intermediates for
# the Global-query backward pass.  A 512-anchor group at 2048 tokens creates
# a multi-GiB temporary even though every anchor is mathematically independent.
# Keep the working group bounded while preserving the same per-anchor state.
_FULL_PREFIX_STATE_ANCHOR_CHUNK_SIZE = 32

# Full-history Global16 attention has shape
# [anchors, selected_layers, query_heads, 16, target_tokens].  At 18K+ tokens
# its score workspace must be streamed even though the Target context remains
# fully visible.  Checkpointing the streamed reduction keeps its temporary
# score tensors out of the backward-save set; it changes compute, not the
# Global16+Raw256 mathematical definition.
_FULL_PREFIX_STATE_DIRECT_MAX_TOKENS = 2048
_FULL_PREFIX_STATE_KEY_CHUNK_SIZE = 512


@dataclass(frozen=True)
class Global16Raw256TrainingReadEvidence:
    """Mechanism evidence for one dense source-group read."""

    source_index: int
    key_storage_ptr: int
    value_storage_ptr: int
    raw_rows: int
    combined_kv_bank_bytes: int = 0


def _storage_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _statistics_as_normalized(statistics) -> tuple[torch.Tensor, torch.Tensor]:
    visible = statistics.normalizer > 0
    safe_normalizer = statistics.normalizer.clamp_min(
        torch.finfo(statistics.normalizer.dtype).tiny
    )
    lse = statistics.max_logits + torch.log(safe_normalizer)
    lse = torch.where(visible, lse, torch.full_like(lse, float("-inf")))
    return statistics.output(), lse


class DenseGlobal16Raw256DraftContext:
    """A differentiable Global16/Raw256 context over shared Target K/V rows."""

    def __init__(
        self,
        *,
        reference: Global16Raw256Reference,
        keys: torch.Tensor,
        values: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        config = reference.config
        if keys.shape != values.shape or keys.ndim != 5:
            raise ValueError("Target K/V must be matching [sources, layers, heads, tokens, dim]")
        if keys.shape[1:] != (
            config.num_selected_layers,
            config.num_key_value_heads,
            keys.shape[-2],
            config.head_dim,
        ):
            raise ValueError("Target K/V does not match the Global16 configuration")
        anchors = int(source_indices.numel())
        if anchors <= 0 or source_indices.ndim != 1:
            raise ValueError("source_indices must be a non-empty rank-1 tensor")
        if anchor_attention_mask.shape != (anchors, keys.shape[-2]):
            raise ValueError("anchor_attention_mask must be [anchors, target_tokens]")
        if anchor_attention_mask.dtype != torch.bool:
            raise ValueError("anchor_attention_mask must use dtype torch.bool")
        if torch.any(source_indices < 0) or torch.any(source_indices >= keys.shape[0]):
            raise ValueError("source_indices contains an invalid Target-K/V source")
        if not anchor_attention_mask.any(dim=-1).all():
            raise ValueError("every Global16 training anchor needs a visible Target prefix")

        self.reference = reference
        self.keys = keys
        self.values = values
        self.anchor_attention_mask = anchor_attention_mask
        self.source_indices = source_indices.to(device=keys.device, dtype=torch.long)
        self.global_queries = reference.global_queries(
            batch_size=anchors,
            device=keys.device,
            dtype=keys.dtype,
        )
        self.memory_state = self._build_full_prefix_state()
        self.evidence: list[Global16Raw256TrainingReadEvidence] = []

    @property
    def batch_size(self) -> int:
        return int(self.source_indices.numel())

    def _source_rows(self):
        for source_index in self.source_indices.unique(sorted=True).tolist():
            rows = torch.nonzero(
                self.source_indices == int(source_index), as_tuple=False
            ).flatten()
            yield int(source_index), rows

    def _expanded_source_kv(
        self,
        *,
        source_index: int,
        count: int,
        layer_index: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Slice + expand only changes tensor metadata.  It preserves the source
        # Target K/V storage pointer for every anchor in this source group.
        if layer_index is None:
            source_keys = self.keys[source_index : source_index + 1]
            source_values = self.values[source_index : source_index + 1]
        else:
            source_keys = self.keys[source_index : source_index + 1, layer_index]
            source_values = self.values[source_index : source_index + 1, layer_index]
        return (
            source_keys.expand(count, *source_keys.shape[1:]),
            source_values.expand(count, *source_values.shape[1:]),
        )

    def _build_full_prefix_state(self) -> Global16MemoryState:
        first = self.reference.config
        accumulator_dtype = (
            torch.float64
            if self.global_queries.dtype == torch.float64
            else torch.float32
        )
        state_max = self.global_queries.new_empty(
            (self.batch_size, first.num_selected_layers, first.num_attention_heads, 16),
            dtype=accumulator_dtype,
        )
        state_normalizer = torch.empty_like(state_max)
        state_weighted_values = self.global_queries.new_empty(
            (
                self.batch_size,
                first.num_selected_layers,
                first.num_attention_heads,
                16,
                first.head_dim,
            ),
            dtype=accumulator_dtype,
        )
        for source_index, rows in self._source_rows():
            for row_chunk in rows.split(_FULL_PREFIX_STATE_ANCHOR_CHUNK_SIZE):
                group_keys, group_values = self._expanded_source_kv(
                    source_index=source_index,
                    count=row_chunk.numel(),
                )
                group_query = self.global_queries.index_select(0, row_chunk)
                group_prefix_mask = self.anchor_attention_mask.index_select(
                    0, row_chunk
                )
                key_chunk_size = (
                    _FULL_PREFIX_STATE_KEY_CHUNK_SIZE
                    if group_keys.shape[-2] > _FULL_PREFIX_STATE_DIRECT_MAX_TOKENS
                    else None
                )
                if key_chunk_size is None or not (
                    torch.is_grad_enabled() and group_query.requires_grad
                ):
                    group_state = self.reference.global_state_from_full_prefix(
                        query=group_query,
                        target_keys=group_keys,
                        target_values=group_values,
                        prefix_mask=group_prefix_mask,
                        key_chunk_size=key_chunk_size,
                    )
                else:
                    def full_prefix_state(
                        query: torch.Tensor,
                        keys: torch.Tensor,
                        values: torch.Tensor,
                        prefix_mask: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        state = self.reference.global_state_from_full_prefix(
                            query=query,
                            target_keys=keys,
                            target_values=values,
                            prefix_mask=prefix_mask,
                            key_chunk_size=key_chunk_size,
                        )
                        return (
                            state.max_logits,
                            state.normalizer,
                            state.weighted_values,
                        )

                    chunk_max_logits, state_normalizer_chunk, state_weighted_values_chunk = checkpoint(
                        full_prefix_state,
                        group_query,
                        group_keys,
                        group_values,
                        group_prefix_mask,
                        use_reentrant=False,
                    )
                    group_state = Global16MemoryState(
                        query=group_query,
                        max_logits=chunk_max_logits,
                        normalizer=state_normalizer_chunk,
                        weighted_values=state_weighted_values_chunk,
                    )
                state_max = state_max.index_copy(
                    0, row_chunk, group_state.max_logits
                )
                state_normalizer = state_normalizer.index_copy(
                    0, row_chunk, group_state.normalizer
                )
                state_weighted_values = state_weighted_values.index_copy(
                    0, row_chunk, group_state.weighted_values
                )
        return Global16MemoryState(
            query=self.global_queries,
            max_logits=state_max,
            normalizer=state_normalizer,
            weighted_values=state_weighted_values,
        )

    def attend(
        self,
        *,
        layer_index: int,
        query: torch.Tensor,
        local_keys: torch.Tensor,
        local_values: torch.Tensor,
        local_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend separately to Global16, raw Target K/V, and Local7."""

        config = self.reference.config
        if query.shape[:2] != (self.batch_size, config.num_attention_heads):
            raise ValueError("draft query does not match the Global16 training context")
        if query.shape[-1] != config.head_dim:
            raise ValueError("draft query head dimension does not match Global16")
        if local_keys.shape != local_values.shape or local_keys.ndim != 4:
            raise ValueError("Local7 K/V must be matching rank-4 tensors")
        if local_keys.shape[0] != self.batch_size or local_keys.shape[-2] > config.local_draft_rows:
            raise ValueError("Local7 must contain at most seven rows per draft query block")
        if local_attention_mask.shape != (
            self.batch_size,
            query.shape[-2],
            local_keys.shape[-2],
        ):
            raise ValueError("Local7 attention mask has an invalid shape")
        if layer_index < 0 or layer_index >= config.num_selected_layers:
            raise ValueError("invalid selected Target layer index")

        output = torch.empty_like(query)
        for source_index, rows in self._source_rows():
            group_query = query.index_select(0, rows)
            group_keys, group_values = self._expanded_source_kv(
                source_index=source_index,
                count=rows.numel(),
                layer_index=layer_index,
            )
            group_prefix_mask = self.anchor_attention_mask.index_select(0, rows)
            lengths = group_prefix_mask.sum(dim=-1, dtype=torch.long)
            token_positions = torch.arange(
                group_prefix_mask.shape[-1], device=group_prefix_mask.device
            )
            raw_mask = group_prefix_mask & token_positions[None, :].ge(
                (lengths - config.raw_target_rows).clamp_min(0)[:, None]
            )
            group_global_query = self.memory_state.query.index_select(0, rows)[:, layer_index]
            group_global_values = self.memory_state.values().index_select(0, rows)[:, layer_index]
            global_statistics = attention_statistics_from_4d(
                query=group_query,
                keys=group_global_query,
                values=group_global_values.to(dtype=group_global_query.dtype),
                attention_mask=torch.ones(
                    (
                        rows.numel(),
                        group_query.shape[-2],
                        config.global_memory_slots,
                    ),
                    dtype=torch.bool,
                    device=group_query.device,
                ),
            )
            raw_statistics = attention_statistics_from_4d(
                query=group_query,
                keys=group_keys,
                values=group_values,
                attention_mask=raw_mask[:, None, :].expand(
                    -1, group_query.shape[-2], -1
                ),
            )
            local_statistics = attention_statistics_from_4d(
                query=group_query,
                keys=local_keys.index_select(0, rows),
                values=local_values.index_select(0, rows),
                attention_mask=local_attention_mask.index_select(0, rows),
            )
            global_output, global_lse = _statistics_as_normalized(global_statistics)
            raw_output, raw_lse = _statistics_as_normalized(raw_statistics)
            local_output, local_lse = _statistics_as_normalized(local_statistics)
            merged_output, _ = merge_normalized_attention(
                (
                    (global_output, global_lse),
                    (raw_output, raw_lse),
                    (local_output, local_lse),
                )
            )
            output = output.index_copy(0, rows, merged_output.to(dtype=query.dtype))
            self.evidence.append(
                Global16Raw256TrainingReadEvidence(
                    source_index=source_index,
                    key_storage_ptr=_storage_ptr(self.keys[source_index]),
                    value_storage_ptr=_storage_ptr(self.values[source_index]),
                    raw_rows=int(raw_mask.sum(dim=-1).max().item()),
                )
            )
        return output


__all__ = [
    "DenseGlobal16Raw256DraftContext",
    "Global16Raw256TrainingReadEvidence",
]
