import logging
import math
import os
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel

from specforge.training_attribution import profile_range

from .draftfreekv_validation import (
    PARALLEL_SHIFTED_DRAFT_MODES,
    normalize_current_hidden_states,
    normalize_kv_attention_mask,
    pop_transformers_metadata,
    reject_unknown_config_fields,
    restrict_target_kv_attention_to_window,
    resolve_draftfreekv_config_fields,
    resolve_target_layer_ids,
    select_latest_visible_target_kv,
)
from .global16_raw256 import (
    Global16Raw256Config,
    Global16Raw256Reference,
    _fixed_equicorrelated_query_pattern,
)
from .global16_raw256_training import DenseGlobal16Raw256DraftContext


_COMPILED_FLEX_ATTENTION = None
logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value (0/1, false/true, no/yes, or off/on)"
    )


@torch.compiler.disable(recursive=False)
def _get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


def _run_packed_flex_attention(*args, **kwargs):
    attention_fn = (
        flex_attention
        if torch.compiler.is_compiling()
        else _get_compiled_flex_attention()
    )
    return attention_fn(*args, **kwargs)


def reference_deduplicated_prefix_attention(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    source_indices: torch.Tensor,
    seed_indices: torch.Tensor,
    anchor_attention_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """FP32 reference for sharing QK rows identified by (source, seed position).

    This is an exactness/reference path, not the production accelerated backend.
    It accepts arbitrary per-anchor visibility masks, including internal holes.
    """

    anchors, heads, window, head_dim = query.shape
    if keys.ndim != 4 or keys.shape[:2] != (values.shape[0], heads):
        raise ValueError("keys and values must have shape [sources, heads, tokens, dim]")
    if keys.shape[:3] != values.shape[:3] or keys.shape[-1] != head_dim:
        raise ValueError("keys and values must share source/head/token dimensions")
    tokens = keys.shape[2]
    if source_indices.shape != (anchors,) or seed_indices.shape != (anchors, window):
        raise ValueError("source_indices or seed_indices has an invalid shape")
    if anchor_attention_mask.shape != (anchors, tokens):
        raise ValueError("anchor_attention_mask has an invalid shape")

    pair_ids = torch.stack(
        [source_indices[:, None].expand(-1, window), seed_indices], dim=-1
    ).reshape(-1, 2)
    unique_pairs, inverse = torch.unique(
        pair_ids, dim=0, sorted=True, return_inverse=True
    )
    flat_query = query.transpose(1, 2).reshape(anchors * window, heads, head_dim)
    representative_indices = torch.stack(
        [
            torch.nonzero(inverse == index, as_tuple=False)[0, 0]
            for index in range(unique_pairs.shape[0])
        ]
    )
    representative_query = flat_query.index_select(0, representative_indices)
    reconstructed_query = representative_query.index_select(0, inverse)
    if not torch.equal(reconstructed_query.detach(), flat_query.detach()):
        raise ValueError(
            "queries sharing a (source, seed position) pair must be exactly equal"
        )

    fp32_query = representative_query.float()
    fp32_keys = keys.index_select(0, unique_pairs[:, 0]).float()
    unique_logits = torch.einsum("uhd,uhtd->uht", fp32_query, fp32_keys) * scale
    deduplicated_logits = unique_logits.index_select(0, inverse).view(
        anchors, window, heads, tokens
    ).permute(0, 2, 1, 3)

    # Preserve the expanded SDPA gradient contract while making the reference
    # forward explicitly use the deduplicated (source, seed position) rows.
    selected_keys = keys.index_select(0, source_indices).float()
    expanded_logits = (
        torch.einsum("ahwd,ahtd->ahwt", query.float(), selected_keys) * scale
    )
    logits = (
        deduplicated_logits.detach()
        + expanded_logits
        - expanded_logits.detach()
    )
    logits = logits.masked_fill(
        ~anchor_attention_mask[:, None, None, :].bool(), float("-inf")
    )
    weights = torch.softmax(logits, dim=-1)
    selected_values = values.index_select(0, source_indices).float()
    return torch.einsum("ahwt,ahtd->ahwd", weights, selected_values)


def _validate_nested_prefix_masks(
    anchor_attention_mask: torch.Tensor,
    source_indices: torch.Tensor,
    num_sources: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-valid masks and physical endpoints for nested prefixes."""

    mask = anchor_attention_mask.bool()
    anchors, tokens = mask.shape
    source_valid = torch.zeros(
        num_sources, tokens, dtype=torch.bool, device=mask.device
    )
    endpoints = torch.empty(anchors, dtype=torch.long, device=mask.device)
    positions = torch.arange(tokens, device=mask.device)
    for source in source_indices.unique(sorted=True):
        rows = torch.nonzero(source_indices == source, as_tuple=False).flatten()
        source_rows = mask.index_select(0, rows)
        valid = source_rows.any(dim=0)
        source_valid[source] = valid
        valid_positions = positions[valid]
        counts = source_rows.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("prefix_scan requires every anchor to see at least one token")
        valid_ranks = valid.cumsum(dim=0)
        expected = valid[None] & (valid_ranks[None] <= counts[:, None])
        if not torch.equal(source_rows, expected):
            raise ValueError(
                "prefix_scan requires masks for each source to be nested prefixes "
                "of one shared source-valid token sequence"
            )
        endpoints.index_copy_(0, rows, valid_positions.index_select(0, counts - 1))
    return source_valid, endpoints


def _deduplicate_query_rows(
    query: torch.Tensor,
    source_indices: torch.Tensor,
    seed_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors, heads, window, head_dim = query.shape
    pair_ids = torch.stack(
        [source_indices[:, None].expand(-1, window), seed_indices], dim=-1
    ).reshape(-1, 2)
    unique_pairs, inverse = torch.unique(
        pair_ids, dim=0, sorted=True, return_inverse=True
    )
    flat_query = query.transpose(1, 2).reshape(-1, heads, head_dim)
    # unique() sorted the pairs, so amin gives a deterministic representative.
    representative_indices = torch.full(
        (unique_pairs.shape[0],),
        flat_query.shape[0],
        dtype=torch.long,
        device=query.device,
    )
    representative_indices.scatter_reduce_(
        0,
        inverse,
        torch.arange(flat_query.shape[0], device=query.device),
        reduce="amin",
        include_self=True,
    )
    representative_query = flat_query.index_select(0, representative_indices)
    if not torch.equal(
        representative_query.index_select(0, inverse).detach(), flat_query.detach()
    ):
        raise ValueError(
            "queries sharing a (source, seed position) pair must be exactly equal"
        )
    return unique_pairs, inverse, representative_indices, representative_query


class _PrefixScanAttention(torch.autograd.Function):
    """Exact nested-prefix attention with query-only custom backward."""

    @staticmethod
    def forward(
        ctx,
        query,
        keys,
        values,
        source_indices,
        seed_indices,
        anchor_attention_mask,
        scale,
        query_chunk_size,
        token_chunk_size,
    ):
        anchors, heads, window, head_dim = query.shape
        source_valid, anchor_endpoints = _validate_nested_prefix_masks(
            anchor_attention_mask, source_indices, keys.shape[0]
        )
        unique_pairs, inverse, representatives, unique_query = _deduplicate_query_rows(
            query, source_indices, seed_indices
        )
        expanded_endpoints = anchor_endpoints[:, None].expand(-1, window).reshape(-1)
        flat_output = torch.empty(
            anchors * window,
            heads,
            values.shape[-1],
            device=query.device,
            dtype=torch.float32,
        )
        for begin in range(0, unique_query.shape[0], query_chunk_size):
            end = min(unique_query.shape[0], begin + query_chunk_size)
            chunk_sources = unique_pairs[begin:end, 0]
            chunk_keys = keys.index_select(0, chunk_sources).float()
            chunk_values = values.index_select(0, chunk_sources).float()
            valid = source_valid.index_select(0, chunk_sources)
            scores = torch.einsum(
                "qhd,qhtd->qht", unique_query[begin:end].float(), chunk_keys
            ).mul_(scale)
            scores.masked_fill_(~valid[:, None], float("-inf"))
            global_max = scores.amax(dim=-1, keepdim=True)
            weights = torch.exp(scores - global_max)
            denominator = weights.cumsum(dim=-1)
            numerator = (weights[..., None] * chunk_values).cumsum(dim=2)

            occurrence_rows = torch.nonzero(
                (inverse >= begin) & (inverse < end), as_tuple=False
            ).flatten()
            local_queries = inverse.index_select(0, occurrence_rows) - begin
            occurrence_endpoints = expanded_endpoints.index_select(0, occurrence_rows)
            selected_denominator = denominator[
                local_queries, :, occurrence_endpoints
            ]
            selected_numerator = numerator[
                local_queries, :, occurrence_endpoints
            ]
            flat_output.index_copy_(
                0,
                occurrence_rows,
                selected_numerator / selected_denominator[..., None],
            )
        ctx.save_for_backward(
            query,
            keys,
            values,
            unique_pairs,
            inverse,
            representatives,
            source_valid,
            expanded_endpoints,
        )
        ctx.scale = scale
        ctx.query_chunk_size = query_chunk_size
        ctx.token_chunk_size = token_chunk_size
        return flat_output.view(anchors, window, heads, -1).transpose(1, 2).to(query.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            keys,
            values,
            unique_pairs,
            inverse,
            representatives,
            source_valid,
            expanded_endpoints,
        ) = ctx.saved_tensors
        anchors, heads, window, head_dim = query.shape
        flat_grad = grad_output.transpose(1, 2).reshape(-1, heads, values.shape[-1]).float()
        unique_count = unique_pairs.shape[0]
        grad_unique = torch.zeros(
            unique_count, heads, head_dim, device=query.device, dtype=torch.float32
        )
        for begin in range(0, unique_count, ctx.query_chunk_size):
            end = min(unique_count, begin + ctx.query_chunk_size)
            chunk_count = end - begin
            chunk_query = query.transpose(1, 2).reshape(-1, heads, head_dim).index_select(
                0, representatives[begin:end]
            ).float()
            chunk_sources = unique_pairs[begin:end, 0]
            chunk_keys = keys.index_select(0, chunk_sources).float()
            chunk_values = values.index_select(0, chunk_sources).float()
            valid = source_valid.index_select(0, chunk_sources)
            scores = torch.einsum(
                "qhd,qhtd->qht", chunk_query, chunk_keys
            ).mul_(ctx.scale)
            scores.masked_fill_(~valid[:, None], float("-inf"))
            global_max = scores.amax(dim=-1, keepdim=True)
            weights = torch.exp(scores - global_max)
            denominator = weights.cumsum(dim=-1)
            numerator = (weights[..., None] * chunk_values).cumsum(dim=2)

            occurrence_rows = torch.nonzero(
                (inverse >= begin) & (inverse < end), as_tuple=False
            ).flatten()
            local_queries = inverse.index_select(0, occurrence_rows) - begin
            occurrence_endpoints = expanded_endpoints.index_select(0, occurrence_rows)
            occurrence_denominator = denominator[
                local_queries, :, occurrence_endpoints
            ]
            occurrence_output = numerator[
                local_queries, :, occurrence_endpoints
            ] / occurrence_denominator[..., None]
            occurrence_grad = flat_grad.index_select(0, occurrence_rows)
            del numerator

            tokens = keys.shape[2]
            scatter_indices = local_queries * tokens + occurrence_endpoints
            endpoint_g = torch.zeros(
                chunk_count * tokens,
                heads,
                values.shape[-1],
                device=query.device,
                dtype=torch.float32,
            )
            endpoint_g.index_add_(
                0,
                scatter_indices,
                occurrence_grad / occurrence_denominator[..., None],
            )
            endpoint_b = torch.zeros(
                chunk_count * tokens,
                heads,
                device=query.device,
                dtype=torch.float32,
            )
            endpoint_b.index_add_(
                0,
                scatter_indices,
                (occurrence_grad * occurrence_output).sum(dim=-1)
                / occurrence_denominator,
            )
            endpoint_g = endpoint_g.view(
                chunk_count, tokens, heads, values.shape[-1]
            )
            endpoint_b = endpoint_b.view(chunk_count, tokens, heads)
            reverse_g = endpoint_g.flip(1).cumsum(dim=1).flip(1).permute(0, 2, 1, 3)
            reverse_b = endpoint_b.flip(1).cumsum(dim=1).flip(1).permute(0, 2, 1)
            coefficient = weights * (
                torch.einsum("qhtd,qhtd->qht", chunk_values, reverse_g)
                - reverse_b
            )
            chunk_grad = ctx.scale * torch.einsum(
                "qht,qhtd->qhd", coefficient, chunk_keys
            )
            grad_unique[begin:end] = chunk_grad

        flat_query_grad = torch.zeros(
            anchors * window, heads, head_dim, device=query.device, dtype=torch.float32
        )
        # One representative receives the sum. Since duplicate rows have the
        # same query graph, this is parameter-gradient equivalent to expanded SDPA.
        flat_query_grad.index_copy_(0, representatives, grad_unique)
        query_grad = flat_query_grad.view(
            anchors, window, heads, head_dim
        ).transpose(1, 2)
        return query_grad.to(query.dtype), None, None, None, None, None, None, None, None


def prefix_scan_attention(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    source_indices: torch.Tensor,
    seed_indices: torch.Tensor,
    anchor_attention_mask: torch.Tensor,
    scale: float,
    query_chunk_size: int = 16,
    token_chunk_size: int = 2048,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Experimental exact backend for nested full-history anchor prefixes."""

    output = _PrefixScanAttention.apply(
        query,
        keys.detach(),
        values.detach(),
        source_indices,
        seed_indices,
        anchor_attention_mask,
        scale,
        query_chunk_size,
        token_chunk_size,
    )
    unique_pairs = torch.unique(
        torch.stack(
            [
                source_indices[:, None].expand_as(seed_indices),
                seed_indices,
            ],
            dim=-1,
        ).reshape(-1, 2),
        dim=0,
    )
    return output, {
        "expanded_query_rows": int(query.shape[0] * query.shape[2]),
        "unique_qk_rows": int(unique_pairs.shape[0]),
        "query_chunk_size": int(query_chunk_size),
        "token_chunk_size": int(token_chunk_size),
    }


class AttentionGeneratedTargetMemory(nn.Module):
    """Build fixed-W per-layer K/V using one Target-attention map per row."""

    def __init__(
        self,
        *,
        num_layers: int,
        hidden_size: int,
        num_key_value_heads: int,
        head_dim: int,
        rank: int,
        residual_rank: Optional[int] = None,
        output_slots: Optional[int] = None,
        query_mode: str = "hidden_r16",
        external_key_mode: str = "akav",
        alpha_init: float = 0.75,
        rope_theta: float = 1_000_000.0,
        anchor_policy: str = "recent",
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.residual_rank = None if residual_rank is None else int(residual_rank)
        self.output_slots = None if output_slots is None else int(output_slots)
        self.query_mode = str(query_mode)
        self.external_key_mode = str(external_key_mode)
        self.anchor_policy = str(anchor_policy)
        self._fold_hybrid_aggregate_sdpa = _env_flag(
            "DFK_FOLD_HYBRID_AGGREGATE_SDPA",
            default=True,
        )
        self._serving_fastpath_trace = _env_flag("DFK_SERVING_FASTPATH_TRACE")
        self._fold_trace_emitted = False
        vectorized_policy_selection = os.environ.get(
            "DFK_VECTORIZED_POLICY_SELECTION", "1"
        )
        if vectorized_policy_selection not in {"0", "1"}:
            raise ValueError(
                "DFK_VECTORIZED_POLICY_SELECTION must be exactly 0 or 1"
            )
        self.vectorized_policy_selection_enabled = (
            vectorized_policy_selection == "1"
        )
        if self.query_mode != "seed_rope_latest":
            self.log_alpha = nn.Parameter(
                torch.full((num_layers,), math.log(float(alpha_init)))
            )
        if self.query_mode == "hidden_r16":
            self.down = nn.ModuleList(
                nn.Linear(hidden_size, rank, bias=False) for _ in range(num_layers)
            )
            self.up = nn.ModuleList(
                nn.Linear(rank, num_key_value_heads * head_dim, bias=False)
                for _ in range(num_layers)
            )
            for up in self.up:
                nn.init.zeros_(up.weight)
        elif self.query_mode in {"seed_r4", "seed_residual"}:
            seed_rank = 4 if self.query_mode == "seed_r4" else self.residual_rank
            if seed_rank is None or seed_rank <= 0:
                raise ValueError("seed_residual requires a positive residual_rank")
            self.seed_down = nn.ModuleList(
                nn.Linear(head_dim, seed_rank, bias=False) for _ in range(num_layers)
            )
            self.seed_up = nn.ModuleList(
                nn.Linear(seed_rank, head_dim, bias=False) for _ in range(num_layers)
            )
            for up in self.seed_up:
                nn.init.zeros_(up.weight)
        elif self.query_mode == "seed_rope_latest":
            if self.head_dim != 128 or float(rope_theta) != 1_000_000.0:
                raise ValueError(
                    "seed_rope_latest requires head_dim=128 and rope_theta=1000000"
                )
            inv_freq = 1.0 / (
                float(rope_theta)
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                    / self.head_dim
                )
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def reset_zero_output(self) -> None:
        for up in getattr(self, "up", ()):
            nn.init.zeros_(up.weight)
        for up in getattr(self, "seed_up", ()):
            nn.init.zeros_(up.weight)

    @staticmethod
    def last_visible_indices(
        attention_mask: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens = attention_mask.shape
        positions = torch.arange(tokens, device=attention_mask.device)[None].expand(
            batch, -1
        )
        sentinel = torch.full_like(positions, tokens)
        ordered = torch.where(attention_mask, positions, sentinel).sort(dim=1).values
        counts = attention_mask.sum(dim=1)
        ranks = counts[:, None] - window_size + torch.arange(
            window_size, device=attention_mask.device
        )[None]
        valid = ranks >= 0
        ranks = ranks.clamp(min=0, max=tokens - 1)
        indices = ordered.gather(1, ranks)
        indices = torch.where(valid, indices, ordered[:, :1].expand_as(indices))
        return indices, valid

    @staticmethod
    def _nearest_unselected(
        target: int,
        historical_count: int,
        selected: set[int],
    ) -> Optional[int]:
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

    @classmethod
    def _policy_rank_selection(
        cls,
        total: int,
        policy: str,
    ) -> list[tuple[int, bool]]:
        recent_count = 128 if policy == "strat128_64_64" else 192
        log_count = 64 if policy == "strat128_64_64" else 32
        segment_count = log_count
        historical_count = max(0, total - recent_count)
        recent_ranks = list(range(historical_count, total))
        selected_historical: set[int] = set()

        if historical_count:
            minimum_distance = recent_count + 1
            maximum_distance = total
            if log_count == 1 or maximum_distance <= minimum_distance:
                distances = [maximum_distance] * log_count
            else:
                ratio = maximum_distance / minimum_distance
                distances = [
                    round(minimum_distance * ratio ** (slot / (log_count - 1)))
                    for slot in range(log_count)
                ]
            for distance in distances:
                target = max(0, min(historical_count - 1, total - int(distance)))
                chosen = cls._nearest_unselected(
                    target, historical_count, selected_historical
                )
                if chosen is not None:
                    selected_historical.add(chosen)

        for slot in range(segment_count):
            if len(selected_historical) >= historical_count:
                break
            target = min(
                historical_count - 1,
                int((slot + 0.5) * historical_count / segment_count),
            )
            chosen = cls._nearest_unselected(
                target, historical_count, selected_historical
            )
            if chosen is not None:
                selected_historical.add(chosen)

        selected = [(rank, False) for rank in selected_historical]
        selected.extend(
            (rank, policy == "hybrid_raw192_agg64") for rank in recent_ranks
        )
        return sorted(selected, key=lambda item: item[0])

    @classmethod
    def policy_visible_indices(
        cls,
        attention_mask: torch.Tensor,
        policy: str,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select distinct visible positions and mark rows copied as raw K/V."""
        if policy == "recent":
            indices, valid = cls.last_visible_indices(attention_mask, window_size)
            return indices, valid, torch.zeros_like(valid)
        if policy not in {"strat128_64_64", "hybrid_raw192_agg64"}:
            raise ValueError("unsupported memory anchor policy")
        if window_size != 256:
            raise ValueError(f"{policy} requires exactly 256 memory rows")

        index_rows = []
        valid_rows = []
        raw_rows = []
        for row_mask in attention_mask:
            visible = torch.nonzero(row_mask, as_tuple=False).flatten()
            total = int(visible.numel())
            selected = cls._policy_rank_selection(total, policy)
            padding = window_size - len(selected)
            fallback = visible[:1] if total else torch.zeros(1, device=row_mask.device, dtype=torch.long)
            rank_tensor = torch.tensor(
                [rank for rank, _ in selected],
                device=row_mask.device,
                dtype=torch.long,
            )
            selected_positions = visible.index_select(0, rank_tensor)
            index_rows.append(
                torch.cat((fallback.expand(padding), selected_positions))
            )
            valid_rows.append(
                torch.cat(
                    (
                        torch.zeros(padding, device=row_mask.device, dtype=torch.bool),
                        torch.ones(len(selected), device=row_mask.device, dtype=torch.bool),
                    )
                )
            )
            raw_rows.append(
                torch.cat(
                    (
                        torch.zeros(padding, device=row_mask.device, dtype=torch.bool),
                        torch.tensor(
                            [is_raw for _, is_raw in selected],
                            device=row_mask.device,
                            dtype=torch.bool,
                        ),
                    )
                )
            )
        return (
            torch.stack(index_rows),
            torch.stack(valid_rows),
            torch.stack(raw_rows),
        )

    @classmethod
    def vectorized_policy_visible_indices(
        cls,
        attention_mask: torch.Tensor,
        policy: str,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch visible-position mapping with one count synchronization."""

        if policy not in {"strat128_64_64", "hybrid_raw192_agg64"}:
            raise ValueError("vectorized selection requires a stratified policy")
        if window_size != 256:
            raise ValueError(f"{policy} requires exactly 256 memory rows")

        batch, tokens = attention_mask.shape
        visible_mask = attention_mask.bool()
        positions = torch.arange(tokens, device=attention_mask.device)[None].expand(
            batch, -1
        )
        ordered = torch.where(
            visible_mask,
            positions,
            torch.full_like(positions, tokens),
        ).sort(dim=1).values
        counts = visible_mask.sum(dim=1, dtype=torch.long)
        count_rows = counts.detach().cpu().tolist()

        plans_by_count: dict[int, tuple[list[int], list[bool], list[bool]]] = {}
        rank_rows = []
        valid_rows = []
        raw_rows = []
        for total in count_rows:
            total = int(total)
            plan = plans_by_count.get(total)
            if plan is None:
                selected = cls._policy_rank_selection(total, policy)
                padding = window_size - len(selected)
                plan = (
                    [0] * padding + [rank for rank, _ in selected],
                    [False] * padding + [True] * len(selected),
                    [False] * padding + [is_raw for _, is_raw in selected],
                )
                plans_by_count[total] = plan
            ranks, valid, raw = plan
            rank_rows.append(ranks)
            valid_rows.append(valid)
            raw_rows.append(raw)

        rank_plan = torch.tensor(rank_rows, dtype=torch.long, device="cpu").to(
            device=attention_mask.device
        )
        valid = torch.tensor(valid_rows, dtype=torch.bool, device="cpu").to(
            device=attention_mask.device
        )
        raw = torch.tensor(raw_rows, dtype=torch.bool, device="cpu").to(
            device=attention_mask.device
        )
        if tokens:
            selected_positions = ordered.gather(1, rank_plan)
            fallback = torch.where(
                counts[:, None] > 0,
                ordered[:, :1],
                torch.zeros((batch, 1), dtype=torch.long, device=attention_mask.device),
            )
            indices = torch.where(valid, selected_positions, fallback)
        else:
            indices = torch.zeros(
                (batch, window_size),
                dtype=torch.long,
                device=attention_mask.device,
            )
        return indices, valid, raw

    @property
    def policy_selection_implementation(self) -> str:
        if (
            self.vectorized_policy_selection_enabled
            and self.anchor_policy in {"strat128_64_64", "hybrid_raw192_agg64"}
        ):
            return "vectorized"
        return "legacy"

    def project_hidden_rows(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        if hidden_rows.ndim != 4 or hidden_rows.shape[2:] != (
            self.num_layers,
            self.hidden_size,
        ):
            raise ValueError(
                "memory_hidden_states must have shape "
                "[anchors, window, layers, hidden_size]"
            )
        anchors, window = hidden_rows.shape[:2]
        refinements = []
        for layer_index in range(self.num_layers):
            hidden = hidden_rows[:, :, layer_index].float()
            hidden = hidden * torch.rsqrt(
                hidden.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            latent = F.silu(self.down[layer_index](hidden.to(hidden_rows.dtype)))
            refinements.append(
                self.up[layer_index](latent).view(
                    anchors,
                    window,
                    self.num_key_value_heads,
                    self.head_dim,
                )
            )
        return torch.stack(refinements, dim=2)

    @staticmethod
    def parameter_free_rms_norm(hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.to(dtype=hidden_states.dtype)

    def rotate_seed_keys_to_latest(self, seeds: torch.Tensor) -> torch.Tensor:
        """Move post-RoPE seed K from its visible rank to the latest rank."""
        window = seeds.shape[-2]
        relative_positions = torch.arange(
            window - 1,
            -1,
            -1,
            device=seeds.device,
            dtype=torch.float32,
        )
        frequencies = relative_positions[:, None] * self.rope_inv_freq[None, :]
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().to(dtype=seeds.dtype)[None, None]
        sine = embedding.sin().to(dtype=seeds.dtype)[None, None]
        return seeds * cosine + _rotate_half(seeds) * sine

    def _should_fold_hybrid_aggregate_sdpa(
        self,
        *,
        independent_sources: bool,
        attention_backend: str,
    ) -> bool:
        return (
            self._fold_hybrid_aggregate_sdpa
            and not self.training
            and independent_sources
            and self.anchor_policy == "hybrid_raw192_agg64"
            and attention_backend == "sdpa"
        )

    def _forward_folded_hybrid_aggregate_sdpa(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        source_indices: torch.Tensor,
        indices: torch.Tensor,
        expected_mask: torch.Tensor,
        raw_row_mask: torch.Tensor,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fold independent Hybrid aggregate rows across selected Target layers."""

        anchors, tokens = anchor_attention_mask.shape
        aggregate_rows = 64
        if window < aggregate_rows:
            raise ValueError("hybrid_raw192_agg64 requires at least 64 memory rows")
        hybrid_aggregate_rows = (
            torch.arange(window, device=keys.device)[None] < aggregate_rows
        ).expand(anchors, -1)
        layer_queries = []
        layer_seeds = []
        layer_value_seeds = []
        for layer_index in range(self.num_layers):
            layer_keys = keys[:, layer_index]
            layer_values = values[:, layer_index]
            seeds = layer_keys.permute(0, 2, 1, 3)[
                source_indices[:, None], indices
            ].transpose(1, 2)
            value_seeds = layer_values.permute(0, 2, 1, 3)[
                source_indices[:, None], indices
            ].transpose(1, 2)
            if self.query_mode == "hidden_r16":
                raise ValueError("hybrid_raw192_agg64 does not support hidden_r16")
            if self.query_mode in {"seed_r4", "seed_residual"}:
                selected_seeds = seeds.permute(0, 2, 1, 3)[
                    hybrid_aggregate_rows
                ]
                normalized_seed = self.parameter_free_rms_norm(selected_seeds)
                selected_delta = self.seed_up[layer_index](
                    F.silu(self.seed_down[layer_index](normalized_seed))
                )
                delta = torch.zeros_like(seeds)
                delta.permute(0, 2, 1, 3)[
                    hybrid_aggregate_rows
                ] = selected_delta
            elif self.query_mode == "seed_rope_latest":
                delta = None
            else:
                delta = 0.0
            if self.query_mode == "seed_rope_latest":
                query = self.rotate_seed_keys_to_latest(seeds)
            else:
                query = (
                    self.log_alpha[layer_index].exp().to(layer_keys.dtype) * seeds
                    + delta
                )
            layer_queries.append(query)
            layer_seeds.append(seeds)
            layer_value_seeds.append(value_seeds)

        output_dim = (
            2 * self.head_dim
            if self.external_key_mode == "akav"
            else self.head_dim
        )
        folded_values = (
            torch.cat([keys, values], dim=-1)
            if self.external_key_mode == "akav"
            else values
        )
        folded_query = torch.stack(layer_queries, dim=1)[..., :aggregate_rows, :]
        folded_output = F.scaled_dot_product_attention(
            folded_query.reshape(
                anchors * self.num_layers,
                self.num_key_value_heads,
                aggregate_rows,
                self.head_dim,
            ),
            keys.reshape(
                anchors * self.num_layers,
                self.num_key_value_heads,
                tokens,
                self.head_dim,
            ),
            folded_values.reshape(
                anchors * self.num_layers,
                self.num_key_value_heads,
                tokens,
                output_dim,
            ),
            attn_mask=anchor_attention_mask[:, None, :]
            .expand(-1, self.num_layers, -1)
            .reshape(anchors * self.num_layers, 1, 1, tokens),
            dropout_p=0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        ).reshape(
            anchors,
            self.num_layers,
            self.num_key_value_heads,
            aggregate_rows,
            output_dim,
        )

        memory_keys = []
        memory_values = []
        valid = expected_mask[:, None, :, None]
        raw = raw_row_mask[:, None, :, None]
        for layer_index, (seeds, value_seeds) in enumerate(
            zip(layer_seeds, layer_value_seeds)
        ):
            layer_output = seeds.new_zeros(
                anchors,
                self.num_key_value_heads,
                window,
                output_dim,
            )
            layer_output[:, :, :aggregate_rows] = folded_output[:, layer_index]
            if self.external_key_mode == "akav":
                memory_key, memory_value = layer_output.split(self.head_dim, dim=-1)
            elif self.external_key_mode == "raw_seed":
                memory_key = seeds.masked_fill(~valid, 0)
                memory_value = torch.where(raw, value_seeds, layer_output)
                memory_value = memory_value.masked_fill(~valid, 0)
            else:
                memory_key = self.parameter_free_rms_norm(layer_queries[layer_index])
                memory_value = layer_output
            memory_keys.append(memory_key)
            memory_values.append(memory_value)

        if self._serving_fastpath_trace and not self._fold_trace_emitted:
            valid_aggregate_rows = int(
                (expected_mask & ~raw_row_mask).sum().detach().cpu().item()
            )
            logger.info(
                "DFK_SERVING_FASTPATH folded_hybrid_aggregate_sdpa hit "
                "aggregate_rows=%d bank_rows=%d source_tokens=%d "
                "valid_aggregate_rows=%d",
                aggregate_rows,
                window,
                tokens,
                valid_aggregate_rows,
            )
            self._fold_trace_emitted = True
        return (
            torch.stack(memory_keys, dim=1),
            torch.stack(memory_values, dim=1),
            expected_mask,
        )

    def forward(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        memory_hidden_states: Optional[torch.Tensor] = None,
        projected_refinements: Optional[torch.Tensor] = None,
        memory_query_mask: Optional[torch.Tensor] = None,
        source_indices: Optional[torch.Tensor] = None,
        attention_backend: str = "sdpa",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, tokens = anchor_attention_mask.shape
        independent_sources = source_indices is None
        if independent_sources:
            source_indices = torch.arange(anchors, device=keys.device)
        source_indices = source_indices.to(device=keys.device, dtype=torch.long)
        window = (
            self.output_slots
            if self.output_slots is not None
            else memory_query_mask.shape[1]
        )
        with profile_range("policy_selection"):
            if self.policy_selection_implementation == "vectorized":
                indices, expected_mask, raw_row_mask = (
                    self.vectorized_policy_visible_indices(
                        anchor_attention_mask,
                        self.anchor_policy,
                        window,
                    )
                )
            else:
                indices, expected_mask, raw_row_mask = self.policy_visible_indices(
                    anchor_attention_mask,
                    self.anchor_policy,
                    window,
                )
        hybrid_aggregate_rows = None
        if self.anchor_policy == "hybrid_raw192_agg64":
            hybrid_aggregate_rows = (
                torch.arange(window, device=keys.device)[None] < 64
            ).expand(anchors, -1)
        if memory_query_mask is not None and not torch.equal(
            expected_mask, memory_query_mask.bool()
        ):
            raise ValueError("memory hidden rows must align with the visible Target K window")
        if self.query_mode == "hidden_r16":
            if (memory_hidden_states is None) == (projected_refinements is None):
                raise ValueError(
                    "provide exactly one of memory_hidden_states or projected_refinements"
                )
            if projected_refinements is None:
                projected_refinements = self.project_hidden_rows(memory_hidden_states)
            expected_shape = (
                anchors,
                window,
                self.num_layers,
                self.num_key_value_heads,
                self.head_dim,
            )
            if projected_refinements.shape != expected_shape:
                raise ValueError("projected memory refinements have an invalid shape")
        elif memory_hidden_states is not None or projected_refinements is not None:
            raise ValueError(f"{self.query_mode} does not consume historical hidden rows")

        if self._should_fold_hybrid_aggregate_sdpa(
            independent_sources=independent_sources,
            attention_backend=attention_backend,
        ):
            return self._forward_folded_hybrid_aggregate_sdpa(
                keys=keys,
                values=values,
                anchor_attention_mask=anchor_attention_mask,
                source_indices=source_indices,
                indices=indices,
                expected_mask=expected_mask,
                raw_row_mask=raw_row_mask,
                window=window,
            )

        memory_keys = []
        memory_values = []
        for layer_index in range(self.num_layers):
            with profile_range("query_projection"):
                layer_keys = keys[:, layer_index]
                layer_values = values[:, layer_index]
                key_by_token = layer_keys.permute(0, 2, 1, 3)
                seeds = key_by_token[source_indices[:, None], indices].transpose(1, 2)
                value_by_token = layer_values.permute(0, 2, 1, 3)
                value_seeds = value_by_token[
                    source_indices[:, None], indices
                ].transpose(1, 2)
                if self.query_mode == "hidden_r16":
                    delta = projected_refinements[:, :, layer_index].transpose(1, 2).to(
                        dtype=layer_keys.dtype
                    )
                elif self.query_mode in {"seed_r4", "seed_residual"}:
                    if self.anchor_policy == "hybrid_raw192_agg64":
                        selected_seeds = seeds.permute(0, 2, 1, 3)[
                            hybrid_aggregate_rows
                        ]
                        normalized_seed = self.parameter_free_rms_norm(selected_seeds)
                        selected_delta = self.seed_up[layer_index](
                            F.silu(self.seed_down[layer_index](normalized_seed))
                        )
                        delta = torch.zeros_like(seeds)
                        delta.permute(0, 2, 1, 3)[
                            hybrid_aggregate_rows
                        ] = selected_delta
                    else:
                        normalized_seed = self.parameter_free_rms_norm(seeds)
                        delta = self.seed_up[layer_index](
                            F.silu(self.seed_down[layer_index](normalized_seed))
                        )
                elif self.query_mode == "seed_rope_latest":
                    delta = None
                else:
                    delta = 0.0
                if self.query_mode == "seed_rope_latest":
                    query = self.rotate_seed_keys_to_latest(seeds)
                else:
                    query = (
                        self.log_alpha[layer_index].exp().to(layer_keys.dtype) * seeds
                        + delta
                    )
            attention_value = (
                torch.cat([layer_keys, layer_values], dim=-1)
                if self.external_key_mode == "akav"
                else layer_values
            )
            output_dim = (
                2 * self.head_dim
                if self.external_key_mode == "akav"
                else self.head_dim
            )
            if attention_backend == "prefix_reference":
                with profile_range("generator_sdpa"):
                    layer_output = reference_deduplicated_prefix_attention(
                        query=query,
                        keys=layer_keys,
                        values=attention_value,
                        source_indices=source_indices,
                        seed_indices=indices,
                        anchor_attention_mask=anchor_attention_mask,
                        scale=self.head_dim**-0.5,
                    ).to(dtype=query.dtype)
            elif attention_backend == "prefix_scan":
                with profile_range("generator_sdpa"):
                    layer_output, self._last_prefix_scan_metadata = prefix_scan_attention(
                        query=query,
                        keys=layer_keys,
                        values=attention_value,
                        source_indices=source_indices,
                        seed_indices=indices,
                        anchor_attention_mask=anchor_attention_mask,
                        scale=self.head_dim**-0.5,
                    )
            elif attention_backend == "sdpa":
                with profile_range("generator_sdpa"):
                    layer_output = query.new_zeros(
                        anchors,
                        self.num_key_value_heads,
                        window,
                        output_dim,
                    )
                    for source_index in source_indices.unique(sorted=True):
                        anchor_rows = torch.nonzero(
                            source_indices == source_index, as_tuple=False
                        ).flatten()
                        selected_rows = (
                            torch.ones_like(expected_mask).index_select(0, anchor_rows)
                            if self.anchor_policy == "recent"
                            else expected_mask.index_select(0, anchor_rows)
                        )
                        if self.anchor_policy == "hybrid_raw192_agg64":
                            selected_rows = hybrid_aggregate_rows.index_select(
                                0, anchor_rows
                            )
                        anchor_offsets, slot_offsets = torch.nonzero(
                            selected_rows, as_tuple=True
                        )
                        if anchor_offsets.numel() == 0:
                            continue
                        selected_query = query.index_select(0, anchor_rows)
                        source_query = selected_query[
                            anchor_offsets, :, slot_offsets
                        ].transpose(0, 1)[None]
                        source_mask = anchor_attention_mask.index_select(
                            0, anchor_rows
                        ).index_select(0, anchor_offsets)[:, None, :].transpose(0, 1)
                        source_mask = source_mask[None]
                        source_row = source_index.view(1)
                        attended = F.scaled_dot_product_attention(
                            source_query,
                            layer_keys.index_select(0, source_row),
                            attention_value.index_select(0, source_row),
                            attn_mask=source_mask,
                            dropout_p=0.0,
                            is_causal=False,
                            scale=self.head_dim**-0.5,
                        )
                        layer_output[
                            anchor_rows.index_select(0, anchor_offsets),
                            :,
                            slot_offsets,
                        ] = attended[0].transpose(0, 1)
            else:
                raise ValueError("unsupported memory attention backend")
            with profile_range("bank_assembly"):
                if self.external_key_mode == "akav":
                    memory_key, memory_value = layer_output.split(self.head_dim, dim=-1)
                elif self.external_key_mode == "raw_seed":
                    valid = expected_mask[:, None, :, None]
                    memory_key = seeds.masked_fill(~valid, 0)
                    if self.anchor_policy == "hybrid_raw192_agg64":
                        raw = raw_row_mask[:, None, :, None]
                        memory_value = torch.where(raw, value_seeds, layer_output)
                    else:
                        memory_value = layer_output
                    memory_value = memory_value.masked_fill(~valid, 0)
                else:
                    memory_key = self.parameter_free_rms_norm(query)
                    memory_value = layer_output
                memory_keys.append(memory_key)
                memory_values.append(memory_value)
        with profile_range("bank_assembly"):
            return (
                torch.stack(memory_keys, dim=1),
                torch.stack(memory_values, dim=1),
                expected_mask,
            )


class DraftFreeKVConfig(PretrainedConfig):
    model_type = "draftfreekv"

    def __init__(
        self,
        target_model_name_or_path: Optional[str] = None,
        target_layer_ids: Optional[list[int]] = None,
        target_layer_policy: Optional[str] = None,
        num_target_layers: Optional[int] = None,
        block_size: int = 8,
        hidden_size: int = 4096,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        target_kv_window_size: Optional[int] = None,
        target_kv_memory_layout: str = "direct",
        target_kv_memory_query_rank: int = 16,
        memory_output_slots: Optional[int] = None,
        memory_source_window: Optional[object] = None,
        memory_query_mode: Optional[str] = None,
        memory_query_residual_rank: Optional[int] = None,
        memory_external_key_mode: Optional[str] = None,
        memory_attention_backend: str = "sdpa",
        memory_anchor_policy: str = "recent",
        vocab_size: int = 32000,
        adapter_rank: int = 512,
        adapter_alpha: float = 512.0,
        transition_rank: Optional[int] = None,
        transition_alpha: Optional[float] = None,
        draft_mode: str = "markov",
        mask_token_id: Optional[int] = None,
        parallel_mixer_layers: int = 5,
        parallel_mixer_rank: Optional[int] = None,
        parallel_mixer_alpha: Optional[float] = None,
        parallel_mixer_num_heads: int = 8,
        parallel_markov_rank: int = 0,
        parallel_markov_head_type: str = "vanilla",
        xpress_num_refinement_steps: int = 4,
        xpress_early_stop: bool = False,
        xpress_activation: str = "silu",
        qwen_intermediate_size: Optional[int] = None,
        qwen_num_attention_heads: Optional[int] = None,
        qwen_rope_theta: float = 1_000_000.0,
        qwen_rms_norm_eps: float = 1e-6,
        qwen_gradient_checkpointing: bool = True,
        training_objective: str = "ce",
        global16_raw256_enabled: bool = False,
        global_memory_slots: int = 16,
        raw_target_rows: int = 256,
        local_draft_rows: int = 7,
        max_accepted_tokens: int = 8,
        global_memory_source_window: str = "full",
        global_checkpoint_compatibility: str = "retrain_required",
        global_query_mode: str = "learned",
        global_query_equicorrelation: float = 0.4,
        global_query_slot_mass_mode: str = "system",
        **kwargs,
    ):
        # Removed in favor of unconditional random initialization. Ignore the
        # serialized field so checkpoints created before the removal still load.
        kwargs.pop("qwen_initialize_from_target", None)
        metadata = pop_transformers_metadata(kwargs)
        reject_unknown_config_fields(kwargs)
        super().__init__(**metadata)

        fields = resolve_draftfreekv_config_fields(
            target_model_name_or_path=target_model_name_or_path,
            target_layer_ids=target_layer_ids,
            target_layer_policy=target_layer_policy,
            num_target_layers=num_target_layers,
            block_size=block_size,
            hidden_size=hidden_size,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            target_kv_window_size=(
                target_kv_window_size
                if target_kv_window_size is not None
                else memory_output_slots
            ),
            target_kv_memory_layout=target_kv_memory_layout,
            target_kv_memory_query_rank=target_kv_memory_query_rank,
            vocab_size=vocab_size,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            transition_rank=transition_rank,
            transition_alpha=transition_alpha,
            draft_mode=draft_mode,
            mask_token_id=mask_token_id,
            parallel_mixer_layers=parallel_mixer_layers,
            parallel_mixer_rank=parallel_mixer_rank,
            parallel_mixer_alpha=parallel_mixer_alpha,
            parallel_mixer_num_heads=parallel_mixer_num_heads,
            parallel_markov_rank=parallel_markov_rank,
            parallel_markov_head_type=parallel_markov_head_type,
            xpress_num_refinement_steps=xpress_num_refinement_steps,
            xpress_early_stop=xpress_early_stop,
            xpress_activation=xpress_activation,
            qwen_intermediate_size=qwen_intermediate_size,
            qwen_num_attention_heads=qwen_num_attention_heads,
            qwen_rope_theta=qwen_rope_theta,
            qwen_rms_norm_eps=qwen_rms_norm_eps,
            qwen_gradient_checkpointing=qwen_gradient_checkpointing,
            training_objective=training_objective,
        )
        for name, value in fields.items():
            setattr(self, name, value)
        self.global16_raw256_enabled = bool(global16_raw256_enabled)
        self.global_memory_slots = int(global_memory_slots)
        self.raw_target_rows = int(raw_target_rows)
        self.local_draft_rows = int(local_draft_rows)
        self.max_accepted_tokens = int(max_accepted_tokens)
        self.global_memory_source_window = str(global_memory_source_window)
        self.global_checkpoint_compatibility = str(global_checkpoint_compatibility)
        self.global_query_mode = str(global_query_mode)
        self.global_query_equicorrelation = float(global_query_equicorrelation)
        self.global_query_slot_mass_mode = str(global_query_slot_mass_mode)
        if self.global16_raw256_enabled:
            if self.draft_mode != "parallel_shifted_qwen":
                raise ValueError("Global16 + Raw256 requires parallel_shifted_qwen")
            if self.target_kv_memory_layout != "direct":
                raise ValueError("Global16 + Raw256 reads direct Target K/V only")
            if target_kv_window_size is not None:
                raise ValueError("Global16 + Raw256 forbids a Target-K/V window")
            if memory_output_slots is not None or memory_source_window is not None:
                raise ValueError("Global16 + Raw256 forbids generated-memory bank fields")
            if self.block_size != 7:
                raise ValueError("Global16 + Raw256 fixes the draft-local source to Local7")
            # The immutable reference config validates all fixed numerical and
            # checkpoint-compatibility constraints in one place.
            Global16Raw256Config(
                target_layer_ids=tuple(self.target_layer_ids),
                num_attention_heads=self.qwen_num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                global_memory_slots=self.global_memory_slots,
                raw_target_rows=self.raw_target_rows,
                local_draft_rows=self.local_draft_rows,
                max_accepted_tokens=self.max_accepted_tokens,
                memory_source_window=self.global_memory_source_window,
                checkpoint_compatibility=self.global_checkpoint_compatibility,
                global_query_mode=self.global_query_mode,
                global_query_equicorrelation=self.global_query_equicorrelation,
                global_query_slot_mass_mode=self.global_query_slot_mass_mode,
            )
        self.target_kv_window_size = target_kv_window_size
        has_new_memory_shape = (
            memory_output_slots is not None or memory_source_window is not None
        )
        has_nonlegacy_memory_mode = memory_query_mode not in {None, "hidden_r16"} or (
            memory_external_key_mode not in {None, "akav"}
        )
        if target_kv_window_size is not None and (
            has_new_memory_shape or has_nonlegacy_memory_mode
        ):
            raise ValueError(
                "target_kv_window_size cannot be mixed with speed-native memory fields"
            )
        if has_new_memory_shape or has_nonlegacy_memory_mode:
            if target_kv_memory_layout != "attention_generated":
                raise ValueError(
                    "speed-native memory fields require attention_generated layout"
                )
            if memory_output_slots is None or memory_source_window is None:
                raise ValueError(
                    "memory_output_slots and memory_source_window must be set together"
                )
            if not isinstance(memory_output_slots, int) or memory_output_slots <= 0:
                raise ValueError("memory_output_slots must be a positive integer")
            if memory_source_window != "full" and (
                not isinstance(memory_source_window, int)
                or memory_source_window <= 0
            ):
                raise ValueError(
                    "memory_source_window must be a positive integer or 'full'"
                )
        self.memory_output_slots = memory_output_slots
        self.memory_source_window = memory_source_window
        self.memory_query_mode = memory_query_mode or "hidden_r16"
        self.memory_query_residual_rank = memory_query_residual_rank
        self.memory_external_key_mode = memory_external_key_mode or "akav"
        self.memory_attention_backend = str(memory_attention_backend)
        self.memory_anchor_policy = str(memory_anchor_policy)
        if self.memory_query_mode not in {
            "hidden_r16",
            "seed_only",
            "seed_r4",
            "seed_residual",
            "seed_rope_latest",
        }:
            raise ValueError("unsupported memory_query_mode")
        if self.memory_query_mode == "seed_residual":
            if (
                not isinstance(memory_query_residual_rank, int)
                or memory_query_residual_rank <= 0
            ):
                raise ValueError(
                    "seed_residual requires memory_query_residual_rank to be a positive integer"
                )
        elif memory_query_residual_rank is not None:
            raise ValueError(
                "memory_query_residual_rank is supported only for seed_residual"
            )
        if self.memory_query_mode == "seed_rope_latest" and (
            self.memory_source_window != "full"
            or self.memory_external_key_mode != "normalized_query"
            or self.memory_attention_backend != "sdpa"
            or self.head_dim != 128
            or float(self.qwen_rope_theta) != 1_000_000.0
        ):
            raise ValueError(
                "seed_rope_latest requires full source, normalized_query, SDPA, "
                "head_dim=128, and qwen_rope_theta=1000000"
            )
        if self.memory_external_key_mode not in {
            "akav",
            "normalized_query",
            "raw_seed",
        }:
            raise ValueError("unsupported memory_external_key_mode")
        if self.memory_external_key_mode == "raw_seed" and (
            self.memory_query_mode != "seed_residual"
            or self.memory_query_residual_rank != 128
            or self.memory_output_slots != 256
            or self.memory_source_window != "full"
            or self.memory_attention_backend != "sdpa"
        ):
            raise ValueError(
                "raw_seed requires seed_residual rank 128, W256, full source, and SDPA"
            )
        if self.memory_attention_backend not in {"sdpa", "prefix_scan"}:
            raise ValueError(
                "memory_attention_backend must be 'sdpa' or experimental 'prefix_scan'"
            )
        if self.memory_anchor_policy not in {
            "recent",
            "strat128_64_64",
            "hybrid_raw192_agg64",
        }:
            raise ValueError("unsupported memory_anchor_policy")
        if self.memory_anchor_policy != "recent" and (
            self.memory_query_mode != "seed_residual"
            or self.memory_query_residual_rank != 128
            or self.memory_output_slots != 256
            or self.memory_source_window != "full"
            or self.memory_external_key_mode != "raw_seed"
            or self.memory_attention_backend != "sdpa"
        ):
            raise ValueError(
                "stratified memory policies require RawK-AV seed_residual r128, "
                "W256, full source, and SDPA"
            )


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states)) * self.scaling

    def forward_selected(
        self,
        hidden_states: torch.Tensor,
        output_indices: torch.Tensor,
    ) -> torch.Tensor:
        if output_indices.ndim != 1:
            raise ValueError("output_indices must be a 1-D tensor")
        projected = self.down(hidden_states)
        selected_weight = self.up.weight.index_select(
            0,
            output_indices.to(device=self.up.weight.device),
        )
        return F.linear(projected, selected_weight) * self.scaling


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        bias: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return normalized.to(dtype=hidden_states.dtype) * self.weight.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


class DraftStateEncoder(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.proj = FeedForward(num_layers * hidden_size, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(self, selected_layer_hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(selected_layer_hidden.flatten(start_dim=1)))


class MarkovEmbedding(nn.Module):
    def __init__(self, hidden_size: int, block_size: int):
        super().__init__()
        self.step_embedding = nn.Embedding(block_size, hidden_size)
        self.proj = FeedForward(2 * hidden_size, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        *,
        previous_token_ids: torch.Tensor,
        step_index: int,
        target_embed_tokens: nn.Module,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        token_ids = previous_token_ids.to(
            device=self.step_embedding.weight.device,
            dtype=torch.long,
        )
        token_state = target_embed_tokens(token_ids).to(dtype=model_dtype)
        step_ids = torch.full_like(token_ids, step_index)
        step_state = self.step_embedding(step_ids).to(dtype=model_dtype)
        return self.norm(self.proj(torch.cat([token_state, step_state], dim=-1)))


class TargetMemoryQuery(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_dim = num_layers * num_kv_heads * head_dim
        self.draft_norm = RMSNorm(hidden_size)
        self.proj = FeedForward(2 * hidden_size, hidden_size, self.query_dim)

    def forward(
        self,
        *,
        draft_state: torch.Tensor,
        markov_state: torch.Tensor,
    ) -> torch.Tensor:
        query = self.proj(
            torch.cat([self.draft_norm(draft_state), markov_state], dim=-1)
        )
        return query.view(
            draft_state.shape[0],
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )


class TargetMemoryReader(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.head_dim = head_dim
        readout_dim = num_layers * num_kv_heads * head_dim
        self.readout = FeedForward(readout_dim, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        *,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("blhd,blhtd->blht", query, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = scores.softmax(dim=-1).to(dtype=values.dtype)
        readout = torch.einsum("blht,blhtd->blhd", weights, values)
        return self.norm(self.readout(readout.flatten(start_dim=1)))


class DraftCell(nn.Module):
    def __init__(self, hidden_size: int, rank: int, alpha: float):
        super().__init__()
        self.input_norm = RMSNorm(3 * hidden_size)
        self.update = LowRankLinear(3 * hidden_size, hidden_size, rank=rank, alpha=alpha)
        self.gate = LowRankLinear(3 * hidden_size, hidden_size, rank=rank, alpha=alpha)
        self.output_norm = RMSNorm(hidden_size)

    def forward(
        self,
        *,
        draft_state: torch.Tensor,
        memory_state: torch.Tensor,
        markov_state: torch.Tensor,
    ) -> torch.Tensor:
        cell_input = torch.cat([draft_state, memory_state, markov_state], dim=-1)
        cell_input = self.input_norm(cell_input)
        update = self.update(cell_input)
        gate = torch.sigmoid(self.gate(cell_input))
        return self.output_norm(draft_state + gate * update)


class ParallelShiftedInput(nn.Module):
    """Build ``[x_t, MASK, ...]`` slots conditioned on the latest target hidden."""

    def __init__(self, hidden_size: int, block_size: int, mask_token_id: int):
        super().__init__()
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.position_embedding = nn.Embedding(block_size, hidden_size)
        self.proj = FeedForward(2 * hidden_size, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def precompute_token_states(
        self,
        *,
        current_token_ids: torch.Tensor,
        target_embed_tokens: nn.Module,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Embed the per-request ``[x_t, MASK, ...]`` token layout."""

        batch_size = current_token_ids.shape[0]
        token_ids = torch.full(
            (batch_size, self.block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        token_ids[:, 0] = current_token_ids.to(
            device=token_ids.device,
            dtype=torch.long,
        )
        return target_embed_tokens(token_ids).to(dtype=model_dtype)

    def forward(
        self,
        *,
        current_token_ids: torch.Tensor,
        latest_target_state: torch.Tensor,
        target_embed_tokens: nn.Module,
        model_dtype: torch.dtype,
        precomputed_token_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if precomputed_token_states is None:
            token_states = self.precompute_token_states(
                current_token_ids=current_token_ids,
                target_embed_tokens=target_embed_tokens,
                model_dtype=model_dtype,
                device=latest_target_state.device,
            )
        else:
            expected_shape = (
                current_token_ids.shape[0],
                self.block_size,
                self.position_embedding.embedding_dim,
            )
            if precomputed_token_states.shape != expected_shape:
                raise ValueError(
                    "precomputed_token_states must have shape "
                    f"{expected_shape}, got {tuple(precomputed_token_states.shape)}"
                )
            if precomputed_token_states.device != latest_target_state.device:
                raise ValueError(
                    "precomputed_token_states must be on "
                    f"{latest_target_state.device}, got "
                    f"{precomputed_token_states.device}"
                )
            if precomputed_token_states.dtype != model_dtype:
                raise ValueError(
                    "precomputed_token_states must have dtype "
                    f"{model_dtype}, got {precomputed_token_states.dtype}"
                )
            token_states = precomputed_token_states
        positions = torch.arange(
            self.block_size,
            device=latest_target_state.device,
        )
        token_states = token_states + self.position_embedding(positions).to(
            dtype=model_dtype
        )[None, :, :]
        latest = latest_target_state[:, None, :].expand(-1, self.block_size, -1)
        return self.norm(self.proj(torch.cat([token_states, latest], dim=-1)))


class ParallelTargetMemoryQuery(nn.Module):
    """Produce one raw-target-KV query for every shifted block position."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.query_dim = num_layers * num_kv_heads * head_dim
        self.block_norm = RMSNorm(hidden_size)
        self.proj = FeedForward(2 * hidden_size, hidden_size, self.query_dim)

    def forward(
        self,
        *,
        block_states: torch.Tensor,
        latest_target_state: torch.Tensor,
    ) -> torch.Tensor:
        latest = latest_target_state[:, None, :].expand_as(block_states)
        query = self.proj(
            torch.cat([self.block_norm(block_states), latest], dim=-1)
        )
        return query.view(
            block_states.shape[0],
            block_states.shape[1],
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )


class ParallelTargetMemoryReader(nn.Module):
    """Read the full verifier-visible target KV history for all slots in parallel."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
    ):
        super().__init__()
        self.head_dim = head_dim
        readout_dim = num_layers * num_kv_heads * head_dim
        self.readout = FeedForward(readout_dim, hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)

    def forward(
        self,
        *,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("bklhd,blhtd->bklht", query, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~attention_mask[:, None, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = scores.float().softmax(dim=-1).to(dtype=values.dtype)
        readout = torch.einsum("bklht,blhtd->bklhd", weights, values)
        return self.norm(self.readout(readout.flatten(start_dim=2)))


class ParallelBlockMixerLayer(nn.Module):
    """A low-rank, bidirectional block mixer with no persistent draft KV cache."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.input_norm = RMSNorm(hidden_size)
        self.q_proj = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.k_proj = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.v_proj = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.o_proj = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.post_attention_norm = RMSNorm(hidden_size)
        self.ffn_update = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.ffn_gate = LowRankLinear(hidden_size, hidden_size, rank, alpha)
        self.output_norm = RMSNorm(hidden_size)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, block, _ = hidden_states.shape
        return hidden_states.view(
            batch,
            block,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(hidden_states)
        query = self._split_heads(self.q_proj(normalized))
        key = self._split_heads(self.k_proj(normalized))
        value = self._split_heads(self.v_proj(normalized))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        # Deliberately no triangular mask: every shifted slot is denoised jointly.
        weights = scores.float().softmax(dim=-1).to(dtype=value.dtype)
        mixed = torch.matmul(weights, value).transpose(1, 2).contiguous()
        mixed = mixed.view_as(hidden_states)
        hidden_states = hidden_states + self.o_proj(mixed)

        normalized = self.post_attention_norm(hidden_states)
        update = self.ffn_update(normalized)
        gate = torch.sigmoid(self.ffn_gate(normalized))
        return self.output_norm(hidden_states + gate * update)


class Qwen3SwiGLUMLP(nn.Module):
    """Full-rank Qwen3 feed-forward block with target-compatible names."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _create_packed_qwen_attention_mask(
    *,
    anchor_prefix_mask: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    block_size: int,
):
    """Build the DSpark-style sparse mask for shared target KV plus local blocks."""

    if anchor_prefix_mask.ndim != 3:
        raise ValueError("anchor_prefix_mask must have shape [batch, anchors, tokens]")
    if anchor_valid_mask.shape != anchor_prefix_mask.shape[:2]:
        raise ValueError("anchor_valid_mask must have shape [batch, anchors]")
    source_batch, num_anchors, target_length = anchor_prefix_mask.shape
    query_length = num_anchors * block_size

    def mask_mod(batch_index, head_index, query_index, key_value_index):
        del head_index
        query_anchor = query_index // block_size
        is_context = key_value_index < target_length
        safe_context_index = key_value_index.clamp(max=target_length - 1)
        context_visible = anchor_prefix_mask[
            batch_index,
            query_anchor,
            safe_context_index,
        ]
        is_local = key_value_index >= target_length
        local_anchor = (key_value_index - target_length) // block_size
        same_local_block = query_anchor == local_anchor
        return anchor_valid_mask[batch_index, query_anchor] & (
            (is_context & context_visible) | (is_local & same_local_block)
        )

    return create_block_mask(
        mask_mod,
        B=source_batch,
        H=None,
        Q_LEN=query_length,
        KV_LEN=target_length + query_length,
        device=anchor_prefix_mask.device,
    )


class Qwen3KVFreeParallelAttention(nn.Module):
    """Qwen3 GQA over target KV history plus a bidirectional local block.

    Target keys are read directly from the verifier cache and have already had
    target Qwen3 K-norm and RoPE applied. Only the local block query/key need
    normalization and rotary embedding here. No local K/V is returned or kept.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _position_embeddings(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = (
            position_ids.to(device=self.inv_freq.device, dtype=torch.float32)[..., None]
            * self.inv_freq[None, None, :]
        )
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)

    def _apply_rotary(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        return hidden_states * cos + _rotate_half(hidden_states) * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, block_size, _ = hidden_states.shape
        if position_ids.shape != (batch_size, block_size):
            raise ValueError("position_ids must have shape [batch, block_size]")
        expected_prefix = (batch_size, self.num_key_value_heads)
        if (
            target_keys.shape[:2] != expected_prefix
            or target_keys.shape != target_values.shape
        ):
            raise ValueError(
                "target_keys/target_values must have shape "
                "[batch, num_key_value_heads, tokens, head_dim]"
            )
        if target_keys.shape[-1] != self.head_dim:
            raise ValueError("target KV head dimension does not match Qwen attention")

        query = self.q_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_attention_heads,
            self.head_dim,
        )
        local_key = self.k_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        local_value = self.v_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        local_key = self.k_norm(local_key).transpose(1, 2)
        local_value = local_value.transpose(1, 2)
        cos, sin = self._position_embeddings(position_ids, dtype=query.dtype)
        query = self._apply_rotary(query, cos, sin)
        local_key = self._apply_rotary(local_key, cos, sin)

        key = torch.cat([target_keys.to(dtype=query.dtype), local_key], dim=2)
        value = torch.cat([target_values.to(dtype=query.dtype), local_value], dim=2)
        local_mask = torch.ones(
            (batch_size, block_size),
            dtype=torch.bool,
            device=target_attention_mask.device,
        )
        attention_mask = torch.cat(
            [target_attention_mask.bool(), local_mask],
            dim=1,
        )[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=self.num_key_value_groups > 1,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            block_size,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attended)

    def forward_global16_raw256(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        global16_context,
        layer_index: int,
    ) -> torch.Tensor:
        """Qwen draft attention over separate Global16, Raw256, and Local7 sources.

        ``global16_context`` is either the differentiable offline adapter or
        the SGLang paged-KV adapter.  Both expose ``attend`` with the same
        source-wise online-softmax definition.  This method intentionally has
        no ``torch.cat`` over K/V: Raw256 remains a borrowed Target cache and
        Local7 stays a separate draft-local source.
        """

        batch_size, block_size, _ = hidden_states.shape
        if block_size != 7:
            raise ValueError("Global16 + Raw256 requires exactly Local7 draft rows")
        if position_ids.shape != (batch_size, block_size):
            raise ValueError("position_ids must have shape [batch, Local7]")
        query = self.q_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_attention_heads,
            self.head_dim,
        )
        local_key = self.k_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        local_value = self.v_proj(hidden_states).view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        local_key = self.k_norm(local_key).transpose(1, 2)
        local_value = local_value.transpose(1, 2)
        prepare_shared_inputs = getattr(
            global16_context,
            "prepare_draft_layer_inputs",
            None,
        )
        if callable(prepare_shared_inputs):
            cos, sin, local_attention_mask = prepare_shared_inputs(
                attention=self,
                position_ids=position_ids,
                dtype=query.dtype,
                batch_size=batch_size,
                block_size=block_size,
            )
        else:
            cos, sin = self._position_embeddings(position_ids, dtype=query.dtype)
            local_attention_mask = torch.ones(
                (batch_size, block_size, block_size),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        query = self._apply_rotary(query, cos, sin)
        local_key = self._apply_rotary(local_key, cos, sin)
        attended = global16_context.attend(
            layer_index=layer_index,
            query=query,
            local_keys=local_key,
            local_values=local_value,
            local_attention_mask=local_attention_mask,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            block_size,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attended)

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        """Attend packed anchor blocks to one shared target KV cache per source."""

        batch_size, query_length, _ = hidden_states.shape
        if position_ids.shape != (batch_size, query_length):
            raise ValueError("position_ids must match packed hidden_states")
        expected_prefix = (batch_size, self.num_key_value_heads)
        if (
            target_keys.shape[:2] != expected_prefix
            or target_keys.shape != target_values.shape
        ):
            raise ValueError(
                "packed target_keys/target_values must have shape "
                "[source_batch, num_key_value_heads, tokens, head_dim]"
            )
        if target_keys.shape[-1] != self.head_dim:
            raise ValueError("target KV head dimension does not match Qwen attention")

        query = self.q_proj(hidden_states).view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.head_dim,
        )
        local_key = self.k_proj(hidden_states).view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        local_value = self.v_proj(hidden_states).view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        local_key = self.k_norm(local_key).transpose(1, 2)
        local_value = local_value.transpose(1, 2)
        cos, sin = self._position_embeddings(position_ids, dtype=query.dtype)
        query = self._apply_rotary(query, cos, sin)
        local_key = self._apply_rotary(local_key, cos, sin)

        key = torch.cat([target_keys.to(dtype=query.dtype), local_key], dim=2)
        value = torch.cat([target_values.to(dtype=query.dtype), local_value], dim=2)
        if self.num_key_value_groups > 1:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        attended = _run_packed_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=self.scaling,
            enable_gqa=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attended)


class Qwen3KVFreeParallelDecoderLayer(nn.Module):
    """A full Qwen3 decoder layer adapted to KV-free parallel drafting."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.self_attn = Qwen3KVFreeParallelAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Qwen3SwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        target_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_keys=target_keys,
            target_values=target_values,
            target_attention_mask=target_attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward_global16_raw256(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        global16_context,
        layer_index: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn.forward_global16_raw256(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            global16_context=global16_context,
            layer_index=layer_index,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward_packed(
        self,
        hidden_states: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        position_ids: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn.forward_packed(
            self.input_layernorm(hidden_states),
            target_keys,
            target_values,
            position_ids,
            block_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class DSparkVanillaMarkovHead(nn.Module):
    """The optional DSpark vanilla low-rank previous-token logit correction."""

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        if markov_rank <= 0:
            raise ValueError("markov_rank must be positive")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_bias(
        self,
        previous_token_ids: torch.Tensor,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent = self.markov_w1(previous_token_ids.long())
        if candidate_token_ids is None:
            return self.markov_w2(latent)
        output_indices = candidate_token_ids.to(device=self.markov_w2.weight.device)
        output_weight = self.markov_w2.weight.index_select(0, output_indices)
        return F.linear(latent, output_weight)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        previous_token_ids: torch.Tensor,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return base_logits + self.compute_bias(
            previous_token_ids,
            candidate_token_ids,
        ).to(dtype=base_logits.dtype)

    def sample_block_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_previous_token_ids: torch.Tensor,
        temperature: float = 0.0,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled_tokens = []
        corrected_logits = []
        previous_token_ids = first_previous_token_ids.long()
        for step_index in range(base_logits.shape[1]):
            step_logits = base_logits[:, step_index, :] + self.compute_bias(
                previous_token_ids,
                candidate_token_ids,
            ).to(dtype=base_logits.dtype)
            corrected_logits.append(step_logits)
            if temperature <= 0:
                sampled_indices = step_logits.argmax(dim=-1)
            else:
                probabilities = (step_logits.float() / temperature).softmax(dim=-1)
                sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
            if candidate_token_ids is None:
                next_tokens = sampled_indices
            else:
                next_tokens = candidate_token_ids.to(
                    device=sampled_indices.device
                ).index_select(0, sampled_indices)
            sampled_tokens.append(next_tokens)
            previous_token_ids = next_tokens
        if not sampled_tokens:
            empty = torch.empty(
                base_logits.shape[0],
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty, base_logits
        return torch.stack(sampled_tokens, dim=1), torch.stack(
            corrected_logits,
            dim=1,
        )

    def sample_block_token_ids(
        self,
        base_logits: torch.Tensor,
        *,
        first_previous_token_ids: torch.Tensor,
        temperature: float = 0.0,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the autoregressive Markov chain without retaining logits."""

        sampled_tokens = []
        previous_token_ids = first_previous_token_ids.long()
        for step_index in range(base_logits.shape[1]):
            step_logits = base_logits[:, step_index, :] + self.compute_bias(
                previous_token_ids,
                candidate_token_ids,
            ).to(dtype=base_logits.dtype)
            if temperature <= 0:
                sampled_indices = step_logits.argmax(dim=-1)
            else:
                probabilities = (step_logits.float() / temperature).softmax(dim=-1)
                sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
            if candidate_token_ids is None:
                next_tokens = sampled_indices
            else:
                next_tokens = candidate_token_ids.to(
                    device=sampled_indices.device
                ).index_select(0, sampled_indices)
            sampled_tokens.append(next_tokens)
            previous_token_ids = next_tokens
        if not sampled_tokens:
            return torch.empty(
                base_logits.shape[0],
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
        return torch.stack(sampled_tokens, dim=1)


class DraftFreeKVModel(PreTrainedModel):
    config_class = DraftFreeKVConfig
    _no_split_modules = ["DraftFreeKVModel", "Qwen3KVFreeParallelDecoderLayer"]

    def __init__(self, config: DraftFreeKVConfig):
        super().__init__(config)
        if not config.target_layer_ids:
            raise ValueError("DraftFreeKVConfig.target_layer_ids must be resolved")
        self.config = config
        self.num_layers = len(config.target_layer_ids)
        self._independent_generated_bank_sdpa = _env_flag(
            "DFK_INDEPENDENT_GENERATED_BANK_SDPA",
            default=True,
        )
        self._serving_fastpath_trace = _env_flag("DFK_SERVING_FASTPATH_TRACE")
        self._independent_trace_emitted = False

        self.draft_state_encoder = DraftStateEncoder(
            num_layers=self.num_layers,
            hidden_size=config.hidden_size,
        )
        if config.draft_mode == "markov":
            # Keep the original module names and parameter layout byte-for-byte
            # compatible with checkpoints created before draft_mode existed.
            self.markov_embedding = MarkovEmbedding(
                hidden_size=config.hidden_size,
                block_size=config.block_size,
            )
            self.memory_query = TargetMemoryQuery(
                num_layers=self.num_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                hidden_size=config.hidden_size,
            )
            self.target_memory_reader = TargetMemoryReader(
                num_layers=self.num_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                hidden_size=config.hidden_size,
            )
            self.draft_cell = DraftCell(
                hidden_size=config.hidden_size,
                rank=config.transition_rank,
                alpha=config.transition_alpha,
            )
        elif config.draft_mode in PARALLEL_SHIFTED_DRAFT_MODES:
            self.parallel_shifted_input = ParallelShiftedInput(
                hidden_size=config.hidden_size,
                block_size=config.block_size,
                mask_token_id=config.mask_token_id,
            )
            if config.draft_mode == "parallel_shifted":
                self.parallel_memory_query = ParallelTargetMemoryQuery(
                    num_layers=self.num_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    hidden_size=config.hidden_size,
                )
                self.parallel_target_memory_reader = ParallelTargetMemoryReader(
                    num_layers=self.num_layers,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    hidden_size=config.hidden_size,
                )
                self.parallel_fusion_cell = DraftCell(
                    hidden_size=config.hidden_size,
                    rank=config.transition_rank,
                    alpha=config.transition_alpha,
                )
                self.parallel_mixers = nn.ModuleList(
                    [
                        ParallelBlockMixerLayer(
                            hidden_size=config.hidden_size,
                            num_heads=config.parallel_mixer_num_heads,
                            rank=config.parallel_mixer_rank,
                            alpha=config.parallel_mixer_alpha,
                        )
                        for _ in range(config.parallel_mixer_layers)
                    ]
                )
            else:
                if config.parallel_mixer_layers != self.num_layers:
                    raise ValueError(
                        "parallel_shifted_qwen requires one decoder layer per "
                        "selected target KV layer"
                    )
                self.parallel_qwen_layers = nn.ModuleList(
                    [
                        Qwen3KVFreeParallelDecoderLayer(
                            hidden_size=config.hidden_size,
                            intermediate_size=config.qwen_intermediate_size,
                            num_attention_heads=config.qwen_num_attention_heads,
                            num_key_value_heads=config.num_key_value_heads,
                            head_dim=config.head_dim,
                            rope_theta=config.qwen_rope_theta,
                            rms_norm_eps=config.qwen_rms_norm_eps,
                        )
                        for _ in range(config.parallel_mixer_layers)
                    ]
                )
                self.attention_generated_target_memory = (
                    AttentionGeneratedTargetMemory(
                        num_layers=self.num_layers,
                        hidden_size=config.hidden_size,
                        num_key_value_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        rank=config.target_kv_memory_query_rank,
                        residual_rank=config.memory_query_residual_rank,
                        output_slots=(
                            config.memory_output_slots
                            or config.target_kv_window_size
                        ),
                        query_mode=config.memory_query_mode,
                        external_key_mode=config.memory_external_key_mode,
                        rope_theta=config.qwen_rope_theta,
                        anchor_policy=config.memory_anchor_policy,
                    )
                    if config.target_kv_memory_layout == "attention_generated"
                    else None
                )
                self.global16_raw256_reference = (
                    Global16Raw256Reference(
                        Global16Raw256Config(
                            target_layer_ids=tuple(config.target_layer_ids),
                            num_attention_heads=config.qwen_num_attention_heads,
                            num_key_value_heads=config.num_key_value_heads,
                            head_dim=config.head_dim,
                            global_memory_slots=config.global_memory_slots,
                            raw_target_rows=config.raw_target_rows,
                            local_draft_rows=config.local_draft_rows,
                            max_accepted_tokens=config.max_accepted_tokens,
                            memory_source_window=config.global_memory_source_window,
                            checkpoint_compatibility=config.global_checkpoint_compatibility,
                            global_query_mode=config.global_query_mode,
                            global_query_equicorrelation=config.global_query_equicorrelation,
                            global_query_slot_mass_mode=config.global_query_slot_mass_mode,
                        )
                    )
                    if config.global16_raw256_enabled
                    else None
                )
            self.parallel_markov_head = (
                DSparkVanillaMarkovHead(
                    vocab_size=config.vocab_size,
                    markov_rank=config.parallel_markov_rank,
                )
                if config.parallel_markov_rank > 0
                else None
            )
        else:
            raise ValueError(f"unsupported DraftFreeKV draft_mode: {config.draft_mode}")
        self.post_init()
        self._enforce_global_query_trainability()
        if getattr(self, "attention_generated_target_memory", None) is not None:
            self.attention_generated_target_memory.reset_zero_output()

    def forward(
        self,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_source_indices: Optional[torch.Tensor] = None,
        anchor_attention_mask: Optional[torch.Tensor] = None,
        current_hidden_states: Optional[torch.Tensor] = None,
        memory_hidden_states: Optional[torch.Tensor] = None,
        memory_query_mask: Optional[torch.Tensor] = None,
        projected_memory_refinements: Optional[torch.Tensor] = None,
        current_token_ids: Optional[torch.Tensor] = None,
        current_position_ids: Optional[torch.Tensor] = None,
        teacher_forcing_token_ids: Optional[torch.Tensor] = None,
        target_final_norm: Optional[nn.Module] = None,
        target_embed_tokens: Optional[nn.Module] = None,
        target_lm_head: Optional[nn.Module] = None,
        candidate_token_ids: Optional[torch.Tensor] = None,
        apply_markov_correction: bool = True,
        global16_paged_context=None,
        return_pre_projection_hidden_states: bool = False,
    ) -> torch.Tensor:
        if target_embed_tokens is None:
            raise ValueError("target_embed_tokens is required")
        if target_lm_head is None:
            raise ValueError("target_lm_head is required and must not be checkpointed")
        if current_token_ids is None:
            raise ValueError("current_token_ids are required")
        if candidate_token_ids is not None and candidate_token_ids.ndim != 1:
            raise ValueError("candidate_token_ids must be a 1-D tensor")
        if return_pre_projection_hidden_states:
            if not self.config.global16_raw256_enabled:
                raise ValueError(
                    "return_pre_projection_hidden_states is supported only by "
                    "Global16 + Raw256 training"
                )
            if candidate_token_ids is not None:
                raise ValueError(
                    "return_pre_projection_hidden_states requires the full vocabulary"
                )
        self._freeze_target_module(target_embed_tokens)
        self._freeze_target_module(target_lm_head)
        self._freeze_target_module(target_final_norm)

        if self.config.global16_raw256_enabled:
            return self._forward_global16_raw256(
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                kv_source_indices=kv_source_indices,
                anchor_attention_mask=anchor_attention_mask,
                current_hidden_states=current_hidden_states,
                current_token_ids=current_token_ids,
                current_position_ids=current_position_ids,
                teacher_forcing_token_ids=teacher_forcing_token_ids,
                target_final_norm=target_final_norm,
                target_embed_tokens=target_embed_tokens,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
                apply_markov_correction=apply_markov_correction,
                global16_paged_context=global16_paged_context,
                return_pre_projection_hidden_states=return_pre_projection_hidden_states,
            )
        if keys is None or values is None:
            raise ValueError("keys and values are required outside Global16 + Raw256 mode")

        attention_mask = normalize_kv_attention_mask(
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            expected_layers=self.num_layers,
            expected_heads=self.config.num_key_value_heads,
            expected_head_dim=self.config.head_dim,
        )
        target_kv_window_size = self.config.target_kv_window_size
        if self.config.memory_output_slots is not None:
            target_kv_window_size = (
                None
                if self.config.memory_source_window == "full"
                else int(self.config.memory_source_window)
            )
        if kv_source_indices is None:
            keys, values, attention_mask = select_latest_visible_target_kv(
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                window_size=target_kv_window_size,
            )
        model_dtype = next(self.parameters()).dtype
        keys = keys.to(dtype=model_dtype)
        values = values.to(dtype=model_dtype)
        anchor_batch_size = keys.shape[0]
        if kv_source_indices is not None:
            if self.config.draft_mode != "parallel_shifted_qwen":
                raise ValueError(
                    "shared target KV training is supported only by parallel_shifted_qwen"
                )
            if kv_source_indices.ndim != 1 or kv_source_indices.numel() == 0:
                raise ValueError("kv_source_indices must be a non-empty 1-D tensor")
            kv_source_indices = kv_source_indices.to(device=keys.device, dtype=torch.long)
            if torch.any(kv_source_indices < 0) or torch.any(
                kv_source_indices >= keys.shape[0]
            ):
                raise ValueError("kv_source_indices contains an invalid source row")
            anchor_batch_size = int(kv_source_indices.numel())
            if anchor_attention_mask is None:
                raise ValueError(
                    "anchor_attention_mask is required with kv_source_indices"
                )
            expected_anchor_mask_shape = (anchor_batch_size, keys.shape[-2])
            if anchor_attention_mask.shape != expected_anchor_mask_shape:
                raise ValueError(
                    "anchor_attention_mask must have shape [anchors, target_tokens]"
                )
            anchor_attention_mask = anchor_attention_mask.to(
                device=keys.device,
                dtype=torch.bool,
            )
        elif anchor_attention_mask is not None:
            raise ValueError(
                "anchor_attention_mask must be omitted without kv_source_indices"
            )
        selected_layer_hidden = normalize_current_hidden_states(
            current_hidden_states=current_hidden_states,
            batch_size=anchor_batch_size,
            expected_layers=self.num_layers,
            hidden_size=self.config.hidden_size,
            dtype=model_dtype,
            device=keys.device,
        )
        teacher_forcing_token_ids = self._validate_teacher_forcing_token_ids(
            teacher_forcing_token_ids,
            batch_size=anchor_batch_size,
            device=keys.device,
        )

        if self.config.draft_mode == "parallel_shifted":
            return self._forward_parallel_shifted(
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                selected_layer_hidden=selected_layer_hidden,
                current_token_ids=current_token_ids,
                teacher_forcing_token_ids=teacher_forcing_token_ids,
                target_final_norm=target_final_norm,
                target_embed_tokens=target_embed_tokens,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
                apply_markov_correction=apply_markov_correction,
                model_dtype=model_dtype,
            )
        if self.config.draft_mode == "parallel_shifted_qwen":
            independent_generated_bank_sdpa = (
                self._independent_generated_bank_sdpa
                and not self.training
                and kv_source_indices is None
                and self.config.target_kv_memory_layout == "attention_generated"
                and self.config.memory_anchor_policy == "hybrid_raw192_agg64"
                and self.config.memory_attention_backend == "sdpa"
            )
            current_position_ids = self._validate_current_position_ids(
                current_position_ids,
                batch_size=anchor_batch_size,
                device=keys.device,
            )
            if anchor_attention_mask is not None:
                anchor_attention_mask = restrict_target_kv_attention_to_window(
                    attention_mask=anchor_attention_mask,
                    current_position_ids=current_position_ids,
                    window_size=target_kv_window_size,
                )
            if self.config.target_kv_memory_layout == "attention_generated":
                if (
                    self.config.memory_query_mode == "hidden_r16"
                    and memory_query_mask is None
                ):
                    raise ValueError(
                        "memory_query_mask is required for attention-generated memory"
                    )
                if memory_query_mask is not None:
                    memory_query_mask = memory_query_mask.to(
                        device=keys.device,
                        dtype=torch.bool,
                    )
                generator_attention_mask = (
                    anchor_attention_mask
                    if kv_source_indices is not None
                    else attention_mask
                )
                keys, values, attention_mask = self.attention_generated_target_memory(
                    keys=keys.detach(),
                    values=values.detach(),
                    anchor_attention_mask=generator_attention_mask,
                    memory_hidden_states=(
                        None
                        if memory_hidden_states is None
                        else memory_hidden_states.to(device=keys.device, dtype=model_dtype)
                    ),
                    projected_refinements=(
                        None
                        if projected_memory_refinements is None
                        else projected_memory_refinements.to(
                            device=keys.device,
                            dtype=model_dtype,
                        )
                    ),
                    memory_query_mask=memory_query_mask,
                    source_indices=kv_source_indices,
                    attention_backend=self.config.memory_attention_backend,
                )
                # Generated banks are anchor-local and fixed-size. H1 lets an
                # independent serving request use the ordinary decoder SDPA/GQA
                # path. Training, shared rows, and the default flag-off route
                # retain the legacy identity-map packed decoder contract.
                if independent_generated_bank_sdpa:
                    kv_source_indices = None
                    anchor_attention_mask = None
                    if (
                        self._serving_fastpath_trace
                        and not self._independent_trace_emitted
                    ):
                        logger.info(
                            "DFK_SERVING_FASTPATH independent_generated_bank_sdpa "
                            "hit bank_rows=%d",
                            attention_mask.shape[-1],
                        )
                        self._independent_trace_emitted = True
                else:
                    kv_source_indices = torch.arange(
                        attention_mask.shape[0], device=keys.device
                    )
                    anchor_attention_mask = attention_mask
            return self._forward_parallel_shifted_qwen(
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                kv_source_indices=kv_source_indices,
                anchor_attention_mask=anchor_attention_mask,
                selected_layer_hidden=selected_layer_hidden,
                current_token_ids=current_token_ids,
                current_position_ids=current_position_ids,
                teacher_forcing_token_ids=teacher_forcing_token_ids,
                target_final_norm=target_final_norm,
                target_embed_tokens=target_embed_tokens,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
                apply_markov_correction=apply_markov_correction,
                model_dtype=model_dtype,
            )

        draft_state = self.draft_state_encoder(selected_layer_hidden)
        previous_token_ids = current_token_ids.to(device=keys.device, dtype=torch.long)
        logits = []
        for step_index in range(self.config.block_size):
            markov_state = self.markov_embedding(
                previous_token_ids=previous_token_ids,
                step_index=step_index,
                target_embed_tokens=target_embed_tokens,
                model_dtype=model_dtype,
            )
            query = self.memory_query(
                draft_state=draft_state,
                markov_state=markov_state,
            )
            memory_state = self.target_memory_reader(
                query=query,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
            )
            draft_state = self.draft_cell(
                draft_state=draft_state,
                memory_state=memory_state,
                markov_state=markov_state,
            )
            step_logits = self._project_logits(
                draft_state,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
            logits.append(step_logits)
            previous_token_ids = self._next_previous_token_ids(
                step_index=step_index,
                step_logits=step_logits,
                teacher_forcing_token_ids=teacher_forcing_token_ids,
                candidate_token_ids=candidate_token_ids,
            )
        return torch.stack(logits, dim=1)

    def _forward_global16_raw256(
        self,
        *,
        keys: Optional[torch.Tensor],
        values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        kv_source_indices: Optional[torch.Tensor],
        anchor_attention_mask: Optional[torch.Tensor],
        current_hidden_states: Optional[torch.Tensor],
        current_token_ids: torch.Tensor,
        current_position_ids: Optional[torch.Tensor],
        teacher_forcing_token_ids: Optional[torch.Tensor],
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
        global16_paged_context,
        return_pre_projection_hidden_states: bool = False,
        return_proposal_ids: bool = False,
        decoder_layers: Optional[nn.ModuleList] = None,
    ) -> torch.Tensor:
        """Run the retrain-only Global16 + Raw256 draft definition.

        Offline training supplies full Target tensors and receives a dense
        shared-K/V context.  SGLang serving supplies a paged context instead;
        it has no Python Target-K/V tensor to gather or copy.
        """

        if self.config.draft_mode != "parallel_shifted_qwen":
            raise RuntimeError("Global16 + Raw256 is defined only for Qwen draft layers")
        if self.global16_raw256_reference is None:
            raise RuntimeError("Global16 + Raw256 configuration has no reference module")
        model_dtype = next(self.parameters()).dtype
        if global16_paged_context is not None:
            if keys is not None or values is not None or attention_mask is not None:
                raise ValueError("paged Global16 serving must not pass gathered Target K/V")
            batch_size = int(current_token_ids.shape[0])
            context_batch_size = int(
                getattr(
                    getattr(global16_paged_context, "verify_context", None),
                    "batch_size",
                    batch_size,
                )
            )
            if context_batch_size != batch_size:
                raise ValueError("paged Global16 context and current-token batch differ")
            device = current_token_ids.device
            selected_layer_hidden = normalize_current_hidden_states(
                current_hidden_states=current_hidden_states,
                batch_size=batch_size,
                expected_layers=self.num_layers,
                hidden_size=self.config.hidden_size,
                dtype=model_dtype,
                device=device,
            )
            current_position_ids = self._validate_current_position_ids(
                current_position_ids,
                batch_size=batch_size,
                device=device,
                validate_values=not (
                    bool(
                        getattr(
                            global16_paged_context,
                            "cuda_graph_static_inputs",
                            False,
                        )
                    )
                    or bool(
                        getattr(
                            global16_paged_context,
                            "position_ids_are_scheduler_owned",
                            False,
                        )
                    )
                ),
            )
            teacher_forcing_token_ids = self._validate_teacher_forcing_token_ids(
                teacher_forcing_token_ids,
                batch_size=batch_size,
                device=device,
            )
            return self._forward_parallel_shifted_qwen_global16_raw256(
                selected_layer_hidden=selected_layer_hidden,
                current_token_ids=current_token_ids,
                current_position_ids=current_position_ids,
                teacher_forcing_token_ids=teacher_forcing_token_ids,
                target_final_norm=target_final_norm,
                target_embed_tokens=target_embed_tokens,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
                apply_markov_correction=apply_markov_correction,
                model_dtype=model_dtype,
                global16_context=global16_paged_context,
                return_pre_projection_hidden_states=return_pre_projection_hidden_states,
                return_proposal_ids=return_proposal_ids,
                decoder_layers=decoder_layers,
            )

        if return_proposal_ids:
            raise ValueError("return_proposal_ids requires paged Global16 serving")
        if keys is None or values is None:
            raise ValueError("offline Global16 training requires Target K/V tensors")
        attention_mask = normalize_kv_attention_mask(
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            expected_layers=self.num_layers,
            expected_heads=self.config.num_key_value_heads,
            expected_head_dim=self.config.head_dim,
        )
        if keys.dtype != model_dtype or values.dtype != model_dtype:
            raise ValueError(
                "Global16 + Raw256 requires Target K/V in the draft model dtype; "
                "casting would violate its borrowed-K/V contract"
            )
        if kv_source_indices is None:
            if anchor_attention_mask is not None:
                raise ValueError("anchor_attention_mask requires kv_source_indices")
            source_indices = torch.arange(keys.shape[0], device=keys.device)
            anchor_mask = attention_mask
            batch_size = keys.shape[0]
        else:
            if kv_source_indices.ndim != 1 or kv_source_indices.numel() == 0:
                raise ValueError("kv_source_indices must be a non-empty rank-1 tensor")
            source_indices = kv_source_indices.to(device=keys.device, dtype=torch.long)
            if torch.any(source_indices < 0) or torch.any(source_indices >= keys.shape[0]):
                raise ValueError("kv_source_indices contains an invalid source row")
            if anchor_attention_mask is None:
                raise ValueError("Global16 shared training requires anchor_attention_mask")
            if anchor_attention_mask.shape != (source_indices.numel(), keys.shape[-2]):
                raise ValueError("anchor_attention_mask must be [anchors, target_tokens]")
            anchor_mask = anchor_attention_mask.to(device=keys.device, dtype=torch.bool)
            batch_size = int(source_indices.numel())

        selected_layer_hidden = normalize_current_hidden_states(
            current_hidden_states=current_hidden_states,
            batch_size=batch_size,
            expected_layers=self.num_layers,
            hidden_size=self.config.hidden_size,
            dtype=model_dtype,
            device=keys.device,
        )
        current_position_ids = self._validate_current_position_ids(
            current_position_ids,
            batch_size=batch_size,
            device=keys.device,
        )
        teacher_forcing_token_ids = self._validate_teacher_forcing_token_ids(
            teacher_forcing_token_ids,
            batch_size=batch_size,
            device=keys.device,
        )
        dense_context = DenseGlobal16Raw256DraftContext(
            reference=self.global16_raw256_reference,
            keys=keys,
            values=values,
            anchor_attention_mask=anchor_mask,
            source_indices=source_indices,
        )
        return self._forward_parallel_shifted_qwen_global16_raw256(
            selected_layer_hidden=selected_layer_hidden,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            teacher_forcing_token_ids=teacher_forcing_token_ids,
            target_final_norm=target_final_norm,
            target_embed_tokens=target_embed_tokens,
            target_lm_head=target_lm_head,
            candidate_token_ids=candidate_token_ids,
            apply_markov_correction=apply_markov_correction,
            model_dtype=model_dtype,
            global16_context=dense_context,
            return_pre_projection_hidden_states=return_pre_projection_hidden_states,
            return_proposal_ids=False,
            decoder_layers=decoder_layers,
        )

    def _forward_parallel_shifted(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        selected_layer_hidden: torch.Tensor,
        current_token_ids: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
        model_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Predict ``x_{t+1:t+K}`` jointly from ``[x_t, MASK^(K-1)]``."""

        latest_target_state = self.draft_state_encoder(selected_layer_hidden)
        block_states = self.parallel_shifted_input(
            current_token_ids=current_token_ids,
            latest_target_state=latest_target_state,
            target_embed_tokens=target_embed_tokens,
            model_dtype=model_dtype,
        )
        query = self.parallel_memory_query(
            block_states=block_states,
            latest_target_state=latest_target_state,
        )
        memory_states = self.parallel_target_memory_reader(
            query=query,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
        )
        latest = latest_target_state[:, None, :].expand_as(block_states)
        block_states = self.parallel_fusion_cell(
            draft_state=block_states,
            memory_state=memory_states,
            markov_state=latest,
        )
        for mixer in self.parallel_mixers:
            block_states = mixer(block_states)

        base_logits = self._project_logits(
            block_states,
            target_final_norm=target_final_norm,
            target_lm_head=target_lm_head,
            candidate_token_ids=candidate_token_ids,
        )
        return self._apply_parallel_markov_correction(
            base_logits=base_logits,
            current_token_ids=current_token_ids,
            teacher_forcing_token_ids=teacher_forcing_token_ids,
            candidate_token_ids=candidate_token_ids,
            apply_markov_correction=apply_markov_correction,
        )

    def _forward_parallel_shifted_qwen(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_source_indices: Optional[torch.Tensor],
        anchor_attention_mask: Optional[torch.Tensor],
        selected_layer_hidden: torch.Tensor,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
        model_dtype: torch.dtype,
        precomputed_token_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run five Qwen3 layers without draft KV state."""

        if kv_source_indices is not None:
            if anchor_attention_mask is None:
                raise ValueError("anchor_attention_mask is required for shared target KV")
            if keys.device.type == "cuda":
                return self._forward_parallel_shifted_qwen_packed(
                    keys=keys,
                    values=values,
                    kv_source_indices=kv_source_indices,
                    anchor_attention_mask=anchor_attention_mask,
                    selected_layer_hidden=selected_layer_hidden,
                    current_token_ids=current_token_ids,
                    current_position_ids=current_position_ids,
                    teacher_forcing_token_ids=teacher_forcing_token_ids,
                    target_final_norm=target_final_norm,
                    target_embed_tokens=target_embed_tokens,
                    target_lm_head=target_lm_head,
                    candidate_token_ids=candidate_token_ids,
                    apply_markov_correction=apply_markov_correction,
                    model_dtype=model_dtype,
                    precomputed_token_states=precomputed_token_states,
                )
            # FlexAttention is CUDA-only. Preserve the exact semantics for CPU
            # validation by expanding only in this inexpensive fallback.
            keys = keys.index_select(0, kv_source_indices)
            values = values.index_select(0, kv_source_indices)
            attention_mask = anchor_attention_mask

        with profile_range("packed_layout_mask"):
            latest_target_state = self.draft_state_encoder(selected_layer_hidden)
            block_states = self.parallel_shifted_input(
                current_token_ids=current_token_ids,
                latest_target_state=latest_target_state,
                target_embed_tokens=target_embed_tokens,
                model_dtype=model_dtype,
                precomputed_token_states=precomputed_token_states,
            )
            position_ids = current_position_ids[:, None] + torch.arange(
                self.config.block_size,
                dtype=torch.long,
                device=current_position_ids.device,
            )[None, :]
        use_checkpointing = bool(
            self.config.qwen_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        with profile_range("qwen_decoder"):
            for layer_index, layer in enumerate(self.parallel_qwen_layers):
                layer_args = (
                    block_states,
                    keys[:, layer_index],
                    values[:, layer_index],
                    attention_mask,
                    position_ids,
                )
                if use_checkpointing:
                    block_states = checkpoint(
                        layer,
                        *layer_args,
                        use_reentrant=False,
                    )
                else:
                    block_states = layer(*layer_args)

        with profile_range("logits_loss"):
            base_logits = self._project_logits(
                block_states,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
        return self._apply_parallel_markov_correction(
            base_logits=base_logits,
            current_token_ids=current_token_ids,
            teacher_forcing_token_ids=teacher_forcing_token_ids,
            candidate_token_ids=candidate_token_ids,
            apply_markov_correction=apply_markov_correction,
        )

    def _forward_parallel_shifted_qwen_global16_raw256(
        self,
        *,
        selected_layer_hidden: torch.Tensor,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
        model_dtype: torch.dtype,
        global16_context,
        return_pre_projection_hidden_states: bool = False,
        return_proposal_ids: bool = False,
        decoder_layers: Optional[nn.ModuleList] = None,
    ) -> torch.Tensor:
        """Run Qwen draft layers with source-wise Global16/Raw256/Local7 reads."""

        with profile_range("global16_raw256_input"):
            latest_target_state = self.draft_state_encoder(selected_layer_hidden)
            block_states = self.parallel_shifted_input(
                current_token_ids=current_token_ids,
                latest_target_state=latest_target_state,
                target_embed_tokens=target_embed_tokens,
                model_dtype=model_dtype,
            )
            position_ids = current_position_ids[:, None] + torch.arange(
                self.config.block_size,
                dtype=torch.long,
                device=current_position_ids.device,
            )[None, :]
        use_checkpointing = bool(
            self.config.qwen_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        active_decoder_layers = (
            self.parallel_qwen_layers
            if decoder_layers is None
            else decoder_layers
        )
        if len(active_decoder_layers) != self.num_layers:
            raise ValueError(
                "Global16 decoder layer count must match selected target layers"
            )
        with profile_range("global16_raw256_qwen_decoder"):
            for layer_index, layer in enumerate(active_decoder_layers):
                if use_checkpointing:
                    block_states = checkpoint(
                        lambda states, decoder_layer=layer, index=layer_index: decoder_layer.forward_global16_raw256(
                            states,
                            position_ids=position_ids,
                            global16_context=global16_context,
                            layer_index=index,
                        ),
                        block_states,
                        use_reentrant=False,
                    )
                else:
                    block_states = layer.forward_global16_raw256(
                        block_states,
                        position_ids=position_ids,
                        global16_context=global16_context,
                        layer_index=layer_index,
                    )
        if return_pre_projection_hidden_states:
            return block_states
        if return_proposal_ids:
            if teacher_forcing_token_ids is not None:
                raise ValueError(
                    "proposal-only serving does not accept teacher forcing"
                )
            if (
                candidate_token_ids is None
                and _env_flag("DFK_GLOBAL16_STREAMING_VOCAB_ARGMAX")
            ):
                streamed_ids = self._sample_full_vocab_proposal_ids_streaming(
                    block_states=block_states,
                    current_token_ids=current_token_ids,
                    target_final_norm=target_final_norm,
                    target_lm_head=target_lm_head,
                    apply_markov_correction=apply_markov_correction,
                )
                if streamed_ids is not None:
                    return streamed_ids
            with profile_range("logits_loss"):
                base_logits = self._project_logits(
                    block_states,
                    target_final_norm=target_final_norm,
                    target_lm_head=target_lm_head,
                    candidate_token_ids=candidate_token_ids,
                )
            if self.parallel_markov_head is not None and apply_markov_correction:
                return self.parallel_markov_head.sample_block_token_ids(
                    base_logits,
                    first_previous_token_ids=current_token_ids,
                    temperature=0.0,
                    candidate_token_ids=candidate_token_ids,
                )
            sampled_indices = base_logits.argmax(dim=-1)
            if candidate_token_ids is None:
                return sampled_indices
            return candidate_token_ids.to(
                device=sampled_indices.device
            ).index_select(0, sampled_indices.reshape(-1)).view_as(
                sampled_indices
            )
        with profile_range("logits_loss"):
            base_logits = self._project_logits(
                block_states,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
        return self._apply_parallel_markov_correction(
            base_logits=base_logits,
            current_token_ids=current_token_ids,
            teacher_forcing_token_ids=teacher_forcing_token_ids,
            candidate_token_ids=candidate_token_ids,
            apply_markov_correction=apply_markov_correction,
        )

    def _forward_parallel_shifted_qwen_packed(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_source_indices: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        selected_layer_hidden: torch.Tensor,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
        model_dtype: torch.dtype,
        precomputed_token_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pack all anchors while keeping target K/V once per source sequence."""

        with profile_range("packed_layout_mask"):
            latest_target_state = self.draft_state_encoder(selected_layer_hidden)
            flat_block_states = self.parallel_shifted_input(
                current_token_ids=current_token_ids,
                latest_target_state=latest_target_state,
                target_embed_tokens=target_embed_tokens,
                model_dtype=model_dtype,
                precomputed_token_states=precomputed_token_states,
            )
            flat_position_ids = current_position_ids[:, None] + torch.arange(
                self.config.block_size,
                dtype=torch.long,
                device=current_position_ids.device,
            )[None, :]

        source_batch_size = keys.shape[0]
        anchor_count = kv_source_indices.numel()
        with profile_range("packed_layout_mask"):
            counts = torch.bincount(
                kv_source_indices,
                minlength=source_batch_size,
            )
            max_anchors = int(counts.max().item())
            slot_indices = torch.empty_like(kv_source_indices)
            for source_index in range(source_batch_size):
                source_rows = torch.nonzero(
                    kv_source_indices == source_index,
                    as_tuple=False,
                ).flatten()
                slot_indices[source_rows] = torch.arange(
                    source_rows.numel(),
                    device=kv_source_indices.device,
                )
            packed_indices = kv_source_indices * max_anchors + slot_indices

        block_size = self.config.block_size
        hidden_size = self.config.hidden_size
        with profile_range("packed_layout_mask"):
            packed_block_states = flat_block_states.new_zeros(
                source_batch_size * max_anchors,
                block_size,
                hidden_size,
            ).index_copy(0, packed_indices, flat_block_states)
            packed_position_ids = flat_position_ids.new_zeros(
                source_batch_size * max_anchors,
                block_size,
            ).index_copy(0, packed_indices, flat_position_ids)
            target_length = keys.shape[-2]
            packed_prefix_mask = anchor_attention_mask.new_zeros(
                source_batch_size * max_anchors,
                target_length,
            ).index_copy(0, packed_indices, anchor_attention_mask)
            anchor_valid_mask = torch.zeros(
                source_batch_size * max_anchors,
                dtype=torch.bool,
                device=keys.device,
            )
            anchor_valid_mask.index_fill_(0, packed_indices, True)

            packed_block_states = packed_block_states.view(
                source_batch_size,
                max_anchors * block_size,
                hidden_size,
            )
            packed_position_ids = packed_position_ids.view(
                source_batch_size,
                max_anchors * block_size,
            )
            packed_prefix_mask = packed_prefix_mask.view(
                source_batch_size,
                max_anchors,
                target_length,
            )
            anchor_valid_mask = anchor_valid_mask.view(
                source_batch_size,
                max_anchors,
            )
            block_mask = _create_packed_qwen_attention_mask(
                anchor_prefix_mask=packed_prefix_mask,
                anchor_valid_mask=anchor_valid_mask,
                block_size=block_size,
            )

        use_checkpointing = bool(
            self.config.qwen_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        with profile_range("qwen_decoder"):
            for layer_index, layer in enumerate(self.parallel_qwen_layers):
                layer_args = (
                    packed_block_states,
                    keys[:, layer_index],
                    values[:, layer_index],
                    packed_position_ids,
                    block_mask,
                )
                if use_checkpointing:
                    packed_block_states = checkpoint(
                        layer.forward_packed,
                        *layer_args,
                        use_reentrant=False,
                    )
                else:
                    packed_block_states = layer.forward_packed(*layer_args)

        with profile_range("packed_layout_mask"):
            packed_block_states = packed_block_states.view(
                source_batch_size,
                max_anchors,
                block_size,
                hidden_size,
            )
            flat_block_states = packed_block_states[
                kv_source_indices,
                slot_indices,
            ]
            if flat_block_states.shape[0] != anchor_count:
                raise RuntimeError("failed to unpack every DraftFreeKV training anchor")
        with profile_range("logits_loss"):
            base_logits = self._project_logits(
                flat_block_states,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
        return self._apply_parallel_markov_correction(
            base_logits=base_logits,
            current_token_ids=current_token_ids,
            teacher_forcing_token_ids=teacher_forcing_token_ids,
            candidate_token_ids=candidate_token_ids,
            apply_markov_correction=apply_markov_correction,
        )

    def _apply_parallel_markov_correction(
        self,
        *,
        base_logits: torch.Tensor,
        current_token_ids: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        candidate_token_ids: Optional[torch.Tensor],
        apply_markov_correction: bool,
    ) -> torch.Tensor:
        if self.parallel_markov_head is None or not apply_markov_correction:
            return base_logits

        current_token_ids = current_token_ids.to(
            device=base_logits.device,
            dtype=torch.long,
        )
        if teacher_forcing_token_ids is not None:
            previous_token_ids = torch.cat(
                [current_token_ids[:, None], teacher_forcing_token_ids[:, :-1]],
                dim=1,
            )
            return self.parallel_markov_head.apply_block_logits(
                base_logits,
                previous_token_ids=previous_token_ids,
                candidate_token_ids=candidate_token_ids,
            )

        _, corrected_logits = self.parallel_markov_head.sample_block_tokens(
            base_logits,
            first_previous_token_ids=current_token_ids,
            temperature=0.0,
            candidate_token_ids=candidate_token_ids,
        )
        return corrected_logits

    def sample_parallel_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_previous_token_ids: torch.Tensor,
        temperature: float = 0.0,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the optional Markov correction with true sampled-token feedback."""

        if self.config.draft_mode not in PARALLEL_SHIFTED_DRAFT_MODES:
            raise ValueError("sample_parallel_tokens requires a parallel shifted mode")
        if self.parallel_markov_head is not None:
            return self.parallel_markov_head.sample_block_tokens(
                base_logits,
                first_previous_token_ids=first_previous_token_ids,
                temperature=temperature,
                candidate_token_ids=candidate_token_ids,
            )
        if temperature <= 0:
            sampled_indices = base_logits.argmax(dim=-1)
        else:
            probabilities = (base_logits.float() / temperature).softmax(dim=-1)
            sampled_indices = torch.multinomial(
                probabilities.view(-1, probabilities.shape[-1]),
                1,
            ).view(*probabilities.shape[:-1])
        if candidate_token_ids is None:
            sampled_tokens = sampled_indices
        else:
            sampled_tokens = candidate_token_ids.to(
                device=sampled_indices.device
            ).index_select(0, sampled_indices.reshape(-1)).view_as(sampled_indices)
        return sampled_tokens, base_logits

    def _project_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        target_final_norm: Optional[nn.Module],
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self._prepare_logits_hidden(
            hidden_states,
            target_final_norm=target_final_norm,
        )
        weight = getattr(target_lm_head, "weight", None)
        if isinstance(weight, torch.Tensor):
            weight = weight.detach()
            bias = getattr(target_lm_head, "bias", None)
            bias = None if bias is None else bias.detach()
            if candidate_token_ids is not None:
                indices = candidate_token_ids.to(device=weight.device)
                weight = weight.index_select(0, indices)
                if bias is not None:
                    bias = bias.index_select(0, indices)
            return F.linear(hidden_states.to(dtype=weight.dtype), weight, bias).float()
        if candidate_token_ids is not None:
            raise ValueError("candidate_token_ids require a weight-backed target_lm_head")
        return target_lm_head(hidden_states).float()

    def _prepare_logits_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        target_final_norm: Optional[nn.Module],
    ) -> torch.Tensor:
        """Apply the frozen target final norm before the vocabulary projection."""
        if target_final_norm is not None:
            norm_dtype = self._module_dtype(target_final_norm)
            original_shape = hidden_states.shape
            flattened_hidden_states = hidden_states.reshape(-1, original_shape[-1])
            norm_input = flattened_hidden_states.to(dtype=norm_dtype)
            norm_weight = getattr(target_final_norm, "weight", None)
            variance_epsilon = getattr(target_final_norm, "variance_epsilon", None)
            if (
                isinstance(norm_weight, torch.Tensor)
                and norm_weight.ndim == 1
                and variance_epsilon is not None
            ):
                # SGLang's target RMSNorm is an inference-only fused kernel.  Its
                # output does not retain the Draft hidden-state autograd path, so
                # use the equivalent PyTorch operation with frozen target weights.
                if hasattr(F, "rms_norm"):
                    norm_output = F.rms_norm(
                        norm_input,
                        (original_shape[-1],),
                        norm_weight.detach(),
                        float(variance_epsilon),
                    )
                else:  # pragma: no cover - compatibility for older CPU torch
                    norm_output = norm_input * torch.rsqrt(
                        norm_input.pow(2).mean(dim=-1, keepdim=True)
                        + float(variance_epsilon)
                    )
                    norm_output = norm_output * norm_weight.detach()
            else:
                norm_output = target_final_norm(norm_input)
            hidden_states = norm_output.reshape(original_shape).to(
                dtype=hidden_states.dtype
            )
        return hidden_states

    def _sample_full_vocab_proposal_ids_streaming(
        self,
        *,
        block_states: torch.Tensor,
        current_token_ids: torch.Tensor,
        target_final_norm: Optional[nn.Module],
        target_lm_head: nn.Module,
        apply_markov_correction: bool,
    ) -> Optional[torch.Tensor]:
        """Select greedy proposal ids without retaining a full vocabulary tensor."""

        weight = getattr(target_lm_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return None
        if block_states.ndim != 3:
            raise ValueError(
                "streaming proposal states must have shape [batch, block, hidden]"
            )
        if current_token_ids.shape != (block_states.shape[0],):
            raise ValueError("streaming proposal current tokens must match batch")

        weight = weight.detach()
        bias = getattr(target_lm_head, "bias", None)
        bias = None if bias is None else bias.detach()
        states = self._prepare_logits_hidden(
            block_states,
            target_final_norm=target_final_norm,
        )
        markov_head = self.parallel_markov_head if apply_markov_correction else None
        vocab_size = int(weight.shape[0])
        chunk_size = min(32768, vocab_size)
        previous_token_ids = current_token_ids.to(
            device=states.device,
            dtype=torch.long,
        )
        sampled_tokens = []

        for step_index in range(states.shape[1]):
            hidden = states[:, step_index].to(dtype=weight.dtype)
            latent = (
                None
                if markov_head is None
                else markov_head.markov_w1(previous_token_ids)
            )
            best_scores = torch.full(
                (hidden.shape[0],),
                float("-inf"),
                dtype=torch.float32,
                device=hidden.device,
            )
            best_tokens = torch.zeros(
                hidden.shape[0],
                dtype=torch.long,
                device=hidden.device,
            )
            for start in range(0, vocab_size, chunk_size):
                end = min(start + chunk_size, vocab_size)
                chunk_bias = None if bias is None else bias[start:end]
                scores = F.linear(hidden, weight[start:end], chunk_bias).float()
                if latent is not None:
                    scores.add_(
                        F.linear(
                            latent,
                            markov_head.markov_w2.weight[start:end],
                        ).to(dtype=scores.dtype)
                    )
                chunk_scores, chunk_indices = scores.max(dim=-1)
                take = chunk_scores > best_scores
                best_scores = torch.where(take, chunk_scores, best_scores)
                best_tokens = torch.where(
                    take,
                    chunk_indices.to(torch.long) + start,
                    best_tokens,
                )
            sampled_tokens.append(best_tokens)
            previous_token_ids = best_tokens
        return torch.stack(sampled_tokens, dim=1)

    def _next_previous_token_ids(
        self,
        *,
        step_index: int,
        step_logits: torch.Tensor,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        candidate_token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if teacher_forcing_token_ids is not None:
            return teacher_forcing_token_ids[:, step_index]
        predicted = step_logits.argmax(dim=-1)
        if candidate_token_ids is None:
            return predicted
        return candidate_token_ids.to(device=predicted.device).index_select(0, predicted)

    def _validate_teacher_forcing_token_ids(
        self,
        teacher_forcing_token_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if teacher_forcing_token_ids is None:
            return None
        expected_shape = (batch_size, self.config.block_size)
        if teacher_forcing_token_ids.shape != expected_shape:
            raise ValueError("teacher_forcing_token_ids must have shape [batch, block_size]")
        return teacher_forcing_token_ids.to(device=device, dtype=torch.long)

    @staticmethod
    def _validate_current_position_ids(
        current_position_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        validate_values: bool = True,
    ) -> torch.Tensor:
        if current_position_ids is None:
            raise ValueError(
                "current_position_ids are required for parallel_shifted_qwen"
            )
        if current_position_ids.shape == (batch_size, 1):
            current_position_ids = current_position_ids[:, 0]
        if current_position_ids.shape != (batch_size,):
            raise ValueError("current_position_ids must have shape [batch]")
        current_position_ids = current_position_ids.to(
            device=device,
            dtype=torch.long,
        )
        # Graph inputs are scheduler-owned fixed buffers. Turning this device
        # predicate into a Python boolean synchronizes the GPU and is illegal
        # during capture. Shape, type, and device checks remain active.
        if validate_values and torch.any(current_position_ids < 0):
            raise ValueError("current_position_ids must be non-negative")
        return current_position_ids

    def _enforce_global_query_trainability(self) -> None:
        """Keep fixed query priors out of optimizers after model initialization."""

        reference = getattr(self, "global16_raw256_reference", None)
        if reference is None:
            return
        query = reference.global_query_generator.learned_query
        query.requires_grad_(self.config.global_query_mode == "learned")

    @torch.no_grad()
    def _validate_fixed_global_query_geometry(self) -> None:
        """Reject mislabeled legacy weights in a fixed-query checkpoint."""

        if self.config.global_query_mode != "fixed_equicorrelated":
            return
        reference = getattr(self, "global16_raw256_reference", None)
        if reference is None:
            raise RuntimeError("fixed Global16 query mode has no reference module")
        query = reference.global_query_generator.learned_query.detach().float()
        norms = query.norm(dim=-1, keepdim=True)
        if bool(torch.any(norms <= torch.finfo(query.dtype).eps).item()):
            raise RuntimeError("fixed Global16 query checkpoint contains a zero row")
        normalized = query / norms
        gram = torch.einsum("lhsd,lhtd->lhst", normalized, normalized)
        slots = int(self.config.global_memory_slots)
        correlation = float(self.config.global_query_equicorrelation)
        expected = (
            (1.0 - correlation) * torch.eye(slots, device=gram.device)
            + correlation * torch.ones((slots, slots), device=gram.device)
        )
        max_error = float((gram - expected).abs().max().item())
        if not math.isfinite(max_error) or max_error > 5e-3:
            raise RuntimeError(
                "fixed Global16 query checkpoint violates the configured "
                "row-normalized equicorrelation Gram matrix: "
                f"max_error={max_error:.6g}"
            )

    def _validate_fixed_global_query_loading_info(
        self,
        loading_info: dict[str, object],
    ) -> None:
        """Require the frozen query tensor to come from the checkpoint itself."""

        if self.config.global_query_mode != "fixed_equicorrelated":
            return
        reference = getattr(self, "global16_raw256_reference", None)
        if reference is None:
            raise RuntimeError("fixed Global16 query mode has no reference module")
        query = reference.global_query_generator.learned_query
        query_key = next(
            (
                name
                for name, parameter in self.named_parameters()
                if parameter is query
            ),
            None,
        )
        if query_key is None:
            raise RuntimeError("cannot identify the fixed Global16 query checkpoint key")
        missing = {str(key) for key in (loading_info.get("missing_keys") or [])}
        mismatched = {
            str(value[0]) if isinstance(value, (tuple, list)) and value else str(value)
            for value in (loading_info.get("mismatched_keys") or [])
        }
        if query_key in missing or query_key in mismatched:
            reason = "missing" if query_key in missing else "shape-mismatched"
            raise RuntimeError(
                f"fixed Global16 query checkpoint key is {reason}: {query_key}; "
                "refusing to substitute a newly initialized query pattern"
            )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        retrofit_requested = _env_flag("DFK_GLOBAL_QUERY_RETROFIT")
        source_query_mode = None
        if retrofit_requested:
            source = kwargs.get("pretrained_model_name_or_path")
            if source is None and args:
                source = args[0]
            if source is None:
                raise RuntimeError(
                    "DFK_GLOBAL_QUERY_RETROFIT requires a checkpoint path"
                )
            try:
                source_config = DraftFreeKVConfig.from_pretrained(source)
            except Exception as exc:
                raise RuntimeError(
                    "DFK_GLOBAL_QUERY_RETROFIT requires a readable checkpoint config"
                ) from exc
            source_query_mode = str(
                getattr(source_config, "global_query_mode", "learned")
            )
            if source_query_mode != "learned":
                raise RuntimeError(
                    "DFK_GLOBAL_QUERY_RETROFIT is only valid for a learned-query "
                    f"initial checkpoint, got {source_query_mode!r}"
                )
            requested_mode = kwargs.get("global_query_mode")
            if requested_mode is not None and str(requested_mode) != "fixed_equicorrelated":
                raise RuntimeError(
                    "DFK_GLOBAL_QUERY_RETROFIT requires fixed_equicorrelated mode"
                )
            kwargs["global_query_mode"] = "fixed_equicorrelated"
            kwargs.setdefault(
                "global_query_equicorrelation",
                float(os.environ.get("DFK_GLOBAL_QUERY_EQUICORRELATION", "0.4")),
            )
            kwargs.setdefault("global_query_slot_mass_mode", "system")
        caller_requested_loading_info = bool(kwargs.get("output_loading_info", False))
        kwargs["output_loading_info"] = True
        result = super().from_pretrained(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("from_pretrained did not return checkpoint loading metadata")
        model, loading_info = result
        if not isinstance(loading_info, dict):
            raise RuntimeError("from_pretrained returned malformed checkpoint loading metadata")
        if retrofit_requested:
            reference = getattr(model, "global16_raw256_reference", None)
            if reference is None or source_query_mode != "learned":
                raise RuntimeError(
                    "DFK_GLOBAL_QUERY_RETROFIT checkpoint has no learned Global16 query"
                )
            with torch.no_grad():
                query = reference.global_query_generator.learned_query
                query.copy_(
                    _fixed_equicorrelated_query_pattern(
                        query,
                        correlation=float(model.config.global_query_equicorrelation),
                    )
                )
        model._enforce_global_query_trainability()
        model._validate_fixed_global_query_loading_info(loading_info)
        model._validate_fixed_global_query_geometry()
        return result if caller_requested_loading_info else model

    @staticmethod
    def _freeze_target_module(module: Optional[nn.Module]) -> None:
        if module is not None:
            module.eval()
            module.requires_grad_(False)

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.float32


__all__ = [
    "DSparkVanillaMarkovHead",
    "DraftFreeKVConfig",
    "DraftFreeKVModel",
    "DraftStateEncoder",
    "MarkovEmbedding",
    "RMSNorm",
    "TargetMemoryQuery",
    "TargetMemoryReader",
    "DraftCell",
    "LowRankLinear",
    "ParallelBlockMixerLayer",
    "ParallelShiftedInput",
    "ParallelTargetMemoryQuery",
    "ParallelTargetMemoryReader",
    "Qwen3KVFreeParallelAttention",
    "Qwen3KVFreeParallelDecoderLayer",
    "Qwen3SwiGLUMLP",
    "resolve_target_layer_ids",
]
