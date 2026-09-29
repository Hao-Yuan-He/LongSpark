"""Paged-FA3 runtime for the retrain-only Global16 + Raw256 contract.

This module owns the SGLang-specific *transport* of the shared mathematical
contract in ``specforge.modeling.draft.global16_raw256``.  It deliberately
duplicates page-table metadata but never gathers, casts, or concatenates Target
K/V.  The Target verify query rows and sixteen Global queries are issued through
one FA3 invocation per selected Target layer, all pointing at the same paged
K/V buffers.

The training implementation uses the PyTorch reference because FA3 has no
backward path.  Both paths exchange normalized attention outputs and
log-normalizers through the same online-softmax merge rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch


GLOBAL16_SLOTS = 16
RAW256_ROWS = 256
LOCAL7_ROWS = 7
MAX_ACCEPTED_ROWS = 8


def select_global16_prefill_rows(reqs, chunked_req) -> tuple[int, ...]:
    """Return scheduler rows whose current EXTEND reaches the full prompt.

    ``chunked_req`` is the scheduler's authoritative marker for a middle
    chunk.  The range check is an independent fail-closed guard for callers
    that construct a batch with a stale marker or a prefix-cache boundary.
    """

    active_rows = []
    for row, req in enumerate(reqs):
        if req is chunked_req:
            continue
        extend_range = getattr(req, "extend_range", None)
        full_fill_ids = getattr(req, "full_untruncated_fill_ids", None)
        if extend_range is None or full_fill_ids is None:
            raise RuntimeError(
                "Global16 prefill selection requires scheduler-owned "
                "extend_range and full_untruncated_fill_ids"
            )
        extend_end = int(extend_range.end)
        full_length = len(full_fill_ids)
        if extend_end < 0 or extend_end > full_length:
            raise RuntimeError("Global16 prefill range lies outside the full prompt")
        if extend_end != full_length:
            continue
        active_rows.append(row)
    return tuple(active_rows)

_DRAFT_MERGE_TOKEN_MAJOR_OUTPUT = (
    os.getenv("DFK_GLOBAL16_DRAFT_MERGE_TOKEN_MAJOR_OUTPUT", "0") == "1"
)
_DRAFT_FUSED_ALL_ATTENTION = (
    os.getenv("DFK_GLOBAL16_DRAFT_FUSED_ALL_ATTENTION", "0") == "1"
)
_DRAFT_SKIP_REDUNDANT_MERGE_CHECK = (
    os.getenv("DFK_GLOBAL16_SKIP_REDUNDANT_MERGE_CHECK", "0") == "1"
)


def _allocate_fused_draft_output(query: torch.Tensor) -> torch.Tensor:
    """Allocate the reusable merge result in the next projection's layout."""

    if not _DRAFT_MERGE_TOKEN_MAJOR_OUTPUT:
        return torch.empty_like(query)
    batch_size, heads, query_rows, head_dim = query.shape
    return torch.empty(
        (batch_size, query_rows, heads, head_dim),
        dtype=query.dtype,
        device=query.device,
    ).transpose(1, 2)


def _reference_helpers():
    """Import the shared math only when the DraftFreeKV mode is activated.

    ``flashattention_backend`` is used by ordinary SGLang deployments too;
    importing this module must not make the optional SpecForge package a global
    SGLang dependency.
    """

    from specforge.modeling.draft.global16_raw256 import (
        attention_statistics_from_4d,
        merge_normalized_attention,
    )

    return attention_statistics_from_4d, merge_normalized_attention


def _incremental_helpers():
    from sglang.srt.speculative.global16_raw256_incremental import (
        cache_position_scores,
        merge_accepted_positions_,
    )

    return cache_position_scores, merge_accepted_positions_


def _draft_fused_helpers():
    from sglang.srt.speculative.global16_raw256_draft_fused import (
        can_use_fused_global_local_raw_merge,
        fused_global_local_raw_merge,
    )

    return (
        can_use_fused_global_local_raw_merge,
        fused_global_local_raw_merge,
    )


def _draft_all_attention_helpers():
    from sglang.srt.speculative.global16_raw256_draft_all_attention import (
        can_use_fused_global_local_raw_attention,
        fused_global_local_raw_attention,
    )

    return (
        can_use_fused_global_local_raw_attention,
        fused_global_local_raw_attention,
    )


def _raw_page_table_helper():
    from sglang.srt.speculative.global16_raw256_draft_fused import (
        build_raw256_page_table,
    )

    return build_raw256_page_table


def _storage_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _flat_lse(
    lse: torch.Tensor,
    *,
    token_count: int,
    num_heads: int,
) -> torch.Tensor:
    """Normalize FA3's LSE layout to ``[tokens, heads]``.

    FA3 normally returns ``[heads, tokens]`` for varlen/paged calls.  Keeping a
    defensive adapter here lets mechanism tests use an ordinary ``[tokens,
    heads]`` fake without changing the production contract.
    """

    if lse.ndim == 2:
        if lse.shape == (num_heads, token_count):
            return lse.transpose(0, 1)
        if lse.shape == (token_count, num_heads):
            return lse
    if lse.ndim == 3 and lse.shape[0] == 1:
        squeezed = lse[0]
        if squeezed.shape == (num_heads, token_count):
            return squeezed.transpose(0, 1)
        if squeezed.shape == (token_count, num_heads):
            return squeezed
    raise RuntimeError(
        "unexpected FA3 logsumexp shape "
        f"{tuple(lse.shape)} for tokens={token_count}, heads={num_heads}"
    )


def _unpack_fa3_result(
    result,
    *,
    token_count: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Global16 FA3 calls must return output and logsumexp")
    output, lse = result[:2]
    if output.ndim != 3 or output.shape[:2] != (token_count, num_heads):
        raise RuntimeError(
            "unexpected FA3 output shape "
            f"{tuple(output.shape)} for tokens={token_count}, heads={num_heads}"
        )
    return output, _flat_lse(lse, token_count=token_count, num_heads=num_heads)


def _cumulative_lengths(lengths: torch.Tensor) -> torch.Tensor:
    if lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ValueError("FA3 sequence lengths must be a rank-1 int32 tensor")
    return torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))


def _statistics_as_normalized(statistics) -> tuple[torch.Tensor, torch.Tensor]:
    visible = statistics.normalizer > 0
    safe_normalizer = statistics.normalizer.clamp_min(
        torch.finfo(statistics.normalizer.dtype).tiny
    )
    lse = statistics.max_logits + torch.log(safe_normalizer)
    lse = torch.where(visible, lse, torch.full_like(lse, float("-inf")))
    return statistics.output(), lse


def validate_global16_raw256_runtime_contract(
    *,
    page_size: int,
    selected_layer_ids: Sequence[int],
    configured_layer_ids: Sequence[int],
    raw_window_size: int,
    configured_raw_rows: int,
    model_local_rows: int,
    runtime_verify_width: int,
    verify_token_num: int,
    tp_size: int,
) -> None:
    """Fail closed when a serving launch drifts from the V1 checkpoint contract."""

    if int(page_size) != 1:
        raise NotImplementedError(
            "Global16 + Raw256 P0 requires --page-size 1 for direct accepted K/V rows"
        )
    if tuple(int(layer_id) for layer_id in selected_layer_ids) != tuple(
        int(layer_id) for layer_id in configured_layer_ids
    ):
        raise ValueError(
            "Global16 selected layers must exactly match the checkpoint configuration"
        )
    if int(raw_window_size) != int(configured_raw_rows):
        raise ValueError("Global16 Raw256 window must exactly match the checkpoint contract")
    if (
        int(model_local_rows) != LOCAL7_ROWS
        or int(runtime_verify_width) != LOCAL7_ROWS + 1
        or int(verify_token_num) > MAX_ACCEPTED_ROWS
    ):
        raise ValueError(
            "Global16 + Raw256 requires model Local7, runtime verify width 8, "
            "and at most eight verify rows"
        )
    if int(tp_size) != 1:
        raise NotImplementedError(
            "Global16 + Raw256 P0 supports TP1 serving; four-GPU training uses data parallelism"
        )


@dataclass(frozen=True)
class Global16Raw256FAEvidence:
    """Concrete evidence that the joint call shares physical Target K/V."""

    layer_id: int
    fa3_invocations: int
    key_cache_storage_ptr: int
    value_cache_storage_ptr: int
    target_query_rows: int
    global_query_rows: int
    duplicated_page_table_rows: int
    combined_kv_bank_bytes: int = 0
    request_layer_base_inits: int = 0
    prefix_rows: int = 0
    query_key_pairs: int = 0


@dataclass(frozen=True)
class Global16Raw256DraftEvidence:
    """Serving-draft evidence; page-table work is not a Target K/V copy."""

    layer_id: int
    raw_key_cache_storage_ptr: int
    raw_value_cache_storage_ptr: int
    raw_rows: int
    global_rows: int = GLOBAL16_SLOTS
    local_rows: int = LOCAL7_ROWS
    combined_kv_bank_bytes: int = 0


@dataclass
class Global16Raw256LayerState:
    """Normalized Global16 attention state for one Target layer."""

    query: torch.Tensor  # [B, Hq, 16, D]
    output: torch.Tensor  # [B, Hq, 16, D]
    logsumexp: torch.Tensor  # [B, Hq, 16]

    def merge_accepted_delta(
        self,
        *,
        output: torch.Tensor,
        logsumexp: torch.Tensor,
    ) -> None:
        _, merge_normalized_attention = _reference_helpers()
        merged_output, merged_lse = merge_normalized_attention(
            ((self.output, self.logsumexp), (output, logsumexp))
        )
        self.output = merged_output
        self.logsumexp = merged_lse


class Global16Raw256VerifyContext:
    """One verify transaction: base scan once, then one accepted delta.

    The context is attached to ``DraftFreeKVVerifyInput``.  It is intentionally
    request-batch scoped: no state is built for hypothetical accept lengths.
    """

    def __init__(
        self,
        *,
        selected_layer_ids: Sequence[int],
        global_queries: torch.Tensor,
        confirmed_prefix_lens: torch.Tensor,
        score_workspace: Optional[torch.Tensor] = None,
        accept_offset_workspace: Optional[torch.Tensor] = None,
        request_slots: Optional[torch.Tensor] = None,
        source_batch_rows: Optional[torch.Tensor] = None,
        source_batch_rows_host: Optional[Sequence[int]] = None,
        validate_confirmed_prefix_values: bool = True,
        collect_fixed_cost_audit: bool = False,
    ) -> None:
        selected_layer_ids = tuple(int(layer_id) for layer_id in selected_layer_ids)
        if not selected_layer_ids or len(set(selected_layer_ids)) != len(selected_layer_ids):
            raise ValueError("selected_layer_ids must be non-empty and unique")
        if global_queries.ndim != 5 or global_queries.shape[1] != len(selected_layer_ids):
            raise ValueError(
                "global_queries must have shape [batch, selected_layers, heads, 16, head_dim]"
            )
        if global_queries.shape[-2] != GLOBAL16_SLOTS:
            raise ValueError("Global16 runtime requires exactly sixteen queries")
        if confirmed_prefix_lens.shape != (global_queries.shape[0],):
            raise ValueError("confirmed_prefix_lens must have one entry per request")
        if confirmed_prefix_lens.dtype not in {torch.int32, torch.int64}:
            raise ValueError("confirmed_prefix_lens must use an integer dtype")
        if validate_confirmed_prefix_values and torch.any(
            confirmed_prefix_lens <= 0
        ):
            raise ValueError("each Global16 query must have a non-empty confirmed prefix")
        self.selected_layer_ids = selected_layer_ids
        self.global_queries = global_queries
        self.confirmed_prefix_lens = confirmed_prefix_lens.to(
            device=global_queries.device,
            dtype=torch.int32,
        )
        if source_batch_rows is not None and (
            source_batch_rows.ndim != 1
            or source_batch_rows.shape[0] != global_queries.shape[0]
        ):
            raise ValueError("source_batch_rows must have one row per Global16 request")
        if source_batch_rows_host is None:
            if source_batch_rows is None:
                source_batch_rows_host = tuple(range(global_queries.shape[0]))
            elif source_batch_rows.device.type != "cpu":
                raise ValueError(
                    "non-CPU source_batch_rows requires source_batch_rows_host"
                )
            else:
                source_batch_rows_host = tuple(
                    int(row) for row in source_batch_rows.tolist()
                )
        else:
            source_batch_rows_host = tuple(int(row) for row in source_batch_rows_host)
        if len(source_batch_rows_host) != global_queries.shape[0]:
            raise ValueError(
                "source_batch_rows_host must have one row per Global16 request"
            )
        if any(row < 0 for row in source_batch_rows_host) or any(
            left >= right
            for left, right in zip(
                source_batch_rows_host,
                source_batch_rows_host[1:],
            )
        ):
            raise ValueError(
                "source_batch_rows_host must be strictly increasing and non-negative"
            )
        if source_batch_rows is not None and source_batch_rows.device.type == "cpu":
            observed_rows = tuple(int(row) for row in source_batch_rows.tolist())
            if observed_rows != source_batch_rows_host:
                raise ValueError(
                    "source_batch_rows and source_batch_rows_host must agree"
                )
        # Canonicalize the device mapping from the validated host tuple.  This
        # prevents the Target-query and page-table paths from observing two
        # different row mappings without synchronizing a CUDA tensor.
        self.source_batch_rows_host = source_batch_rows_host
        self.source_batch_rows = torch.tensor(
            source_batch_rows_host,
            device=global_queries.device,
            dtype=torch.long,
        )
        self.collect_fixed_cost_audit = bool(collect_fixed_cost_audit)
        self.fixed_cost_score_events: dict[
            int, tuple[torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._layer_index = {layer_id: index for index, layer_id in enumerate(selected_layer_ids)}
        if score_workspace is not None:
            expected = (
                global_queries.shape[0],
                len(selected_layer_ids),
                global_queries.shape[2],
                GLOBAL16_SLOTS,
                MAX_ACCEPTED_ROWS,
            )
            if score_workspace.shape != expected or score_workspace.dtype != torch.float32:
                raise ValueError(
                    f"Global16 score workspace must be FP32 {expected}, "
                    f"got {tuple(score_workspace.shape)} {score_workspace.dtype}"
                )
            if score_workspace.device != global_queries.device:
                raise ValueError("Global16 score workspace must share the query device")
        if accept_offset_workspace is not None:
            if (
                accept_offset_workspace.shape != (global_queries.shape[0],)
                or accept_offset_workspace.dtype != torch.int32
                or accept_offset_workspace.device != global_queries.device
            ):
                raise ValueError(
                    "Global16 accept-offset workspace must be one int32 row per request"
                )
        if request_slots is not None:
            if (
                request_slots.shape != (global_queries.shape[0],)
                or request_slots.dtype != torch.int32
                or request_slots.device != global_queries.device
            ):
                raise ValueError(
                    "Global16 request slots must be one device int32 row per request"
                )
        self.score_workspace = score_workspace
        self.accept_offset_workspace = accept_offset_workspace
        self.request_slots = request_slots
        self.state_pool_backed = False
        self.incremental_from_prior = False
        self.layer_states: dict[int, Global16Raw256LayerState] = {}
        self.layer_objects: dict[int, object] = {}
        self.joint_evidence: dict[int, Global16Raw256FAEvidence] = {}
        self.delta_evidence: dict[int, Global16Raw256FAEvidence] = {}
        self.position_evidence: dict[int, object] = {}
        self.cuda_graph_capture_context = False
        self.committed = False

    @property
    def batch_size(self) -> int:
        return int(self.global_queries.shape[0])

    @property
    def num_attention_heads(self) -> int:
        return int(self.global_queries.shape[2])

    def applies_to(self, layer_id: int) -> bool:
        return int(layer_id) in self._layer_index

    def _query_for_layer(self, layer_id: int) -> torch.Tensor:
        return self.global_queries[:, self._layer_index[int(layer_id)]]

    def fixed_cost_audit_totals(self) -> tuple[int, int, int, int]:
        """Return prefill scan counts accumulated by the selected layers."""

        if not self.collect_fixed_cost_audit:
            return (0, 0, 0, 0)
        request_layer_inits = sum(
            evidence.request_layer_base_inits
            for evidence in self.joint_evidence.values()
        )
        prefix_rows = sum(evidence.prefix_rows for evidence in self.joint_evidence.values())
        query_key_pairs = sum(
            evidence.query_key_pairs for evidence in self.joint_evidence.values()
        )
        return (
            self.batch_size,
            request_layer_inits,
            prefix_rows,
            query_key_pairs,
        )

    def run_joint_target_and_global_attention(
        self,
        *,
        layer_id: int,
        layer,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_table: torch.Tensor,
        target_cache_seqlens: torch.Tensor,
        target_cu_seqlens_q: torch.Tensor,
        flash_attn_with_kvcache: Callable,
        softmax_scale: float,
        window_size: tuple[int, int],
        softcap: float,
        k_descale=None,
        v_descale=None,
        num_splits: int = 0,
        extra_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run one FA3 call with Target rows plus sixteen 1-row Global reads.

        The sixteen Global rows are deliberately represented as sixteen
        independent qlen=1 sequences.  With bottom-right causal alignment,
        each one can see the whole confirmed prefix; packing them as one qlen=16
        sequence would incorrectly hide the prefix tail from earlier rows.
        """

        layer_id = int(layer_id)
        if self.incremental_from_prior:
            raise RuntimeError(
                "incremental Global16 verify must not rescan the confirmed prefix"
            )
        if not self.applies_to(layer_id):
            raise ValueError(f"layer {layer_id} is not selected for Global16")
        if layer_id in self.layer_states:
            raise RuntimeError("Global16 base state may be computed only once per verify")
        if self.committed:
            raise RuntimeError("cannot start a new Global16 base scan after commit")
        q_flat = q.contiguous().view(
            -1, int(layer.tp_q_head_num), int(layer.head_dim)
        )
        if q_flat.shape[1] != self.num_attention_heads:
            raise ValueError("Target and Global query head counts do not match")
        if target_cache_seqlens.ndim != 1:
            raise ValueError("Target cache lengths must have one row per request")
        target_batch_size = int(target_cache_seqlens.shape[0])
        if target_cu_seqlens_q.shape != (target_batch_size + 1,):
            raise ValueError("Target query offsets must have batch_size + 1 rows")
        target_lengths = target_cu_seqlens_q[1:] - target_cu_seqlens_q[:-1]
        if torch.any(target_lengths <= 0):
            raise ValueError("each Target verify request must contain at least one query")
        if int(target_lengths.sum().item()) != q_flat.shape[0]:
            raise ValueError("Target query offsets do not describe q")
        if page_table.shape[0] != target_batch_size:
            raise ValueError("Target page table must have one row per request")

        global_query = self._query_for_layer(layer_id).permute(0, 2, 1, 3)
        query_parts = []
        target_indices = []
        global_indices = []
        source_rows = self.source_batch_rows_host
        context_row_by_source = {
            int(source_row): context_row
            for context_row, source_row in enumerate(source_rows)
        }
        if any(row < 0 or row >= target_batch_size for row in source_rows):
            raise ValueError("source_batch_rows must fit the Target batch")
        offset = 0
        target_offset = 0
        for batch_index, target_length in enumerate(target_lengths.tolist()):
            target_length = int(target_length)
            query_parts.append(q_flat[target_offset : target_offset + target_length])
            target_indices.extend(range(offset, offset + target_length))
            offset += target_length
            target_offset += target_length
            context_row = context_row_by_source.get(batch_index)
            if context_row is not None:
                for global_index in range(GLOBAL16_SLOTS):
                    query_parts.append(
                        global_query[context_row, global_index : global_index + 1]
                    )
                    global_indices.append(offset)
                    offset += 1
        joint_q = torch.cat(query_parts, dim=0)
        joint_query_lengths_parts = []
        joint_cache_lengths_parts = []
        joint_page_table_parts = []
        for batch_index in range(target_batch_size):
            joint_query_lengths_parts.append(
                target_lengths[batch_index : batch_index + 1].to(dtype=torch.int32)
            )
            joint_cache_lengths_parts.append(
                target_cache_seqlens[batch_index : batch_index + 1].to(dtype=torch.int32)
            )
            joint_page_table_parts.append(page_table[batch_index : batch_index + 1])
            context_row = context_row_by_source.get(batch_index)
            if context_row is not None:
                joint_query_lengths_parts.append(
                    torch.ones(
                        (GLOBAL16_SLOTS,),
                        dtype=torch.int32,
                        device=target_lengths.device,
                    )
                )
                joint_cache_lengths_parts.append(
                    self.confirmed_prefix_lens[context_row : context_row + 1].expand(
                        GLOBAL16_SLOTS
                    )
                )
                joint_page_table_parts.append(
                    page_table[batch_index : batch_index + 1].expand(
                        GLOBAL16_SLOTS, -1
                    )
                )
        # Every Target row remains one sequence; active rows append sixteen
        # independent qlen=1 Global sequences after it.
        joint_query_lengths = torch.cat(joint_query_lengths_parts)
        joint_cu_q = _cumulative_lengths(joint_query_lengths)
        joint_cache_seqlens = torch.cat(joint_cache_lengths_parts)
        joint_page_table = torch.cat(joint_page_table_parts, dim=0)
        joint_cu_k = _cumulative_lengths(joint_cache_seqlens)
        call_kwargs = dict(extra_kwargs or {})
        result = flash_attn_with_kvcache(
            q=joint_q,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=joint_page_table,
            cache_seqlens=joint_cache_seqlens,
            cu_seqlens_q=joint_cu_q,
            cu_seqlens_k_new=joint_cu_k,
            max_seqlen_q=int(target_lengths.max().item()),
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            softcap=softcap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=True,
            num_splits=num_splits,
            **call_kwargs,
        )
        joint_output, joint_lse = _unpack_fa3_result(
            result,
            token_count=joint_q.shape[0],
            num_heads=q_flat.shape[1],
        )
        target_index_tensor = torch.tensor(
            target_indices, device=q_flat.device, dtype=torch.long
        )
        global_index_tensor = torch.tensor(
            global_indices, device=q_flat.device, dtype=torch.long
        )
        target_output = joint_output.index_select(0, target_index_tensor).view_as(q_flat)
        global_output = joint_output.index_select(0, global_index_tensor).view(
            self.batch_size,
            GLOBAL16_SLOTS,
            q_flat.shape[1],
            q_flat.shape[2],
        ).permute(0, 2, 1, 3).float().contiguous()
        global_lse = joint_lse.index_select(0, global_index_tensor).view(
            self.batch_size,
            GLOBAL16_SLOTS,
            q_flat.shape[1],
        ).permute(0, 2, 1).float().contiguous()
        self.layer_states[layer_id] = Global16Raw256LayerState(
            query=self._query_for_layer(layer_id),
            output=global_output,
            logsumexp=global_lse,
        )
        self.layer_objects[layer_id] = layer
        self.joint_evidence[layer_id] = Global16Raw256FAEvidence(
            layer_id=layer_id,
            fa3_invocations=1,
            key_cache_storage_ptr=_storage_ptr(key_cache),
            value_cache_storage_ptr=_storage_ptr(value_cache),
            target_query_rows=int(q_flat.shape[0]),
            global_query_rows=self.batch_size * GLOBAL16_SLOTS,
            duplicated_page_table_rows=int(joint_page_table.shape[0]),
            request_layer_base_inits=self.batch_size,
            prefix_rows=(
                int(self.confirmed_prefix_lens.sum().item())
                if self.collect_fixed_cost_audit
                else 0
            ),
            query_key_pairs=(
                int(self.confirmed_prefix_lens.sum().item()) * GLOBAL16_SLOTS
                if self.collect_fixed_cost_audit
                else 0
            ),
        )
        return target_output.view_as(q)

    def run_prefill_global_attention(
        self,
        *,
        layer_id: int,
        layer,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        flash_attn_with_kvcache: Callable,
        softmax_scale: float,
        softcap: float,
        k_descale=None,
        v_descale=None,
        num_splits: int = 0,
        extra_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize one layer by scanning the confirmed prefix once.

        This is the prefill-only half of the incremental serving contract.
        Steady-state Target verify never calls it or rereads historical K/V.
        """

        layer_id = int(layer_id)
        if self.incremental_from_prior or self.committed:
            raise RuntimeError("Global16 prefill scan requires a fresh context")
        if not self.applies_to(layer_id):
            raise ValueError(f"layer {layer_id} is not selected for Global16")
        if layer_id in self.layer_states:
            raise RuntimeError("Global16 prefill state may be initialized only once")
        if page_table.ndim != 2:
            raise ValueError("Global16 prefill page table must be two-dimensional")
        cache_seqlens = cache_seqlens.to(
            device=self.global_queries.device,
            dtype=torch.int32,
        )
        if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != page_table.shape[0]:
            raise ValueError("Global16 prefill lengths must match the batch")

        source_rows = self.source_batch_rows.to(device=page_table.device)
        page_table = page_table.index_select(0, source_rows)
        cache_seqlens = cache_seqlens.index_select(0, source_rows)

        query = self._query_for_layer(layer_id).permute(0, 2, 1, 3)
        query = query.reshape(
            self.batch_size * GLOBAL16_SLOTS,
            self.num_attention_heads,
            query.shape[-1],
        )
        cu_seqlens_q = torch.arange(
            0,
            (self.batch_size + 1) * GLOBAL16_SLOTS,
            GLOBAL16_SLOTS,
            device=query.device,
            dtype=torch.int32,
        )
        result = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=_cumulative_lengths(cache_seqlens),
            max_seqlen_q=GLOBAL16_SLOTS,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            softcap=softcap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=True,
            num_splits=num_splits,
            **dict(extra_kwargs or {}),
        )
        output, logsumexp = _unpack_fa3_result(
            result,
            token_count=query.shape[0],
            num_heads=self.num_attention_heads,
        )
        output = output.view(
            self.batch_size,
            GLOBAL16_SLOTS,
            self.num_attention_heads,
            query.shape[-1],
        ).permute(0, 2, 1, 3).float().contiguous()
        logsumexp = logsumexp.view(
            self.batch_size,
            GLOBAL16_SLOTS,
            self.num_attention_heads,
        ).permute(0, 2, 1).float().contiguous()
        self.layer_states[layer_id] = Global16Raw256LayerState(
            query=self._query_for_layer(layer_id),
            output=output,
            logsumexp=logsumexp,
        )
        self.layer_objects[layer_id] = layer
        self.joint_evidence[layer_id] = Global16Raw256FAEvidence(
            layer_id=layer_id,
            fa3_invocations=1,
            key_cache_storage_ptr=_storage_ptr(key_cache),
            value_cache_storage_ptr=_storage_ptr(value_cache),
            target_query_rows=0,
            global_query_rows=self.batch_size * GLOBAL16_SLOTS,
            duplicated_page_table_rows=0,
            request_layer_base_inits=self.batch_size,
            prefix_rows=(
                int(self.confirmed_prefix_lens.sum().item())
                if self.collect_fixed_cost_audit
                else 0
            ),
            query_key_pairs=(
                int(self.confirmed_prefix_lens.sum().item()) * GLOBAL16_SLOTS
                if self.collect_fixed_cost_audit
                else 0
            ),
        )

    def require_base_state(self) -> None:
        missing = [layer_id for layer_id in self.selected_layer_ids if layer_id not in self.layer_states]
        if missing:
            raise RuntimeError(f"Global16 base state is missing selected layers: {missing}")

    def commit_base_only(self) -> None:
        """Mark a prefill base scan as committed without an acceptance delta."""

        self.require_base_state()
        if self.committed:
            raise RuntimeError("Global16 base state is already committed")
        self.committed = True

    @classmethod
    def _from_state_rows(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        layer_rows: Sequence[dict[int, Global16Raw256LayerState]],
        layer_objects: dict[int, object],
        committed: bool,
        score_workspace: Optional[torch.Tensor] = None,
        accept_offset_workspace: Optional[torch.Tensor] = None,
    ) -> "Global16Raw256VerifyContext":
        """Assemble independently stored requests in the scheduler's row order."""

        if not layer_rows:
            raise ValueError("at least one committed Global16 row is required")
        selected_layer_ids = tuple(int(layer_id) for layer_id in selected_layer_ids)
        query_by_layer = []
        for layer_id in selected_layer_ids:
            if any(layer_id not in rows for rows in layer_rows):
                raise ValueError(f"committed Global16 row lacks selected layer {layer_id}")
            query_by_layer.append(
                torch.cat([rows[layer_id].query for rows in layer_rows], dim=0)
            )
        global_queries = torch.stack(query_by_layer, dim=1)
        context = cls(
            selected_layer_ids=selected_layer_ids,
            global_queries=global_queries,
            # Draft reads do not use this field; a positive placeholder keeps
            # the batch state structurally valid without pretending it is a
            # new verification base scan.
            confirmed_prefix_lens=torch.ones(
                (len(layer_rows),), device=global_queries.device, dtype=torch.int32
            ),
            score_workspace=score_workspace,
            accept_offset_workspace=accept_offset_workspace,
        )
        for layer_id in selected_layer_ids:
            context.layer_states[layer_id] = Global16Raw256LayerState(
                query=global_queries[:, context._layer_index[layer_id]],
                output=torch.cat([rows[layer_id].output for rows in layer_rows], dim=0),
                logsumexp=torch.cat(
                    [rows[layer_id].logsumexp for rows in layer_rows], dim=0
                ),
            )
        context.layer_objects = dict(layer_objects)
        context.incremental_from_prior = not committed
        context.committed = committed
        return context

    @classmethod
    def _from_state_pool(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        global_queries: torch.Tensor,
        state_output_pool: torch.Tensor,
        state_lse_pool: torch.Tensor,
        request_slots: torch.Tensor,
        layer_objects: dict[int, object],
        committed: bool,
        score_workspace: Optional[torch.Tensor] = None,
        accept_offset_workspace: Optional[torch.Tensor] = None,
    ) -> "Global16Raw256VerifyContext":
        """Reference persistent request-slot state without batch copies."""

        selected_layer_ids = tuple(int(layer_id) for layer_id in selected_layer_ids)
        batch, selected_layers, heads, global_rows, head_dim = global_queries.shape
        if selected_layers != len(selected_layer_ids):
            raise ValueError("Global16 query layer count does not match selected layers")
        expected_output_tail = (
            len(selected_layer_ids),
            heads,
            global_rows,
            head_dim,
        )
        if (
            state_output_pool.ndim != 5
            or state_output_pool.shape[1:] != expected_output_tail
            or state_output_pool.dtype != torch.float32
        ):
            raise ValueError(
                "Global16 output pool must be FP32 "
                f"[capacity, {expected_output_tail}]"
            )
        if (
            state_lse_pool.shape
            != (
                state_output_pool.shape[0],
                len(selected_layer_ids),
                heads,
                global_rows,
            )
            or state_lse_pool.dtype != torch.float32
        ):
            raise ValueError("Global16 LSE pool does not match the output pool")
        if (
            state_output_pool.device != global_queries.device
            or state_lse_pool.device != global_queries.device
        ):
            raise ValueError("Global16 state pool must share the query device")
        if request_slots.shape != (batch,) or request_slots.dtype != torch.int32:
            raise ValueError("Global16 state pool needs one int32 slot per request")

        context = cls(
            selected_layer_ids=selected_layer_ids,
            global_queries=global_queries,
            confirmed_prefix_lens=torch.ones(
                (batch,), device=global_queries.device, dtype=torch.int32
            ),
            score_workspace=score_workspace,
            accept_offset_workspace=accept_offset_workspace,
            request_slots=request_slots,
            # This vector is constructed as positive ones immediately above.
            # Re-reading it as a Python bool would serialize the CUDA stream
            # twice in every speculative round (committed + prior contexts).
            validate_confirmed_prefix_values=False,
        )
        for layer_index, layer_id in enumerate(selected_layer_ids):
            context.layer_states[layer_id] = Global16Raw256LayerState(
                query=global_queries[:, layer_index],
                output=state_output_pool[:, layer_index],
                logsumexp=state_lse_pool[:, layer_index],
            )
        context.layer_objects = dict(layer_objects)
        context.state_pool_backed = True
        context.incremental_from_prior = not committed
        context.committed = committed
        return context

    @classmethod
    def from_committed_rows(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        layer_rows: Sequence[dict[int, Global16Raw256LayerState]],
        layer_objects: dict[int, object],
    ) -> "Global16Raw256VerifyContext":
        """Assemble committed rows for one Draft batch."""

        return cls._from_state_rows(
            selected_layer_ids=selected_layer_ids,
            layer_rows=layer_rows,
            layer_objects=layer_objects,
            committed=True,
        )

    @classmethod
    def from_prior_rows(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        layer_rows: Sequence[dict[int, Global16Raw256LayerState]],
        layer_objects: dict[int, object],
        score_workspace: torch.Tensor,
        accept_offset_workspace: torch.Tensor,
    ) -> "Global16Raw256VerifyContext":
        """Restore persistent state for a verify without a full-prefix scan."""

        return cls._from_state_rows(
            selected_layer_ids=selected_layer_ids,
            layer_rows=layer_rows,
            layer_objects=layer_objects,
            committed=False,
            score_workspace=score_workspace,
            accept_offset_workspace=accept_offset_workspace,
        )

    @classmethod
    def from_committed_pool(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        global_queries: torch.Tensor,
        state_output_pool: torch.Tensor,
        state_lse_pool: torch.Tensor,
        request_slots: torch.Tensor,
        layer_objects: dict[int, object],
    ) -> "Global16Raw256VerifyContext":
        """Reference committed request-slot state for Draft attention."""

        return cls._from_state_pool(
            selected_layer_ids=selected_layer_ids,
            global_queries=global_queries,
            state_output_pool=state_output_pool,
            state_lse_pool=state_lse_pool,
            request_slots=request_slots,
            layer_objects=layer_objects,
            committed=True,
        )

    @classmethod
    def from_prior_pool(
        cls,
        *,
        selected_layer_ids: Sequence[int],
        global_queries: torch.Tensor,
        state_output_pool: torch.Tensor,
        state_lse_pool: torch.Tensor,
        request_slots: torch.Tensor,
        layer_objects: dict[int, object],
        score_workspace: torch.Tensor,
        accept_offset_workspace: torch.Tensor,
    ) -> "Global16Raw256VerifyContext":
        """Reference persistent state for verify without gather/scatter."""

        return cls._from_state_pool(
            selected_layer_ids=selected_layer_ids,
            global_queries=global_queries,
            state_output_pool=state_output_pool,
            state_lse_pool=state_lse_pool,
            request_slots=request_slots,
            layer_objects=layer_objects,
            committed=False,
            score_workspace=score_workspace,
            accept_offset_workspace=accept_offset_workspace,
        )

    def begin_incremental_verify(
        self,
        *,
        score_workspace: torch.Tensor,
        accept_offset_workspace: torch.Tensor,
    ) -> "Global16Raw256VerifyContext":
        """Transfer this round's committed Draft view to its Target verify.

        Called after proposal has consumed the paged Draft context. Request
        slots and state-pool views still describe the same batch; no new row
        mapping, placeholder prefix tensor or layer-state views are needed.
        Capture contexts retain their separate lifetime and never use this.
        """
        if not self.state_pool_backed or not self.committed:
            raise RuntimeError("incremental verify requires a committed pool view")
        if self.cuda_graph_capture_context or self.position_evidence:
            raise RuntimeError("incremental verify cannot reuse a capture or scored context")
        expected = (*self.global_queries.shape[:4], MAX_ACCEPTED_ROWS)
        if (score_workspace.shape != expected
                or score_workspace.dtype != torch.float32
                or score_workspace.device != self.global_queries.device):
            raise ValueError("incremental score workspace does not match the query batch")
        if (accept_offset_workspace.shape != (self.batch_size,)
                or accept_offset_workspace.dtype != torch.int32
                or accept_offset_workspace.device != self.global_queries.device):
            raise ValueError("incremental offsets must have one int32 device row per request")
        self.score_workspace = score_workspace
        self.accept_offset_workspace = accept_offset_workspace
        self.incremental_from_prior = True
        self.committed = False
        return self

    def cache_verify_position_scores(
        self,
        *,
        layer_id: int,
        layer,
        new_keys: torch.Tensor,
        softmax_scale: float,
        softcap: float,
        cache_position_scores_fn: Optional[Callable] = None,
    ) -> None:
        """Cache Global16 scores when this verify layer produces its eight K rows."""

        layer_id = int(layer_id)
        if not self.incremental_from_prior or self.committed:
            raise RuntimeError("position scores require an active incremental verify")
        if layer_id not in self.layer_states:
            raise RuntimeError(f"Global16 prior state is missing selected layer {layer_id}")
        if (
            layer_id in self.position_evidence
            and not self.cuda_graph_capture_context
        ):
            raise RuntimeError(f"Global16 layer {layer_id} cached position scores twice")
        if self.score_workspace is None:
            raise RuntimeError("incremental Global16 verify has no score workspace")
        if new_keys.ndim != 3 or new_keys.shape[0] != self.batch_size * MAX_ACCEPTED_ROWS:
            raise ValueError(
                "Target verify K must be flat [batch * 8, kv_heads, head_dim]"
            )
        cache_position_scores_fn = (
            _incremental_helpers()[0]
            if cache_position_scores_fn is None
            else cache_position_scores_fn
        )
        state = self.layer_states[layer_id]
        output = self.score_workspace[:, self._layer_index[layer_id]]
        score_events = None
        if self.collect_fixed_cost_audit and self.global_queries.device.type == "cuda":
            score_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            score_events[0].record()
        # KNorm/RoPE may return a strided view. Materialize the fixed eight-row
        # selected-layer slice here so the score audit covers the complete
        # Global16 update rather than only the following kernel.
        new_keys = new_keys.contiguous()
        keys = new_keys.view(
            self.batch_size,
            MAX_ACCEPTED_ROWS,
            new_keys.shape[1],
            new_keys.shape[2],
        )
        self.position_evidence[layer_id] = cache_position_scores_fn(
            layer_id=layer_id,
            global_query=state.query,
            new_keys=keys,
            output=output,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )
        if score_events is not None:
            score_events[1].record()
            self.fixed_cost_score_events[layer_id] = score_events
        self.layer_objects[layer_id] = layer

    def fixed_cost_score_update_ms(self) -> float:
        """Return summed CUDA time for the per-layer Global-query score kernels."""

        if not self.collect_fixed_cost_audit:
            return 0.0
        if len(self.fixed_cost_score_events) != len(self.selected_layer_ids):
            raise RuntimeError(
                "fixed-cost timing is missing a selected-layer score kernel"
            )
        return sum(
            start.elapsed_time(end)
            for start, end in self.fixed_cost_score_events.values()
        )

    def mark_position_scores_captured_by_cuda_graph(self) -> None:
        """Expose score work already replayed by the Target CUDA graph."""

        if not self.incremental_from_prior or self.committed:
            raise RuntimeError(
                "CUDA-graph position scores require an active incremental verify"
            )
        if self.score_workspace is None:
            raise RuntimeError("incremental Global16 verify has no score workspace")
        missing_layers = [
            layer_id
            for layer_id in self.selected_layer_ids
            if layer_id not in self.layer_objects
        ]
        if missing_layers:
            raise RuntimeError(
                "Global16 Target graph has no layer objects for "
                f"{missing_layers}"
            )
        if self.position_evidence:
            raise RuntimeError(
                "Global16 position scores cannot be marked after eager caching"
            )
        self.position_evidence.update(
            {
                layer_id: "cuda_graph_replay"
                for layer_id in self.selected_layer_ids
            }
        )

    def commit_accepted_delta(
        self,
        *,
        accepted_cache_locs: torch.Tensor,
        accept_lens: torch.Tensor,
        token_to_kv_pool,
        page_size: int,
        merge_accepted_positions_fn: Optional[Callable] = None,
    ) -> None:
        """Merge cached scores and only the accepted Target V rows into state."""

        self.require_base_state()
        if not self.incremental_from_prior:
            raise RuntimeError(
                "accepted position merge requires persistent Global16 prior state"
            )
        if self.committed:
            raise RuntimeError("Global16 accepted delta may be committed only once")
        if page_size != 1:
            raise NotImplementedError("Global16 P0 requires page_size=1 for exact accepted rows")
        missing = [
            layer_id
            for layer_id in self.selected_layer_ids
            if layer_id not in self.position_evidence
        ]
        if missing:
            raise RuntimeError(
                f"Global16 position scores are missing selected layers: {missing}"
            )
        if self.score_workspace is None or self.accept_offset_workspace is None:
            raise RuntimeError("Global16 incremental workspaces are unavailable")
        accept_lens = accept_lens.to(device=self.global_queries.device, dtype=torch.int32)
        if accept_lens.shape != (self.batch_size,):
            raise ValueError("accept_lens must have one row per request")
        # Production values come directly from DraftFreeKVVerifyInput.verify().
        # Avoid synchronizing CUDA merely to repeat that validation here.
        if accept_lens.device.type != "cuda":
            if torch.any(accept_lens <= 0) or torch.any(
                accept_lens > MAX_ACCEPTED_ROWS
            ):
                raise ValueError(
                    "accepted Target K/V delta must contain between one and eight rows"
                )
            if accepted_cache_locs.ndim == 2:
                if accepted_cache_locs.shape != (
                    self.batch_size,
                    MAX_ACCEPTED_ROWS,
                ):
                    raise ValueError(
                        "fixed verify cache locations must have shape [batch, 8]"
                    )
            elif int(accept_lens.sum().item()) != accepted_cache_locs.numel():
                raise ValueError("accepted cache locations do not match accept_lens")
        merge_accepted_positions_fn = (
            _incremental_helpers()[1]
            if merge_accepted_positions_fn is None
            else merge_accepted_positions_fn
        )
        for layer_id in self.selected_layer_ids:
            state = self.layer_states[layer_id]
            layer = self.layer_objects[layer_id]
            _, value_cache = token_to_kv_pool.get_kv_buffer(layer_id)
            value_cache = value_cache.view(-1, page_size, layer.tp_v_head_num, layer.v_head_dim)
            merge_accepted_positions_fn(
                state_output=state.output,
                state_lse=state.logsumexp,
                position_scores=self.score_workspace[
                    :, self._layer_index[layer_id]
                ],
                value_cache=value_cache,
                accepted_cache_locs=accepted_cache_locs,
                accept_lens=accept_lens,
                accept_offsets=self.accept_offset_workspace,
                state_slots=self.request_slots if self.state_pool_backed else None,
            )
        self.committed = True


class Global16Raw256PagedDraftContext:
    """Direct paged Raw256 reader used by the frozen SpecForge draft head."""

    def __init__(
        self,
        *,
        verify_context: Global16Raw256VerifyContext,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_token_pool,
        token_to_kv_pool,
        page_size: int,
        flash_attn_with_kvcache: Callable,
        num_splits: int = 0,
        host_seq_lens: Optional[Sequence[int]] = None,
        page_table_workspace: Optional[torch.Tensor] = None,
        raw_length_workspace: Optional[torch.Tensor] = None,
        cuda_graph_static_inputs: bool = False,
        position_ids_are_scheduler_owned: bool = False,
    ) -> None:
        if not verify_context.committed:
            raise RuntimeError("draft attention requires committed Global16 state")
        if page_size != 1:
            raise NotImplementedError("Global16 P0 requires page_size=1 for Raw256 views")
        if req_pool_indices.shape != (verify_context.batch_size,) or seq_lens.shape != (
            verify_context.batch_size,
        ):
            raise ValueError("paged draft metadata must have one row per Global16 request")
        self.verify_context = verify_context
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens.to(device=verify_context.global_queries.device, dtype=torch.int32)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.page_size = int(page_size)
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self.num_splits = int(num_splits)
        self.cuda_graph_static_inputs = bool(cuda_graph_static_inputs)
        self.position_ids_are_scheduler_owned = bool(
            position_ids_are_scheduler_owned
        )
        self.evidence: list[Global16Raw256DraftEvidence] = []
        self._page_table_workspace = page_table_workspace
        self._raw_length_workspace = raw_length_workspace
        self._page_table, self._raw_lengths = self._build_raw256_page_table(
            host_seq_lens=host_seq_lens
        )
        self._query_rows: Optional[int] = None
        self._raw_cu_seqlens_q: Optional[torch.Tensor] = None
        self._global_attention_mask: Optional[torch.Tensor] = None
        self._fused_output: Optional[torch.Tensor] = None
        self.fused_merge_layers = 0
        self.fused_all_attention_layers = 0
        self.shared_draft_input_layers = 0
        self.shared_draft_input_builds = 0
        self._shared_position_ids: Optional[torch.Tensor] = None
        self._shared_position_dtype: Optional[torch.dtype] = None
        self._shared_position_cos: Optional[torch.Tensor] = None
        self._shared_position_sin: Optional[torch.Tensor] = None
        self._shared_local_attention_mask: Optional[torch.Tensor] = None

    @property
    def raw_page_table(self) -> torch.Tensor:
        return self._page_table

    @property
    def raw_lengths(self) -> torch.Tensor:
        return self._raw_lengths

    @property
    def max_raw_rows(self) -> int:
        return int(self._max_raw_rows)

    def _build_raw256_page_table(
        self,
        *,
        host_seq_lens: Optional[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_lengths = self.seq_lens.clamp(max=RAW256_ROWS)
        if host_seq_lens is None:
            max_raw = int(raw_lengths.max().item())
        else:
            normalized_host_lens = [int(length) for length in host_seq_lens]
            if len(normalized_host_lens) != self.verify_context.batch_size:
                raise ValueError("host sequence lengths must match the Draft batch")
            max_raw = min(max(normalized_host_lens), RAW256_ROWS)
        if max_raw <= 0:
            raise RuntimeError("Raw256 requires a non-empty committed Target prefix")
        # Keep this as a host scalar.  Reading raw_lengths.max().item() inside
        # every Draft layer would serialize the CUDA stream five times per
        # proposal merely to populate debug evidence.
        self._max_raw_rows = max_raw
        if (
            self._page_table_workspace is not None
            or self._raw_length_workspace is not None
        ):
            if (
                self._page_table_workspace is None
                or self._raw_length_workspace is None
                or self._page_table_workspace.shape
                != (self.verify_context.batch_size, RAW256_ROWS)
                or self._page_table_workspace.dtype != torch.int32
                or self._page_table_workspace.device
                != self.verify_context.global_queries.device
                or self._raw_length_workspace.shape
                != (self.verify_context.batch_size,)
                or self._raw_length_workspace.dtype != torch.int32
                or self._raw_length_workspace.device
                != self.verify_context.global_queries.device
            ):
                raise ValueError(
                    "Raw256 page-table workspaces must be CUDA int32 [B,256] "
                    "and [B]"
                )
            return _raw_page_table_helper()(
                req_to_token=self.req_to_token_pool.req_to_token,
                req_pool_indices=self.req_pool_indices,
                seq_lens=self.seq_lens,
                page_table=self._page_table_workspace,
                raw_lengths=self._raw_length_workspace,
            )
        full_table = self.req_to_token_pool.req_to_token[self.req_pool_indices]
        starts = self.seq_lens.to(dtype=torch.long) - raw_lengths.to(dtype=torch.long)
        positions = torch.arange(max_raw, device=full_table.device, dtype=torch.long)
        indices = starts[:, None] + positions[None, :]
        safe_indices = indices.clamp(max=full_table.shape[1] - 1)
        page_table = full_table.gather(1, safe_indices).to(dtype=torch.int32)
        return page_table, raw_lengths

    def prepare_draft_layer_inputs(
        self,
        *,
        attention,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
        batch_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build proposal-wide RoPE and Local7 metadata once for all five layers."""

        if not bool(
            getattr(attention, "_dfk_global16_shared_rope_verified", False)
        ):
            raise RuntimeError(
                "Global16 shared Draft inputs require a verified common RoPE "
                "definition"
            )
        expected_shape = (int(batch_size), int(block_size))
        if position_ids.shape != expected_shape:
            raise ValueError(
                "Global16 shared position_ids must have shape "
                f"{expected_shape}, got {tuple(position_ids.shape)}"
            )
        if block_size != LOCAL7_ROWS:
            raise ValueError("Global16 shared Draft inputs require Local7")
        if position_ids.device != self.verify_context.global_queries.device:
            raise ValueError(
                "Global16 shared Draft inputs must use the verify-context device"
            )
        if self.shared_draft_input_layers >= len(
            self.verify_context.selected_layer_ids
        ):
            raise RuntimeError(
                "Global16 Draft context was reused beyond its selected layers"
            )

        if self._shared_position_ids is None:
            self._shared_position_ids = position_ids
            self._shared_position_dtype = dtype
            (
                self._shared_position_cos,
                self._shared_position_sin,
            ) = attention._position_embeddings(position_ids, dtype=dtype)
            self._shared_local_attention_mask = torch.ones(
                (batch_size, block_size, block_size),
                dtype=torch.bool,
                device=position_ids.device,
            )
            self.shared_draft_input_builds = 1
        elif (
            position_ids is not self._shared_position_ids
            or dtype != self._shared_position_dtype
        ):
            raise RuntimeError(
                "all selected Draft layers must share one position tensor and dtype"
            )

        self.shared_draft_input_layers += 1
        assert self._shared_position_cos is not None
        assert self._shared_position_sin is not None
        assert self._shared_local_attention_mask is not None
        return (
            self._shared_position_cos,
            self._shared_position_sin,
            self._shared_local_attention_mask,
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
        """Merge Global16, direct Raw256 and Local7 without a K/V bank."""

        if query.ndim != 4:
            raise ValueError("draft query must have shape [batch, heads, rows, dim]")
        if local_keys.shape != local_values.shape or local_keys.ndim != 4:
            raise ValueError("Local7 K/V must be matching rank-4 tensors")
        if local_keys.shape[-2] > LOCAL7_ROWS:
            raise ValueError("Global16 + Raw256 permits at most Local7 K/V rows")
        layer_id = self.verify_context.selected_layer_ids[int(layer_index)]
        state = self.verify_context.layer_states[layer_id]
        if query.shape[:2] != state.query.shape[:2] or query.shape[-1] != state.query.shape[-1]:
            raise ValueError("draft query shape does not match committed Global16 state")
        if local_attention_mask.shape != (
            query.shape[0], query.shape[-2], local_keys.shape[-2]
        ):
            raise ValueError("Local7 attention mask has an invalid shape")

        page_table, raw_lengths = self._page_table, self._raw_lengths
        layer = self.verify_context.layer_objects[layer_id]
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer_id)
        key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
        batch_size, heads, query_rows, head_dim = query.shape
        if _DRAFT_FUSED_ALL_ATTENTION:
            fixed_cuda_candidate = bool(
                self.verify_context.state_pool_backed
                and query.is_cuda
                and query.dtype == torch.bfloat16
                and query.shape[-2:] == (LOCAL7_ROWS, 128)
                and float(layer.logit_cap) == 0.0
            )
            if self._fused_output is None and fixed_cuda_candidate:
                self._fused_output = _allocate_fused_draft_output(query)
            if fixed_cuda_candidate and self._fused_output is not None:
                can_fuse_all, fuse_all = _draft_all_attention_helpers()
                if can_fuse_all(
                    query=query,
                    global_keys=state.query,
                    global_values=state.output,
                    state_slots=self.verify_context.request_slots,
                    local_keys=local_keys,
                    local_values=local_values,
                    local_attention_mask=local_attention_mask,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    page_table=page_table,
                    raw_lengths=raw_lengths,
                    output=self._fused_output,
                ):
                    merged_output = fuse_all(
                        query=query,
                        global_keys=state.query,
                        global_values=state.output,
                        state_slots=self.verify_context.request_slots,
                        local_keys=local_keys,
                        local_values=local_values,
                        local_attention_mask=local_attention_mask,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        page_table=page_table,
                        raw_lengths=raw_lengths,
                        output=self._fused_output,
                        softmax_scale=layer.scaling,
                        softcap=layer.logit_cap,
                    )
                    self.fused_merge_layers += 1
                    self.fused_all_attention_layers += 1
                    self.evidence.append(
                        Global16Raw256DraftEvidence(
                            layer_id=layer_id,
                            raw_key_cache_storage_ptr=_storage_ptr(key_cache),
                            raw_value_cache_storage_ptr=_storage_ptr(value_cache),
                            raw_rows=self._max_raw_rows,
                        )
                    )
                    return merged_output.to(dtype=query.dtype)
        if self._query_rows is None:
            self._query_rows = query_rows
            self._raw_cu_seqlens_q = torch.arange(
                0,
                batch_size * query_rows + 1,
                query_rows,
                device=query.device,
                dtype=torch.int32,
            )
            self._global_attention_mask = torch.ones(
                (batch_size, query_rows, GLOBAL16_SLOTS),
                dtype=torch.bool,
                device=query.device,
            )
        elif self._query_rows != query_rows:
            raise RuntimeError(
                "all selected Draft layers must use the same query-row count"
            )
        raw_q = query.permute(0, 2, 1, 3).reshape(batch_size * query_rows, heads, head_dim)
        raw_result = self.flash_attn_with_kvcache(
            q=raw_q,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=page_table,
            cache_seqlens=raw_lengths,
            cu_seqlens_q=self._raw_cu_seqlens_q,
            max_seqlen_q=query_rows,
            softmax_scale=layer.scaling,
            causal=False,
            window_size=(-1, -1),
            softcap=layer.logit_cap,
            return_softmax_lse=True,
            num_splits=self.num_splits,
        )
        raw_output, raw_lse = _unpack_fa3_result(
            raw_result,
            token_count=raw_q.shape[0],
            num_heads=heads,
        )
        fixed_cuda_candidate = bool(
            self.verify_context.state_pool_backed
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and query.shape[-2:] == (LOCAL7_ROWS, 128)
        )
        if self._fused_output is None and fixed_cuda_candidate:
            self._fused_output = _allocate_fused_draft_output(query)
        fused = False
        fused_merge = None
        if fixed_cuda_candidate and self._fused_output is not None:
            can_fuse, fused_merge = _draft_fused_helpers()
            fused = can_fuse(
                query=query,
                global_keys=state.query,
                global_values=state.output,
                state_slots=self.verify_context.request_slots,
                local_keys=local_keys,
                local_values=local_values,
                local_attention_mask=local_attention_mask,
                raw_output=raw_output,
                raw_lse=raw_lse,
                output=self._fused_output,
            )
        if fused:
            assert fused_merge is not None
            merged_output = fused_merge(
                query=query,
                global_keys=state.query,
                global_values=state.output,
                state_slots=self.verify_context.request_slots,
                local_keys=local_keys,
                local_values=local_values,
                local_attention_mask=local_attention_mask,
                raw_output=raw_output,
                raw_lse=raw_lse,
                output=self._fused_output,
                softmax_scale=layer.scaling,
                softcap=layer.logit_cap,
                validate_contract=not _DRAFT_SKIP_REDUNDANT_MERGE_CHECK,
            )
            self.fused_merge_layers += 1
        else:
            raw_output = raw_output.view(
                batch_size, query_rows, heads, head_dim
            ).permute(0, 2, 1, 3).contiguous()
            raw_lse = raw_lse.view(
                batch_size, query_rows, heads
            ).permute(0, 2, 1).contiguous()
            attention_statistics_from_4d, merge_normalized_attention = (
                _reference_helpers()
            )
            # The fallback materializes only compact Global16 values in
            # scheduler order; it never gathers Target K/V history.
            state_output = (
                state.output.index_select(0, self.verify_context.request_slots)
                if self.verify_context.state_pool_backed
                else state.output
            )
            global_statistics = attention_statistics_from_4d(
                query=query,
                keys=state.query,
                values=state_output.to(dtype=state.query.dtype),
                attention_mask=self._global_attention_mask,
            )
            local_statistics = attention_statistics_from_4d(
                query=query,
                keys=local_keys,
                values=local_values,
                attention_mask=local_attention_mask,
            )
            global_output, global_lse = _statistics_as_normalized(
                global_statistics
            )
            local_output, local_lse = _statistics_as_normalized(
                local_statistics
            )
            merged_output, _ = merge_normalized_attention(
                (
                    (global_output, global_lse),
                    (raw_output, raw_lse),
                    (local_output, local_lse),
                )
            )
        self.evidence.append(
            Global16Raw256DraftEvidence(
                layer_id=layer_id,
                raw_key_cache_storage_ptr=_storage_ptr(key_cache),
                raw_value_cache_storage_ptr=_storage_ptr(value_cache),
                raw_rows=self._max_raw_rows,
            )
        )
        return merged_output.to(dtype=query.dtype)


def context_from_spec_info(forward_batch) -> Optional[Global16Raw256VerifyContext]:
    """Return the transaction attached by ``DraftFreeKVWorker`` if present."""

    context = getattr(forward_batch, "global16_raw256_context", None)
    if context is not None:
        if not isinstance(context, Global16Raw256VerifyContext):
            raise TypeError("global16_raw256_context has an unexpected type")
        return context
    spec_info = getattr(forward_batch, "spec_info", None)
    context = getattr(spec_info, "global16_raw256_context", None)
    if context is None:
        return None
    if not isinstance(context, Global16Raw256VerifyContext):
        raise TypeError("global16_raw256_context has an unexpected type")
    return context


__all__ = [
    "GLOBAL16_SLOTS",
    "LOCAL7_ROWS",
    "MAX_ACCEPTED_ROWS",
    "RAW256_ROWS",
    "Global16Raw256DraftEvidence",
    "Global16Raw256FAEvidence",
    "Global16Raw256LayerState",
    "Global16Raw256PagedDraftContext",
    "Global16Raw256VerifyContext",
    "context_from_spec_info",
    "select_global16_prefill_rows",
    "validate_global16_raw256_runtime_contract",
]
