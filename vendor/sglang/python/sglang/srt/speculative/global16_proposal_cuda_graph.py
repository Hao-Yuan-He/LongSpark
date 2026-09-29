from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def global16_proposal_cuda_graph_enabled(
    server_args,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve the proposal-graph switch without coupling it to Target graphs."""

    environ = os.environ if environ is None else environ
    value = environ.get("DFK_GLOBAL16_PROPOSAL_CUDA_GRAPH")
    if value is None:
        return not bool(getattr(server_args, "disable_cuda_graph", False))
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        "DFK_GLOBAL16_PROPOSAL_CUDA_GRAPH must be one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}"
    )


def resolve_global16_proposal_graph_batch_sizes(
    server_args,
    *,
    capacity: int,
) -> list[int]:
    """Use the same decode batch-size tiers as SGLang's model graph runner."""

    cuda_graph_config = getattr(server_args, "cuda_graph_config", None)
    decode_config = getattr(cuda_graph_config, "decode", None)
    batch_sizes = getattr(decode_config, "bs", None)
    if batch_sizes is None:
        batch_sizes = getattr(server_args, "cuda_graph_bs", None)
    if not batch_sizes:
        raise RuntimeError(
            "Global16 proposal CUDA graph needs finalized SGLang decode "
            "batch-size tiers"
        )
    resolved = sorted(
        {
            int(batch_size)
            for batch_size in batch_sizes
            if 0 < int(batch_size) <= int(capacity)
        }
    )
    if not resolved:
        raise RuntimeError(
            "Global16 proposal CUDA graph has no batch-size tier within the "
            f"request-pool capacity {capacity}"
        )
    # SGLang's default decode graph tiers stop at 256 even when the request
    # pool can hold a larger batch.  LongSpark's proposal runner is separate
    # from the Target graph, so extend only its tiers to cover the whole pool.
    # Above 256, 32-request spacing bounds padding without recording every
    # possible shrinking-batch size.  The exact capacity is always retained as
    # the final tier so no valid request batch falls back to eager execution.
    if resolved[-1] < int(capacity):
        next_batch_size = ((resolved[-1] // 32) + 1) * 32
        resolved.extend(
            range(
                next_batch_size,
                int(capacity),
                32,
            )
        )
        resolved.append(int(capacity))
    return resolved


def copy_and_pad_rows_(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    graph_batch_size: int,
) -> None:
    """Copy active rows and duplicate the last request into graph padding."""

    batch_size = int(source.shape[0])
    graph_batch_size = int(graph_batch_size)
    if batch_size <= 0 or batch_size > graph_batch_size:
        raise ValueError(
            f"cannot pad batch {batch_size} to graph batch {graph_batch_size}"
        )
    if destination.shape[0] < graph_batch_size:
        raise ValueError("proposal graph destination is smaller than its batch tier")
    if destination.shape[1:] != source.shape[1:]:
        raise ValueError(
            "proposal graph source/destination row shapes differ: "
            f"{tuple(source.shape[1:])} vs {tuple(destination.shape[1:])}"
        )
    destination[:batch_size].copy_(source)
    if graph_batch_size > batch_size:
        destination[batch_size:graph_batch_size].copy_(
            source[-1:].expand(graph_batch_size - batch_size, *source.shape[1:])
        )


def attach_global16_target_graph_capture_context(
    *,
    spec_info,
    model_runner,
    num_tokens: int,
    num_tokens_per_bs: int,
) -> None:
    """Attach LongSpark's fixed Target-attention state to a capture batch."""

    if spec_info is None:
        return
    spec_algorithm = getattr(model_runner, "spec_algorithm", None)
    if (
        spec_algorithm is None
        or not spec_algorithm.is_draft_free_kv()
        or bool(getattr(model_runner, "is_draft_worker", False))
    ):
        return
    context_factory = getattr(
        model_runner,
        "global16_raw256_cuda_graph_context_factory",
        None,
    )
    if context_factory is None:
        return
    num_tokens = int(num_tokens)
    num_tokens_per_bs = int(num_tokens_per_bs)
    if num_tokens_per_bs <= 0 or num_tokens % num_tokens_per_bs:
        raise ValueError(
            "Global16 Target graph tokens must be an exact multiple of its "
            "verify width"
        )
    spec_info.global16_raw256_context = context_factory(
        batch_size=num_tokens // num_tokens_per_bs,
    )


@dataclass
class _ProposalGraphInputs:
    current_token_ids: torch.Tensor
    current_position_ids: torch.Tensor
    conditioning_hidden_states: Optional[torch.Tensor]
    request_slots: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    raw_page_table: torch.Tensor
    raw_lengths: torch.Tensor


@dataclass
class _ProposalGraphState:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    context: object


class Global16ProposalCudaGraphRunner:
    """Capture a fixed-shape Global16 callback returning tokens or base logits.

    Returned rows are borrowed until the next call. Callers must finish using
    them on the current stream before replaying another tier from this pool.
    """

    def __init__(
        self,
        *,
        proposal_fn: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], object],
            torch.Tensor,
        ],
        device: torch.device,
        capture_batch_sizes: list[int],
        context_factory: Callable[..., object],
        selected_layer_count: int,
        conditioning_tail_shape: Optional[tuple[int, ...]] = None,
        conditioning_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not capture_batch_sizes:
            raise ValueError("proposal CUDA graph requires at least one batch tier")
        self.proposal_fn = proposal_fn
        self.device = torch.device(device)
        self.capture_batch_sizes = sorted(
            {int(batch_size) for batch_size in capture_batch_sizes}
        )
        self.context_factory = context_factory
        self.selected_layer_count = int(selected_layer_count)
        self.conditioning_tail_shape = conditioning_tail_shape
        self.conditioning_dtype = conditioning_dtype
        self.max_batch_size = max(self.capture_batch_sizes)
        self._states: dict[int, _ProposalGraphState] = {}
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._pool = torch.cuda.graph_pool_handle()
        self._inputs = self._allocate_inputs()
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0

    def _allocate_inputs(self) -> _ProposalGraphInputs:
        max_batch_size = self.max_batch_size
        return _ProposalGraphInputs(
            current_token_ids=torch.empty(
                max_batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            current_position_ids=torch.empty(
                max_batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            conditioning_hidden_states=(
                None
                if self.conditioning_tail_shape is None
                else torch.empty(
                    max_batch_size,
                    *self.conditioning_tail_shape,
                    dtype=self.conditioning_dtype,
                    device=self.device,
                )
            ),
            request_slots=torch.empty(
                max_batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
            req_pool_indices=torch.empty(
                max_batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
            seq_lens=torch.empty(
                max_batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
            raw_page_table=torch.empty(
                max_batch_size,
                256,
                dtype=torch.int32,
                device=self.device,
            ),
            raw_lengths=torch.empty(
                max_batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
        )

    def _graph_batch_size(self, batch_size: int) -> Optional[int]:
        index = bisect.bisect_left(self.capture_batch_sizes, int(batch_size))
        if index == len(self.capture_batch_sizes):
            return None
        return self.capture_batch_sizes[index]

    def _prepare_dynamic_inputs(
        self,
        *,
        graph_batch_size: int,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        conditioning_hidden_states: Optional[torch.Tensor],
        paged_context,
    ) -> None:
        copy_and_pad_rows_(
            self._inputs.current_token_ids,
            current_token_ids,
            graph_batch_size=graph_batch_size,
        )
        copy_and_pad_rows_(
            self._inputs.current_position_ids,
            current_position_ids,
            graph_batch_size=graph_batch_size,
        )
        if self._inputs.conditioning_hidden_states is None:
            if conditioning_hidden_states is not None:
                raise ValueError(
                    "proposal graph received unexpected hidden conditioning"
                )
        else:
            if conditioning_hidden_states is None:
                raise ValueError(
                    "proposal graph requires hidden conditioning for this checkpoint"
                )
            copy_and_pad_rows_(
                self._inputs.conditioning_hidden_states,
                conditioning_hidden_states,
                graph_batch_size=graph_batch_size,
            )
        copy_and_pad_rows_(
            self._inputs.request_slots,
            paged_context.verify_context.request_slots,
            graph_batch_size=graph_batch_size,
        )
        copy_and_pad_rows_(
            self._inputs.raw_page_table,
            paged_context.raw_page_table,
            graph_batch_size=graph_batch_size,
        )
        copy_and_pad_rows_(
            self._inputs.raw_lengths,
            paged_context.raw_lengths,
            graph_batch_size=graph_batch_size,
        )

    def _prepare_capture_metadata(
        self,
        *,
        graph_batch_size: int,
        paged_context,
    ) -> None:
        copy_and_pad_rows_(
            self._inputs.req_pool_indices,
            paged_context.req_pool_indices,
            graph_batch_size=graph_batch_size,
        )
        copy_and_pad_rows_(
            self._inputs.seq_lens,
            paged_context.seq_lens,
            graph_batch_size=graph_batch_size,
        )

    def _new_context(self, graph_batch_size: int, max_raw_rows: int):
        return self.context_factory(
            request_slots=self._inputs.request_slots[:graph_batch_size],
            req_pool_indices=self._inputs.req_pool_indices[:graph_batch_size],
            seq_lens=self._inputs.seq_lens[:graph_batch_size],
            page_table_workspace=self._inputs.raw_page_table[:graph_batch_size],
            raw_length_workspace=self._inputs.raw_lengths[:graph_batch_size],
            host_seq_lens=[int(max_raw_rows)] * graph_batch_size,
        )

    def _run_once(self, graph_batch_size: int, context) -> torch.Tensor:
        current_token_ids = self._inputs.current_token_ids[:graph_batch_size]
        return self.proposal_fn(
            current_token_ids,
            self._inputs.current_position_ids[:graph_batch_size],
            (
                None
                if self._inputs.conditioning_hidden_states is None
                else self._inputs.conditioning_hidden_states[:graph_batch_size]
            ),
            context,
        )

    def _capture(
        self,
        *,
        graph_batch_size: int,
        paged_context,
    ) -> _ProposalGraphState:
        self._prepare_capture_metadata(
            graph_batch_size=graph_batch_size,
            paged_context=paged_context,
        )
        max_raw_rows = int(paged_context.max_raw_rows)
        current_stream = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._capture_stream):
            for _ in range(2):
                warmup_context = self._new_context(
                    graph_batch_size,
                    max_raw_rows,
                )
                self._run_once(graph_batch_size, warmup_context)
            capture_context = self._new_context(
                graph_batch_size,
                max_raw_rows,
            )
        self._capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            output = self._run_once(graph_batch_size, capture_context)
        current_stream.wait_stream(self._capture_stream)

        if (
            int(capture_context.fused_merge_layers) != self.selected_layer_count
            or int(capture_context.shared_draft_input_layers)
            != self.selected_layer_count
            or int(capture_context.shared_draft_input_builds) != 1
        ):
            raise RuntimeError(
                "Global16 proposal graph did not capture the fused five-layer "
                "serving path"
            )
        state = _ProposalGraphState(
            graph=graph,
            output=output,
            context=capture_context,
        )
        self._states[graph_batch_size] = state
        self.capture_count += 1
        return state

    def try_propose(
        self,
        *,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        conditioning_hidden_states: Optional[torch.Tensor],
        paged_context,
    ) -> Optional[torch.Tensor]:
        batch_size = int(current_token_ids.shape[0])
        if current_position_ids.shape != (batch_size,):
            raise ValueError(
                "proposal graph current positions must have one row per request"
            )
        graph_batch_size = self._graph_batch_size(batch_size)
        if graph_batch_size is None:
            self.fallback_count += 1
            return None

        self._prepare_dynamic_inputs(
            graph_batch_size=graph_batch_size,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            conditioning_hidden_states=conditioning_hidden_states,
            paged_context=paged_context,
        )
        state = self._states.get(graph_batch_size)
        if state is None:
            state = self._capture(
                graph_batch_size=graph_batch_size,
                paged_context=paged_context,
            )
        # CUDA stream capture records the kernels but does not promise that the
        # newly allocated output contains this request's result.  This matters
        # when a shrinking live batch reaches a graph tier for the first time:
        # returning the capture buffer directly can expose uninitialized token
        # ids.  Every request, including the one that lazily captures a tier,
        # therefore consumes an explicit replay.
        state.graph.replay()
        self.replay_count += 1
        return state.output[:batch_size]

    def stats(self) -> dict[str, object]:
        return {
            "captures": int(self.capture_count),
            "replays": int(self.replay_count),
            "fallbacks": int(self.fallback_count),
            "captured_batch_sizes": tuple(sorted(self._states)),
        }

    def reset_stats(self) -> None:
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0


__all__ = [
    "Global16ProposalCudaGraphRunner",
    "attach_global16_target_graph_capture_context",
    "copy_and_pad_rows_",
    "global16_proposal_cuda_graph_enabled",
    "resolve_global16_proposal_graph_batch_sizes",
]
