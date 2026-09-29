from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.speculative.draft_free_kv_candidate_memory import (
    HEAD_DIM,
    HYBRID_AGGREGATE_ROWS,
    HYBRID_BANK_ROWS,
    HYBRID_RAW_ROWS,
    HYBRID_RAW_UNION_ROWS,
    VERIFY_WIDTH,
    HybridRaw192Agg64PolicyLUT,
    build_draft_free_kv_candidate_memory_plan,
)
from sglang.srt.speculative.draft_free_kv_native_memory import (
    DraftFreeKVLocalMemoryBank,
    build_draft_free_kv_candidate_native_layer_parts,
)


DFK_P8_L0_TAIL_SHADOW_ENV = "DFK_P8_L0_TAIL_SHADOW"
DFK_P8_L0_TAIL_ACTIVE_ENV = "DFK_P8_L0_TAIL_ACTIVE"
DFK_P8_L0_TAIL_ACTIVE_VALIDATE_ENV = "DFK_P8_L0_TAIL_ACTIVE_VALIDATE"
DFK_P8_L0_ATTRIBUTION_ENV = "DFK_P8_L0_ATTRIBUTION"
DFK_P8_L0_TAIL_SHADOW_PREFIX = "DFK_P8_L0_TAIL_SHADOW "
DFK_P8_L0_TAIL_ACTIVE_PREFIX = "DFK_P8_L0_TAIL_ACTIVE "
DFK_P8_L0_ATTRIBUTION_PREFIX = "DFK_P8_L0_ATTRIBUTION "

P8_L0_TAIL_MODE = "p8_l0_tail_shadow"
P8_L0_TAIL_ACTIVE_MODE = "p8_l0_tail_active"
P8_L0_OVERLAP_ACTIVE_MODE = "p8_l0_overlap_active"
P8_L0_TAIL_ENABLED_LAYERS = (0,)
P8_L0_TAIL_ACTIVE_SELECTED_LAYERS = (0, 9, 18, 26, 35)
P8_L0_TAIL_WORKSPACE_VERSION = 1
P8_L0_TAIL_MAX_BATCH = 16
P8_L0_TAIL_ACTIVE_EVENT_LIMIT = 4
P8_L0_TAIL_AGGREGATE_THRESHOLDS = {
    "mean": 5e-4,
    "p99": 0.01,
    "max": 0.125,
}
P8_L0_ATTRIBUTION_PREFIX_BUCKETS = (
    ("[0,192)", 0, 192),
    ("[192,256)", 192, 256),
    ("[256,512)", 256, 512),
    ("[512,1024)", 512, 1024),
    ("[1024,2048)", 1024, 2048),
    ("[2048,+inf)", 2048, None),
)

_HOOK_SIGNATURE_ATTR = "_dfk_p8_tail_hook_signature"
_HOOK_OWNER_ATTR = "_dfk_p8_tail_hook_owner"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def p8_l0_tail_shadow_enabled() -> bool:
    return _env_flag(DFK_P8_L0_TAIL_SHADOW_ENV)


def p8_l0_tail_active_enabled() -> bool:
    return _env_flag(DFK_P8_L0_TAIL_ACTIVE_ENV)


def p8_l0_tail_active_validate_enabled() -> bool:
    return _env_flag(DFK_P8_L0_TAIL_ACTIVE_VALIDATE_ENV)


def p8_l0_attribution_enabled() -> bool:
    return _env_flag(DFK_P8_L0_ATTRIBUTION_ENV)


def validate_p8_l0_attribution_mode(
    *,
    enabled: bool,
    tail_active: bool,
    tail_active_validate: bool,
    overlap_active_validate: bool,
) -> None:
    if not enabled:
        return
    if not tail_active:
        raise ValueError(
            f"{DFK_P8_L0_ATTRIBUTION_ENV} requires DFK_P8_L0_TAIL_ACTIVE=1."
        )
    conflicts = [
        name
        for name, active in (
            ("DFK_P8_L0_TAIL_ACTIVE_VALIDATE", tail_active_validate),
            ("DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE", overlap_active_validate),
        )
        if active
    ]
    if conflicts:
        raise ValueError(
            f"{DFK_P8_L0_ATTRIBUTION_ENV} cannot run with " + ", ".join(conflicts)
        )


@dataclass(frozen=True)
class P8L0TailHookSignature:
    mode: str = P8_L0_TAIL_MODE
    enabled_layers: tuple[int, ...] = P8_L0_TAIL_ENABLED_LAYERS
    workspace_version: int = P8_L0_TAIL_WORKSPACE_VERSION


P8_L0_TAIL_HOOK_SIGNATURE = P8L0TailHookSignature()
P8_L0_TAIL_ACTIVE_HOOK_SIGNATURE = P8L0TailHookSignature(mode=P8_L0_TAIL_ACTIVE_MODE)
P8_L0_OVERLAP_ACTIVE_HOOK_SIGNATURE = P8L0TailHookSignature(
    mode=P8_L0_OVERLAP_ACTIVE_MODE
)


def validate_p8_l0_tail_shadow_mode(
    *,
    enabled: bool,
    native_memory_active: bool,
    native_memory_shadow: bool,
    candidate_memory_shadow: bool,
    accept_zero_probe: bool,
    critical_path_trace: bool,
    future_active: bool,
    active_validate: bool = False,
    server_args,
    selected_layers: tuple[int, ...],
    tp_size: int,
) -> None:
    """Fail closed before allocating capture-tail shadow or active state."""

    if enabled and future_active:
        raise ValueError(
            f"{DFK_P8_L0_TAIL_SHADOW_ENV} and {DFK_P8_L0_TAIL_ACTIVE_ENV} "
            "are mutually exclusive."
        )
    if active_validate and not future_active:
        raise ValueError(
            f"{DFK_P8_L0_TAIL_ACTIVE_VALIDATE_ENV} requires "
            f"{DFK_P8_L0_TAIL_ACTIVE_ENV}=1."
        )
    if not enabled and not future_active:
        return
    mode_env = DFK_P8_L0_TAIL_SHADOW_ENV if enabled else DFK_P8_L0_TAIL_ACTIVE_ENV
    if not native_memory_active:
        raise ValueError(f"{mode_env} requires DFK_NATIVE_MEMORY_ACTIVE=1.")
    conflicts = [
        name
        for name, active in (
            ("DFK_NATIVE_MEMORY_SHADOW", native_memory_shadow),
            ("DFK_CANDIDATE_MEMORY_SHADOW", candidate_memory_shadow),
            ("DFK_ACCEPT_ZERO_PROBE", accept_zero_probe),
            ("DFK_CRITICAL_PATH_TRACE", critical_path_trace),
        )
        if active
    ]
    if conflicts:
        raise ValueError(f"{mode_env} cannot run with " + ", ".join(conflicts))
    if not bool(server_args.disable_overlap_schedule):
        raise ValueError(
            f"{mode_env} requires synchronous scheduling "
            "(--disable-overlap-schedule)."
        )
    if bool(getattr(server_args, "enable_two_batch_overlap", False)):
        raise ValueError(f"{mode_env} does not support two-batch overlap.")
    if int(server_args.page_size) != 1:
        raise ValueError(f"{mode_env} requires page_size == 1.")
    if int(server_args.speculative_num_draft_tokens) != VERIFY_WIDTH:
        raise ValueError(f"{mode_env} requires verify width {VERIFY_WIDTH}.")
    if not selected_layers or int(selected_layers[0]) != 0:
        raise ValueError(f"{mode_env} requires selected target layer 0 first.")
    if future_active and tuple(selected_layers) != P8_L0_TAIL_ACTIVE_SELECTED_LAYERS:
        raise ValueError(
            f"{DFK_P8_L0_TAIL_ACTIVE_ENV} requires selected target layers "
            f"{P8_L0_TAIL_ACTIVE_SELECTED_LAYERS}."
        )
    if int(tp_size) not in (1, 2):
        raise ValueError(f"{mode_env} supports TP1 or TP2 only.")

    graph_config = getattr(server_args, "cuda_graph_config", None)
    decode_config = getattr(graph_config, "decode", None)
    if (
        decode_config is None
        or getattr(decode_config, "backend", "disabled") == "disabled"
    ):
        raise ValueError(f"{mode_env} requires target decode CUDA graph.")
    tiers = tuple(int(value) for value in (getattr(decode_config, "bs", None) or ()))
    if not tiers or min(tiers) <= 0:
        raise ValueError(f"{mode_env} requires positive graph batch tiers.")
    configured_max = getattr(decode_config, "max_bs", None)
    maxima = [max(tiers)]
    if configured_max is not None:
        maxima.append(int(configured_max))
    if max(maxima) > P8_L0_TAIL_MAX_BATCH:
        raise ValueError(
            f"{mode_env} requires graph max tier <= "
            f"{P8_L0_TAIL_MAX_BATCH}, got tiers={tiers}, max_bs={configured_max}."
        )


@dataclass(frozen=True)
class P8L0TailRound:
    generation: int
    batch_size: int


@dataclass(frozen=True)
class P8L0TailPreparedBank:
    bank: DraftFreeKVLocalMemoryBank
    generation: int
    batch_size: int
    graph_batch_size: int
    commit_lens: torch.Tensor
    selected_indices: torch.Tensor
    source_end: torch.Tensor
    req_pool_indices: torch.Tensor
    selected_cache_locs: torch.Tensor
    row_checks: dict[str, bool]
    slot_checks: dict[str, bool]


@dataclass(frozen=True)
class P8L0TailActiveIdentity:
    hook_signature: P8L0TailHookSignature
    generation: int
    batch_size: int
    request_ids: tuple[object, ...]
    req_generations: tuple[int, ...]
    req_pool_indices: tuple[int, ...]
    source_end: tuple[int, ...]
    selected_cache_locs: tuple[int, ...]


@dataclass(frozen=True)
class P8L0TailActiveHandoff:
    bank: DraftFreeKVLocalMemoryBank
    identity: P8L0TailActiveIdentity


def _new_p8_l0_attribution_state() -> dict:
    return {
        "reported": False,
        "proposal_take_attempt_rounds": 0,
        "armed_graph_rounds": 0,
        "completed_staged_producer_rounds": 0,
        "staged_rows": 0,
        "consumed_composed_rounds": 0,
        "consumed_rows": 0,
        "unarmed_rounds": 0,
        "eager_fallback_rounds": 0,
        "missing_host_length_rounds": 0,
        "proposal_actual_batch_histogram": {},
        "staged_actual_batch_histogram": {},
        "captured_graph_tier_histogram": {},
        "committed_prefix_row_histogram": {
            label: 0 for label, _lower, _upper in P8_L0_ATTRIBUTION_PREFIX_BUCKETS
        },
        "committed_prefix_row_count": 0,
        "committed_prefix_row_sum": 0,
        "committed_prefix_row_min": None,
        "committed_prefix_row_max": None,
    }


class P8L0TailWorkspace:
    """One per-rank B16 external workspace shared by every graph tier."""

    def __init__(
        self,
        *,
        local_kv_heads: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_batch_size: int = P8_L0_TAIL_MAX_BATCH,
    ) -> None:
        if max_batch_size != P8_L0_TAIL_MAX_BATCH:
            raise ValueError(f"P8 L0 tail workspace requires B{P8_L0_TAIL_MAX_BATCH}.")
        if local_kv_heads <= 0:
            raise ValueError("local_kv_heads must be positive.")
        if not dtype.is_floating_point:
            raise ValueError("P8 L0 tail workspace dtype must be floating point.")
        self.max_batch_size = max_batch_size
        self.local_kv_heads = int(local_kv_heads)
        self.dtype = dtype
        self.device = torch.device(device)

        def empty(*shape, dtype=dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)

        def zeros(*shape, dtype=dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        # Replay inputs. They are armed immediately before target verify replay.
        self.input_armed = zeros(dtype=torch.bool)
        self.input_actual_batch = zeros(dtype=torch.int32)
        self.input_generation = torch.full(
            (), -1, dtype=torch.int64, device=self.device
        )
        self.input_row_valid = zeros(max_batch_size, dtype=torch.bool)
        self.input_prefix_lens = zeros(max_batch_size, dtype=torch.int32)
        self.input_req_pool_indices = zeros(max_batch_size, dtype=torch.int32)
        self.input_verify_cache_locs = zeros(
            max_batch_size, VERIFY_WIDTH, dtype=torch.int64
        )

        # Replay outputs. No eight full [aggregate+raw] banks are stored.
        self.output_armed = zeros(dtype=torch.bool)
        self.output_actual_batch = zeros(dtype=torch.int32)
        self.output_graph_batch = zeros(dtype=torch.int32)
        self.output_generation = torch.full(
            (max_batch_size,), -1, dtype=torch.int64, device=self.device
        )
        self.output_row_valid = zeros(max_batch_size, dtype=torch.bool)
        self.output_prefix_lens = zeros(max_batch_size, dtype=torch.int32)
        self.output_req_pool_indices = zeros(max_batch_size, dtype=torch.int32)
        self.output_verify_cache_locs = zeros(
            max_batch_size, VERIFY_WIDTH, dtype=torch.int64
        )
        self.output_source_end = zeros(max_batch_size, VERIFY_WIDTH, dtype=torch.int64)
        self.output_req_match = zeros(max_batch_size, dtype=torch.bool)
        self.output_prefix_match = zeros(max_batch_size, dtype=torch.bool)
        self.output_slot_match = zeros(max_batch_size, dtype=torch.bool)

        aggregate_union_rows = VERIFY_WIDTH * HYBRID_AGGREGATE_ROWS
        self.aggregate_values = empty(
            max_batch_size,
            VERIFY_WIDTH,
            local_kv_heads,
            HYBRID_AGGREGATE_ROWS,
            HEAD_DIM,
        )
        self.raw_union_keys = empty(
            max_batch_size,
            HYBRID_RAW_UNION_ROWS,
            local_kv_heads,
            HEAD_DIM,
        )
        self.raw_union_values = empty(
            max_batch_size,
            HYBRID_RAW_UNION_ROWS,
            local_kv_heads,
            HEAD_DIM,
        )
        self.aggregate_union_locs = zeros(
            max_batch_size, aggregate_union_rows, dtype=torch.int32
        )
        self.aggregate_union_inverse = zeros(
            max_batch_size,
            VERIFY_WIDTH,
            HYBRID_AGGREGATE_ROWS,
            dtype=torch.int64,
        )
        self.aggregate_valid = zeros(
            max_batch_size,
            VERIFY_WIDTH,
            HYBRID_AGGREGATE_ROWS,
            dtype=torch.bool,
        )
        self.aggregate_union_valid = zeros(
            max_batch_size, aggregate_union_rows, dtype=torch.bool
        )
        self.raw_union_inverse = zeros(
            max_batch_size,
            VERIFY_WIDTH,
            HYBRID_RAW_ROWS,
            dtype=torch.int64,
        )
        self.raw_valid = zeros(
            max_batch_size, VERIFY_WIDTH, HYBRID_RAW_ROWS, dtype=torch.bool
        )
        self.raw_union_valid = zeros(
            max_batch_size, HYBRID_RAW_UNION_ROWS, dtype=torch.bool
        )

    def invalidate_output(self) -> None:
        self.output_armed.zero_()
        self.output_actual_batch.zero_()
        self.output_graph_batch.zero_()
        self.output_generation.fill_(-1)
        self.output_row_valid.zero_()

    def disarm(self) -> None:
        self.input_armed.zero_()
        self.input_actual_batch.zero_()
        self.input_row_valid.zero_()


def register_p8_l0_tail_capture_hook(model_runner, owner) -> None:
    """Register exactly once, before target decode graph construction."""

    if getattr(model_runner, "decode_cuda_graph_runner", None) is not None:
        raise RuntimeError(
            "DFK P8 L0 capture-tail hook registration is too late: target decode "
            "graph runner already exists."
        )
    owner_signature = getattr(owner, "hook_signature", P8_L0_TAIL_HOOK_SIGNATURE)
    if not isinstance(owner_signature, P8L0TailHookSignature):
        raise RuntimeError("DFK P8 L0 capture-tail owner has an invalid signature.")
    existing_signature = getattr(model_runner, _HOOK_SIGNATURE_ATTR, None)
    if existing_signature is not None:
        if existing_signature != owner_signature:
            raise RuntimeError(
                "DFK P8 L0 capture-tail found a stale or mixed hook signature: "
                f"{existing_signature!r}."
            )
        raise RuntimeError("DFK P8 L0 capture-tail hook was registered twice.")
    hooks = getattr(model_runner, "capture_tail_hooks", None)
    if not isinstance(hooks, list):
        raise RuntimeError("target model runner has no capture_tail_hooks list.")
    hook = owner.capture_hook
    if any(existing == hook for existing in hooks):
        raise RuntimeError("DFK P8 L0 capture-tail hook was registered twice.")
    hooks.append(hook)
    setattr(model_runner, _HOOK_SIGNATURE_ATTR, owner_signature)
    setattr(model_runner, _HOOK_OWNER_ATTR, owner)


class P8L0TailShadow:
    """Default-off capture/replay shadow; never supplies proposal memory."""

    hook_signature = P8_L0_TAIL_HOOK_SIGNATURE

    def __init__(
        self,
        *,
        tp_rank: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seed_down,
        seed_up,
        log_alpha: torch.Tensor,
        req_to_token: torch.Tensor,
        policy_lut: HybridRaw192Agg64PolicyLUT,
        emit: Callable[[str], None],
    ) -> None:
        if key_cache.dim() != 3 or key_cache.shape[2] != HEAD_DIM:
            raise ValueError("P8 L0 tail requires rank-3 target key cache.")
        if value_cache.shape != key_cache.shape:
            raise ValueError("P8 L0 tail key/value cache shapes must match.")
        if req_to_token.dtype != torch.int32 or not req_to_token.is_contiguous():
            raise ValueError("P8 L0 tail requires contiguous int32 req_to_token.")
        if policy_lut.device != key_cache.device:
            raise ValueError("P8 L0 tail policy LUT must share the cache device.")
        self.tp_rank = int(tp_rank)
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.seed_down = seed_down
        self.seed_up = seed_up
        self.log_alpha = log_alpha
        self.req_to_token = req_to_token
        self.policy_lut = policy_lut
        self.emit = emit
        self.workspace = P8L0TailWorkspace(
            local_kv_heads=key_cache.shape[1],
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        self._generation = 0
        self._in_flight = False
        self._pending = True
        self._prepared: P8L0TailPreparedBank | None = None
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []
        self._captured_graph_tiers: set[int] = set()

    @property
    def pending(self) -> bool:
        return self._pending

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def capture_hook(
        self, graph_runner, _output, forward_batch, num_tokens: int
    ) -> None:
        graph_batch = self._validate_capture_call(
            graph_runner, forward_batch, num_tokens
        )
        self._produce_and_store_capture_outputs(graph_batch)
        self._stamp_capture_output_identity(forward_batch, graph_batch)

    def _validate_capture_call(
        self, graph_runner, forward_batch, num_tokens: int
    ) -> int:
        """Validate and register one serial or overlapped graph-build call."""

        graph_batch = int(forward_batch.batch_size)
        if graph_batch <= 0 or graph_batch > self.workspace.max_batch_size:
            raise RuntimeError(
                f"P8 L0 tail graph batch is out of range: {graph_batch}."
            )
        if int(num_tokens) != graph_batch * VERIFY_WIDTH:
            raise RuntimeError(
                "P8 L0 tail hook requires all8 target verify capture, got "
                f"batch={graph_batch}, num_tokens={num_tokens}."
            )
        if getattr(graph_runner, "model_runner", None) is None:
            raise RuntimeError("P8 L0 tail hook requires a target graph runner.")
        self._captured_graph_tiers.add(graph_batch)
        return graph_batch

    def _produce_and_store_capture_outputs(self, graph_batch: int) -> None:
        """Run the shared candidate plan, native producer, and workspace copies."""

        workspace = self.workspace
        plan = build_draft_free_kv_candidate_memory_plan(
            workspace.input_prefix_lens[:graph_batch],
            workspace.input_req_pool_indices[:graph_batch],
            self.req_to_token,
            policy_lut=self.policy_lut,
        )
        parts = build_draft_free_kv_candidate_native_layer_parts(
            self.key_cache,
            self.value_cache,
            self.seed_down,
            self.seed_up,
            self.log_alpha,
            plan,
            self.req_to_token,
        )

        workspace.aggregate_values[:graph_batch].copy_(parts.aggregate_values)
        workspace.raw_union_keys[:graph_batch].copy_(parts.raw_union_keys)
        workspace.raw_union_values[:graph_batch].copy_(parts.raw_union_values)
        workspace.aggregate_union_locs[:graph_batch].copy_(plan.aggregate_union_locs)
        workspace.aggregate_union_inverse[:graph_batch].copy_(
            plan.aggregate_union_inverse
        )
        workspace.aggregate_valid[:graph_batch].copy_(plan.aggregate_valid)
        workspace.aggregate_union_valid[:graph_batch].copy_(plan.aggregate_union_valid)
        workspace.raw_union_inverse[:graph_batch].copy_(plan.raw_union_inverse)
        workspace.raw_valid[:graph_batch].copy_(plan.raw_valid)
        workspace.raw_union_valid[:graph_batch].copy_(plan.raw_union_valid)
        workspace.output_source_end[:graph_batch].copy_(plan.source_end)

    def _stamp_capture_output_identity(self, forward_batch, graph_batch: int) -> None:
        """Apply the shared A2 output-identity stamp after producer completion."""

        workspace = self.workspace
        workspace.output_armed.copy_(workspace.input_armed)
        workspace.output_actual_batch.copy_(workspace.input_actual_batch)
        workspace.output_graph_batch.fill_(graph_batch)
        workspace.output_generation[:graph_batch].copy_(
            workspace.input_generation.expand(graph_batch)
        )
        workspace.output_row_valid[:graph_batch].copy_(
            workspace.input_row_valid[:graph_batch]
        )
        workspace.output_prefix_lens[:graph_batch].copy_(
            workspace.input_prefix_lens[:graph_batch]
        )
        workspace.output_req_pool_indices[:graph_batch].copy_(
            workspace.input_req_pool_indices[:graph_batch]
        )
        workspace.output_verify_cache_locs[:graph_batch].copy_(
            workspace.input_verify_cache_locs[:graph_batch]
        )

        graph_req = forward_batch.req_pool_indices[:graph_batch].to(torch.int32)
        graph_prefix = forward_batch.seq_lens[:graph_batch].to(torch.int32)
        graph_slots = forward_batch.out_cache_loc[: graph_batch * VERIFY_WIDTH].view(
            graph_batch, VERIFY_WIDTH
        )
        workspace.output_req_match[:graph_batch].copy_(
            graph_req == workspace.input_req_pool_indices[:graph_batch]
        )
        workspace.output_prefix_match[:graph_batch].copy_(
            graph_prefix == workspace.input_prefix_lens[:graph_batch]
        )
        workspace.output_slot_match[:graph_batch].copy_(
            (
                graph_slots.to(torch.int64)
                == workspace.input_verify_cache_locs[:graph_batch]
            ).all(dim=1)
        )

    def validate_capture_complete(self, *, expected_tiers: tuple[int, ...]) -> None:
        expected = set(int(tier) for tier in expected_tiers)
        if not expected or self._captured_graph_tiers != expected:
            raise RuntimeError(
                "P8 L0 capture-tail graph tiers were not captured exactly once per "
                f"configured shape: expected={sorted(expected)}, "
                f"observed={sorted(self._captured_graph_tiers)}."
            )

    def arm(
        self,
        *,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        verify_cache_locs: torch.Tensor,
    ) -> P8L0TailRound:
        if not self._pending:
            raise RuntimeError("P8 L0 tail shadow already completed.")
        if self._in_flight or self._prepared is not None:
            raise RuntimeError("P8 L0 tail workspace permits one in-flight round.")
        batch_size = int(prefix_lens.numel())
        if not 1 <= batch_size <= self.workspace.max_batch_size:
            raise ValueError("P8 L0 tail batch size must be between 1 and 16.")
        expected_vector = (batch_size,)
        if tuple(prefix_lens.shape) != expected_vector:
            raise ValueError(f"prefix_lens must have shape {expected_vector}.")
        if tuple(req_pool_indices.shape) != expected_vector:
            raise ValueError(f"req_pool_indices must have shape {expected_vector}.")
        if tuple(verify_cache_locs.shape) != (batch_size, VERIFY_WIDTH):
            raise ValueError(
                "verify_cache_locs must have shape " f"{(batch_size, VERIFY_WIDTH)}."
            )
        for name, tensor in (
            ("prefix_lens", prefix_lens),
            ("req_pool_indices", req_pool_indices),
            ("verify_cache_locs", verify_cache_locs),
        ):
            if tensor.device != self.workspace.device:
                raise ValueError(
                    f"{name} must be on {self.workspace.device}, got {tensor.device}."
                )

        self._generation += 1
        workspace = self.workspace
        workspace.disarm()
        workspace.invalidate_output()
        workspace.input_prefix_lens.zero_()
        workspace.input_req_pool_indices.zero_()
        workspace.input_verify_cache_locs.zero_()
        workspace.input_prefix_lens[:batch_size].copy_(prefix_lens.to(torch.int32))
        workspace.input_req_pool_indices[:batch_size].copy_(
            req_pool_indices.to(torch.int32)
        )
        workspace.input_verify_cache_locs[:batch_size].copy_(
            verify_cache_locs.to(torch.int64)
        )
        workspace.input_row_valid[:batch_size].fill_(True)
        workspace.input_actual_batch.fill_(batch_size)
        workspace.input_generation.fill_(self._generation)
        workspace.input_armed.fill_(True)
        self._in_flight = True
        return P8L0TailRound(generation=self._generation, batch_size=batch_size)

    def note_fallback(self, round_state: P8L0TailRound, *, reason: str) -> None:
        self._require_round(round_state)
        self._fallback_count += 1
        if reason not in self._fallback_reasons:
            self._fallback_reasons.append(reason)
        self.workspace.disarm()
        self.workspace.invalidate_output()
        self._in_flight = False

    def note_unarmed_fallback(self, *, reason: str) -> None:
        if self._in_flight or self._prepared is not None:
            raise RuntimeError("P8 L0 fallback collided with an active workspace.")
        self._fallback_count += 1
        if reason not in self._fallback_reasons:
            self._fallback_reasons.append(reason)
        self.workspace.disarm()
        self.workspace.invalidate_output()

    def _require_round(self, round_state: P8L0TailRound) -> None:
        if not self._in_flight:
            raise RuntimeError("P8 L0 tail has no in-flight round.")
        if round_state.generation != self._generation:
            raise RuntimeError("P8 L0 tail generation mismatch.")
        if not 1 <= round_state.batch_size <= self.workspace.max_batch_size:
            raise RuntimeError("P8 L0 tail round batch is invalid.")

    def _capture_checks(
        self, round_state: P8L0TailRound
    ) -> tuple[int, dict[str, bool], dict[str, bool]]:
        workspace = self.workspace
        batch_size = round_state.batch_size
        graph_batch = int(workspace.output_graph_batch.item())
        row_checks = {
            "armed": bool(workspace.output_armed.item()),
            "actual_batch": int(workspace.output_actual_batch.item()) == batch_size,
            "graph_tier": batch_size <= graph_batch <= workspace.max_batch_size,
            "row_valid": bool(workspace.output_row_valid[:batch_size].all().item()),
            "padded_rows_invalid": bool(
                (~workspace.output_row_valid[batch_size:graph_batch]).all().item()
            ),
            "generation": bool(
                (workspace.output_generation[:batch_size] == round_state.generation)
                .all()
                .item()
            ),
            "req_identity": bool(workspace.output_req_match[:batch_size].all().item()),
            "prefix_identity": bool(
                workspace.output_prefix_match[:batch_size].all().item()
            ),
        }
        slot_checks = {
            "graph_slots_match_arm": bool(
                workspace.output_slot_match[:batch_size].all().item()
            ),
            "stamped_slots_match_arm": torch.equal(
                workspace.output_verify_cache_locs[:batch_size],
                workspace.input_verify_cache_locs[:batch_size],
            ),
        }
        if not all(row_checks.values()) or not all(slot_checks.values()):
            raise RuntimeError(
                "P8 L0 tail replay identity mismatch: "
                f"rows={row_checks}, slots={slot_checks}."
            )
        return graph_batch, row_checks, slot_checks

    @torch.inference_mode()
    def select_and_copy(
        self,
        round_state: P8L0TailRound,
        *,
        commit_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> P8L0TailPreparedBank:
        """Compact selected L0 bank before scheduler publication can reuse pages."""

        self._require_round(round_state)
        try:
            graph_batch, row_checks, slot_checks = self._capture_checks(round_state)
            batch_size = round_state.batch_size
            for name, tensor in (
                ("commit_lens", commit_lens),
                ("new_seq_lens", new_seq_lens),
                ("req_pool_indices", req_pool_indices),
            ):
                if tuple(tensor.shape) != (batch_size,):
                    raise ValueError(f"{name} must have shape {(batch_size,)}.")
                if tensor.device != self.workspace.device:
                    raise ValueError(f"{name} must be on {self.workspace.device}.")

            commit_lens_i64 = commit_lens.to(torch.int64)
            if bool(
                ((commit_lens_i64 < 1) | (commit_lens_i64 > VERIFY_WIDTH)).any().item()
            ):
                raise RuntimeError("P8 L0 selected commit length must be in [1, 8].")
            selected = commit_lens_i64 - 1
            workspace = self.workspace
            prefix = workspace.output_prefix_lens[:batch_size].to(torch.int64)
            new_seq_i64 = new_seq_lens.to(torch.int64)
            if not torch.equal(new_seq_i64 - prefix, commit_lens_i64):
                raise RuntimeError("P8 L0 committed delta does not match commit_lens.")
            row = torch.arange(batch_size, device=workspace.device)
            selected_source_end = workspace.output_source_end[:batch_size][
                row, selected
            ]
            if not torch.equal(selected_source_end, new_seq_i64):
                raise RuntimeError("P8 L0 selected source_end is not committed length.")
            stamped_reqs = workspace.output_req_pool_indices[:batch_size]
            if not torch.equal(stamped_reqs, req_pool_indices.to(torch.int32)):
                raise RuntimeError("P8 L0 request rows changed before publication.")
            selected_cache_locs = workspace.output_verify_cache_locs[:batch_size][
                row, selected
            ]
            committed_cache_locs = self.req_to_token[
                stamped_reqs.to(torch.int64), new_seq_i64 - 1
            ].to(torch.int64)
            slot_checks["selected_slot_is_committed"] = torch.equal(
                selected_cache_locs, committed_cache_locs
            )
            if not slot_checks["selected_slot_is_committed"]:
                raise RuntimeError("P8 L0 selected verify slot is not committed.")

            aggregate_inverse = workspace.aggregate_union_inverse[:batch_size][
                row, selected
            ]
            aggregate_locs = workspace.aggregate_union_locs[:batch_size].gather(
                1, aggregate_inverse
            )
            aggregate_keys = self.key_cache.index_select(
                0, aggregate_locs.reshape(-1).to(torch.int64)
            ).view(
                batch_size,
                HYBRID_AGGREGATE_ROWS,
                workspace.local_kv_heads,
                HEAD_DIM,
            )
            aggregate_keys = aggregate_keys.permute(0, 2, 1, 3)
            aggregate_values = workspace.aggregate_values[:batch_size][row, selected]

            raw_inverse = workspace.raw_union_inverse[:batch_size][row, selected]
            raw_gather = raw_inverse[:, :, None, None].expand(
                -1, -1, workspace.local_kv_heads, HEAD_DIM
            )
            raw_keys = workspace.raw_union_keys[:batch_size].gather(1, raw_gather)
            raw_values = workspace.raw_union_values[:batch_size].gather(1, raw_gather)
            raw_keys = raw_keys.permute(0, 2, 1, 3)
            raw_values = raw_values.permute(0, 2, 1, 3)
            valid_mask = torch.cat(
                (
                    workspace.aggregate_valid[:batch_size][row, selected],
                    workspace.raw_valid[:batch_size][row, selected],
                ),
                dim=1,
            )
            valid_values = valid_mask[:, None, :, None]
            keys = torch.cat((aggregate_keys, raw_keys), dim=2).masked_fill(
                ~valid_values, 0
            )
            values = torch.cat((aggregate_values, raw_values), dim=2).masked_fill(
                ~valid_values, 0
            )
            expected_shape = (
                batch_size,
                workspace.local_kv_heads,
                HYBRID_BANK_ROWS,
                HEAD_DIM,
            )
            if (
                tuple(keys.shape) != expected_shape
                or tuple(values.shape) != expected_shape
            ):
                raise RuntimeError("P8 L0 compact bank shape invariant failed.")

            prepared = P8L0TailPreparedBank(
                bank=DraftFreeKVLocalMemoryBank(
                    keys=keys.clone(),
                    values=values.clone(),
                    valid_mask=valid_mask.clone(),
                ),
                generation=round_state.generation,
                batch_size=batch_size,
                graph_batch_size=graph_batch,
                commit_lens=commit_lens_i64.clone(),
                selected_indices=selected.clone(),
                source_end=selected_source_end.clone(),
                req_pool_indices=stamped_reqs.clone(),
                selected_cache_locs=selected_cache_locs.clone(),
                row_checks=row_checks,
                slot_checks=slot_checks,
            )
            self._prepared = prepared
            return prepared
        finally:
            self.workspace.disarm()
            self._in_flight = False

    def compare_and_emit(
        self,
        prepared: P8L0TailPreparedBank,
        *,
        reference_bank: DraftFreeKVLocalMemoryBank,
        target_result_can_run_cuda_graph: bool,
    ) -> bool:
        if not self._pending:
            return False
        if self._prepared is not prepared:
            raise RuntimeError("P8 L0 shadow-ready bank identity mismatch.")
        if not target_result_can_run_cuda_graph:
            raise RuntimeError("P8 L0 captured bank cannot be consumed after fallback.")
        candidate = prepared.bank
        if candidate.keys.shape != reference_bank.keys.shape:
            raise RuntimeError("P8 L0 reference key shape mismatch.")
        if candidate.values.shape != reference_bank.values.shape:
            raise RuntimeError("P8 L0 reference value shape mismatch.")
        if candidate.valid_mask.shape != reference_bank.valid_mask.shape:
            raise RuntimeError("P8 L0 reference mask shape mismatch.")

        key_exact = torch.equal(candidate.keys, reference_bank.keys)
        raw_exact = torch.equal(
            candidate.values[:, :, HYBRID_AGGREGATE_ROWS:],
            reference_bank.values[:, :, HYBRID_AGGREGATE_ROWS:],
        )
        mask_exact = torch.equal(candidate.valid_mask, reference_bank.valid_mask)
        aggregate_mask = reference_bank.valid_mask[
            :, None, :HYBRID_AGGREGATE_ROWS, None
        ].expand_as(reference_bank.values[:, :, :HYBRID_AGGREGATE_ROWS])
        candidate_aggregate = (
            candidate.values[:, :, :HYBRID_AGGREGATE_ROWS][aggregate_mask]
            .detach()
            .float()
            .cpu()
        )
        reference_aggregate = (
            reference_bank.values[:, :, :HYBRID_AGGREGATE_ROWS][aggregate_mask]
            .detach()
            .float()
            .cpu()
        )
        delta = (candidate_aggregate - reference_aggregate).abs()

        def statistic(kind: str, quantile: float | None = None) -> float:
            if delta.numel() == 0:
                return 0.0
            if quantile is not None:
                return float(torch.quantile(delta, quantile).item())
            return float(getattr(delta, kind)().item())

        aggregate_error = {
            "max": statistic("max"),
            "mean": statistic("mean"),
            "p99": statistic("quantile", 0.99),
            "p999": statistic("quantile", 0.999),
        }
        aggregate_exact = torch.equal(candidate_aggregate, reference_aggregate)
        structural_exact = key_exact and raw_exact and mask_exact
        aggregate_within_threshold = all(
            aggregate_error[name] <= threshold
            for name, threshold in P8_L0_TAIL_AGGREGATE_THRESHOLDS.items()
        )
        gate_pass = structural_exact and aggregate_within_threshold
        payload = {
            "schema": "dfk_p8_l0_tail_shadow_v1",
            "schema_version": 1,
            "mode": P8_L0_TAIL_MODE,
            "workspace_version": P8_L0_TAIL_WORKSPACE_VERSION,
            "enabled_layers": list(P8_L0_TAIL_ENABLED_LAYERS),
            "scope": "worker_first_graph_verify_pre_publish",
            "tp_rank": self.tp_rank,
            "target_result_can_run_cuda_graph": True,
            "batch_size": prepared.batch_size,
            "graph_batch_size": prepared.graph_batch_size,
            "generation": prepared.generation,
            "commit_lens": prepared.commit_lens.cpu().tolist(),
            "selected_indices": prepared.selected_indices.cpu().tolist(),
            "source_end": prepared.source_end.cpu().tolist(),
            "req_pool_indices": prepared.req_pool_indices.cpu().tolist(),
            "selected_cache_locs": prepared.selected_cache_locs.cpu().tolist(),
            "eligibility": {
                "eligible": True,
                "fallback_count": self._fallback_count,
                "fallback_reasons": list(self._fallback_reasons),
            },
            "row_checks": prepared.row_checks,
            "slot_checks": prepared.slot_checks,
            "memory_shape": list(candidate.keys.shape),
            "exact_flags": {
                "keys": key_exact,
                "raw_values": raw_exact,
                "mask": mask_exact,
                "aggregate_values": aggregate_exact,
                "structural_required": structural_exact,
                "required": gate_pass,
                "all": structural_exact and aggregate_exact,
            },
            "aggregate_abs_error": aggregate_error,
            "aggregate_gate": {
                "thresholds": dict(P8_L0_TAIL_AGGREGATE_THRESHOLDS),
                "within_threshold": aggregate_within_threshold,
            },
            "reference_aggregate_abs_max": (
                0.0
                if reference_aggregate.numel() == 0
                else float(reference_aggregate.abs().max().item())
            ),
        }
        self._pending = False
        self._prepared = None
        self.emit(
            DFK_P8_L0_TAIL_SHADOW_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        if not structural_exact:
            raise RuntimeError("P8 L0 capture-tail failed exact key/raw/mask parity.")
        if not aggregate_within_threshold:
            raise RuntimeError(
                "P8 L0 capture-tail aggregate error exceeded the validated "
                f"thresholds: stats={aggregate_error}, "
                f"thresholds={P8_L0_TAIL_AGGREGATE_THRESHOLDS}."
            )
        return True


class P8L0TailActive(P8L0TailShadow):
    """Serial whole-batch ownership controller for one next-round L0 bank."""

    hook_signature = P8_L0_TAIL_ACTIVE_HOOK_SIGNATURE
    _KNOWN_FALLBACK_REASONS = (
        "no_prepared",
        "double_consume",
        "batch_size_mismatch",
        "request_id_mismatch",
        "req_pool_indices_mismatch",
        "source_end_mismatch",
        "req_generation_mismatch",
        "committed_slot_mismatch",
        "signature_mismatch",
        "controller_generation_mismatch",
        "prepared_generation_mismatch",
        "identity_invalid",
        "prepare_for_verify_not_graph_eligible",
        "target_forward_eager_fallback",
        "prefill",
        "idle",
        "clear_cache_pool",
        "eos",
        "cancel",
        "proposal_exception",
        "validation_failed",
        "other",
    )

    def __init__(
        self,
        *,
        tp_rank: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seed_down,
        seed_up,
        log_alpha: torch.Tensor,
        req_to_token: torch.Tensor,
        req_generation: torch.Tensor,
        policy_lut: HybridRaw192Agg64PolicyLUT,
        emit: Callable[[str], None],
        validate: bool,
        attribution_enabled: bool = False,
    ) -> None:
        super().__init__(
            tp_rank=tp_rank,
            key_cache=key_cache,
            value_cache=value_cache,
            seed_down=seed_down,
            seed_up=seed_up,
            log_alpha=log_alpha,
            req_to_token=req_to_token,
            policy_lut=policy_lut,
            emit=emit,
        )
        if (
            req_generation.dim() != 1
            or req_generation.dtype != torch.int64
            or req_generation.device.type != "cpu"
            or req_generation.numel() != req_to_token.shape[0]
        ):
            raise ValueError(
                "P8 L0 active requires the host int64 ReqToTokenPool "
                "req_generation vector."
            )
        self.req_generation = req_generation
        self.validate = bool(validate)
        self._attribution_enabled = bool(attribution_enabled)
        if self._attribution_enabled:
            self._attribution = _new_p8_l0_attribution_state()
        self._active_identity: P8L0TailActiveIdentity | None = None
        self._ready_generation: int | None = None
        self._last_consumed_identity: P8L0TailActiveIdentity | None = None
        self._last_consumed_generation = 0
        self._outstanding_handoff_generation: int | None = None
        self._handoff_validation: dict | None = None
        self._fallback_reason_counts = {
            reason: 0 for reason in self._KNOWN_FALLBACK_REASONS
        }
        self._event_count = 0
        self._bootstrap_emitted = False

    @property
    def has_prepared(self) -> bool:
        return self._prepared is not None

    @property
    def fallback_reason_counts(self) -> dict[str, int]:
        return {
            reason: count
            for reason, count in self._fallback_reason_counts.items()
            if count
        }

    @property
    def attribution_enabled(self) -> bool:
        return self._attribution_enabled

    @staticmethod
    def _increment_attribution_histogram(histogram: dict, value: int) -> None:
        key = int(value)
        histogram[key] = histogram.get(key, 0) + 1

    def _note_attribution_proposal_attempt(
        self, *, batch_size: int, host_prefix_lens
    ) -> None:
        state = self._attribution
        state["proposal_take_attempt_rounds"] += 1
        self._increment_attribution_histogram(
            state["proposal_actual_batch_histogram"], batch_size
        )

        values = None
        if host_prefix_lens is not None:
            host_device = getattr(host_prefix_lens, "device", None)
            if host_device is not None and getattr(host_device, "type", None) != "cpu":
                values = None
            elif isinstance(host_prefix_lens, torch.Tensor):
                if (
                    host_prefix_lens.device.type == "cpu"
                    and host_prefix_lens.dim() == 1
                ):
                    values = host_prefix_lens.tolist()
            else:
                try:
                    values = list(host_prefix_lens)
                except (TypeError, ValueError):
                    values = None
        if values is None or len(values) != int(batch_size):
            state["missing_host_length_rounds"] += 1
            return
        try:
            lengths = tuple(int(value) for value in values)
        except (TypeError, ValueError, OverflowError):
            state["missing_host_length_rounds"] += 1
            return
        if any(length < 0 for length in lengths):
            state["missing_host_length_rounds"] += 1
            return

        prefix_histogram = state["committed_prefix_row_histogram"]
        for length in lengths:
            for label, lower, upper in P8_L0_ATTRIBUTION_PREFIX_BUCKETS:
                if length >= lower and (upper is None or length < upper):
                    prefix_histogram[label] += 1
                    break
        state["committed_prefix_row_count"] += len(lengths)
        state["committed_prefix_row_sum"] += sum(lengths)
        minimum = min(lengths)
        maximum = max(lengths)
        current_minimum = state["committed_prefix_row_min"]
        current_maximum = state["committed_prefix_row_max"]
        state["committed_prefix_row_min"] = (
            minimum if current_minimum is None else min(current_minimum, minimum)
        )
        state["committed_prefix_row_max"] = (
            maximum if current_maximum is None else max(current_maximum, maximum)
        )

    def report_attribution(self) -> bool:
        if not self._attribution_enabled:
            return False
        state = self._attribution
        if state["reported"]:
            return False
        state["reported"] = True
        fallback_reason_counts = self.fallback_reason_counts

        def ratio(numerator: int, denominator: int) -> float | None:
            return None if denominator == 0 else numerator / denominator

        staged_rounds = state["completed_staged_producer_rounds"]
        staged_rows = state["staged_rows"]
        consumed_rounds = state["consumed_composed_rounds"]
        consumed_rows = state["consumed_rows"]
        signature = self.hook_signature
        payload = {
            "schema": "dfk_p8_l0_attribution_v1",
            "schema_version": 1,
            "diagnostic_only": True,
            "reason": "clear_cache_pool",
            "mode": signature.mode,
            "signature": {
                "mode": signature.mode,
                "enabled_layers": list(signature.enabled_layers),
                "workspace_version": signature.workspace_version,
            },
            "tp_rank": self.tp_rank,
            "counts": {
                "proposal_take_attempt_rounds": state["proposal_take_attempt_rounds"],
                "armed_graph_rounds": state["armed_graph_rounds"],
                "completed_staged_producer_rounds": staged_rounds,
                "staged_rows": staged_rows,
                "consumed_composed_rounds": consumed_rounds,
                "consumed_rows": consumed_rows,
                "unarmed_rounds": state["unarmed_rounds"],
                "eager_fallback_rounds": state["eager_fallback_rounds"],
                "fallback_rounds": sum(fallback_reason_counts.values()),
                "missing_host_length_rounds": state["missing_host_length_rounds"],
            },
            "fallback_reason_counts": fallback_reason_counts,
            "histograms": {
                "proposal_actual_batch": dict(state["proposal_actual_batch_histogram"]),
                "staged_actual_batch": dict(state["staged_actual_batch_histogram"]),
                "captured_graph_tier": dict(state["captured_graph_tier_histogram"]),
                "committed_prefix_rows": dict(state["committed_prefix_row_histogram"]),
            },
            "committed_prefix_rows": {
                "row_count": state["committed_prefix_row_count"],
                "sum": state["committed_prefix_row_sum"],
                "min": state["committed_prefix_row_min"],
                "max": state["committed_prefix_row_max"],
                "missing_host_length_rounds": state["missing_host_length_rounds"],
            },
            "ratios": {
                "staged_over_proposal_rounds": ratio(
                    staged_rounds, state["proposal_take_attempt_rounds"]
                ),
                "consumed_over_staged_rounds": ratio(consumed_rounds, staged_rounds),
                "consumed_rows_over_staged_rows": ratio(consumed_rows, staged_rows),
            },
            "state": {
                "outstanding": self._outstanding_handoff_generation is not None,
                "outstanding_generation": self._outstanding_handoff_generation,
                "prepared": self._prepared is not None,
                "prepared_generation": (
                    None if self._prepared is None else self._prepared.generation
                ),
                "in_flight": self._in_flight,
                "pending": self._pending,
            },
        }
        self.emit(
            DFK_P8_L0_ATTRIBUTION_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return True

    def arm(
        self,
        *,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        verify_cache_locs: torch.Tensor,
    ) -> P8L0TailRound:
        round_state = super().arm(
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            verify_cache_locs=verify_cache_locs,
        )
        if self._attribution_enabled:
            self._attribution["armed_graph_rounds"] += 1
        return round_state

    def _record_reason(self, reason: str) -> str:
        bounded_reason = reason if reason in self._fallback_reason_counts else "other"
        self._fallback_reason_counts[bounded_reason] += 1
        return bounded_reason

    def _emit_lifecycle(self, event: str, **fields) -> None:
        if self._event_count >= P8_L0_TAIL_ACTIVE_EVENT_LIMIT:
            return
        payload = {
            "schema": "dfk_p8_l0_tail_active_lifecycle_v1",
            "schema_version": 1,
            "mode": P8_L0_TAIL_ACTIVE_MODE,
            "workspace_version": P8_L0_TAIL_WORKSPACE_VERSION,
            "enabled_layers": list(P8_L0_TAIL_ENABLED_LAYERS),
            "tp_rank": self.tp_rank,
            "event": event,
            "fallback_reason_counts": self.fallback_reason_counts,
        }
        payload.update(fields)
        self._event_count += 1
        self.emit(
            DFK_P8_L0_TAIL_ACTIVE_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    @staticmethod
    def _json_request_ids(request_ids: tuple[object, ...]) -> list:
        return [
            (
                request_id
                if request_id is None or isinstance(request_id, (str, int, float, bool))
                else str(request_id)
            )
            for request_id in request_ids
        ]

    def _identity_payload(self, identity: P8L0TailActiveIdentity) -> dict:
        signature = identity.hook_signature
        return {
            "generation": identity.generation,
            "controller_generation": self._generation,
            "signature": {
                "mode": signature.mode,
                "enabled_layers": list(signature.enabled_layers),
                "workspace_version": signature.workspace_version,
            },
            "request_ids": self._json_request_ids(identity.request_ids),
            "req_generations": list(identity.req_generations),
            "req_pool_indices": list(identity.req_pool_indices),
            "source_end": list(identity.source_end),
            "selected_cache_locs": list(identity.selected_cache_locs),
        }

    @staticmethod
    def _int_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in tensor.detach().cpu().tolist())

    @staticmethod
    def _request_id_tuple(request_ids) -> tuple[object, ...]:
        return tuple(request_ids)

    def _current_identity_values(
        self,
        *,
        committed_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        request_ids,
    ) -> (
        tuple[
            tuple[object, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
        ]
        | None
    ):
        if committed_seq_lens.dim() != 1 or req_pool_indices.dim() != 1:
            return None
        batch_size = int(committed_seq_lens.numel())
        if req_pool_indices.numel() != batch_size:
            return None
        ids = self._request_id_tuple(request_ids)
        if len(ids) != batch_size:
            return None
        rows = self._int_tuple(req_pool_indices)
        source_end = self._int_tuple(committed_seq_lens)
        if any(row <= 0 or row >= self.req_generation.numel() for row in rows):
            return None
        if any(
            length <= 0 or length > self.req_to_token.shape[1] for length in source_end
        ):
            return None
        row_index = req_pool_indices.to(
            device=self.req_to_token.device, dtype=torch.int64
        )
        end_index = committed_seq_lens.to(
            device=self.req_to_token.device, dtype=torch.int64
        )
        committed_slots = self._int_tuple(
            self.req_to_token[row_index, end_index - 1].to(torch.int64)
        )
        generation_index = torch.tensor(rows, dtype=torch.int64)
        req_generations = self._int_tuple(
            self.req_generation.index_select(0, generation_index)
        )
        return ids, req_generations, rows, source_end, committed_slots

    def _discard_prepared(self, *, reason: str, emit_event: bool) -> None:
        bounded_reason = self._record_reason(reason)
        identity = self._active_identity
        self._prepared = None
        self._active_identity = None
        self._ready_generation = None
        self.workspace.disarm()
        self.workspace.invalidate_output()
        if emit_event:
            self._emit_lifecycle(
                "whole_batch_fallback",
                eligible=False,
                reason=bounded_reason,
                generation=None if identity is None else identity.generation,
            )

    def note_fallback(self, round_state: P8L0TailRound, *, reason: str) -> None:
        super().note_fallback(round_state, reason=reason)
        bounded_reason = self._record_reason(reason)
        if (
            self._attribution_enabled
            and bounded_reason == "target_forward_eager_fallback"
        ):
            self._attribution["eager_fallback_rounds"] += 1

    def note_unarmed_fallback(self, *, reason: str) -> None:
        super().note_unarmed_fallback(reason=reason)
        self._record_reason(reason)
        if self._attribution_enabled:
            self._attribution["unarmed_rounds"] += 1

    def select_and_stage(
        self,
        round_state: P8L0TailRound,
        *,
        commit_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        request_ids,
    ) -> P8L0TailPreparedBank:
        prepared = super().select_and_copy(
            round_state,
            commit_lens=commit_lens,
            new_seq_lens=new_seq_lens,
            req_pool_indices=req_pool_indices,
        )
        current = self._current_identity_values(
            committed_seq_lens=new_seq_lens,
            req_pool_indices=req_pool_indices,
            request_ids=request_ids,
        )
        if current is None:
            self._discard_prepared(reason="identity_invalid", emit_event=True)
            raise RuntimeError("P8 L0 active could not snapshot scheduler identity.")
        ids, req_generations, rows, source_end, committed_slots = current
        if source_end != self._int_tuple(prepared.source_end):
            self._discard_prepared(reason="source_end_mismatch", emit_event=True)
            raise RuntimeError("P8 L0 active staged source_end changed unexpectedly.")
        if rows != self._int_tuple(prepared.req_pool_indices):
            self._discard_prepared(reason="req_pool_indices_mismatch", emit_event=True)
            raise RuntimeError("P8 L0 active staged request rows changed unexpectedly.")
        if committed_slots != self._int_tuple(prepared.selected_cache_locs):
            self._discard_prepared(reason="committed_slot_mismatch", emit_event=True)
            raise RuntimeError(
                "P8 L0 active staged committed slots changed unexpectedly."
            )
        self._active_identity = P8L0TailActiveIdentity(
            hook_signature=self.hook_signature,
            generation=prepared.generation,
            batch_size=prepared.batch_size,
            request_ids=ids,
            req_generations=req_generations,
            req_pool_indices=rows,
            source_end=source_end,
            selected_cache_locs=committed_slots,
        )
        self._ready_generation = None
        return prepared

    @staticmethod
    def _validate_bank_against_oracle(
        candidate: DraftFreeKVLocalMemoryBank,
        reference: DraftFreeKVLocalMemoryBank,
    ) -> dict:
        if (
            candidate.keys.shape != reference.keys.shape
            or candidate.values.shape != reference.values.shape
            or candidate.valid_mask.shape != reference.valid_mask.shape
        ):
            raise RuntimeError("P8 L0 active validation oracle shape mismatch.")
        key_exact = torch.equal(candidate.keys, reference.keys)
        raw_exact = torch.equal(
            candidate.values[:, :, HYBRID_AGGREGATE_ROWS:],
            reference.values[:, :, HYBRID_AGGREGATE_ROWS:],
        )
        mask_exact = torch.equal(candidate.valid_mask, reference.valid_mask)
        aggregate_mask = reference.valid_mask[
            :, None, :HYBRID_AGGREGATE_ROWS, None
        ].expand_as(reference.values[:, :, :HYBRID_AGGREGATE_ROWS])
        candidate_aggregate = (
            candidate.values[:, :, :HYBRID_AGGREGATE_ROWS][aggregate_mask]
            .detach()
            .float()
            .cpu()
        )
        reference_aggregate = (
            reference.values[:, :, :HYBRID_AGGREGATE_ROWS][aggregate_mask]
            .detach()
            .float()
            .cpu()
        )
        delta = (candidate_aggregate - reference_aggregate).abs()

        def statistic(kind: str, quantile: float | None = None) -> float:
            if delta.numel() == 0:
                return 0.0
            if quantile is not None:
                return float(torch.quantile(delta, quantile).item())
            return float(getattr(delta, kind)().item())

        aggregate_error = {
            "max": statistic("max"),
            "mean": statistic("mean"),
            "p99": statistic("quantile", 0.99),
            "p999": statistic("quantile", 0.999),
        }
        structural_exact = key_exact and raw_exact and mask_exact
        aggregate_within_threshold = all(
            aggregate_error[name] <= threshold
            for name, threshold in P8_L0_TAIL_AGGREGATE_THRESHOLDS.items()
        )
        return {
            "exact_flags": {
                "keys": key_exact,
                "raw_values": raw_exact,
                "mask": mask_exact,
                "aggregate_values": torch.equal(
                    candidate_aggregate, reference_aggregate
                ),
                "structural_required": structural_exact,
                "required": structural_exact and aggregate_within_threshold,
            },
            "aggregate_abs_error": aggregate_error,
            "aggregate_gate": {
                "thresholds": dict(P8_L0_TAIL_AGGREGATE_THRESHOLDS),
                "within_threshold": aggregate_within_threshold,
            },
        }

    def finish_prepare(
        self,
        prepared: P8L0TailPreparedBank,
    ) -> None:
        if self._prepared is not prepared or self._active_identity is None:
            raise RuntimeError("P8 L0 active prepared ownership mismatch.")
        if self._ready_generation is not None:
            raise RuntimeError("P8 L0 active prepared bank was already finalized.")
        self._ready_generation = prepared.generation
        identity = self._active_identity
        self._emit_lifecycle(
            "prepared",
            eligible=True,
            batch_size=prepared.batch_size,
            graph_batch_size=prepared.graph_batch_size,
            commit_lens=self._int_tuple(prepared.commit_lens),
            selected_indices=self._int_tuple(prepared.selected_indices),
            row_checks=prepared.row_checks,
            slot_checks=prepared.slot_checks,
            memory_shape=list(prepared.bank.keys.shape),
            validation_phase="consume" if self.validate else None,
            **self._identity_payload(identity),
        )
        if self._attribution_enabled:
            state = self._attribution
            state["completed_staged_producer_rounds"] += 1
            state["staged_rows"] += int(prepared.batch_size)
            self._increment_attribution_histogram(
                state["staged_actual_batch_histogram"], prepared.batch_size
            )
            self._increment_attribution_histogram(
                state["captured_graph_tier_histogram"], prepared.graph_batch_size
            )

    def take_for_proposal(
        self,
        *,
        committed_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        request_ids,
        host_prefix_lens=None,
    ) -> P8L0TailActiveHandoff | None:
        if self._attribution_enabled:
            self._note_attribution_proposal_attempt(
                batch_size=len(request_ids),
                host_prefix_lens=host_prefix_lens,
            )
        current = self._current_identity_values(
            committed_seq_lens=committed_seq_lens,
            req_pool_indices=req_pool_indices,
            request_ids=request_ids,
        )
        if self._prepared is None:
            reason = "no_prepared"
            previous = self._last_consumed_identity
            if current is not None and previous is not None:
                ids, generations, rows, source_end, committed_slots = current
                if (
                    ids == previous.request_ids
                    and generations == previous.req_generations
                    and rows == previous.req_pool_indices
                    and source_end == previous.source_end
                    and committed_slots == previous.selected_cache_locs
                ):
                    reason = "double_consume"
            bounded_reason = self._record_reason(reason)
            if (
                not self._bootstrap_emitted
                and self._generation == 0
                and self._last_consumed_generation == 0
            ):
                self._bootstrap_emitted = True
                self._emit_lifecycle(
                    "bootstrap_fallback",
                    eligible=False,
                    reason=bounded_reason,
                    batch_size=int(committed_seq_lens.numel()),
                )
            return None

        identity = self._active_identity
        if current is None or identity is None:
            self._discard_prepared(reason="identity_invalid", emit_event=True)
            return None
        ids, generations, rows, source_end, committed_slots = current
        mismatch = None
        if identity.hook_signature != self.hook_signature:
            mismatch = "signature_mismatch"
        elif identity.generation != self._generation:
            mismatch = "controller_generation_mismatch"
        elif identity.generation <= self._last_consumed_generation:
            mismatch = "prepared_generation_mismatch"
        elif self._ready_generation != identity.generation:
            mismatch = "prepared_generation_mismatch"
        elif identity.batch_size != len(rows):
            mismatch = "batch_size_mismatch"
        elif ids != identity.request_ids:
            mismatch = "request_id_mismatch"
        elif rows != identity.req_pool_indices:
            mismatch = "req_pool_indices_mismatch"
        elif source_end != identity.source_end:
            mismatch = "source_end_mismatch"
        elif generations != identity.req_generations:
            mismatch = "req_generation_mismatch"
        elif committed_slots != identity.selected_cache_locs:
            mismatch = "committed_slot_mismatch"
        elif self._prepared.generation != identity.generation:
            mismatch = "prepared_generation_mismatch"
        if mismatch is not None:
            self._discard_prepared(reason=mismatch, emit_event=True)
            return None

        handoff = P8L0TailActiveHandoff(
            bank=self._prepared.bank,
            identity=identity,
        )
        self._prepared = None
        self._active_identity = None
        self._ready_generation = None
        self._last_consumed_identity = identity
        self._last_consumed_generation = identity.generation
        self._outstanding_handoff_generation = identity.generation
        self._handoff_validation = None
        return handoff

    def _abort_handoff(
        self,
        handoff: P8L0TailActiveHandoff,
        *,
        reason: str,
        validation: dict | None = None,
    ) -> bool:
        generation = handoff.identity.generation
        if self._outstanding_handoff_generation != generation:
            return False
        self._outstanding_handoff_generation = None
        self._handoff_validation = None
        bounded_reason = self._record_reason(reason)
        self._emit_lifecycle(
            "ownership_aborted",
            eligible=False,
            reason=bounded_reason,
            validation=validation,
            **self._identity_payload(handoff.identity),
        )
        return True

    def is_handoff_outstanding(self, handoff: P8L0TailActiveHandoff) -> bool:
        return self._outstanding_handoff_generation == handoff.identity.generation

    def validate_consumed_handoff(
        self,
        handoff: P8L0TailActiveHandoff,
        *,
        reference_bank: DraftFreeKVLocalMemoryBank,
    ) -> dict:
        if not self.validate:
            raise RuntimeError("P8 L0 active consume validation is disabled.")
        if not self.is_handoff_outstanding(handoff):
            raise RuntimeError("P8 L0 active validation received a stale handoff.")
        try:
            validation = self._validate_bank_against_oracle(
                handoff.bank, reference_bank
            )
        except Exception:
            self._abort_handoff(handoff, reason="validation_failed")
            raise
        if not validation["exact_flags"]["structural_required"]:
            self._abort_handoff(
                handoff,
                reason="validation_failed",
                validation=validation,
            )
            raise RuntimeError(
                "P8 L0 active consume validation failed exact key/raw/mask parity."
            )
        if not validation["aggregate_gate"]["within_threshold"]:
            self._abort_handoff(
                handoff,
                reason="validation_failed",
                validation=validation,
            )
            raise RuntimeError(
                "P8 L0 active consume validation aggregate error exceeded thresholds."
            )
        self._handoff_validation = validation
        return validation

    def note_composed(
        self,
        handoff: P8L0TailActiveHandoff,
        *,
        j1_layers: tuple[int, ...],
        tp_all_gather_count: int,
    ) -> None:
        generation = handoff.identity.generation
        if self._outstanding_handoff_generation != generation:
            raise RuntimeError("P8 L0 active handoff was already finalized.")
        if self.validate and self._handoff_validation is None:
            raise RuntimeError(
                "P8 L0 active strict handoff was not validated at consumption."
            )
        expected_j1 = P8_L0_TAIL_ACTIVE_SELECTED_LAYERS[1:]
        if tuple(j1_layers) != expected_j1:
            raise RuntimeError(
                f"P8 L0 active expected J1 layers {expected_j1}, got {j1_layers}."
            )
        if int(tp_all_gather_count) not in (0, 1):
            raise RuntimeError("P8 L0 active permits at most one TP all-gather.")
        if self.key_cache.shape[1] == 4 and int(tp_all_gather_count) != 1:
            raise RuntimeError("P8 L0 active TP2 composition requires one all-gather.")
        if self.key_cache.shape[1] == 8 and int(tp_all_gather_count) != 0:
            raise RuntimeError("P8 L0 active TP1 composition must not all-gather.")
        validation = self._handoff_validation
        self._outstanding_handoff_generation = None
        self._handoff_validation = None
        self._emit_lifecycle(
            "consumed",
            eligible=True,
            batch_size=handoff.identity.batch_size,
            reused_layers=[0],
            j1_layers=list(j1_layers),
            stack_count=1,
            tp_all_gather_count=int(tp_all_gather_count),
            validation=validation,
            **self._identity_payload(handoff.identity),
        )
        if self._attribution_enabled:
            self._attribution["consumed_composed_rounds"] += 1
            self._attribution["consumed_rows"] += int(handoff.identity.batch_size)

    def note_proposal_exception(self, handoff: P8L0TailActiveHandoff) -> None:
        if not self._abort_handoff(handoff, reason="proposal_exception"):
            raise RuntimeError("P8 L0 active exception finalized a stale handoff.")

    def cancel(self, *, reason: str) -> bool:
        outstanding_generation = self._outstanding_handoff_generation
        had_ownership = (
            self._prepared is not None
            or self._in_flight
            or outstanding_generation is not None
        )
        if self._prepared is not None:
            self._discard_prepared(reason=reason, emit_event=True)
        else:
            self.workspace.disarm()
            self.workspace.invalidate_output()
            if had_ownership:
                bounded_reason = self._record_reason(reason)
                if outstanding_generation is not None:
                    self._emit_lifecycle(
                        "ownership_aborted",
                        eligible=False,
                        generation=outstanding_generation,
                        reason=bounded_reason,
                    )
        self._active_identity = None
        self._ready_generation = None
        self._outstanding_handoff_generation = None
        self._handoff_validation = None
        self._in_flight = False
        return had_ownership


class P8L0TailOverlapActive(P8L0TailActive):
    """A2 ownership with a distinct immutable B1 graph-hook signature."""

    hook_signature = P8_L0_OVERLAP_ACTIVE_HOOK_SIGNATURE
