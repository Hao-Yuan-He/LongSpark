from __future__ import annotations

import json
import math
import os

import torch


DFK_P8_L0_AFTER_LAYER_RETURN_EVENT_ENV = "DFK_P8_L0_AFTER_LAYER_RETURN_EVENT"
DFK_P8_L0_AFTER_LAYER_RETURN_EVENT_PREFIX = "DFK_P8_L0_AFTER_LAYER_RETURN_EVENT "
DFK_P8_L0_OVERLAP_PRODUCER_ENV_PREFIX = "DFK_P8_L0_OVERLAP"
DFK_P8_L0_OVERLAP_ACTIVE_ENV = "DFK_P8_L0_OVERLAP_ACTIVE"
DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE_ENV = "DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE"
DFK_P8_L0_OVERLAP_ACTIVE_PREFIX = "DFK_P8_L0_OVERLAP_ACTIVE "

P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH = 8
P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER = 16
P8_L0_AFTER_LAYER_RETURN_SELECTED_LAYERS = (0, 9, 18, 26, 35)
P8_L0_AFTER_LAYER_RETURN_HOOK_POSITION = "after_entire_decoder_layer_return"
P8_L0_AFTER_LAYER_RETURN_HOOK_SIGNATURE = (
    "dfk_p8_l0_after_layer_return_event_v1",
    0,
    P8_L0_AFTER_LAYER_RETURN_HOOK_POSITION,
    P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH,
    "cuda_external_timing_events_current_graph_stream",
)
P8_L0_OVERLAP_ACTIVE_PRODUCTION_TIERS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
)
P8_L0_OVERLAP_ACTIVE_CURRENT_KERNEL_SCOPE = "current_kernel_existing_candidate_plan_native_layer_parts_into_existing_b16_workspace"
P8_L0_OVERLAP_ACTIVE_LAYER_HOOK_SIGNATURE = (
    "dfk_p8_l0_overlap_active_layer_hook_v1",
    0,
    P8_L0_AFTER_LAYER_RETURN_HOOK_POSITION,
    P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH,
    "graph_capture_only_aux_stream_producer_join_before_a2_stamp",
)

_SIGNATURE_ATTR = "_dfk_p8_l0_after_layer_return_signature"
_OWNER_ATTR = "_dfk_p8_l0_after_layer_return_owner"


def _env_enabled(name: str, *, environ=None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def p8_l0_after_layer_return_event_enabled() -> bool:
    return _env_enabled(DFK_P8_L0_AFTER_LAYER_RETURN_EVENT_ENV)


def p8_l0_overlap_active_enabled() -> bool:
    return _env_enabled(DFK_P8_L0_OVERLAP_ACTIVE_ENV)


def p8_l0_overlap_active_validate_enabled() -> bool:
    return _env_enabled(DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE_ENV)


def enabled_p8_overlap_producer_envs(*, environ=None) -> tuple[str, ...]:
    source = os.environ if environ is None else environ
    return tuple(
        sorted(
            name
            for name in source
            if name.startswith(DFK_P8_L0_OVERLAP_PRODUCER_ENV_PREFIX)
            and _env_enabled(name, environ=source)
        )
    )


def enabled_other_p8_overlap_producer_envs(*, environ=None) -> tuple[str, ...]:
    return tuple(
        name
        for name in enabled_p8_overlap_producer_envs(environ=environ)
        if name
        not in (
            DFK_P8_L0_OVERLAP_ACTIVE_ENV,
            DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE_ENV,
        )
    )


def validate_p8_l0_after_layer_return_event_mode(
    *,
    enabled: bool,
    native_memory_active: bool,
    native_memory_shadow: bool,
    candidate_memory_shadow: bool,
    accept_zero_probe: bool,
    critical_path_trace: bool,
    p8_tail_shadow: bool,
    p8_tail_active: bool,
    p8_tail_active_validate: bool,
    overlap_producer_envs: tuple[str, ...],
    selected_layers: tuple[int, ...],
    pp_size: int,
    verify_width: int,
    graph_backend,
    graph_tiers: tuple[int, ...],
) -> None:
    """Fail closed before allocating timing events or registering hooks."""

    if not enabled:
        return
    mode_env = DFK_P8_L0_AFTER_LAYER_RETURN_EVENT_ENV
    if not native_memory_active:
        raise ValueError(f"{mode_env} requires DFK_NATIVE_MEMORY_ACTIVE=1.")
    conflicts = [
        name
        for name, active in (
            ("DFK_NATIVE_MEMORY_SHADOW", native_memory_shadow),
            ("DFK_CANDIDATE_MEMORY_SHADOW", candidate_memory_shadow),
            ("DFK_ACCEPT_ZERO_PROBE", accept_zero_probe),
            ("DFK_CRITICAL_PATH_TRACE", critical_path_trace),
            ("DFK_P8_L0_TAIL_SHADOW", p8_tail_shadow),
            ("DFK_P8_L0_TAIL_ACTIVE", p8_tail_active),
            ("DFK_P8_L0_TAIL_ACTIVE_VALIDATE", p8_tail_active_validate),
        )
        if active
    ]
    conflicts.extend(str(name) for name in overlap_producer_envs)
    if conflicts:
        raise ValueError(f"{mode_env} cannot run with " + ", ".join(conflicts))
    if tuple(int(layer) for layer in selected_layers) != (
        P8_L0_AFTER_LAYER_RETURN_SELECTED_LAYERS
    ):
        raise ValueError(
            f"{mode_env} requires selected target layers "
            f"{P8_L0_AFTER_LAYER_RETURN_SELECTED_LAYERS}."
        )
    if int(pp_size) != 1:
        raise ValueError(f"{mode_env} requires the PP1 target model contract.")
    if int(verify_width) != P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH:
        raise ValueError(
            f"{mode_env} requires verify width "
            f"{P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH}."
        )
    if graph_backend is None or str(graph_backend).lower() != "full":
        raise ValueError(f"{mode_env} requires full target decode CUDA graph.")
    tiers = tuple(int(tier) for tier in graph_tiers)
    if not tiers or min(tiers) <= 0:
        raise ValueError(f"{mode_env} requires positive graph batch tiers.")
    if len(set(tiers)) != len(tiers):
        raise ValueError(f"{mode_env} requires unique graph batch tiers.")
    if P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER not in tiers:
        raise ValueError(f"{mode_env} requires a B16 target graph tier.")


def validate_p8_l0_overlap_active_mode(
    *,
    enabled: bool,
    validate: bool,
    native_memory_active: bool,
    native_memory_shadow: bool,
    candidate_memory_shadow: bool,
    accept_zero_probe: bool,
    critical_path_trace: bool,
    after_layer_return_event: bool,
    p8_tail_shadow: bool,
    p8_tail_active: bool,
    p8_tail_active_validate: bool,
    other_overlap_producer_envs: tuple[str, ...],
    selected_layers: tuple[int, ...],
    pp_size: int,
    tp_size: int,
    verify_width: int,
    proposal_width: int,
    disable_overlap_schedule: bool,
    enable_two_batch_overlap: bool,
    page_size: int,
    target_attention_backends: tuple[str, str],
    draft_attention_backend: str,
    graph_backend,
    graph_tiers: tuple[int, ...],
) -> None:
    """Fail closed before allocating B1 streams, events, or graph hooks."""

    if validate and not enabled:
        raise ValueError(
            f"{DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE_ENV} requires "
            f"{DFK_P8_L0_OVERLAP_ACTIVE_ENV}=1."
        )
    if not enabled:
        return
    mode_env = DFK_P8_L0_OVERLAP_ACTIVE_ENV
    if not native_memory_active:
        raise ValueError(f"{mode_env} requires DFK_NATIVE_MEMORY_ACTIVE=1.")
    if not p8_tail_active:
        raise ValueError(f"{mode_env} requires DFK_P8_L0_TAIL_ACTIVE=1.")
    if validate and not p8_tail_active_validate:
        raise ValueError(
            f"{DFK_P8_L0_OVERLAP_ACTIVE_VALIDATE_ENV} requires "
            "DFK_P8_L0_TAIL_ACTIVE_VALIDATE=1."
        )
    conflicts = [
        name
        for name, active in (
            ("DFK_P8_L0_AFTER_LAYER_RETURN_EVENT", after_layer_return_event),
            ("DFK_P8_L0_TAIL_SHADOW", p8_tail_shadow),
            ("DFK_NATIVE_MEMORY_SHADOW", native_memory_shadow),
            ("DFK_CANDIDATE_MEMORY_SHADOW", candidate_memory_shadow),
            ("DFK_ACCEPT_ZERO_PROBE", accept_zero_probe),
            ("DFK_CRITICAL_PATH_TRACE", critical_path_trace),
        )
        if active
    ]
    conflicts.extend(str(name) for name in other_overlap_producer_envs)
    if conflicts:
        raise ValueError(f"{mode_env} cannot run with " + ", ".join(conflicts))
    if tuple(int(layer) for layer in selected_layers) != (
        P8_L0_AFTER_LAYER_RETURN_SELECTED_LAYERS
    ):
        raise ValueError(
            f"{mode_env} requires selected target layers "
            f"{P8_L0_AFTER_LAYER_RETURN_SELECTED_LAYERS}."
        )
    if int(pp_size) != 1:
        raise ValueError(f"{mode_env} requires the PP1 target model contract.")
    if int(tp_size) not in (1, 2):
        raise ValueError(f"{mode_env} supports TP1 or TP2 only.")
    if int(verify_width) != P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH:
        raise ValueError(
            f"{mode_env} requires verify width "
            f"{P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH}."
        )
    if int(proposal_width) != P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH - 1:
        raise ValueError(f"{mode_env} requires proposal width 7.")
    if not bool(disable_overlap_schedule):
        raise ValueError(f"{mode_env} requires synchronous scheduling.")
    if bool(enable_two_batch_overlap):
        raise ValueError(f"{mode_env} does not support two-batch overlap.")
    if int(page_size) != 1:
        raise ValueError(f"{mode_env} requires page_size == 1.")
    if tuple(str(value) for value in target_attention_backends) != ("fa3", "fa3"):
        raise ValueError(f"{mode_env} requires target prefill/decode FA3.")
    if str(draft_attention_backend) != "fa3":
        raise ValueError(f"{mode_env} requires draft FA3.")
    if graph_backend is None or str(graph_backend).lower() != "full":
        raise ValueError(f"{mode_env} requires full target decode CUDA graph.")
    tiers = tuple(int(tier) for tier in graph_tiers)
    if tiers != P8_L0_OVERLAP_ACTIVE_PRODUCTION_TIERS:
        raise ValueError(
            f"{mode_env} requires exact production graph tiers "
            f"{P8_L0_OVERLAP_ACTIVE_PRODUCTION_TIERS}, got {tiers}."
        )


def _new_cuda_timing_event():
    # external=True materializes an event-record node in each CUDA graph. This
    # keeps the retained event timestamp observable after replay; the internal
    # event form represents only graph-local cross-stream dependencies.
    return torch.cuda.Event(enable_timing=True, external=True)


def _new_cuda_dependency_event():
    return torch.cuda.Event(enable_timing=False, external=False)


def _new_cuda_aux_stream():
    return torch.cuda.Stream()


def _synchronize_cuda_once() -> None:
    torch.cuda.synchronize()


class _EventPair:
    def __init__(self, *, layer0_return, target_tail):
        self.layer0_return = layer0_return
        self.target_tail = target_tail


class _OverlapDependencyPair:
    def __init__(self, *, producer_ready, producer_done):
        self.producer_ready = producer_ready
        self.producer_done = producer_done


class _OverlapTimingEvents:
    def __init__(
        self,
        *,
        layer0_return,
        aux_start,
        producer_done,
        target_tail_arrival,
        joined,
    ):
        self.layer0_return = layer0_return
        self.aux_start = aux_start
        self.producer_done = producer_done
        self.target_tail_arrival = target_tail_arrival
        self.joined = joined


class P8L0AfterLayerReturnEvent:
    """One-shot graph seam timing owner; it performs no overlap work."""

    hook_signature = P8_L0_AFTER_LAYER_RETURN_HOOK_SIGNATURE

    def __init__(
        self,
        *,
        tp_rank: int,
        expected_tiers: tuple[int, ...],
        event_factory=_new_cuda_timing_event,
        synchronize=_synchronize_cuda_once,
        emit,
    ):
        tiers = tuple(int(tier) for tier in expected_tiers)
        if not tiers or min(tiers) <= 0 or len(set(tiers)) != len(tiers):
            raise ValueError("P8 after-layer-return requires unique positive tiers.")
        if P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER not in tiers:
            raise ValueError("P8 after-layer-return requires a B16 graph tier.")
        self.tp_rank = int(tp_rank)
        self.expected_tiers = tiers
        self._events = {
            tier: _EventPair(
                layer0_return=event_factory(),
                target_tail=event_factory(),
            )
            for tier in tiers
        }
        self._synchronize = synchronize
        self.emit = emit
        self._layer0_seen_tiers: set[int] = set()
        self._captured_tiers: set[int] = set()
        self._capture_validated = False
        self._pending = True
        self._registered_model_runner = None
        self._layer0_module = None
        self.layer0_hook_handle = None

    @property
    def pending(self) -> bool:
        return self._pending

    @property
    def captured_tiers(self) -> tuple[int, ...]:
        return tuple(sorted(self._captured_tiers))

    @staticmethod
    def _target_verify_tier(forward_batch) -> int | None:
        mode = getattr(forward_batch, "forward_mode", None)
        is_target_verify = getattr(mode, "is_target_verify", None)
        if not callable(is_target_verify) or not bool(is_target_verify()):
            return None
        tier = int(getattr(forward_batch, "batch_size", 0))
        if tier <= 0:
            raise RuntimeError(
                "P8 after-layer-return target verify has invalid batch size."
            )
        return tier

    def layer0_return_hook(self, module, inputs, _output) -> None:
        if self._layer0_module is not None and module is not self._layer0_module:
            raise RuntimeError(
                "P8 after-layer-return hook ran on the wrong decoder layer."
            )
        if len(inputs) < 3:
            raise RuntimeError(
                "P8 after-layer-return decoder layer input topology changed."
            )
        forward_batch = inputs[2]
        tier = self._target_verify_tier(forward_batch)
        if tier is None:
            return
        hidden_states = inputs[1]
        shape = getattr(hidden_states, "shape", ())
        expected_tokens = tier * P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH
        if not shape or int(shape[0]) != expected_tokens:
            raise RuntimeError(
                "P8 after-layer-return hook requires all8 target verify shape, "
                f"got batch={tier}, tokens={None if not shape else int(shape[0])}."
            )
        event_pair = self._events.get(tier)
        if event_pair is None:
            # Eager target verifies outside captured tiers are irrelevant. A graph
            # capture for an unexpected tier fails in the capture-tail hook.
            return
        # No stream argument: Event.record() uses the current target graph stream.
        event_pair.layer0_return.record()
        self._layer0_seen_tiers.add(tier)

    def capture_tail_hook(
        self, graph_runner, _output, forward_batch, num_tokens: int
    ) -> None:
        tier = self._target_verify_tier(forward_batch)
        if tier is None:
            return
        if (
            self._registered_model_runner is None
            or getattr(graph_runner, "model_runner", None)
            is not self._registered_model_runner
        ):
            raise RuntimeError(
                "P8 after-layer-return capture-tail received the wrong graph runner."
            )
        expected_tokens = tier * P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH
        if int(num_tokens) != expected_tokens:
            raise RuntimeError(
                "P8 after-layer-return capture-tail requires all8 target verify "
                f"shape, got batch={tier}, num_tokens={num_tokens}."
            )
        event_pair = self._events.get(tier)
        if event_pair is None:
            raise RuntimeError(
                "P8 after-layer-return captured an unsupported graph tier: " f"{tier}."
            )
        if tier not in self._layer0_seen_tiers:
            raise RuntimeError(
                "P8 after-layer-return capture-tail is missing the layer0 return "
                f"event for tier {tier}."
            )
        # This is the same current target graph stream as the model capture tail.
        event_pair.target_tail.record()
        self._captured_tiers.add(tier)

    def validate_capture_complete(self, *, expected_tiers: tuple[int, ...]) -> None:
        runtime_tiers = tuple(int(tier) for tier in expected_tiers)
        if runtime_tiers != self.expected_tiers:
            raise RuntimeError(
                "P8 after-layer-return runtime tiers changed after event "
                f"preallocation: allocated={self.expected_tiers}, "
                f"runtime={runtime_tiers}."
            )
        if self._captured_tiers != set(runtime_tiers):
            raise RuntimeError(
                "P8 after-layer-return capture tiers are incomplete: "
                f"expected={sorted(runtime_tiers)}, "
                f"observed={sorted(self._captured_tiers)}."
            )
        self._capture_validated = True

    def maybe_measure(
        self,
        *,
        batch_size: int,
        target_result_can_run_cuda_graph: bool,
    ) -> bool:
        if not self._pending:
            return False
        if int(batch_size) != P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER or not bool(
            target_result_can_run_cuda_graph
        ):
            return False

        # Claim the one-shot before the explicitly enabled diagnostic sync.
        self._pending = False
        if not self._capture_validated:
            raise RuntimeError(
                "P8 after-layer-return capture completeness was not validated."
            )
        tier = P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER
        event_pair = self._events.get(tier)
        if event_pair is None or tier not in self._captured_tiers:
            raise RuntimeError(
                "P8 after-layer-return B16 timing events were not captured."
            )

        self._synchronize()
        if not bool(event_pair.layer0_return.query()) or not bool(
            event_pair.target_tail.query()
        ):
            raise RuntimeError(
                "P8 after-layer-return B16 timing events were not recorded."
            )
        elapsed_ms = float(
            event_pair.layer0_return.elapsed_time(event_pair.target_tail)
        )
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise RuntimeError(
                "P8 after-layer-return timing must be finite positive CUDA ms, "
                f"got {elapsed_ms!r}."
            )
        payload = {
            "schema": "dfk_p8_l0_after_layer_return_event_v1",
            "schema_version": 1,
            "scope": "graph_safe_seam_slack_measurement_only",
            "tp_rank": self.tp_rank,
            "tier": tier,
            "layer0": 0,
            "hook_position": P8_L0_AFTER_LAYER_RETURN_HOOK_POSITION,
            "captured_tiers": list(self.captured_tiers),
            "timing": {
                "layer0_return_to_target_tail_cuda_ms": elapsed_ms,
            },
            "event_semantics": ("cuda_external_timing_events_current_graph_stream"),
            "graph": True,
            "diagnostic_round_perturbed": True,
        }
        self.emit(
            DFK_P8_L0_AFTER_LAYER_RETURN_EVENT_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return True


class P8L0OverlapActive:
    """Graph-captured B1 producer fork/join around the shared A2 tail work."""

    layer_hook_signature = P8_L0_OVERLAP_ACTIVE_LAYER_HOOK_SIGNATURE

    def __init__(
        self,
        *,
        tp_rank: int,
        expected_tiers: tuple[int, ...],
        tail_active,
        validate: bool,
        dependency_event_factory=_new_cuda_dependency_event,
        timing_event_factory=_new_cuda_timing_event,
        stream_factory=_new_cuda_aux_stream,
        stream_context=torch.cuda.stream,
        is_capturing=torch.cuda.is_current_stream_capturing,
        synchronize=_synchronize_cuda_once,
        emit,
    ):
        tiers = tuple(int(tier) for tier in expected_tiers)
        if not tiers or min(tiers) <= 0 or len(set(tiers)) != len(tiers):
            raise ValueError("P8 overlap active requires unique positive tiers.")
        if P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER not in tiers:
            raise ValueError("P8 overlap active requires a B16 graph tier.")
        hook_signature = getattr(tail_active, "hook_signature", None)
        if getattr(hook_signature, "mode", None) != "p8_l0_overlap_active":
            raise ValueError(
                "P8 overlap active requires the distinct overlap tail signature."
            )
        self.tp_rank = int(tp_rank)
        self.expected_tiers = tiers
        self.tail_active = tail_active
        # register_p8_l0_tail_capture_hook validates this immutable signature.
        self.hook_signature = hook_signature
        self.validate = bool(validate)
        self.aux_stream = stream_factory()
        self._dependencies = {
            tier: _OverlapDependencyPair(
                producer_ready=dependency_event_factory(),
                producer_done=dependency_event_factory(),
            )
            for tier in tiers
        }
        self._timing = (
            _OverlapTimingEvents(
                layer0_return=timing_event_factory(),
                aux_start=timing_event_factory(),
                producer_done=timing_event_factory(),
                target_tail_arrival=timing_event_factory(),
                joined=timing_event_factory(),
            )
            if self.validate
            else None
        )
        self._stream_context = stream_context
        self._is_capturing = is_capturing
        self._synchronize = synchronize
        self.emit = emit
        self._forked_tiers: set[int] = set()
        self._joined_tiers: set[int] = set()
        self._capture_validated = False
        self._timing_pending = self.validate
        self._registered_model_runner = None
        self._layer0_module = None
        self.layer0_hook_handle = None

    @property
    def timing_pending(self) -> bool:
        return self._timing_pending

    @property
    def captured_tiers(self) -> tuple[int, ...]:
        return tuple(sorted(self._joined_tiers))

    def layer0_return_hook(self, module, inputs, _output) -> None:
        if self._layer0_module is not None and module is not self._layer0_module:
            raise RuntimeError("P8 overlap active hook ran on the wrong decoder layer.")
        if len(inputs) < 3:
            raise RuntimeError("P8 overlap active decoder layer topology changed.")
        forward_batch = inputs[2]
        tier = P8L0AfterLayerReturnEvent._target_verify_tier(forward_batch)
        if tier is None:
            return
        hidden_states = inputs[1]
        shape = getattr(hidden_states, "shape", ())
        expected_tokens = tier * P8_L0_AFTER_LAYER_RETURN_VERIFY_WIDTH
        if not shape or int(shape[0]) != expected_tokens:
            raise RuntimeError(
                "P8 overlap active requires all8 target verify shape, got "
                f"batch={tier}, tokens={None if not shape else int(shape[0])}."
            )
        # FullCudaGraphBackend executes non-capturing warmups. They deliberately
        # remain on the serial A2 capture-tail path to warm the existing kernels.
        if not bool(self._is_capturing()):
            return
        dependency = self._dependencies.get(tier)
        if dependency is None:
            raise RuntimeError(
                f"P8 overlap active captured unsupported graph tier {tier}."
            )
        if tier in self._forked_tiers:
            raise RuntimeError(
                f"P8 overlap active forked graph tier {tier} more than once."
            )
        timing = (
            self._timing if tier == P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER else None
        )
        if timing is not None:
            timing.layer0_return.record()
        dependency.producer_ready.record()
        with self._stream_context(self.aux_stream):
            dependency.producer_ready.wait()
            if timing is not None:
                timing.aux_start.record()
            self.tail_active._produce_and_store_capture_outputs(tier)
            if timing is not None:
                timing.producer_done.record()
            dependency.producer_done.record()
        self._forked_tiers.add(tier)

    def capture_hook(
        self, graph_runner, output, forward_batch, num_tokens: int
    ) -> None:
        # The two FullCudaGraphBackend warmups reuse the unchanged serial path.
        # No auxiliary work is launched unless an actual graph capture is active.
        if not bool(self._is_capturing()):
            self.tail_active.capture_hook(
                graph_runner, output, forward_batch, num_tokens
            )
            return
        if (
            self._registered_model_runner is None
            or getattr(graph_runner, "model_runner", None)
            is not self._registered_model_runner
        ):
            raise RuntimeError(
                "P8 overlap active capture-tail received the wrong graph runner."
            )
        tier = self.tail_active._validate_capture_call(
            graph_runner, forward_batch, num_tokens
        )
        if tier not in self._forked_tiers:
            raise RuntimeError(
                "P8 overlap active capture-tail is missing its layer0 producer "
                f"fork for tier {tier}."
            )
        if tier in self._joined_tiers:
            raise RuntimeError(
                f"P8 overlap active joined graph tier {tier} more than once."
            )
        dependency = self._dependencies.get(tier)
        if dependency is None:
            raise RuntimeError(
                f"P8 overlap active captured unsupported graph tier {tier}."
            )
        timing = (
            self._timing if tier == P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER else None
        )
        if timing is not None:
            timing.target_tail_arrival.record()
        # Event.wait() without a stream argument targets the current graph stream.
        dependency.producer_done.wait()
        if timing is not None:
            timing.joined.record()
        self.tail_active._stamp_capture_output_identity(forward_batch, tier)
        self._joined_tiers.add(tier)

    def validate_capture_complete(self, *, expected_tiers: tuple[int, ...]) -> None:
        runtime_tiers = tuple(int(tier) for tier in expected_tiers)
        if runtime_tiers != self.expected_tiers:
            raise RuntimeError(
                "P8 overlap active runtime tiers changed after preallocation: "
                f"allocated={self.expected_tiers}, runtime={runtime_tiers}."
            )
        expected = set(runtime_tiers)
        if self._forked_tiers != expected or self._joined_tiers != expected:
            raise RuntimeError(
                "P8 overlap active capture tiers are incomplete: "
                f"expected={sorted(expected)}, forked={sorted(self._forked_tiers)}, "
                f"joined={sorted(self._joined_tiers)}."
            )
        self.tail_active.validate_capture_complete(expected_tiers=runtime_tiers)
        self._capture_validated = True

    @staticmethod
    def _elapsed_ms(start_event, end_event, *, field: str, allow_zero: bool) -> float:
        value = float(start_event.elapsed_time(end_event))
        valid = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
        if not valid:
            relation = "finite nonnegative" if allow_zero else "finite positive"
            raise RuntimeError(
                f"P8 overlap active {field} must be {relation} CUDA ms, "
                f"got {value!r}."
            )
        return value

    def maybe_measure(
        self,
        *,
        batch_size: int,
        target_result_can_run_cuda_graph: bool,
    ) -> bool:
        if not self.validate or not self._timing_pending:
            return False
        if int(batch_size) != P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER or not bool(
            target_result_can_run_cuda_graph
        ):
            return False
        self._timing_pending = False
        if not self._capture_validated:
            raise RuntimeError(
                "P8 overlap active capture completeness was not validated."
            )
        tier = P8_L0_AFTER_LAYER_RETURN_DIAGNOSTIC_TIER
        if tier not in self._joined_tiers or self._timing is None:
            raise RuntimeError("P8 overlap active B16 timing was not captured.")

        self._synchronize()
        timing = self._timing
        events = (
            timing.layer0_return,
            timing.aux_start,
            timing.producer_done,
            timing.target_tail_arrival,
            timing.joined,
        )
        if not all(bool(event.query()) for event in events):
            raise RuntimeError("P8 overlap active B16 timing events were not recorded.")
        values = {
            "layer0_return_to_aux_start_cuda_ms": self._elapsed_ms(
                timing.layer0_return,
                timing.aux_start,
                field="layer0_return_to_aux_start",
                allow_zero=True,
            ),
            "aux_start_to_producer_done_cuda_ms": self._elapsed_ms(
                timing.aux_start,
                timing.producer_done,
                field="aux_start_to_producer_done",
                allow_zero=False,
            ),
            "layer0_return_to_target_tail_arrival_cuda_ms": self._elapsed_ms(
                timing.layer0_return,
                timing.target_tail_arrival,
                field="layer0_return_to_target_tail_arrival",
                allow_zero=False,
            ),
            "target_tail_arrival_to_join_cuda_ms": self._elapsed_ms(
                timing.target_tail_arrival,
                timing.joined,
                field="target_tail_arrival_to_join",
                allow_zero=True,
            ),
            "layer0_return_to_join_cuda_ms": self._elapsed_ms(
                timing.layer0_return,
                timing.joined,
                field="layer0_return_to_join",
                allow_zero=False,
            ),
        }
        payload = {
            "schema": "dfk_p8_l0_overlap_active_v1",
            "schema_version": 1,
            "scope": P8_L0_OVERLAP_ACTIVE_CURRENT_KERNEL_SCOPE,
            "tp_rank": self.tp_rank,
            "tier": tier,
            "layer0": 0,
            "hook_position": P8_L0_AFTER_LAYER_RETURN_HOOK_POSITION,
            "captured_tiers": list(self.captured_tiers),
            "timing": values,
            "timing_unit": "cuda_ms",
            "event_semantics": "graph_captured_aux_stream_fork_join_external_timing",
            "graph": True,
            "diagnostic_round_perturbed": True,
        }
        self.emit(
            DFK_P8_L0_OVERLAP_ACTIVE_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return True


def resolve_qwen3_decoder_layer0(model_runner):
    if int(getattr(model_runner, "pp_size", 0)) != 1:
        raise RuntimeError("P8 after-layer-return requires the PP1 model runner.")
    causal_lm = getattr(model_runner, "model", None)
    if type(causal_lm).__name__ != "Qwen3ForCausalLM":
        raise RuntimeError(
            "P8 after-layer-return requires Qwen3ForCausalLM target architecture."
        )
    model = getattr(causal_lm, "model", None)
    if type(model).__name__ != "Qwen3Model":
        raise RuntimeError(
            "P8 after-layer-return requires Qwen3ForCausalLM.model topology."
        )
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError(
            "P8 after-layer-return could not resolve target decoder layer 0."
        )
    layer0 = layers[0]
    if type(layer0).__name__ != "Qwen3DecoderLayer" or not callable(
        getattr(layer0, "register_forward_hook", None)
    ):
        raise RuntimeError(
            "P8 after-layer-return target decoder layer 0 topology is unsupported."
        )
    return layer0


def register_p8_l0_after_layer_return_hooks(model_runner, owner) -> None:
    """Register the layer-return and capture-tail hooks before graph creation."""

    if getattr(model_runner, "decode_cuda_graph_runner", None) is not None:
        raise RuntimeError(
            "P8 after-layer-return hook registration is too late: target decode "
            "graph runner already exists."
        )
    if not isinstance(owner, P8L0AfterLayerReturnEvent):
        raise RuntimeError("P8 after-layer-return owner has an invalid type.")
    existing_signature = getattr(model_runner, _SIGNATURE_ATTR, None)
    if existing_signature is not None:
        if existing_signature != owner.hook_signature:
            raise RuntimeError(
                "P8 after-layer-return found a stale or mixed hook signature: "
                f"{existing_signature!r}."
            )
        raise RuntimeError("P8 after-layer-return hooks were registered twice.")
    if owner.layer0_hook_handle is not None:
        raise RuntimeError("P8 after-layer-return hooks were registered twice.")
    capture_tail_hooks = getattr(model_runner, "capture_tail_hooks", None)
    if not isinstance(capture_tail_hooks, list):
        raise RuntimeError("target model runner has no capture_tail_hooks list.")
    if any(existing == owner.capture_tail_hook for existing in capture_tail_hooks):
        raise RuntimeError("P8 after-layer-return hooks were registered twice.")
    layer0 = resolve_qwen3_decoder_layer0(model_runner)

    handle = layer0.register_forward_hook(owner.layer0_return_hook)
    capture_tail_hooks.append(owner.capture_tail_hook)
    owner.layer0_hook_handle = handle
    owner._registered_model_runner = model_runner
    owner._layer0_module = layer0
    setattr(model_runner, _SIGNATURE_ATTR, owner.hook_signature)
    setattr(model_runner, _OWNER_ATTR, owner)


def register_p8_l0_overlap_active_hooks(
    model_runner, owner, *, register_capture_tail_hook
) -> None:
    """Register the B1 layer fork and A2-compatible capture-tail join once."""

    if getattr(model_runner, "decode_cuda_graph_runner", None) is not None:
        raise RuntimeError(
            "P8 overlap active hook registration is too late: target decode "
            "graph runner already exists."
        )
    if not isinstance(owner, P8L0OverlapActive):
        raise RuntimeError("P8 overlap active owner has an invalid type.")
    existing_signature = getattr(model_runner, _SIGNATURE_ATTR, None)
    if existing_signature is not None:
        if existing_signature != owner.layer_hook_signature:
            raise RuntimeError(
                "P8 overlap active found a stale or mixed layer-hook signature: "
                f"{existing_signature!r}."
            )
        raise RuntimeError("P8 overlap active hooks were registered twice.")
    if owner.layer0_hook_handle is not None:
        raise RuntimeError("P8 overlap active hooks were registered twice.")
    if not callable(register_capture_tail_hook):
        raise RuntimeError("P8 overlap active capture-tail registrar is invalid.")
    layer0 = resolve_qwen3_decoder_layer0(model_runner)

    register_capture_tail_hook(model_runner, owner)
    handle = layer0.register_forward_hook(owner.layer0_return_hook)
    owner.layer0_hook_handle = handle
    owner._registered_model_runner = model_runner
    owner._layer0_module = layer0
    setattr(model_runner, _SIGNATURE_ATTR, owner.layer_hook_signature)
    setattr(model_runner, _OWNER_ATTR, owner)
