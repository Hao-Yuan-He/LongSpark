"""Opt-in training attribution and test-only parity guards.

The runtime profiler is intentionally inert unless ``DFK_TRAIN_PROFILE`` is
set.  Parity helpers live here as well so optimization experiments share one
fail-closed comparison contract without changing the training algorithm.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn


PROFILE_ENV = "DFK_TRAIN_PROFILE"
PROFILE_SCHEMA_VERSION = 1
REGISTERED_RANGES = (
    "target_request_prep",
    "target_prefill",
    "hidden_reshape_repack",
    "local_kv_extraction",
    "tp_gather",
    "anchor_sampling_mask",
    "policy_selection",
    "query_projection",
    "generator_sdpa",
    "bank_assembly",
    "packed_layout_mask",
    "qwen_decoder",
    "global16_raw256_input",
    "global16_raw256_qwen_decoder",
    "logits_loss",
    "backward",
    "optimizer",
)


class _NoopRange(AbstractContextManager):
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NOOP_RANGE = _NoopRange()
_ACTIVE_PROFILER: Optional["TrainingProfiler"] = None


class _TorchCudaBackend:
    def __init__(self, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError("DFK training profiling requires a CUDA device")
        self.device = device

    @staticmethod
    def new_event():
        return torch.cuda.Event(enable_timing=True)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats(self.device)

    def max_memory_allocated(self) -> int:
        return int(torch.cuda.max_memory_allocated(self.device))

    def max_memory_reserved(self) -> int:
        return int(torch.cuda.max_memory_reserved(self.device))


@dataclass(frozen=True)
class _ProfileSettings:
    output_path: Path
    warmup_steps: int
    measured_steps: int


def _parse_profile_settings(raw: str, default_output_path: Path) -> _ProfileSettings:
    if raw.strip().startswith("{"):
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{PROFILE_ENV} JSON must be an object")
        unknown = set(payload) - {"output_path", "warmup_steps", "measured_steps"}
        if unknown:
            raise ValueError(f"unsupported {PROFILE_ENV} fields: {sorted(unknown)}")
        output_path = Path(payload.get("output_path") or default_output_path)
        warmup_steps = int(payload.get("warmup_steps", 1))
        measured_steps = int(payload.get("measured_steps", 5))
    else:
        output_path = default_output_path if raw.strip() == "1" else Path(raw)
        warmup_steps = 1
        measured_steps = 5
    if warmup_steps < 0:
        raise ValueError("profile warmup_steps must be non-negative")
    if measured_steps <= 0:
        raise ValueError("profile measured_steps must be positive")
    return _ProfileSettings(
        output_path=output_path.expanduser().resolve(),
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )


class _EventRange(AbstractContextManager):
    def __init__(self, profiler: "TrainingProfiler", name: str):
        self.profiler = profiler
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = self.profiler._backend.new_event()
        self.start.record()
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.start is not None:
            end = self.profiler._backend.new_event()
            end.record()
            self.profiler._events[self.name].append((self.start, end))
        return False


class TrainingProfiler:
    """Collect one fixed-size CUDA-event attribution window."""

    def __init__(
        self,
        *,
        output_path: Path,
        warmup_steps: int,
        measured_steps: int,
        configuration: Mapping[str, Any],
        backend=None,
        cpu_clock: Optional[Callable[[], float]] = None,
    ):
        self.output_path = Path(output_path)
        self.warmup_steps = int(warmup_steps)
        self.measured_steps = int(measured_steps)
        self.configuration = _jsonable(dict(configuration))
        self._backend = backend
        self._cpu_clock = time.perf_counter if cpu_clock is None else cpu_clock
        self._seen_steps = 0
        self._measured_steps = 0
        self._measuring = False
        self._completed = False
        self._window_start = None
        self._cpu_window_start = None
        self._device = None
        self._dtype = None
        self._source_shapes: list[dict[str, Any]] = []
        self._source_shape_recorded_this_step = False
        self._events = {name: [] for name in REGISTERED_RANGES}

    @property
    def measuring(self) -> bool:
        return self._measuring and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    def begin_step(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self._completed or self._measuring:
            if self._measuring:
                raise RuntimeError("profile step already active")
            return
        self._seen_steps += 1
        if self._seen_steps <= self.warmup_steps:
            return
        if self._backend is None:
            self._backend = _TorchCudaBackend(torch.device(device))
        if self._measured_steps == 0:
            self._cpu_window_start = self._cpu_clock()
            self._device = str(device)
            self._dtype = str(dtype).removeprefix("torch.")
            self._backend.reset_peak_memory()
            self._window_start = self._backend.new_event()
            self._window_start.record()
        self._source_shape_recorded_this_step = False
        self._measuring = True

    def record_source_token_counts(
        self,
        visible_token_counts: Sequence[int],
        *,
        padded_tokens: int,
    ) -> None:
        if not self.measuring:
            return
        if self._source_shape_recorded_this_step:
            raise RuntimeError("source token counts already recorded for this profile step")
        counts = [int(count) for count in visible_token_counts]
        padded_tokens = int(padded_tokens)
        if padded_tokens <= 0 or not counts:
            raise ValueError("profile source shape must contain positive token dimensions")
        if any(count <= 0 or count > padded_tokens for count in counts):
            raise ValueError("visible token counts must be inside the padded token length")
        self._source_shapes.append(
            {
                "visible_token_counts": counts,
                "padded_tokens": padded_tokens,
            }
        )
        self._source_shape_recorded_this_step = True

    def range(self, name: str) -> AbstractContextManager:
        if name not in self._events:
            raise ValueError(f"unregistered training profile range: {name}")
        if not self.measuring:
            return _NOOP_RANGE
        return _EventRange(self, name)

    def end_step(self) -> Optional[dict[str, Any]]:
        if not self._measuring:
            return None
        self._measuring = False
        self._measured_steps += 1
        if self._measured_steps < self.measured_steps:
            return None
        window_end = self._backend.new_event()
        window_end.record()
        # This is the sole synchronization point, after the complete window.
        self._backend.synchronize()
        cpu_window_end = self._cpu_clock()
        wall_ms = float(self._window_start.elapsed_time(window_end))
        cpu_wall_ms = (cpu_window_end - self._cpu_window_start) * 1000.0
        ranges = {}
        accounted_ms = 0.0
        for name in REGISTERED_RANGES:
            durations = [float(start.elapsed_time(end)) for start, end in self._events[name]]
            total_ms = sum(durations)
            accounted_ms += total_ms
            ranges[name] = {
                "count": len(durations),
                "total_ms": round(total_ms, 6),
            }
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "enabled": True,
            "completed": True,
            "device": self._device,
            "dtype": self._dtype,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self._measured_steps,
            "ranges": ranges,
            "measured_cuda_wall_ms": round(wall_ms, 6),
            "measured_cpu_wall_ms": round(cpu_wall_ms, 6),
            "cpu_minus_cuda_wall_ms": round(cpu_wall_ms - wall_ms, 6),
            "accounted_cuda_ms": round(accounted_ms, 6),
            "unaccounted_cuda_ms": round(max(0.0, wall_ms - accounted_ms), 6),
            "max_memory_allocated_bytes": self._backend.max_memory_allocated(),
            "max_memory_reserved_bytes": self._backend.max_memory_reserved(),
            "configuration": self.configuration,
            "source_shapes": self._source_shapes,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        self._completed = True
        return payload


def initialize_training_profiler(
    *,
    default_output_path: Path,
    configuration: Mapping[str, Any],
    environ: Optional[Mapping[str, str]] = None,
    cpu_clock: Optional[Callable[[], float]] = None,
) -> Optional[TrainingProfiler]:
    """Create no object and touch no output when the opt-in switch is absent."""

    raw = (os.environ if environ is None else environ).get(PROFILE_ENV, "").strip()
    if not raw or raw == "0":
        return None
    settings = _parse_profile_settings(raw, Path(default_output_path))
    output_path = settings.output_path
    world_size = int(configuration.get("world_size", 1))
    if world_size > 1:
        rank = int(configuration.get("rank", 0))
        output_path = output_path.with_name(
            f"{output_path.stem}.rank{rank}{output_path.suffix}"
        )
    return TrainingProfiler(
        output_path=output_path,
        warmup_steps=settings.warmup_steps,
        measured_steps=settings.measured_steps,
        configuration=configuration,
        cpu_clock=cpu_clock,
    )


def set_active_training_profiler(profiler: Optional[TrainingProfiler]) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_active_training_profiler() -> Optional[TrainingProfiler]:
    return _ACTIVE_PROFILER


def profile_range(name: str) -> AbstractContextManager:
    profiler = _ACTIVE_PROFILER
    if profiler is None or not profiler.measuring:
        return _NOOP_RANGE
    return profiler.range(name)


def record_source_token_counts(
    token_rows: Sequence[Sequence[int]],
    *,
    padded_tokens: int,
) -> None:
    profiler = _ACTIVE_PROFILER
    if profiler is None or not profiler.measuring:
        return
    profiler.record_source_token_counts(
        [len(row) for row in token_rows],
        padded_tokens=padded_tokens,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def assert_exact_tensor_map(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> None:
    """Compare row plans, indices, masks, and other discrete contracts exactly."""

    if set(reference) != set(candidate):
        raise AssertionError(
            f"exact tensor fields differ: reference={sorted(reference)} "
            f"candidate={sorted(candidate)}"
        )
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"exact tensor mismatch: {name}")


def assert_exact_masked_rows(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    row_axis: int = -2,
    name: str = "raw_rows",
) -> None:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"{name} shape or dtype mismatch")
    axis = row_axis % reference.ndim
    mask = row_mask.bool()
    if mask.ndim == 1:
        if mask.numel() != reference.shape[axis]:
            raise ValueError("row_mask length does not match row axis")
        selected_reference = reference.index_select(axis, torch.nonzero(mask).flatten())
        selected_candidate = candidate.index_select(axis, torch.nonzero(mask).flatten())
    elif mask.ndim == 2 and mask.shape == (reference.shape[0], reference.shape[axis]):
        moved_reference = reference.movedim((0, axis), (0, 1))
        moved_candidate = candidate.movedim((0, axis), (0, 1))
        selected_reference = moved_reference[mask]
        selected_candidate = moved_candidate[mask]
    else:
        raise ValueError("row_mask must address the row axis globally or per batch")
    if not torch.equal(selected_reference, selected_candidate):
        raise AssertionError(f"exact tensor mismatch: {name}")


def tensor_error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise AssertionError("tensor shapes differ")
    left = reference.detach().float()
    right = candidate.detach().float()
    absolute = (left - right).abs()
    denominator = left.abs().clamp_min(torch.finfo(torch.float32).tiny)
    relative = absolute / denominator
    return {
        "max_abs": float(absolute.max().item()) if absolute.numel() else 0.0,
        "mean_abs": float(absolute.mean().item()) if absolute.numel() else 0.0,
        "max_rel": float(relative.max().item()) if relative.numel() else 0.0,
    }


def load_frozen_tolerances(
    path: Path,
    *,
    required_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    tolerance_path = Path(path)
    if not tolerance_path.is_file():
        raise FileNotFoundError(f"frozen tolerance file not found: {tolerance_path}")
    with tolerance_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("thresholds"), dict):
        raise ValueError("tolerance.json must contain schema_version=1 and thresholds")
    thresholds = payload["thresholds"]
    missing = [name for name in required_names if name not in thresholds]
    if missing:
        raise ValueError(f"frozen tolerances missing required entries: {missing}")
    normalized = {}
    for name in required_names:
        entry = thresholds[name]
        if not isinstance(entry, dict) or set(entry) != {"atol", "rtol"}:
            raise ValueError(f"invalid frozen tolerance entry: {name}")
        atol = float(entry["atol"])
        rtol = float(entry["rtol"])
        if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
            raise ValueError(f"invalid frozen tolerance values: {name}")
        normalized[name] = {"atol": atol, "rtol": rtol}
    return normalized


def assert_tensor_with_tolerance(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerances: Mapping[str, Mapping[str, float]],
) -> None:
    if name not in tolerances:
        raise ValueError(f"frozen tolerance missing required entry: {name}")
    tolerance = tolerances[name]
    if reference.shape != candidate.shape or not torch.allclose(
        reference,
        candidate,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
    ):
        raise AssertionError(f"tensor tolerance mismatch: {name}")


def observe_legacy_envelopes(
    pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return observations only; callers must review and freeze thresholds separately."""

    return {
        "schema_version": 1,
        "kind": "legacy_vs_legacy_observation",
        "metadata": _jsonable(metadata),
        "observed": {
            name: tensor_error_metrics(left, right)
            for name, (left, right) in sorted(pairs.items())
        },
    }


def snapshot_parameter_gradients(model: nn.Module) -> dict[str, Optional[torch.Tensor]]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def assert_gradient_parity(
    reference: Mapping[str, Optional[torch.Tensor]],
    candidate: Mapping[str, Optional[torch.Tensor]],
    tolerances: Mapping[str, Mapping[str, float]],
) -> None:
    if set(reference) != set(candidate):
        raise AssertionError("trainable parameter names differ")
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if (left is None) != (right is None):
            raise AssertionError(f"gradient presence mismatch: {name}")
        if left is not None:
            assert_tensor_with_tolerance(f"gradient:{name}", left, right, tolerances)


def _iter_tensors(value: Any, prefix: str = "output"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _iter_tensors(getattr(value, field.name), f"{prefix}.{field.name}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{index}]")


def assert_target_detached(
    output: Any,
    *,
    target_modules: Sequence[nn.Module] = (),
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    for name, tensor in _iter_tensors(output):
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise AssertionError(f"target tensor retains autograd history: {name}")
    target_parameters = {}
    for module_index, module in enumerate(target_modules):
        for name, parameter in module.named_parameters():
            target_parameters[id(parameter)] = f"module[{module_index}].{name}"
            if parameter.requires_grad or parameter.grad is not None:
                raise AssertionError(
                    f"target parameter is not detached: module[{module_index}].{name}"
                )
    if optimizer is not None:
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                target_name = target_parameters.get(id(parameter))
                if target_name is not None:
                    raise AssertionError(
                        "target parameter appears in optimizer param_group: "
                        f"{target_name} group={group_index} parameter={parameter_index}"
                    )


def snapshot_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter)
        if not state:
            snapshot[name] = {"present": False}
            continue
        missing = {"step", "exp_avg", "exp_avg_sq"} - set(state)
        if missing:
            raise AssertionError(f"optimizer state missing {sorted(missing)} for {name}")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            step = float(step.detach().cpu().item())
        snapshot[name] = {
            "present": True,
            "step": float(step),
            "exp_avg": state["exp_avg"].detach().clone(),
            "exp_avg_sq": state["exp_avg_sq"].detach().clone(),
        }
    return snapshot


def snapshot_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "parameters": {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "gradients": snapshot_parameter_gradients(model),
        "optimizer": snapshot_optimizer_state(model, optimizer),
    }


def assert_two_step_training_parity(
    reference_steps: Sequence[Mapping[str, Any]],
    candidate_steps: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, Mapping[str, float]],
) -> None:
    if len(reference_steps) != 2 or len(candidate_steps) != 2:
        raise AssertionError("two-step parity requires exactly two snapshots per side")
    for step_index, (reference, candidate) in enumerate(
        zip(reference_steps, candidate_steps), start=1
    ):
        reference_parameters = reference["parameters"]
        candidate_parameters = candidate["parameters"]
        if set(reference_parameters) != set(candidate_parameters):
            raise AssertionError(f"step {step_index} trainable parameter names differ")
        for name in sorted(reference_parameters):
            assert_tensor_with_tolerance(
                f"parameter:{name}",
                reference_parameters[name],
                candidate_parameters[name],
                tolerances,
            )
        assert_gradient_parity(
            reference["gradients"], candidate["gradients"], tolerances
        )
        reference_optimizer = reference["optimizer"]
        candidate_optimizer = candidate["optimizer"]
        if set(reference_optimizer) != set(candidate_optimizer):
            raise AssertionError(f"step {step_index} optimizer parameter names differ")
        for name in sorted(reference_optimizer):
            left = reference_optimizer[name]
            right = candidate_optimizer[name]
            if bool(left["present"]) != bool(right["present"]):
                raise AssertionError(f"optimizer state presence mismatch: {name}")
            if not left["present"]:
                continue
            if left["step"] != right["step"]:
                raise AssertionError(f"optimizer step mismatch: {name}")
            for state_name in ("exp_avg", "exp_avg_sq"):
                assert_tensor_with_tolerance(
                    f"optimizer:{state_name}:{name}",
                    left[state_name],
                    right[state_name],
                    tolerances,
                )


__all__ = [
    "PROFILE_ENV",
    "PROFILE_SCHEMA_VERSION",
    "REGISTERED_RANGES",
    "TrainingProfiler",
    "assert_exact_masked_rows",
    "assert_exact_tensor_map",
    "assert_gradient_parity",
    "assert_target_detached",
    "assert_tensor_with_tolerance",
    "assert_two_step_training_parity",
    "get_active_training_profiler",
    "initialize_training_profiler",
    "load_frozen_tolerances",
    "observe_legacy_envelopes",
    "profile_range",
    "record_source_token_counts",
    "set_active_training_profiler",
    "snapshot_optimizer_state",
    "snapshot_parameter_gradients",
    "snapshot_training_step",
    "tensor_error_metrics",
]
