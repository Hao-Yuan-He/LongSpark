"""Reference contract for the Global16 + Raw256 DraftFreeKV design.

This module is deliberately a PyTorch reference, not a replacement FA3 kernel.
It owns the mathematical definition shared by offline training and the serving
session facade.  In particular, it never materializes a 16 + 256 K/V bank:

* Global16 is an online-softmax state over the full committed Target prefix.
* Raw256 is a borrowed view of the Target K/V cache plus a range mask.
* Local7 is a separate, short draft-local source.

The serving integration can replace the reference attention implementation with
one FA3 kernel, but must preserve these state and aliasing contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import nn


GLOBAL16_RAW256_SCHEME = "global16_raw256_v1"
GLOBAL16_QUERY_MODES = frozenset(("learned", "fixed_equicorrelated"))
GLOBAL16_SLOT_MASS_MODES = frozenset(("system", "baseline16"))


@dataclass(frozen=True)
class Global16Raw256Config:
    """Immutable, retrain-only contract for Global16 + Raw256."""

    target_layer_ids: tuple[int, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    global_memory_slots: int = 16
    raw_target_rows: int = 256
    local_draft_rows: int = 7
    max_accepted_tokens: int = 8
    memory_source_window: str = "full"
    scheme: str = GLOBAL16_RAW256_SCHEME
    checkpoint_compatibility: str = "retrain_required"
    global_query_mode: str = "learned"
    global_query_equicorrelation: float = 0.4
    global_query_slot_mass_mode: str = "system"

    def __post_init__(self) -> None:
        layer_ids = tuple(int(layer_id) for layer_id in self.target_layer_ids)
        object.__setattr__(self, "target_layer_ids", layer_ids)
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("target_layer_ids must be a non-empty unique sequence")
        if any(layer_id < 0 for layer_id in layer_ids):
            raise ValueError("target_layer_ids must be non-negative")
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.global_memory_slots != 16:
            raise ValueError("Global16 requires exactly 16 global memory queries")
        if self.raw_target_rows != 256:
            raise ValueError("Raw256 requires exactly 256 Target-KV rows")
        if self.local_draft_rows != 7:
            raise ValueError("Local7 requires exactly 7 draft-local rows")
        if self.max_accepted_tokens != 8:
            raise ValueError("accepted Target-K/V delta is capped at exactly 8 rows")
        if self.memory_source_window != "full":
            raise ValueError("Global16 queries must read the full Target prefix")
        if self.scheme != GLOBAL16_RAW256_SCHEME:
            raise ValueError(f"scheme must be {GLOBAL16_RAW256_SCHEME!r}")
        if self.checkpoint_compatibility != "retrain_required":
            raise ValueError("Global16 + Raw256 does not support legacy checkpoints")
        if self.global_query_mode not in GLOBAL16_QUERY_MODES:
            raise ValueError(
                "global_query_mode must be one of "
                + ", ".join(sorted(GLOBAL16_QUERY_MODES))
            )
        correlation = float(self.global_query_equicorrelation)
        lower_bound = -1.0 / (self.global_memory_slots - 1)
        if not lower_bound < correlation < 1.0:
            raise ValueError(
                "global_query_equicorrelation must lie strictly between "
                f"{lower_bound} and 1"
            )
        object.__setattr__(self, "global_query_equicorrelation", correlation)
        if self.global_query_slot_mass_mode not in GLOBAL16_SLOT_MASS_MODES:
            raise ValueError(
                "global_query_slot_mass_mode must be one of "
                + ", ".join(sorted(GLOBAL16_SLOT_MASS_MODES))
            )

    @property
    def num_selected_layers(self) -> int:
        return len(self.target_layer_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_layer_ids": list(self.target_layer_ids),
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "global_memory_slots": self.global_memory_slots,
            "raw_target_rows": self.raw_target_rows,
            "local_draft_rows": self.local_draft_rows,
            "max_accepted_tokens": self.max_accepted_tokens,
            "memory_source_window": self.memory_source_window,
            "scheme": self.scheme,
            "checkpoint_compatibility": self.checkpoint_compatibility,
            "global_query_mode": self.global_query_mode,
            "global_query_equicorrelation": self.global_query_equicorrelation,
            "global_query_slot_mass_mode": self.global_query_slot_mass_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Global16Raw256Config":
        known = {
            "target_layer_ids",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "global_memory_slots",
            "raw_target_rows",
            "local_draft_rows",
            "max_accepted_tokens",
            "memory_source_window",
            "scheme",
            "checkpoint_compatibility",
            "global_query_mode",
            "global_query_equicorrelation",
            "global_query_slot_mass_mode",
        }
        unknown = set(payload).difference(known)
        if unknown:
            raise ValueError(
                "unsupported Global16 + Raw256 config fields: "
                + ", ".join(sorted(unknown))
            )
        required = {
            "target_layer_ids",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError("missing Global16 + Raw256 config fields: " + ", ".join(sorted(missing)))
        return cls(
            target_layer_ids=tuple(payload["target_layer_ids"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(payload["head_dim"]),
            global_memory_slots=int(payload.get("global_memory_slots", 16)),
            raw_target_rows=int(payload.get("raw_target_rows", 256)),
            local_draft_rows=int(payload.get("local_draft_rows", 7)),
            max_accepted_tokens=int(payload.get("max_accepted_tokens", 8)),
            memory_source_window=str(payload.get("memory_source_window", "full")),
            scheme=str(payload.get("scheme", GLOBAL16_RAW256_SCHEME)),
            checkpoint_compatibility=str(
                payload.get("checkpoint_compatibility", "retrain_required")
            ),
            global_query_mode=str(payload.get("global_query_mode", "learned")),
            global_query_equicorrelation=float(
                payload.get("global_query_equicorrelation", 0.4)
            ),
            global_query_slot_mass_mode=str(
                payload.get("global_query_slot_mass_mode", "system")
            ),
        )

    @property
    def global_slot_logit_bias(self) -> float:
        """Bias keeping aggregate Global-source mass at the R=16 scale."""

        if self.global_query_slot_mass_mode == "system":
            return 0.0
        return float(torch.log(torch.tensor(16.0 / self.global_memory_slots)))


def _fixed_equicorrelated_query_pattern(
    input_query: torch.Tensor,
    *,
    correlation: float,
) -> torch.Tensor:
    """Return the closest row-normalized equicorrelated bank.

    The row directions of ``input_query`` are the input to an orthogonal
    Procrustes projection.  If ``S = G**1/2`` for
    ``G = (1-rho)I + rho 11^T`` and ``U`` contains the input row directions,
    the returned unit-row bank is ``S W`` where ``W`` solves
    ``argmax_W tr(W (S U)^T)`` subject to ``W W^T = I``.  This is the
    closest bank in Frobenius norm among banks with Gram ``G``.  The input
    row norms are restored afterwards so a retrofit preserves the checkpoint
    tensor's per-row scale while its normalized geometry is fixed.

    For the canonical fixed-from-init path, ``input_query`` is the same
    baseline random draw used by the learned mode.  Thus the RNG stream,
    state-dict key, and tensor shape remain unchanged.
    """

    slots, head_dim = input_query.shape[-2:]
    if head_dim < slots:
        raise ValueError("fixed Global16 pattern requires head_dim >= slots")

    # CPU SVD has no fp16/bfloat16 implementation, and low-precision
    # factorization would unnecessarily degrade the prescribed Gram geometry.
    compute_dtype = (
        torch.float64 if input_query.dtype == torch.float64 else torch.float32
    )
    flattened = input_query.to(dtype=compute_dtype).reshape(-1, slots, head_dim)
    row_norms = flattened.norm(dim=-1, keepdim=True)
    if torch.any(row_norms <= torch.finfo(compute_dtype).eps):
        raise ValueError("fixed Global16 projection requires non-zero query rows")
    normalized_input = flattened / row_norms

    dtype = flattened.dtype
    device = input_query.device
    identity = torch.eye(slots, dtype=dtype, device=device)
    shared_projector = torch.full(
        (slots, slots),
        1.0 / slots,
        dtype=dtype,
        device=device,
    )
    centered_projector = identity - shared_projector
    gram_root = (
        (1.0 - correlation) ** 0.5 * centered_projector
        + (1.0 + (slots - 1) * correlation) ** 0.5 * shared_projector
    )
    # Batched SVD gives the polar factor W of S U.  The resulting S W has
    # exact target Gram in exact arithmetic and is the closest feasible bank.
    projected = torch.matmul(gram_root, normalized_input)
    left, _, right_transpose = torch.linalg.svd(projected, full_matrices=False)
    orthonormal_rows = torch.matmul(left, right_transpose)
    unit_pattern = torch.matmul(gram_root, orthonormal_rows)
    fixed = unit_pattern * row_norms
    return fixed.reshape_as(input_query).to(dtype=input_query.dtype)


def _storage_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _require_bool_mask(name: str, mask: torch.Tensor, shape: tuple[int, ...]) -> None:
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool")


def _validate_target_kv(
    config: Global16Raw256Config,
    keys: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[int, int]:
    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError(
            "Target K/V must have identical shape [batch, layers, kv_heads, tokens, head_dim]"
        )
    batch, layers, kv_heads, tokens, head_dim = keys.shape
    expected = (
        config.num_selected_layers,
        config.num_key_value_heads,
        config.head_dim,
    )
    if (layers, kv_heads, head_dim) != expected:
        raise ValueError(
            "Target K/V layer/head shape does not match Global16 + Raw256 config"
        )
    if tokens <= 0:
        raise ValueError("Target K/V token dimension must be non-empty")
    if keys.dtype != values.dtype:
        raise ValueError("Target keys and values must use the same dtype")
    _require_bool_mask("valid_mask", valid_mask, (batch, tokens))
    if not valid_mask.any(dim=-1).all():
        raise ValueError("every Target-K/V row must contain one committed token")
    return batch, tokens


def _require_left_packed_prefix(mask: torch.Tensor, *, name: str) -> None:
    lengths = mask.sum(dim=-1, dtype=torch.long)
    positions = torch.arange(mask.shape[-1], device=mask.device)
    expected = positions[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError(f"{name} must be a left-packed Target-prefix mask")


@dataclass(frozen=True)
class Raw256TargetKVReference:
    """A borrow-only Raw256 reference to a full Target K/V cache.

    ``keys`` and ``values`` deliberately remain the caller's tensors.  A range
    mask selects the most recent 256 valid rows at attention time; this avoids
    an index-select/gather bank and lets a serving kernel consume the underlying
    paged Target cache directly.
    """

    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor
    raw_rows: int = 256

    @classmethod
    def from_target_cache(
        cls,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        raw_rows: int = 256,
    ) -> "Raw256TargetKVReference":
        if raw_rows != 256:
            raise ValueError("Raw256 reference must expose exactly 256 rows")
        if keys.shape != values.shape or keys.ndim != 5:
            raise ValueError("Raw256 keys and values must be rank-5 matching tensors")
        _require_bool_mask("valid_mask", valid_mask, (keys.shape[0], keys.shape[-2]))
        _require_left_packed_prefix(valid_mask, name="Raw256 valid_mask")
        return cls(keys=keys, values=values, valid_mask=valid_mask, raw_rows=raw_rows)

    @property
    def key_storage_ptr(self) -> int:
        return _storage_ptr(self.keys)

    @property
    def value_storage_ptr(self) -> int:
        return _storage_ptr(self.values)

    def window_mask(self) -> torch.Tensor:
        """Return a mask for the latest 256 rows without gathering K/V."""

        lengths = self.valid_mask.sum(dim=-1, dtype=torch.long)
        positions = torch.arange(self.valid_mask.shape[-1], device=self.keys.device)
        start = (lengths - self.raw_rows).clamp_min(0)
        return self.valid_mask & positions[None, :].ge(start[:, None])


@dataclass(frozen=True)
class Local7KV:
    """Borrowed short draft-local K/V.  It is never folded into Raw256."""

    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class AttentionStatistics:
    max_logits: torch.Tensor
    normalizer: torch.Tensor
    weighted_values: torch.Tensor

    def output(self) -> torch.Tensor:
        return self.weighted_values / self.normalizer.clamp_min(
            torch.finfo(self.weighted_values.dtype).tiny
        )[..., None]


def attention_statistics_from_4d(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> AttentionStatistics:
    """Return online-softmax statistics for one selected Target layer.

    The canonical contract stores selected layers explicitly.  The Qwen draft
    decoder processes those layers one at a time, so this narrow adapter keeps
    the exact same grouped-attention mathematics without manufacturing a
    singleton K/V bank or an artificial combined source.

    Args:
        query: ``[B, Hq, Rq, D]``.
        keys/values: ``[B, Hkv, Rk, D]``.
        attention_mask: ``[B, Rq, Rk]``.
    """

    if query.ndim != 4 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("4-D query/K/V tensors are required")
    statistics = _attention_statistics(
        query=query[:, None],
        keys=keys[:, None],
        values=values[:, None],
        attention_mask=attention_mask,
    )
    return AttentionStatistics(
        max_logits=statistics.max_logits[:, 0],
        normalizer=statistics.normalizer[:, 0],
        weighted_values=statistics.weighted_values[:, 0],
    )


def merge_normalized_attention(
    parts: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention outputs using their log normalizers.

    FA3 exposes a normalized output plus ``logsumexp`` rather than the
    reference ``(max, normalizer, weighted_values)`` triple.  This is the
    exact online-softmax representation with ``max=logsumexp`` and
    ``normalizer=1``.  Keeping this conversion here means PyTorch training and
    SGLang serving use one merge rule, rather than similar-looking copies.

    Each tuple is ``(output[..., D], logsumexp[...])``.  Fully-masked sources
    may use ``-inf`` logsumexp; they contribute zero weight.
    """

    if not parts:
        raise ValueError("at least one normalized attention part is required")
    first_output, first_lse = parts[0]
    if first_output.shape[:-1] != first_lse.shape:
        raise ValueError("normalized output and logsumexp shapes do not align")
    for output, lse in parts[1:]:
        if output.shape != first_output.shape or lse.shape != first_lse.shape:
            raise ValueError("all normalized attention parts must have matching shapes")
    accumulator_dtype = (
        torch.float64
        if any(
            output.dtype == torch.float64 or lse.dtype == torch.float64
            for output, lse in parts
        )
        else torch.float32
    )
    lse_stack = torch.stack([lse.to(dtype=accumulator_dtype) for _, lse in parts], dim=0)
    merged_lse = torch.logsumexp(lse_stack, dim=0)
    safe_lse = torch.where(
        torch.isfinite(merged_lse), merged_lse, torch.zeros_like(merged_lse)
    )
    merged_output = torch.zeros_like(first_output, dtype=accumulator_dtype)
    for output, lse in parts:
        contribution = torch.where(
            torch.isfinite(lse),
            torch.exp(lse.to(dtype=accumulator_dtype) - safe_lse),
            torch.zeros_like(safe_lse),
        )
        merged_output = merged_output + contribution[..., None] * output.to(
            dtype=accumulator_dtype
        )
    return merged_output.to(dtype=first_output.dtype), merged_lse


def _attention_statistics(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    key_chunk_size: Optional[int] = None,
) -> AttentionStatistics:
    """GQA attention statistics without expanding, concatenating, or copying K/V.

    The only dtype promotion is applied to score/accumulator scratch tensors;
    K and V are consumed in their original storage and dtype.
    """

    if query.ndim != 5 or keys.ndim != 5 or values.shape != keys.shape:
        raise ValueError("query must be rank 5 and keys/values must be matching rank 5")
    batch, layers, query_heads, query_rows, head_dim = query.shape
    key_batch, key_layers, kv_heads, key_rows, key_head_dim = keys.shape
    if (key_batch, key_layers, key_head_dim) != (batch, layers, head_dim):
        raise ValueError("query and K/V batch/layer/head dimensions must align")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads for grouped attention")
    _require_bool_mask(
        "attention_mask",
        attention_mask,
        (batch, query_rows, key_rows),
    )
    if key_chunk_size is not None:
        if isinstance(key_chunk_size, bool) or not isinstance(key_chunk_size, int):
            raise ValueError("key_chunk_size must be a positive integer or None")
        if key_chunk_size < 1:
            raise ValueError("key_chunk_size must be a positive integer or None")
        if key_chunk_size < key_rows:
            # This is an exact online-softmax reduction over the *entire*
            # prefix, not a context window.  K/V slices are views, so Raw256
            # still reads the original Target-cache storage without a gather or
            # a combined bank.
            merged: Optional[AttentionStatistics] = None
            for start in range(0, key_rows, key_chunk_size):
                stop = min(start + key_chunk_size, key_rows)
                part = _attention_statistics(
                    query=query,
                    keys=keys[..., start:stop, :],
                    values=values[..., start:stop, :],
                    attention_mask=attention_mask[..., start:stop],
                )
                merged = (
                    part
                    if merged is None
                    else _merge_attention_statistics((merged, part))
                )
            assert merged is not None
            return merged

    # Never cast K/V: doing so would turn a Raw256 reference into a copy.
    query_for_scores = (
        query if query.dtype == keys.dtype else query.to(dtype=keys.dtype)
    )
    groups = query_heads // kv_heads
    grouped_query = query_for_scores.reshape(
        batch, layers, kv_heads, groups, query_rows, head_dim
    )
    scores = torch.einsum("blhgrd,blhtd->blhgrt", grouped_query, keys).reshape(
        batch, layers, query_heads, query_rows, key_rows
    )
    accumulator_dtype = (
        torch.float64
        if keys.dtype == torch.float64 or query.dtype == torch.float64
        else torch.float32
    )
    scores = scores.to(dtype=accumulator_dtype) * (head_dim**-0.5)
    expanded_mask = attention_mask[:, None, None, :, :]
    has_visible_key = attention_mask.any(dim=-1)[:, None, None, :]
    masked_scores = scores.masked_fill(~expanded_mask, float("-inf"))
    max_logits = masked_scores.max(dim=-1).values
    safe_max = torch.where(has_visible_key, max_logits, torch.zeros_like(max_logits))
    weights = torch.exp(scores - safe_max[..., None]).masked_fill(
        ~expanded_mask,
        0,
    )
    normalizer = weights.sum(dim=-1)
    grouped_weights = weights.to(dtype=values.dtype).reshape(
        batch, layers, kv_heads, groups, query_rows, key_rows
    )
    weighted_values = torch.einsum(
        "blhgrt,blhtd->blhgrd",
        grouped_weights,
        values,
    ).reshape(batch, layers, query_heads, query_rows, head_dim)
    weighted_values = weighted_values.to(dtype=accumulator_dtype)
    return AttentionStatistics(
        max_logits=torch.where(has_visible_key, max_logits, torch.zeros_like(max_logits)),
        normalizer=normalizer,
        weighted_values=weighted_values,
    )


def _merge_attention_statistics(
    parts: Sequence[AttentionStatistics],
) -> AttentionStatistics:
    if not parts:
        raise ValueError("at least one attention source is required")
    max_logits = torch.zeros_like(parts[0].max_logits)
    has_visible_key = torch.zeros_like(parts[0].normalizer, dtype=torch.bool)
    for part in parts[1:]:
        if part.max_logits.shape != max_logits.shape:
            raise ValueError("attention statistics parts must have matching shapes")
    for part in parts:
        visible = part.normalizer > 0
        first_visible = visible & ~has_visible_key
        max_logits = torch.where(first_visible, part.max_logits, max_logits)
        max_logits = torch.where(
            visible & has_visible_key,
            torch.maximum(max_logits, part.max_logits),
            max_logits,
        )
        has_visible_key = has_visible_key | visible
    normalizer = torch.zeros_like(parts[0].normalizer)
    weighted_values = torch.zeros_like(parts[0].weighted_values)
    for part in parts:
        visible = part.normalizer > 0
        scale = torch.where(
            visible,
            torch.exp(part.max_logits - max_logits),
            torch.zeros_like(part.normalizer),
        )
        normalizer = normalizer + scale * part.normalizer
        weighted_values = weighted_values + scale[..., None] * part.weighted_values
    return AttentionStatistics(max_logits, normalizer, weighted_values)


@dataclass(frozen=True)
class Global16MemoryState:
    """Sufficient online-softmax state for one frozen Global16 query snapshot."""

    query: torch.Tensor
    max_logits: torch.Tensor
    normalizer: torch.Tensor
    weighted_values: torch.Tensor

    def statistics(self) -> AttentionStatistics:
        return AttentionStatistics(
            max_logits=self.max_logits,
            normalizer=self.normalizer,
            weighted_values=self.weighted_values,
        )

    def values(self) -> torch.Tensor:
        return self.statistics().output()

    @classmethod
    def from_normalized_output_and_logsumexp(
        cls,
        *,
        query: torch.Tensor,
        output: torch.Tensor,
        logsumexp: torch.Tensor,
    ) -> "Global16MemoryState":
        """Build a state from FA3's ``(output, logsumexp)`` result.

        FA3 returns a normalized attention value and the logarithm of its
        normalizer.  Encoding it as ``(m=logsumexp, z=1, a=output)`` is an
        algebraically equivalent online-softmax state, so an accepted delta
        can be merged by :func:`_merge_attention_statistics` without a second
        full-prefix read.
        """

        if query.ndim != 5 or output.shape != query.shape:
            raise ValueError("query and normalized output must have matching rank-5 shapes")
        if logsumexp.shape != query.shape[:-1]:
            raise ValueError("logsumexp must match query without head_dim")
        if not torch.isfinite(logsumexp).all():
            raise ValueError("Global16 attention requires at least one visible Target K/V row")
        return cls(
            query=query,
            max_logits=logsumexp.to(dtype=torch.float32),
            normalizer=torch.ones_like(logsumexp, dtype=torch.float32),
            weighted_values=output.to(dtype=torch.float32),
        )


@dataclass(frozen=True)
class JointTargetMemoryReadEvidence:
    """Mechanism evidence emitted by the reference's one shared K/V read."""

    attention_passes: int
    key_storage_ptr: int
    value_storage_ptr: int
    target_query_rows: int
    global_query_rows: int
    combined_kv_bank_bytes: int = 0


@dataclass(frozen=True)
class JointTargetMemoryResult:
    target_output: torch.Tensor
    base_state: Global16MemoryState
    evidence: JointTargetMemoryReadEvidence


@dataclass(frozen=True)
class DraftAttentionReadEvidence:
    raw_key_storage_ptr: int
    raw_value_storage_ptr: int
    global_rows: int
    raw_rows: int
    local_rows: int
    combined_kv_bank_bytes: int = 0


class Global16QueryGenerator(nn.Module):
    """Candidate-independent Global16 queries.

    V1 intentionally uses only stable learned queries.  Committed memory is
    carried as the Global16 *value* state, not fed back into a new query.  That
    keeps independently sampled training anchors and SGLang serving on the
    same definition and avoids untrained state-residual parameters.  A future
    state-conditioned query is a distinct retrain-only scheme, not a silent
    extension of this checkpoint format.
    """

    def __init__(self, config: Global16Raw256Config):
        super().__init__()
        self.config = config
        shape = (
            config.num_selected_layers,
            config.num_attention_heads,
            config.global_memory_slots,
            config.head_dim,
        )
        self.learned_query = nn.Parameter(
            torch.empty(shape),
            requires_grad=config.global_query_mode == "learned",
        )
        # Keep the baseline draw in both modes.  Fixed mode transforms this
        # draw below, preserving RNG progression and checkpoint shape/key.
        nn.init.normal_(self.learned_query, mean=0.0, std=config.head_dim**-0.5)
        # ``from_pretrained`` may construct the module under Transformers'
        # meta-device context before checkpoint tensors are materialized.  In
        # that path there is no data to project: learned checkpoints are
        # projected after loading, while fixed checkpoints load and validate
        # their already-projected tensor.  Fresh, materialized initialization
        # still applies the fixed geometry here.
        if (
            config.global_query_mode == "fixed_equicorrelated"
            and self.learned_query.device.type != "meta"
        ):
            with torch.no_grad():
                self.learned_query.copy_(
                    _fixed_equicorrelated_query_pattern(
                        self.learned_query,
                        correlation=config.global_query_equicorrelation,
                    )
                )

    def forward(
        self,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        return self.learned_query[None].expand(batch_size, -1, -1, -1, -1)


class Global16Raw256Reference(nn.Module):
    """Single mathematical implementation used by training and serving facade."""

    def __init__(self, config: Global16Raw256Config):
        super().__init__()
        self.config = config
        self.global_query_generator = Global16QueryGenerator(config)

    def global_queries(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.global_query_generator(
            batch_size=batch_size,
        ).to(device=device, dtype=dtype)

    def _validate_query(
        self,
        query: torch.Tensor,
        *,
        batch_size: int,
    ) -> None:
        expected = (
            batch_size,
            self.config.num_selected_layers,
            self.config.num_attention_heads,
            self.config.global_memory_slots,
            self.config.head_dim,
        )
        if query.shape != expected:
            raise ValueError(f"Global16 query must have shape {expected}, got {tuple(query.shape)}")

    def joint_target_and_global_attention(
        self,
        *,
        target_queries: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        confirmed_prefix_mask: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        global_queries: Optional[torch.Tensor] = None,
    ) -> JointTargetMemoryResult:
        """Read Target and Global16 rows through one actual K/V operand pair."""

        batch_size, token_count = _validate_target_kv(
            self.config,
            target_keys,
            target_values,
            confirmed_prefix_mask,
        )
        if target_queries.ndim != 5:
            raise ValueError("target_queries must have shape [batch, layers, heads, rows, dim]")
        if target_queries.shape[:3] != (
            batch_size,
            self.config.num_selected_layers,
            self.config.num_attention_heads,
        ) or target_queries.shape[-1] != self.config.head_dim:
            raise ValueError("target_queries do not match Global16 + Raw256 configuration")
        target_rows = target_queries.shape[-2]
        if target_rows <= 0:
            raise ValueError("target_queries must contain at least one row")
        if target_attention_mask is None:
            target_attention_mask = confirmed_prefix_mask[:, None, :].expand(
                -1,
                target_rows,
                -1,
            )
        _require_bool_mask(
            "target_attention_mask",
            target_attention_mask,
            (batch_size, target_rows, token_count),
        )
        if not target_attention_mask.any(dim=-1).all():
            raise ValueError("every Target query row must see at least one key")
        if global_queries is None:
            global_queries = self.global_queries(
                batch_size=batch_size,
                device=target_keys.device,
                dtype=target_keys.dtype,
            )
        self._validate_query(global_queries, batch_size=batch_size)
        global_mask = confirmed_prefix_mask[:, None, :].expand(
            -1,
            self.config.global_memory_slots,
            -1,
        )

        # Query rows are joined; K/V are neither concatenated nor copied.
        joined_queries = torch.cat((target_queries, global_queries), dim=-2)
        joined_mask = torch.cat((target_attention_mask, global_mask), dim=1)
        statistics = _attention_statistics(
            query=joined_queries,
            keys=target_keys,
            values=target_values,
            attention_mask=joined_mask,
        )
        target_output = statistics.output()[..., :target_rows, :].to(
            dtype=target_queries.dtype
        )
        global_slice = slice(target_rows, target_rows + self.config.global_memory_slots)
        base_state = Global16MemoryState(
            query=global_queries,
            max_logits=statistics.max_logits[..., global_slice],
            normalizer=statistics.normalizer[..., global_slice],
            weighted_values=statistics.weighted_values[..., global_slice, :],
        )
        return JointTargetMemoryResult(
            target_output=target_output,
            base_state=base_state,
            evidence=JointTargetMemoryReadEvidence(
                attention_passes=1,
                key_storage_ptr=_storage_ptr(target_keys),
                value_storage_ptr=_storage_ptr(target_values),
                target_query_rows=target_rows,
                global_query_rows=self.config.global_memory_slots,
            ),
        )

    def global_state_from_full_prefix(
        self,
        *,
        query: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        prefix_mask: torch.Tensor,
        key_chunk_size: Optional[int] = None,
    ) -> Global16MemoryState:
        """Full-prefix reference used to validate accepted-delta updates.

        ``key_chunk_size`` only bounds temporary score workspace.  Every
        chunk participates in the same online-softmax reduction, so it does
        not impose a Target-K/V window or alter the Global16 definition.
        """

        batch_size, token_count = _validate_target_kv(
            self.config,
            target_keys,
            target_values,
            prefix_mask,
        )
        self._validate_query(query, batch_size=batch_size)
        mask = prefix_mask[:, None, :].expand(
            -1,
            self.config.global_memory_slots,
            token_count,
        )
        statistics = _attention_statistics(
            query=query,
            keys=target_keys,
            values=target_values,
            attention_mask=mask,
            key_chunk_size=key_chunk_size,
        )
        return Global16MemoryState(
            query=query,
            max_logits=statistics.max_logits,
            normalizer=statistics.normalizer,
            weighted_values=statistics.weighted_values,
        )

    def commit_accepted_delta(
        self,
        *,
        base_state: Global16MemoryState,
        accepted_keys: torch.Tensor,
        accepted_values: torch.Tensor,
        accepted_mask: torch.Tensor,
    ) -> Global16MemoryState:
        """Merge only the confirmed (at most eight) Target K/V rows into state."""

        if accepted_keys.shape != accepted_values.shape or accepted_keys.ndim != 5:
            raise ValueError("accepted Target K/V must be matching rank-5 tensors")
        batch_size, layers, heads, accepted_rows, head_dim = accepted_keys.shape
        expected = (
            self.config.num_selected_layers,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        if (layers, heads, head_dim) != expected:
            raise ValueError("accepted Target K/V do not match Global16 + Raw256 config")
        if accepted_rows > self.config.max_accepted_tokens:
            raise ValueError("accepted Target K/V delta exceeds the eight-token contract")
        _require_bool_mask("accepted_mask", accepted_mask, (batch_size, accepted_rows))
        _require_left_packed_prefix(accepted_mask, name="accepted_mask")
        self._validate_query(base_state.query, batch_size=batch_size)
        delta_mask = accepted_mask[:, None, :].expand(
            -1,
            self.config.global_memory_slots,
            -1,
        )
        delta_statistics = _attention_statistics(
            query=base_state.query,
            keys=accepted_keys,
            values=accepted_values,
            attention_mask=delta_mask,
        )
        merged = _merge_attention_statistics(
            (base_state.statistics(), delta_statistics)
        )
        return Global16MemoryState(
            query=base_state.query,
            max_logits=merged.max_logits,
            normalizer=merged.normalizer,
            weighted_values=merged.weighted_values,
        )

    def training_verify_and_commit(
        self,
        *,
        target_queries: torch.Tensor,
        confirmed_keys: torch.Tensor,
        confirmed_values: torch.Tensor,
        confirmed_prefix_mask: torch.Tensor,
        accepted_keys: torch.Tensor,
        accepted_values: torch.Tensor,
        accepted_mask: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
    ) -> JointTargetMemoryResult:
        """Training path: exactly the serving begin/commit mathematics."""

        result = self.joint_target_and_global_attention(
            target_queries=target_queries,
            target_keys=confirmed_keys,
            target_values=confirmed_values,
            confirmed_prefix_mask=confirmed_prefix_mask,
            target_attention_mask=target_attention_mask,
        )
        committed = self.commit_accepted_delta(
            base_state=result.base_state,
            accepted_keys=accepted_keys,
            accepted_values=accepted_values,
            accepted_mask=accepted_mask,
        )
        return JointTargetMemoryResult(
            target_output=result.target_output,
            base_state=committed,
            evidence=result.evidence,
        )

    def _validate_local7(self, local: Local7KV, *, batch_size: int) -> None:
        if local.keys.shape != local.values.shape or local.keys.ndim != 5:
            raise ValueError("Local7 K/V must be matching rank-5 tensors")
        expected_prefix = (
            batch_size,
            self.config.num_selected_layers,
            self.config.num_key_value_heads,
        )
        if local.keys.shape[:3] != expected_prefix or local.keys.shape[-1] != self.config.head_dim:
            raise ValueError("Local7 K/V do not match Global16 + Raw256 config")
        if local.keys.shape[-2] > self.config.local_draft_rows:
            raise ValueError("Local7 may not contain more than seven rows")
        _require_bool_mask(
            "Local7 valid_mask",
            local.valid_mask,
            (batch_size, local.keys.shape[-2]),
        )

    def draft_attention(
        self,
        *,
        draft_queries: torch.Tensor,
        memory_state: Global16MemoryState,
        raw256: Raw256TargetKVReference,
        local7: Optional[Local7KV] = None,
    ) -> tuple[torch.Tensor, DraftAttentionReadEvidence]:
        """Attend over Global16, borrowed Raw256, and Local7 without a K/V bank."""

        batch_size, _ = _validate_target_kv(
            self.config,
            raw256.keys,
            raw256.values,
            raw256.valid_mask,
        )
        if raw256.raw_rows != self.config.raw_target_rows:
            raise ValueError("Raw256 reference does not match the configured raw window")
        if draft_queries.ndim != 5 or draft_queries.shape[:3] != (
            batch_size,
            self.config.num_selected_layers,
            self.config.num_attention_heads,
        ) or draft_queries.shape[-1] != self.config.head_dim:
            raise ValueError("draft_queries do not match Global16 + Raw256 config")
        self._validate_query(memory_state.query, batch_size=batch_size)
        draft_rows = draft_queries.shape[-2]
        global_mask = torch.ones(
            (batch_size, draft_rows, self.config.global_memory_slots),
            dtype=torch.bool,
            device=draft_queries.device,
        )
        source_statistics = [
            _attention_statistics(
                query=draft_queries,
                keys=memory_state.query,
                values=memory_state.values().to(dtype=memory_state.query.dtype),
                attention_mask=global_mask,
            ),
            _attention_statistics(
                query=draft_queries,
                keys=raw256.keys,
                values=raw256.values,
                attention_mask=raw256.window_mask()[:, None, :].expand(
                    -1,
                    draft_rows,
                    -1,
                ),
            ),
        ]
        local_rows = 0
        if local7 is not None:
            self._validate_local7(local7, batch_size=batch_size)
            local_rows = local7.keys.shape[-2]
            source_statistics.append(
                _attention_statistics(
                    query=draft_queries,
                    keys=local7.keys,
                    values=local7.values,
                    attention_mask=local7.valid_mask[:, None, :].expand(
                        -1,
                        draft_rows,
                        -1,
                    ),
                )
            )
        merged = _merge_attention_statistics(source_statistics)
        output = merged.output().to(dtype=draft_queries.dtype)
        raw_rows = int(raw256.window_mask().sum(dim=-1).max().item())
        return output, DraftAttentionReadEvidence(
            raw_key_storage_ptr=raw256.key_storage_ptr,
            raw_value_storage_ptr=raw256.value_storage_ptr,
            global_rows=self.config.global_memory_slots,
            raw_rows=raw_rows,
            local_rows=local_rows,
        )


@dataclass(frozen=True)
class Global16Raw256WorkspaceBytes:
    joint_query_rows: int
    global_query_snapshot: int
    global_attention_state: int
    accepted_delta_scores: int
    raw256_target_kv_copy: int
    local7_target_kv_copy: int
    legacy_combined_272_kv_bank: int

    @property
    def total_owned(self) -> int:
        return (
            self.joint_query_rows
            + self.global_query_snapshot
            + self.global_attention_state
            + self.accepted_delta_scores
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "joint_query_rows": self.joint_query_rows,
            "global_query_snapshot": self.global_query_snapshot,
            "global_attention_state": self.global_attention_state,
            "accepted_delta_scores": self.accepted_delta_scores,
            "raw256_target_kv_copy": self.raw256_target_kv_copy,
            "local7_target_kv_copy": self.local7_target_kv_copy,
            "legacy_combined_272_kv_bank": self.legacy_combined_272_kv_bank,
            "total_owned": self.total_owned,
        }


def workspace_byte_accounting(
    *,
    config: Global16Raw256Config,
    batch_size: int,
    target_query_rows: int,
    dtype: torch.dtype,
) -> Global16Raw256WorkspaceBytes:
    """Report owned workspace only; Target K/V and Local7 inputs are borrowed."""

    if batch_size <= 0 or target_query_rows <= 0:
        raise ValueError("batch_size and target_query_rows must be positive")
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    cells = (
        batch_size
        * config.num_selected_layers
        * config.num_attention_heads
        * config.global_memory_slots
    )
    return Global16Raw256WorkspaceBytes(
        joint_query_rows=(
            batch_size
            * config.num_selected_layers
            * config.num_attention_heads
            * (target_query_rows + config.global_memory_slots)
            * config.head_dim
            * dtype_bytes
        ),
        global_query_snapshot=cells * config.head_dim * dtype_bytes,
        # max, normalizer, and weighted V accumulator are FP32 reference state.
        global_attention_state=cells * (2 + config.head_dim) * 4,
        accepted_delta_scores=cells * config.max_accepted_tokens * 4,
        raw256_target_kv_copy=0,
        local7_target_kv_copy=0,
        legacy_combined_272_kv_bank=0,
    )


class Global16Raw256ServingSession:
    """Serving lifecycle facade: one joint base scan, then one accepted delta."""

    def __init__(self, reference: Global16Raw256Reference):
        self.reference = reference
        self._committed_state: Optional[Global16MemoryState] = None
        self._pending_base_state: Optional[Global16MemoryState] = None

    @property
    def has_pending_verify(self) -> bool:
        return self._pending_base_state is not None

    @property
    def committed_state(self) -> Optional[Global16MemoryState]:
        return self._committed_state

    def begin_verify(
        self,
        *,
        target_queries: torch.Tensor,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        confirmed_prefix_mask: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
    ) -> JointTargetMemoryResult:
        if self._pending_base_state is not None:
            raise RuntimeError("commit or abandon the current Global16 verification first")
        result = self.reference.joint_target_and_global_attention(
            target_queries=target_queries,
            target_keys=target_keys,
            target_values=target_values,
            confirmed_prefix_mask=confirmed_prefix_mask,
            target_attention_mask=target_attention_mask,
        )
        self._pending_base_state = result.base_state
        return result

    def commit_accepted_delta(
        self,
        *,
        accepted_keys: torch.Tensor,
        accepted_values: torch.Tensor,
        accepted_mask: torch.Tensor,
    ) -> Global16MemoryState:
        if self._pending_base_state is None:
            raise RuntimeError("begin_verify must run before committing accepted Target K/V")
        committed = self.reference.commit_accepted_delta(
            base_state=self._pending_base_state,
            accepted_keys=accepted_keys,
            accepted_values=accepted_values,
            accepted_mask=accepted_mask,
        )
        self._pending_base_state = None
        self._committed_state = committed
        return committed

    def draft_attention(
        self,
        *,
        draft_queries: torch.Tensor,
        raw256: Raw256TargetKVReference,
        local7: Optional[Local7KV] = None,
    ) -> tuple[torch.Tensor, DraftAttentionReadEvidence]:
        if self._committed_state is None:
            raise RuntimeError("Global16 draft attention requires committed memory state")
        return self.reference.draft_attention(
            draft_queries=draft_queries,
            memory_state=self._committed_state,
            raw256=raw256,
            local7=local7,
        )


__all__ = [
    "GLOBAL16_RAW256_SCHEME",
    "AttentionStatistics",
    "attention_statistics_from_4d",
    "DraftAttentionReadEvidence",
    "Global16MemoryState",
    "Global16QueryGenerator",
    "Global16Raw256Config",
    "Global16Raw256Reference",
    "Global16Raw256ServingSession",
    "Global16Raw256WorkspaceBytes",
    "JointTargetMemoryReadEvidence",
    "JointTargetMemoryResult",
    "Local7KV",
    "Raw256TargetKVReference",
    "merge_normalized_attention",
    "workspace_byte_accounting",
]
