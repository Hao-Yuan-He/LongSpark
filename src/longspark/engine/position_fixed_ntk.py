"""Fixed NTK-aware RoPE scaling (alpha=4, 140K context) and its exactness audits."""

from __future__ import annotations

import math

import torch

from sglang.srt.utils.hf_transformers_utils import get_context_length


ALPHA = 4.0
ORIGINAL_CONTEXT = 32768
SERVICE_CONTEXT = 140288
ORIGINAL_THETA = 1_000_000.0
ROTARY_DIM = 128
MAX_POSITION_EMBEDDINGS = 140672
EFFECTIVE_BASE = ORIGINAL_THETA * ALPHA ** (ROTARY_DIM / (ROTARY_DIM - 2))
MSCALE = 1.0
# Model-config flag that marks a checkpoint override as using this adapter.
FLAGS = {
    "fixed_ntk": "longspark_fixed_ntk_v1",
}
POSITIONS = (
    0,
    1,
    32767,
    32768,
    49151,
    65535,
    65536,
    73727,
    73728,
    73735,
    73791,
    74751,
    131071,
    131072,
    139263,
    139264,
    139271,
    139327,
    140287,
    140671,
)


def reference_inv_freq(device):
    """Return fixed NTK inverse frequencies in FP32, independently of cache code."""
    exponents = (
        torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=device) / ROTARY_DIM
    )
    return 1.0 / (EFFECTIVE_BASE**exponents)


def reference_cos_sin(inv_freq, positions):
    """Compute the full-dimension fixed NTK position embeddings in FP32."""
    angles = positions.to(device=inv_freq.device, dtype=torch.float32)[..., None]
    angles = angles * inv_freq
    doubled = torch.cat((angles, angles), dim=-1)
    return doubled.cos() * MSCALE, doubled.sin() * MSCALE


def configured_modes(config):
    if config is None:
        return set()
    enabled = set()
    for mode, flag in FLAGS.items():
        value = getattr(config, flag, False)
        if type(value) is not bool:
            raise RuntimeError(f"Position adapter flag {flag} must be boolean")
        if value:
            enabled.add(mode)
    return enabled


def validate_mode(config, expected):
    """Confirm exactly the expected static position adapter is enabled before graph capture."""
    if configured_modes(config) != {expected}:
        raise RuntimeError(f"Expected exactly the {expected} position adapter")


def _validate_rope_parameters(name, values):
    if not isinstance(values, dict):
        raise RuntimeError(
            f"LongSpark fixed NTK requires actual {name} to be a mapping: {values!r}"
        )
    if "rope_type" not in values and "type" not in values:
        raise RuntimeError(f"LongSpark fixed NTK requires {name}.rope_type=default")
    rope_type = values.get("rope_type", values.get("type", "default"))
    if (
        values.get("rope_type", rope_type) != rope_type
        or values.get("type", rope_type) != rope_type
    ):
        raise RuntimeError(f"LongSpark fixed NTK has conflicting {name} type aliases")
    if rope_type != "default":
        raise RuntimeError(
            f"LongSpark fixed NTK rejects scaled or stale rotary scaling {name}: {values!r}"
        )
    theta = values.get("rope_theta")
    if (
        not isinstance(theta, (int, float))
        or isinstance(theta, bool)
        or not math.isfinite(float(theta))
    ):
        raise RuntimeError(f"LongSpark fixed NTK requires finite {name}.rope_theta")
    if not math.isclose(float(theta), EFFECTIVE_BASE, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            f"LongSpark fixed NTK requires effective rope_theta={EFFECTIVE_BASE}, "
            f"got {theta!r} in {name}"
        )
    # These keys would select a different frequency transform even if a caller
    # mislabeled the map as default.
    stale_keys = {
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
        "beta_fast",
        "beta_slow",
        "attention_factor",
        "mscale",
    }
    if stale_keys.intersection(values):
        raise RuntimeError(f"LongSpark fixed NTK rejects scaled {name}: {values!r}")


def validate_config(config, context_length):
    """Validate the resolved Target/DS configuration before graph capture."""
    validate_mode(config, "fixed_ntk")
    if get_context_length(config) != MAX_POSITION_EMBEDDINGS:
        raise RuntimeError("Fixed NTK derived context differs")
    for name in ("rope_parameters", "rope_scaling"):
        _validate_rope_parameters(name, getattr(config, name, None))
    max_positions = getattr(config, "max_position_embeddings", None)
    layers = getattr(config, "num_hidden_layers", None)
    if max_positions != MAX_POSITION_EMBEDDINGS or context_length != SERVICE_CONTEXT:
        raise RuntimeError(
            f"LongSpark fixed NTK requires max_position_embeddings={MAX_POSITION_EMBEDDINGS} and "
            f"service context={SERVICE_CONTEXT}, got max_position_embeddings="
            f"{max_positions!r}, context_length={context_length!r}"
        )
    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        raise RuntimeError(
            f"LongSpark fixed NTK requires a positive layer count, got {layers!r}"
        )


def _check_close(actual, expected, *, message, rtol=0.0, atol=0.0):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise RuntimeError(message) from exc


def _reference_cache(inv_freq, rows):
    """Build the complete cache with the constructor's FP32 operation shape."""
    positions = torch.arange(rows, device=inv_freq.device, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((freqs.cos()[:, : ROTARY_DIM // 2],
                      freqs.sin()[:, : ROTARY_DIM // 2]), dim=-1)


def _audit_rotary(owner, rope, *, positions):
    if type(rope).__name__ != "RotaryEmbedding":
        raise RuntimeError(
            "LongSpark fixed NTK requires the actual default RotaryEmbedding, "
            f"got {type(rope).__name__}"
        )
    if int(getattr(rope, "rotary_dim", -1)) != ROTARY_DIM or not bool(
        getattr(rope, "is_neox_style", False)
    ):
        raise RuntimeError("LongSpark fixed NTK rotary dimension/style differs")
    if int(getattr(rope, "max_position_embeddings", -1)) != MAX_POSITION_EMBEDDINGS:
        raise RuntimeError("LongSpark fixed NTK rotary cache maximum differs")
    base = getattr(rope, "base", None)
    if not isinstance(base, (int, float)) or not math.isclose(
        float(base), EFFECTIVE_BASE, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"LongSpark fixed NTK resolved rotary base differs: {base!r}"
        )
    if hasattr(rope, "mscale") and not math.isclose(
        float(rope.mscale), MSCALE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("LongSpark fixed NTK rotary magnitude differs")

    # The common-fusion gates are held off for this fixed runtime comparison.
    if getattr(owner, "_dspark_fused_qk_norm_rope", False):
        raise RuntimeError("LongSpark fixed NTK requires DSpark RoPE fusion disabled")
    cache = getattr(rope, "cos_sin_cache", None)
    if not isinstance(cache, torch.Tensor):
        raise RuntimeError("LongSpark fixed NTK rotary cache is not a Tensor")
    if cache.ndim != 2 or tuple(cache.shape) != (MAX_POSITION_EMBEDDINGS, ROTARY_DIM):
        raise RuntimeError(
            "Unexpected LongSpark fixed NTK cache shape: " f"{tuple(cache.shape)}"
        )
    if not cache.is_contiguous():
        raise RuntimeError("LongSpark fixed NTK rotary cache must be contiguous")
    if cache.dtype != torch.float32:
        raise RuntimeError("LongSpark fixed NTK cache must remain FP32")

    with torch.device(cache.device):
        expected_freq = reference_inv_freq(cache.device)
        actual_freq = rope._compute_inv_freq(base).to(device=cache.device)
    _check_close(
        actual_freq,
        expected_freq,
        message="LongSpark fixed NTK resolved inverse frequencies differ",
    )
    selected = cache.index_select(0, positions)
    expected_cache = _reference_cache(expected_freq, MAX_POSITION_EMBEDDINGS).index_select(
        0, positions
    )
    _check_close(
        selected,
        expected_cache,
        message="LongSpark fixed NTK rotary cache differs at a checked position",
        rtol=0.0,
        atol=0.0,
    )
    return float((selected.float() - expected_cache.float()).abs().max())


def audit_model_rotaries(model, config, context_length):
    """Audit every resolved Target/DS rotary owner and its actual cache."""
    validate_config(config, context_length)
    owners = [module for module in model.modules() if hasattr(module, "rotary_emb")]
    expected_layers = int(config.num_hidden_layers)
    if len(owners) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} rotary layers, found {len(owners)}"
        )
    positions = torch.tensor(POSITIONS, dtype=torch.long)
    errors = []
    caches = set()
    cache_signatures = set()
    for owner in owners:
        rope = owner.rotary_emb
        if not isinstance(getattr(rope, "cos_sin_cache", None), torch.Tensor):
            raise RuntimeError("LongSpark fixed NTK rotary owner has no actual cache")
        positions_on_device = positions.to(device=rope.cos_sin_cache.device)
        errors.append(_audit_rotary(owner, rope, positions=positions_on_device))
        caches.add(id(rope))
        cache_signatures.add(
            (str(rope.cos_sin_cache.dtype), str(rope.cos_sin_cache.device))
        )
    if len(cache_signatures) != 1:
        raise RuntimeError(
            f"LongSpark fixed NTK rotary owners disagree on cache dtype/device: {cache_signatures!r}"
        )
    report = dict(
        position_variant="fixed_ntk",
        derived_context_length=get_context_length(config),
        rotary_max_position_embeddings=owners[0].rotary_emb.max_position_embeddings,
        layer_count=len(owners),
        unique_rotary_caches=len(caches),
        cache_rows=owners[0].rotary_emb.cos_sin_cache.shape[0],
        cache_max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        max_position_embeddings=owners[0].rotary_emb.max_position_embeddings,
        cache_dtype=str(owners[0].rotary_emb.cos_sin_cache.dtype),
        cache_device=str(owners[0].rotary_emb.cos_sin_cache.device),
        native=False,
        context_length=context_length,
        alpha=ALPHA,
        factor=ALPHA,
        original_context=ORIGINAL_CONTEXT,
        original_theta=ORIGINAL_THETA,
        theta_original=ORIGINAL_THETA,
        theta=EFFECTIVE_BASE,
        theta_effective=EFFECTIVE_BASE,
        effective_base=EFFECTIVE_BASE,
        rotary_dim=ROTARY_DIM,
        mscale=MSCALE,
        inv_freq_dtype="float32",
        checked_positions=list(POSITIONS),
        max_cache_error=max(errors),
        ordinary_rope_fusion=False,
        weights_changed=False,
    )
    model._longspark_fixed_ntk_audit = report
    return report


def set_fixed_inv_freq(attention, inv_freq_fn=reference_inv_freq):
    old = getattr(attention, "inv_freq", None)
    if not isinstance(old, torch.Tensor):
        raise RuntimeError("LongSpark fixed NTK Draft attention has no RoPE buffer")
    expected = inv_freq_fn(old.device)
    if tuple(old.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"LongSpark fixed NTK Draft RoPE shape differs: {tuple(old.shape)}"
        )
    # Keep the existing buffer registration and persistence policy.  The
    # SpecForge buffer is nonpersistent, so this adds no checkpoint state.
    if "inv_freq" in getattr(attention, "_buffers", {}):
        if "inv_freq" not in attention._non_persistent_buffers_set:
            raise RuntimeError("Static RoPE cannot modify a persistent checkpoint buffer")
        attention._buffers["inv_freq"] = expected
    else:
        attention.register_buffer("inv_freq", expected, persistent=False)
    attention._longspark_mscale = MSCALE


def audit_longspark(layers, native=False, *, inv_freq_fn=reference_inv_freq):
    """Audit all five shared Draft/native fixed NTK frequency buffers."""
    if len(layers) != 5:
        raise RuntimeError(
            f"LongSpark fixed NTK requires five shared layers, got {len(layers)}"
        )
    expected_freq = None
    errors = []
    positions = None
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        inv_freq = getattr(attention, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or inv_freq.dtype != torch.float32:
            raise RuntimeError(
                f"LongSpark fixed NTK layer {layer_index} frequencies must remain FP32"
            )
        if tuple(inv_freq.shape) != (ROTARY_DIM // 2,):
            raise RuntimeError("LongSpark fixed NTK Draft frequency shape differs")
        if getattr(attention, "_longspark_mscale", None) != MSCALE:
            raise RuntimeError(
                f"LongSpark fixed NTK layer {layer_index} magnitude is not explicit 1.0"
            )
        if native and getattr(attention, "fused_qk_norm_rope_enabled", False):
            raise RuntimeError(
                "LongSpark fixed NTK requires native RoPE fusion disabled"
            )
        expected_freq = inv_freq_fn(inv_freq.device)
        _check_close(
            inv_freq,
            expected_freq,
            message=f"LongSpark fixed NTK layer {layer_index} frequencies differ",
            rtol=0.0,
            atol=0.0,
        )
        if not callable(getattr(attention, "_position_embeddings", None)):
            raise RuntimeError(
                "LongSpark fixed NTK layer has no position embedding method"
            )
        positions = torch.tensor(
            POSITIONS, device=inv_freq.device, dtype=torch.long
        ).view(1, -1)
        expected = reference_cos_sin(expected_freq, positions)
        actual = attention._position_embeddings(positions, dtype=torch.float32)
        if not isinstance(actual, tuple) or len(actual) != 2:
            raise RuntimeError(
                "LongSpark fixed NTK position embedding shape is invalid"
            )
        for value, reference in zip(actual, expected):
            _check_close(
                value,
                reference,
                message=f"LongSpark fixed NTK layer {layer_index} position embeddings differ",
                rtol=0.0,
                atol=0.0,
            )
            errors.append(float((value - reference).abs().max()))
    return dict(
        position_variant="fixed_ntk",
        context_length=SERVICE_CONTEXT,
        cache_max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        layer_count=len(layers),
        native=native,
        alpha=ALPHA,
        factor=ALPHA,
        original_theta=ORIGINAL_THETA,
        theta_original=ORIGINAL_THETA,
        theta=EFFECTIVE_BASE,
        theta_effective=EFFECTIVE_BASE,
        effective_base=EFFECTIVE_BASE,
        mscale=MSCALE,
        checked_positions=list(POSITIONS),
        max_rotary_error=max(errors),
        inv_freq_dtype="float32",
        ordinary_rope_fusion=False,
        weights_changed=False,
    )


__all__ = [
    "ALPHA",
    "EFFECTIVE_BASE",
    "MAX_POSITION_EMBEDDINGS",
    "MSCALE",
    "POSITIONS",
    "audit_longspark",
    "audit_model_rotaries",
    "reference_cos_sin",
    "reference_inv_freq",
    "set_fixed_inv_freq",
    "validate_config",
    "validate_mode",
]
