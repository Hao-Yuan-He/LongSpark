"""Adapt the frozen fixed-NTK audit to independently loaded target/draft ranks."""
import math

from . import longspark_fixed_ntk as ntk


def adapter(variant):
    if variant == 'native':
        from . import native_adapter
        return native_adapter
    if variant == 'fixed_ntk':
        return ntk
    raise ValueError(f'unsupported position variant: {variant}')


def model_override(variant='fixed_ntk'):
    if variant == 'native':
        return adapter(variant).model_override()
    adapter(variant)
    return dict(max_position_embeddings=ntk.MAX_POSITION_EMBEDDINGS,
                rope_parameters=dict(rope_type="default", rope_theta=ntk.EFFECTIVE_BASE),
                rope_scaling=dict(rope_type="default", rope_theta=ntk.EFFECTIVE_BASE),
                longspark_yarn_v1=False, longspark_fixed_ntk_v1=True,
                longspark_linear_pi_v1=False)


def audit_runner(runner, variant='fixed_ntk'):
    if variant == 'native':
        return adapter(variant).audit_runner(runner)
    adapter(variant)
    return ntk.audit_model_rotaries(runner.model, runner.model_config.hf_text_config,
                                    runner.model_config.context_len)


def configure_split_longspark(model, target_ready, variant='fixed_ntk'):
    if variant == 'native':
        if not target_ready or any(r.get('position_audit', {}).get('position_variant') != 'native' for r in target_ready):
            raise RuntimeError('missing remote native position audit')
        return adapter(variant).audit_longspark(model.parallel_qwen_layers)
    adapter(variant)
    if not target_ready:
        raise RuntimeError("missing actual remote Target RoPE audit")
    for rank in target_ready:
        audit = rank.get("position_audit", {})
        if (audit.get("position_variant") != "fixed_ntk"
                or audit.get("context_length") != ntk.SERVICE_CONTEXT
                or audit.get("max_cache_error") != 0.
                or audit.get("effective_base") != ntk.EFFECTIVE_BASE
                or audit.get("inv_freq_dtype") != "float32"):
            raise RuntimeError(f"Target rank RoPE audit does not match frozen NTK: {audit}")
    if len(model.parallel_qwen_layers) != 5 or not math.isclose(
            model.config.qwen_rope_theta, ntk.ORIGINAL_THETA, abs_tol=1e-6):
        raise RuntimeError("unexpected LongSpark checkpoint configuration")
    # Explicitly override only nonpersistent buffers, after checkpoint restoration.
    for layer in model.parallel_qwen_layers:
        ntk._set_fixed_inv_freq(layer.self_attn)
    return ntk.audit_longspark(model.parallel_qwen_layers)


def configure_native(layers, variant='fixed_ntk'):
    mode = adapter(variant)
    if variant == 'native':
        return mode.audit_longspark(layers, native=True)
    for layer in layers:
        layer.self_attn._longspark_mscale = ntk.MSCALE
    return mode.audit_longspark(layers, native=True)
