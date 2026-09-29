"""Unscaled (native) Qwen3 RoPE; no shared module or checkpoint mutation."""
import torch

PUBLIC_CONTEXT = 40960
SERVICE_CONTEXT = PUBLIC_CONTEXT + 8


def model_override():
    params = dict(rope_type='default', rope_theta=1_000_000.)
    return dict(max_position_embeddings=SERVICE_CONTEXT, rope_theta=1_000_000.,
                rope_parameters=dict(params), rope_scaling=dict(params),
                longspark_fixed_ntk_v1=False, longspark_yarn_v1=False,
                longspark_linear_pi_v1=False)


def audit_runner(runner):
    owners = [m for m in runner.model.modules() if hasattr(m, 'rotary_emb')]
    if len(owners) != runner.model_config.hf_text_config.num_hidden_layers:
        raise RuntimeError('native rotary layer count differs')
    for owner in owners:
        rope = owner.rotary_emb
        if type(rope).__name__ != 'RotaryEmbedding' or rope.base != 1_000_000.:
            raise RuntimeError('expected unscaled Qwen3 native RoPE')
        if rope.cos_sin_cache.shape[0] < SERVICE_CONTEXT:
            raise RuntimeError('native cache too short')
        cache = rope.cos_sin_cache
        positions = torch.tensor([0, 32767, 40959, 40960, 40967], device=cache.device)
        freq = 1. / (1_000_000. ** (torch.arange(0, 128, 2, device=cache.device).float() / 128))
        phase = torch.einsum('i,j->ij', positions.float(), freq)
        expected = torch.cat([phase.cos(), phase.sin()], dim=-1).to(cache.dtype)
        torch.testing.assert_close(cache[positions], expected, atol=0, rtol=0)
    return dict(position_variant='native', theta=1_000_000., context_length=SERVICE_CONTEXT,
                generation_context_limit=PUBLIC_CONTEXT, scratch_rows=8,
                layer_count=len(owners), weights_changed=False, boundary_cache_error=0.)


def audit_longspark(layers, native=False):
    for layer in layers:
        a = layer.self_attn
        expected = 1. / (1_000_000. ** (torch.arange(0, 128, 2, device=a.inv_freq.device).float() / 128))
        torch.testing.assert_close(a.inv_freq, expected, atol=0, rtol=0)
    return dict(position_variant='native', theta=1_000_000., native=native,
                layer_count=len(layers), weights_changed=False)
