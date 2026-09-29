"""Opt-in Qwen3 verify8 preparation with the existing BF16/cache contract."""

import os

import torch
import triton
import triton.language as tl


def target_verify_prepare_mode():
    mode = os.getenv("SGLANG_TARGET_VERIFY_PREPARE", "off")
    if mode not in ("off", "rope_store", "norm_rope_store"):
        raise ValueError(f"Unsupported Target verification preparation: {mode}")
    return mode


def target_verify_pool(attention, positions, hidden_states, forward_batch):
    """Return the supported pool, or leave the established path untouched."""
    if (attention.target_verify_prepare_mode == "off"
            or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or not forward_batch.forward_mode.is_target_verify()):
        return None
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.speculative.draft_free_kv_mask_policy import _unwrap_full_attention_backend

    args = get_global_server_args()
    backend = _unwrap_full_attention_backend(get_attn_backend())
    rope = attention.rotary_emb
    pool = getattr(backend, "token_to_kv_pool", None)
    if not (
        args.tp_size == 1 and args.speculative_num_draft_tokens == 8
        and args.speculative_algorithm in ("DSPARK", "DRAFT_FREE_KV")
        and not args.enable_deterministic_inference
        and args.rl_on_policy_target is None
        and type(backend).__name__ == "FlashAttentionBackend"
        and backend.fa_impl_ver == 3 and backend.page_size == 1
        and type(pool).__name__ == "MHATokenToKVPool"
        and pool.dtype == pool.store_dtype == torch.bfloat16
        and attention.head_dim == 128
        and attention.num_heads == 32 and attention.num_kv_heads == 8
        and attention.q_norm.variance_epsilon == attention.k_norm.variance_epsilon
        and attention.q_norm.weight.dtype == attention.k_norm.weight.dtype == torch.bfloat16
        and attention.attn.k_scale is None and attention.attn.v_scale is None
        and not attention.attn.is_cross_attention
        and getattr(attention.attn, "sliding_window_size", None) in (None, -1)
        and rope.rotary_dim == 128 and rope.is_neox_style
        and not rope.use_fallback_kernel
        and rope.cos_sin_cache.dtype == torch.float32 and rope.cos_sin_cache.is_contiguous()
        and positions.ndim == 1 and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and positions.numel() == hidden_states.shape[0]
        and forward_batch.out_cache_loc.numel() == positions.numel()
        and forward_batch.out_cache_loc.is_contiguous()
        and forward_batch.out_cache_loc.dtype == positions.dtype
        and getattr(forward_batch.spec_info, "ragged_verify_layout", None) is None
    ):
        return None
    return pool


@triton.jit
def _target_verify_norm_rope_store(
    QKV, QW, KW, POS, COS_SIN, LOC, KC, VC,
    TOKENS, CACHE_ROWS: tl.constexpr, POS_ROWS: tl.constexpr,
    EPS: tl.constexpr, HEADS: tl.constexpr,
):
    # One warp per 128-wide head; several heads share one program. Preserve
    # the BF16 norm output before applying the cached FP32 RoPE coefficients.
    ids = tl.program_id(0) * HEADS + tl.arange(0, HEADS)
    token, head = ids // 40, ids % 40
    d = tl.arange(0, 128)
    valid = token < TOKENS
    offsets = token[:, None] * 6144 + head[:, None] * 128 + d[None, :]
    x = tl.load(QKV + offsets, valid[:, None], 0).to(tl.float32)
    qw = tl.load(QW + d).to(tl.float32)
    kw = tl.load(KW + d).to(tl.float32)
    w = tl.where(head[:, None] < 32, qw[None, :], kw[None, :])
    inv = tl.rsqrt(tl.sum(x * x, 1) / 128.0 + EPS)
    norm = (x * inv[:, None] * w).to(tl.bfloat16).to(tl.float32)
    paired = tl.gather(norm, tl.broadcast_to(((d + 64) % 128)[None, :], (HEADS, 128)), 1)
    pos = tl.load(POS + token, valid, 0).to(tl.int64)
    loc = tl.load(LOC + token, valid, 0).to(tl.int64)
    pos_valid = valid & (pos >= 0) & (pos < POS_ROWS)
    c = tl.load(COS_SIN + pos[:, None] * 128 + (d % 64)[None, :], pos_valid[:, None], 1)
    s = tl.load(COS_SIN + pos[:, None] * 128 + 64 + (d % 64)[None, :], pos_valid[:, None], 0)
    out = norm * c + tl.where(d[None, :] < 64, -paired, paired) * s
    tl.store(QKV + offsets, out, valid[:, None])
    # Slot zero is the pool's padding sink. Skip its duplicate writes; retain
    # the post-RoPE packed K for LongSpark's selected-layer Global16 hook.
    write = valid & (head >= 32) & (loc > 0) & (loc < CACHE_ROWS)
    cache_offsets = loc[:, None] * 1024 + (head[:, None] - 32) * 128 + d[None, :]
    tl.store(KC + cache_offsets, out, write[:, None])
    values = tl.load(QKV + offsets + 1024, write[:, None], 0)
    tl.store(VC + cache_offsets, values, write[:, None])


def fused_norm_rope_store(qkv, q_weight, k_weight, positions, cos_sin, cache_loc,
                          key_cache, value_cache, eps, *, heads_per_program=4):
    assert qkv.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
    assert qkv.ndim == 2 and qkv.shape[1] == 6144 and qkv.is_contiguous()
    assert key_cache.is_contiguous() and value_cache.is_contiguous()
    assert key_cache.shape == value_cache.shape and key_cache.shape[-2:] == (8, 128)
    assert cos_sin.dtype == torch.float32 and cos_sin.shape[1] == 128
    assert q_weight.numel() == k_weight.numel() == 128
    assert positions.numel() == cache_loc.numel() == qkv.shape[0]
    if qkv.shape[0]:
        _target_verify_norm_rope_store[(triton.cdiv(qkv.shape[0] * 40, heads_per_program),)](
            qkv, q_weight, k_weight, positions, cos_sin, cache_loc, key_cache, value_cache,
            qkv.shape[0], key_cache.shape[0], cos_sin.shape[0], eps, heads_per_program,
            num_warps=4, enable_fp_fusion=False,
        )


def prepare_target_verify(attention, positions, hidden_states, forward_batch, pool):
    if not getattr(attention, '_target_verify_prepare_reported', False):
        import logging
        logging.getLogger(__name__).info('Target verify prepare selected mode=%s layer=%s',
                                        attention.target_verify_prepare_mode,attention.attn.layer_id)
        attention._target_verify_prepare_reported = True
    qkv, _ = attention.qkv_proj(hidden_states)
    q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
    if attention.target_verify_prepare_mode == "norm_rope_store":
        key_cache, value_cache = pool.get_kv_buffer(attention.attn.layer_id)
        fused_norm_rope_store(
            qkv, attention.q_norm.weight, attention.k_norm.weight, positions,
            attention.rotary_emb.cos_sin_cache, forward_batch.out_cache_loc,
            key_cache, value_cache, attention.q_norm.variance_epsilon,
        )
    else:
        from sglang.jit_kernel.rope import FusedSetKVBufferArg
        from sglang.srt.models.utils import apply_qk_norm

        q, k = apply_qk_norm(q, k, attention.q_norm, attention.k_norm,
                             attention.head_dim, attention.alt_stream)
        kc, vc = pool.get_kv_buffer(attention.attn.layer_id)
        q, k = attention.rotary_emb(
            positions, q, k, fused_set_kv_buffer_arg=FusedSetKVBufferArg(
                value=v, k_buffer=kc.view(kc.shape[0], -1),
                v_buffer=vc.view(vc.shape[0], -1), cache_loc=forward_batch.out_cache_loc,
            ),
        )
    return q, k, v
