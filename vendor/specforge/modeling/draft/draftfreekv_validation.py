from typing import Optional

import torch


PARALLEL_SHIFTED_DRAFT_MODES = {
    "parallel_shifted",
    "parallel_shifted_qwen",
}


TRANSFORMERS_METADATA_KEYS = {
    "_name_or_path",
    "_commit_hash",
    "architectures",
    "attn_implementation",
    "dtype",
    "model_type",
    "torch_dtype",
    "transformers_version",
}


def pop_transformers_metadata(kwargs: dict) -> dict:
    return {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in TRANSFORMERS_METADATA_KEYS
    }


def reject_unknown_config_fields(kwargs: dict) -> None:
    if kwargs:
        raise ValueError(
            "unsupported DraftFreeKV config fields: " + ", ".join(sorted(kwargs))
        )


def resolve_target_layer_ids(
    num_target_layers: int,
    target_layer_ids: Optional[list[int]] = None,
    policy: Optional[str] = None,
) -> list[int]:
    _require_positive_int("num_target_layers", num_target_layers)

    if target_layer_ids is not None:
        resolved = [int(layer_id) for layer_id in target_layer_ids]
    elif policy is not None:
        resolved = _resolve_layer_policy(num_target_layers, policy)
    else:
        raise ValueError("target_layer_ids or policy must be provided")

    if not resolved:
        raise ValueError("target_layer_ids must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError("target_layer_ids must be unique")
    if any(layer_id < 0 or layer_id >= num_target_layers for layer_id in resolved):
        raise ValueError("target_layer_ids must be within num_target_layers")
    return resolved


def resolve_draftfreekv_config_fields(
    *,
    target_model_name_or_path: Optional[str],
    target_layer_ids: Optional[list[int]],
    target_layer_policy: Optional[str],
    num_target_layers: Optional[int],
    block_size: int,
    hidden_size: int,
    num_key_value_heads: int,
    head_dim: int,
    target_kv_window_size: Optional[int],
    target_kv_memory_layout: str,
    target_kv_memory_query_rank: int,
    vocab_size: int,
    adapter_rank: int,
    adapter_alpha: float,
    transition_rank: Optional[int],
    transition_alpha: Optional[float],
    draft_mode: str,
    mask_token_id: Optional[int],
    parallel_mixer_layers: int,
    parallel_mixer_rank: Optional[int],
    parallel_mixer_alpha: Optional[float],
    parallel_mixer_num_heads: int,
    parallel_markov_rank: int,
    parallel_markov_head_type: str,
    xpress_num_refinement_steps: int,
    xpress_early_stop: bool,
    xpress_activation: str,
    qwen_intermediate_size: Optional[int],
    qwen_num_attention_heads: Optional[int],
    qwen_rope_theta: float,
    qwen_rms_norm_eps: float,
    qwen_gradient_checkpointing: bool,
    training_objective: str,
) -> dict:
    _require_positive_int("block_size", block_size)
    _require_positive_int("hidden_size", hidden_size)
    _require_positive_int("num_key_value_heads", num_key_value_heads)
    _require_positive_int("head_dim", head_dim)
    if target_kv_window_size is not None:
        _require_positive_int("target_kv_window_size", target_kv_window_size)
        target_kv_window_size = int(target_kv_window_size)
    target_kv_memory_layout = str(target_kv_memory_layout).lower()
    if target_kv_memory_layout not in {"direct", "attention_generated"}:
        raise ValueError(
            "target_kv_memory_layout must be 'direct' or 'attention_generated'"
        )
    _require_positive_int("target_kv_memory_query_rank", target_kv_memory_query_rank)
    _require_positive_int("vocab_size", vocab_size)
    _require_positive_int("adapter_rank", adapter_rank)
    _require_positive_float("adapter_alpha", adapter_alpha)

    transition_rank = _default_rank(transition_rank, adapter_rank, "transition_rank")
    transition_alpha = _default_alpha(
        transition_alpha,
        adapter_alpha,
        "transition_alpha",
    )
    draft_mode = str(draft_mode).lower()
    if draft_mode not in {"markov", *PARALLEL_SHIFTED_DRAFT_MODES}:
        raise ValueError(f"unsupported DraftFreeKV draft_mode: {draft_mode}")

    parallel_mixer_layers = int(parallel_mixer_layers)
    if parallel_mixer_layers < 0:
        raise ValueError("parallel_mixer_layers must be non-negative")
    parallel_mixer_rank = _default_rank(
        parallel_mixer_rank,
        adapter_rank,
        "parallel_mixer_rank",
    )
    parallel_mixer_alpha = _default_alpha(
        parallel_mixer_alpha,
        adapter_alpha,
        "parallel_mixer_alpha",
    )
    _require_positive_int("parallel_mixer_num_heads", parallel_mixer_num_heads)
    parallel_markov_rank = int(parallel_markov_rank)
    if parallel_markov_rank < 0:
        raise ValueError("parallel_markov_rank must be non-negative")
    parallel_markov_head_type = str(parallel_markov_head_type).lower()
    if parallel_markov_head_type != "vanilla":
        raise ValueError(
            "parallel_markov_head_type currently supports only 'vanilla'"
        )
    xpress_num_refinement_steps = int(xpress_num_refinement_steps)
    if xpress_num_refinement_steps <= 0:
        raise ValueError("xpress_num_refinement_steps must be positive")
    xpress_early_stop = bool(xpress_early_stop)
    xpress_activation = str(xpress_activation).lower()
    if xpress_activation not in {"silu", "gelu"}:
        raise ValueError("xpress_activation must be 'silu' or 'gelu'")
    if draft_mode in PARALLEL_SHIFTED_DRAFT_MODES:
        if (
            draft_mode == "parallel_shifted"
            and hidden_size % int(parallel_mixer_num_heads) != 0
        ):
            raise ValueError(
                "hidden_size must be divisible by parallel_mixer_num_heads"
            )
        if mask_token_id is None:
            raise ValueError("mask_token_id is required for parallel shifted modes")
        mask_token_id = int(mask_token_id)
        if mask_token_id < 0 or mask_token_id >= vocab_size:
            raise ValueError("mask_token_id must be within vocab_size")
    elif parallel_markov_rank > 0:
        raise ValueError(
            "parallel_markov_rank can be enabled only in parallel shifted modes"
        )

    if draft_mode == "parallel_shifted_qwen":
        if qwen_intermediate_size is None:
            raise ValueError(
                "qwen_intermediate_size is required for parallel_shifted_qwen"
            )
        if qwen_num_attention_heads is None:
            raise ValueError(
                "qwen_num_attention_heads is required for parallel_shifted_qwen"
            )
    qwen_intermediate_size = int(qwen_intermediate_size or (4 * hidden_size))
    qwen_num_attention_heads = int(
        qwen_num_attention_heads or (hidden_size // head_dim)
    )
    _require_positive_int("qwen_intermediate_size", qwen_intermediate_size)
    _require_positive_int("qwen_num_attention_heads", qwen_num_attention_heads)
    _require_positive_float("qwen_rope_theta", qwen_rope_theta)
    _require_positive_float("qwen_rms_norm_eps", qwen_rms_norm_eps)
    if draft_mode == "parallel_shifted_qwen":
        if parallel_mixer_layers <= 0:
            raise ValueError(
                "parallel_mixer_layers must be positive for parallel_shifted_qwen"
            )
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for Qwen rotary embeddings")
        if qwen_num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "qwen_num_attention_heads must be divisible by num_key_value_heads"
            )
    elif target_kv_memory_layout != "direct":
        raise ValueError(
            "attention-generated Target memory requires parallel_shifted_qwen"
        )
    if target_kv_memory_layout == "attention_generated" and target_kv_window_size is None:
        raise ValueError(
            "target_kv_window_size is required for attention-generated Target memory"
        )
    target_layer_ids = _resolve_config_layer_ids(
        target_layer_ids,
        target_layer_policy,
        num_target_layers,
    )
    if training_objective not in {
        "ce",
        "kl",
        "kd",
        "rkl",
        "dflash_rkl",
        "dspark_l1",
        "lk_alpha",
        "lk_hybrid",
        "on_policy_ce",
        "on_policy_kl",
    }:
        raise ValueError(f"unsupported DraftFreeKV training_objective: {training_objective}")

    return {
        "target_model_name_or_path": target_model_name_or_path,
        "target_layer_ids": target_layer_ids,
        "target_layer_policy": target_layer_policy,
        "num_target_layers": num_target_layers,
        "block_size": block_size,
        "hidden_size": hidden_size,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "target_kv_window_size": target_kv_window_size,
        "target_kv_memory_layout": target_kv_memory_layout,
        "target_kv_memory_query_rank": int(target_kv_memory_query_rank),
        "vocab_size": vocab_size,
        "adapter_rank": adapter_rank,
        "adapter_alpha": adapter_alpha,
        "transition_rank": transition_rank,
        "transition_alpha": transition_alpha,
        "draft_mode": draft_mode,
        "mask_token_id": mask_token_id,
        "parallel_mixer_layers": parallel_mixer_layers,
        "parallel_mixer_rank": parallel_mixer_rank,
        "parallel_mixer_alpha": parallel_mixer_alpha,
        "parallel_mixer_num_heads": int(parallel_mixer_num_heads),
        "parallel_markov_rank": parallel_markov_rank,
        "parallel_markov_head_type": parallel_markov_head_type,
        "xpress_num_refinement_steps": xpress_num_refinement_steps,
        "xpress_early_stop": xpress_early_stop,
        "xpress_activation": xpress_activation,
        "qwen_intermediate_size": qwen_intermediate_size,
        "qwen_num_attention_heads": qwen_num_attention_heads,
        "qwen_rope_theta": float(qwen_rope_theta),
        "qwen_rms_norm_eps": float(qwen_rms_norm_eps),
        "qwen_gradient_checkpointing": bool(qwen_gradient_checkpointing),
        "training_objective": training_objective,
    }


def normalize_kv_attention_mask(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    expected_layers: int,
    expected_heads: int,
    expected_head_dim: int,
) -> torch.Tensor:
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape")
    if keys.ndim != 5:
        raise ValueError(
            "keys and values must have shape [batch, layers, heads, tokens, head_dim]"
        )

    batch, layers, heads, tokens, head_dim = keys.shape
    if layers != expected_layers:
        raise ValueError(f"expected {expected_layers} selected layers, received {layers}")
    if heads != expected_heads:
        raise ValueError(f"expected {expected_heads} KV heads, received {heads}")
    if head_dim != expected_head_dim:
        raise ValueError(f"expected head_dim={expected_head_dim}, received {head_dim}")
    if tokens <= 0:
        raise ValueError("KV token dimension must be non-empty")

    if attention_mask is None:
        return torch.ones((batch, tokens), dtype=torch.bool, device=keys.device)
    if attention_mask.shape != (batch, tokens):
        raise ValueError("attention_mask must have shape [batch, tokens]")
    attention_mask = attention_mask.to(device=keys.device, dtype=torch.bool)
    if not attention_mask.any(dim=-1).all():
        raise ValueError("each batch row must expose at least one KV token")
    return attention_mask


def select_latest_visible_target_kv(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    window_size: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact dense target KV rows to their latest visible tokens."""

    if window_size is None:
        return keys, values, attention_mask
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("target KV window size must be positive")
    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError(
            "keys and values must have shape [batch, layers, heads, tokens, head_dim]"
        )
    batch_size, layers, heads, token_count, head_dim = keys.shape
    if attention_mask.shape != (batch_size, token_count):
        raise ValueError("attention_mask must have shape [batch, tokens]")
    if token_count <= window_size:
        return keys, values, attention_mask.bool()

    # Select the latest visible positions even with padding or anchor-prefix
    # masks. Reversing topk restores chronological order and left-pads short rows,
    # matching the SGLang reader layout.
    token_positions = torch.arange(token_count, device=attention_mask.device)
    visible_positions = token_positions[None, :].expand(batch_size, -1).masked_fill(
        ~attention_mask.bool(),
        -1,
    )
    selected_positions = visible_positions.topk(
        k=window_size,
        dim=-1,
        largest=True,
        sorted=True,
    ).values.flip(-1)
    selected_mask = selected_positions.ge(0)
    safe_positions = selected_positions.clamp_min(0).to(device=keys.device)
    gather_index = safe_positions[:, None, None, :, None].expand(
        batch_size,
        layers,
        heads,
        window_size,
        head_dim,
    )
    keys = keys.gather(dim=-2, index=gather_index)
    values = values.gather(dim=-2, index=gather_index.to(device=values.device))
    return keys, values, selected_mask.to(device=keys.device)


def restrict_target_kv_attention_to_window(
    *,
    attention_mask: torch.Tensor,
    current_position_ids: torch.Tensor,
    window_size: Optional[int],
) -> torch.Tensor:
    """Apply an anchor-relative latest-N window without duplicating shared KV."""

    if window_size is None:
        return attention_mask
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("target KV window size must be positive")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, tokens]")
    if current_position_ids.shape != (attention_mask.shape[0],):
        raise ValueError("current_position_ids must have shape [batch]")
    token_positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    )[None, :]
    current_position_ids = current_position_ids.to(
        device=attention_mask.device,
        dtype=torch.long,
    )
    window_start = (current_position_ids - window_size).clamp_min(0)[:, None]
    prefix_end = current_position_ids[:, None]
    return (
        attention_mask.bool()
        & token_positions.ge(window_start)
        & token_positions.lt(prefix_end)
    )


def normalize_current_hidden_states(
    *,
    current_hidden_states: Optional[torch.Tensor],
    batch_size: int,
    expected_layers: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if current_hidden_states is None:
        raise ValueError("current_hidden_states are required")
    expected_shape = (batch_size, expected_layers, hidden_size)
    if current_hidden_states.shape != expected_shape:
        raise ValueError(
            "current_hidden_states must have shape [batch, layers, hidden_size]"
        )
    return current_hidden_states.to(device=device, dtype=dtype)


def _resolve_layer_policy(num_target_layers: int, policy: str) -> list[int]:
    if policy.startswith("uniform_"):
        count = int(policy.removeprefix("uniform_"))
        _require_positive_int("uniform layer count", count)
        if count == 1:
            return [num_target_layers // 2]
        span = num_target_layers - 1
        return [round(index * span / (count - 1)) for index in range(count)]
    if policy.startswith("last_"):
        count = int(policy.removeprefix("last_"))
        _require_positive_int("last layer count", count)
        if count > num_target_layers:
            raise ValueError("last layer count cannot exceed num_target_layers")
        return list(range(num_target_layers - count, num_target_layers))
    raise ValueError(f"unknown target layer policy: {policy}")


def _resolve_config_layer_ids(
    target_layer_ids: Optional[list[int]],
    target_layer_policy: Optional[str],
    num_target_layers: Optional[int],
) -> list[int]:
    if target_layer_ids is not None and num_target_layers is not None:
        return resolve_target_layer_ids(
            num_target_layers=num_target_layers,
            target_layer_ids=target_layer_ids,
        )
    if target_layer_ids is None and target_layer_policy is not None:
        if num_target_layers is None:
            raise ValueError("num_target_layers is required for target_layer_policy")
        return resolve_target_layer_ids(
            num_target_layers=num_target_layers,
            policy=target_layer_policy,
        )
    return [] if target_layer_ids is None else list(target_layer_ids)


def _default_rank(value: Optional[int], default: int, name: str) -> int:
    rank = default if value is None else value
    _require_positive_int(name, rank)
    return rank


def _default_alpha(value: Optional[float], default: float, name: str) -> float:
    alpha = default if value is None else value
    _require_positive_float(name, alpha)
    return float(alpha)


def _require_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_float(name: str, value: float) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive")
