"""Conservative policy for skipping an unused target-verify mask."""

from __future__ import annotations

from typing import Any


def _unwrap_full_attention_backend(attention_backend: Any) -> Any:
    backend = attention_backend
    for _ in range(4):
        # Resolve the backend that actually handles TARGET_VERIFY. A hybrid
        # prefill choice must not disable the proven FA3 verification fast path.
        if type(backend).__name__ == "HybridAttnBackend":
            mode = backend.model_runner.server_args.speculative_attention_mode
            backend = (
                backend.decode_backend if mode == "decode" else backend.prefill_backend
            )
            continue
        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    return backend


def resolve_draft_free_kv_verify_mask_policy(
    attention_backend: Any,
    *,
    topk: int,
    fixed_width_linear: bool,
    cuda_graph_enabled: bool,
    overlap_enabled: bool,
) -> tuple[str, bool]:
    """Return the resolved backend name and whether to build a custom mask.

    FA3 uses built-in causal attention for fixed-width, topk=1 verification in
    eager execution, graph capture and graph replay (including padded batches).
    Keep cuda_graph_enabled in the interface for existing policy callers; it
    does not change whether this attention path consumes the custom mask.
    """

    backend = _unwrap_full_attention_backend(attention_backend)
    backend_name = type(backend).__name__
    flash_attention_version = getattr(backend, "fa_impl_ver", None)
    built_in_causal_path_is_proven = (
        backend_name == "FlashAttentionBackend"
        and flash_attention_version == 3
        and int(topk) == 1
        and bool(fixed_width_linear)
        and not bool(overlap_enabled)
    )
    return backend_name, not built_in_causal_path_is_proven


__all__ = ["resolve_draft_free_kv_verify_mask_policy"]
