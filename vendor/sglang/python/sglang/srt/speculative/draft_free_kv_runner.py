from __future__ import annotations

import json
import logging
import inspect
import math
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from sglang.srt.speculative.draft_free_kv_utils import DraftFreeKVManifest

logger = logging.getLogger(__name__)


def _install_global16_sglang_rmsnorm(draft_model: nn.Module) -> int:
    from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

    class ShapePreservingSGLangRMSNorm(SGLangRMSNorm):
        """Restore higher-rank shape after SGLang's rank-2 RMSNorm kernels."""

        def forward(self, x, *args, **kwargs):
            output = super().forward(x, *args, **kwargs)
            if not isinstance(output, torch.Tensor) or output.shape == x.shape:
                return output
            if output.numel() != x.numel():
                raise RuntimeError(
                    "Global16 SGLang RMSNorm changed the hidden-state element count: "
                    f"input={tuple(x.shape)} output={tuple(output.shape)}"
                )
            return output.reshape_as(x)

    executable_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if executable_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [executable_bin, *[entry for entry in path_entries if entry]]
        )
    replaced = 0

    def visit(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            is_exact_hidden_norm = (
                child.__class__.__name__ == "RMSNorm"
                and child.__class__.__module__.startswith("specforge.")
                and hasattr(child, "eps")
                and isinstance(getattr(child, "weight", None), nn.Parameter)
                and int(child.weight.numel()) >= 1024
            )
            if not is_exact_hidden_norm:
                visit(child)
                continue
            weight = child.weight
            replacement = ShapePreservingSGLangRMSNorm(
                int(weight.numel()),
                eps=float(child.eps),
                cast_x_before_out_mul=True,
                weight_dtype=weight.dtype,
            ).to(device=weight.device)
            replacement.weight = weight
            replacement.train(child.training)
            setattr(parent, name, replacement)
            replaced += 1

    visit(draft_model)
    if replaced <= 0:
        raise RuntimeError(
            "Global16 serving did not find any exact hidden-state RMSNorm modules"
        )
    return replaced


def _restore_dfk_parallel_qwen_inv_freq(draft_model: nn.Module) -> int:
    """Restore config-derived RoPE buffers omitted from the checkpoint.

    ``inv_freq`` is registered as a non-persistent buffer by SpecForge, so it
    is not present in ``model.safetensors``.  Transformers' low-memory loading
    path may instantiate those buffers without running their value
    initialization.  Rebuild them from the checkpoint config before the
    serving-side shared-RoPE validation runs.
    """
    layers = getattr(draft_model, "parallel_qwen_layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise RuntimeError(
            "DraftFreeKV RoPE restoration requires parallel Qwen layers"
        )
    config = getattr(draft_model, "config", None)
    theta = float(getattr(config, "qwen_rope_theta", 0.0))
    if not math.isfinite(theta) or theta <= 0.0:
        raise RuntimeError(
            "DraftFreeKV RoPE restoration requires positive qwen_rope_theta"
        )

    restored = 0
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            inv_freq = getattr(attention, "inv_freq", None)
            if not isinstance(inv_freq, torch.Tensor):
                raise RuntimeError(
                    f"DraftFreeKV layer {layer_index} has no RoPE buffer"
                )
            head_dim = int(getattr(attention, "head_dim", inv_freq.numel() * 2))
            expected = 1.0 / (
                theta
                ** (
                    torch.arange(
                        0,
                        head_dim,
                        2,
                        dtype=torch.float32,
                        device=inv_freq.device,
                    )
                    / head_dim
                )
            )
            if expected.shape != inv_freq.shape:
                raise RuntimeError(
                    "DraftFreeKV layer "
                    f"{layer_index} RoPE shape mismatch: "
                    f"expected {tuple(expected.shape)}, got {tuple(inv_freq.shape)}"
                )
            inv_freq.copy_(expected.to(dtype=inv_freq.dtype))
            restored += 1
    return restored


def _verify_global16_shared_rope(draft_model: nn.Module) -> int:
    layers = getattr(draft_model, "parallel_qwen_layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != 5:
        raise RuntimeError(
            "Global16 shared Draft inputs require exactly five Qwen layers"
        )
    attentions = [getattr(layer, "self_attn", None) for layer in layers]
    if any(not isinstance(attention, nn.Module) for attention in attentions):
        raise RuntimeError("Global16 shared Draft inputs require five attentions")
    reference = getattr(attentions[0], "inv_freq", None)
    if not isinstance(reference, torch.Tensor):
        raise RuntimeError("Global16 shared Draft inputs require an RoPE buffer")
    for layer_index, attention in enumerate(attentions):
        inv_freq = getattr(attention, "inv_freq", None)
        if (
            not isinstance(inv_freq, torch.Tensor)
            or inv_freq.shape != reference.shape
            or inv_freq.dtype != reference.dtype
            or inv_freq.device != reference.device
            or not torch.equal(inv_freq, reference)
        ):
            raise RuntimeError(
                "Global16 Draft layer "
                f"{layer_index} has a different RoPE definition"
            )
        attention._dfk_global16_shared_rope_verified = True
    return len(attentions)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def sample_markov_draft_block(
    base_logits: torch.Tensor,
    *,
    current_token_ids: torch.Tensor,
    markov_head,
    sampling_info,
    candidate_token_ids: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Sample every draft position and retain the probabilities used later.

    The previous-token correction is a chain: the token sampled at one
    position must condition the next position.  Greedy requests keep the
    existing argmax behavior, while sampling requests use their own
    temperature.
    """

    if base_logits.ndim != 3:
        raise ValueError(
            "LongSpark sampling logits must be [batch, steps, vocabulary]"
        )
    batch_size = int(base_logits.shape[0])
    if current_token_ids.shape != (batch_size,):
        raise ValueError("LongSpark sampling needs one current token per request")
    if candidate_token_ids is not None:
        if (
            candidate_token_ids.ndim != 1
            or candidate_token_ids.numel() != base_logits.shape[-1]
        ):
            raise ValueError(
                "LongSpark candidate vocabulary does not match its draft logits"
            )
        candidate_token_ids = candidate_token_ids.to(
            device=base_logits.device,
            dtype=torch.long,
        )

    if sampling_info is None:
        greedy_mask = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=base_logits.device,
        )
        temperatures = torch.ones(
            batch_size,
            dtype=torch.float32,
            device=base_logits.device,
        )
        any_sampling = False
    else:
        greedy_mask = sampling_info.top_ks.view(-1).to(
            device=base_logits.device
        ) <= 1
        temperatures = (
            sampling_info.temperatures.view(-1)
            .to(device=base_logits.device, dtype=torch.float32)
            .clamp_min(1e-5)
        )
        if greedy_mask.shape != (batch_size,) or temperatures.shape != (
            batch_size,
        ):
            raise ValueError("LongSpark sampling metadata does not match the batch")
        any_sampling = not bool(sampling_info.is_all_greedy)

    previous_token_ids = current_token_ids.to(
        device=base_logits.device,
        dtype=torch.long,
    )
    sampled_tokens = []
    if sampling_info is None:
        any_greedy = False
    else:
        any_greedy_attr = getattr(sampling_info, "is_any_greedy", None)
        any_greedy = (
            bool(greedy_mask.any())
            if any_greedy_attr is None
            else bool(any_greedy_attr)
        )
    all_sampling = any_sampling and not any_greedy
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    capture_sampled_q = (
        all_sampling
        and not getattr(sampling_info, "need_top_k_sampling", False)
        and not getattr(sampling_info, "need_top_p_sampling", False)
        and not getattr(sampling_info, "need_min_p_sampling", False)
        and not getattr(sampling_info, "has_custom_logit_processor", False)
        and getattr(sampling_info, "acc_linear_penalties", None) is None
        and not (penalizer is not None and penalizer.is_required)
        and getattr(sampling_info, "vocab_mask", None) is None
        and getattr(sampling_info, "logit_bias", None) is None
    )
    sampled_token_probs = [] if capture_sampled_q else None
    fast_sampler = None
    sampling_workspace = None
    if (
        capture_sampled_q
        and candidate_token_ids is None
        and base_logits.is_cuda
        and base_logits.dtype in (torch.bfloat16, torch.float32)
        and base_logits.stride(-1) == 1
        and _env_flag("DFK_GLOBAL16_FAST_SAMPLING")
    ):
        from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens_with_q import (
            create_sample_step_workspace,
            sample_step_tokens_with_q,
        )

        fast_sampler = sample_step_tokens_with_q
        sampling_workspace = create_sample_step_workspace(
            batch=batch_size, vocab=base_logits.shape[-1], device=base_logits.device
        )
    for step_index in range(base_logits.shape[1]):
        step_logits = base_logits[:, step_index, :]
        if markov_head is not None:
            # ``base_logits`` is a fresh head result owned by this routine;
            # fold the Markov correction into it so the corrected-logit
            # carrier does not require seven row copies plus a final stack.
            step_logits.add_(
                markov_head.compute_bias(
                    previous_token_ids,
                    candidate_token_ids,
                ).to(dtype=step_logits.dtype)
            )
        if fast_sampler is not None:
            next_indices, sampled_q = fast_sampler(
                step_logits=step_logits,
                temperatures=temperatures,
                workspace=sampling_workspace,
            )
            sampled_token_probs.append(sampled_q)
        elif any_sampling:
            probabilities = torch.softmax(
                step_logits.float() / temperatures[:, None],
                dim=-1,
            )
            sampled_indices = torch.multinomial(
                probabilities,
                num_samples=1,
            ).squeeze(-1)
            if all_sampling:
                next_indices = sampled_indices
            else:
                argmax_indices = torch.argmax(step_logits, dim=-1)
                next_indices = torch.where(
                    greedy_mask,
                    argmax_indices,
                    sampled_indices,
                )
            if sampled_token_probs is not None:
                sampled_token_probs.append(
                    probabilities.gather(1, next_indices[:, None]).squeeze(1)
                )
        else:
            next_indices = torch.argmax(step_logits, dim=-1)
        if candidate_token_ids is None:
            next_token_ids = next_indices
        else:
            next_token_ids = candidate_token_ids.index_select(0, next_indices)
        sampled_tokens.append(next_token_ids)
        previous_token_ids = next_token_ids

    if not sampled_tokens:
        return (
            torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=base_logits.device,
            ),
            base_logits,
            temperatures,
            greedy_mask,
            None,
        )
    return (
        torch.stack(sampled_tokens, dim=1),
        base_logits,
        temperatures,
        greedy_mask,
        None
        if sampled_token_probs is None
        else torch.stack(sampled_token_probs, dim=1),
    )


def _supported_forward_kwargs(module: nn.Module) -> set[str] | None:
    signature = inspect.signature(module.forward)
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return None
    return set(parameters)


def _filter_supported_forward_kwargs(
    module: nn.Module,
    kwargs: dict,
) -> dict:
    supported_forward_kwargs = _supported_forward_kwargs(module)
    if supported_forward_kwargs is None:
        return kwargs
    return {name: value for name, value in kwargs.items() if name in supported_forward_kwargs}


def _infer_specforge_conditioning_requirements(
    draft_model,
) -> tuple[bool, bool, bool, bool]:
    forward_parameters = inspect.signature(draft_model.forward).parameters
    accepts_current_hidden = "current_hidden_states" in forward_parameters
    accepts_current_token = "current_token_ids" in forward_parameters
    accepts_current_position = "current_position_ids" in forward_parameters
    condition_source = getattr(
        draft_model.config,
        "current_condition_source",
        "target_current_hidden_legacy",
    )
    config_requires_current_hidden = (
        condition_source
        in {
            "target_current_hidden_legacy",
            "prefix_last_hidden",
            "prefix_last_hidden_plus_current_token",
            "target_hidden",
        }
        or bool(getattr(draft_model.config, "use_current_hidden_conditioning", False))
    )
    requires_current_hidden = bool(
        config_requires_current_hidden
        or (accepts_current_hidden and not accepts_current_token)
    )
    requires_hidden = bool(
        getattr(draft_model.config, "use_hidden_conditioning", False)
        or requires_current_hidden
    )
    requires_current_token = bool(
        condition_source == "prefix_last_hidden_plus_current_token"
        or accepts_current_token
    )
    requires_current_position = bool(
        getattr(draft_model.config, "draft_mode", "markov")
        == "parallel_shifted_qwen"
        and accepts_current_position
    )
    return (
        requires_current_hidden,
        requires_hidden,
        requires_current_token,
        requires_current_position,
    )


class KVReader(ABC):
    @abstractmethod
    def read(self, batch, *, selected_layers: tuple[int, ...], window_size: int):
        pass


class HeadRunner(ABC):
    @abstractmethod
    def propose(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        block_size: int,
        conditioning_hidden_states: torch.Tensor | None = None,
        current_token_ids: torch.Tensor | None = None,
        current_position_ids: torch.Tensor | None = None,
    ):
        pass


class TorchKVReader(KVReader):
    def __init__(self, target_worker):
        self.target_worker = target_worker
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()

    def _kv_buffer(self, layer_id: int, *, is_key: bool):
        return (
            self.token_to_kv_pool.get_key_buffer(layer_id)
            if is_key
            else self.token_to_kv_pool.get_value_buffer(layer_id)
        )

    def read(self, batch, *, selected_layers: tuple[int, ...], window_size: int):
        if batch.forward_mode.is_idle():
            raise RuntimeError("DRAFT_FREE_KV cannot read KV for an idle batch")
        bs = batch.batch_size()
        seq_lens = batch.seq_lens
        max_window = int(torch.clamp(seq_lens, max=window_size).max().item())
        if max_window <= 0:
            raise RuntimeError("DRAFT_FREE_KV requires committed target KV")

        loc_rows = []
        mask_rows = []
        for i in range(bs):
            seq_len = int(seq_lens[i].item())
            take = min(seq_len, int(window_size))
            start = seq_len - take
            req_pool_idx = batch.req_pool_indices[i]
            locs = self.req_to_token_pool.req_to_token[req_pool_idx, start:seq_len].to(torch.long)
            if take < max_window:
                pad = locs.new_full((max_window - take,), int(locs[-1].item()))
                locs = torch.cat([pad, locs], dim=0)
                mask = torch.cat(
                    [
                        torch.zeros((max_window - take,), dtype=torch.bool, device=locs.device),
                        torch.ones((take,), dtype=torch.bool, device=locs.device),
                    ],
                    dim=0,
                )
            else:
                mask = torch.ones((max_window,), dtype=torch.bool, device=locs.device)
            loc_rows.append(locs)
            mask_rows.append(mask)

        loc_matrix = torch.stack(loc_rows, dim=0)
        attention_mask = torch.stack(mask_rows, dim=0)
        keys_per_layer = []
        values_per_layer = []
        for layer_id in selected_layers:
            keys_per_layer.append(self._kv_buffer(int(layer_id), is_key=True)[loc_matrix])
            values_per_layer.append(self._kv_buffer(int(layer_id), is_key=False)[loc_matrix])

        keys = torch.stack(keys_per_layer, dim=1)
        values = torch.stack(values_per_layer, dim=1)
        if keys.ndim != 5:
            raise RuntimeError(f"Unexpected target KV buffer shape after gather: {tuple(keys.shape)}")
        keys = keys.permute(0, 1, 3, 2, 4).contiguous()
        values = values.permute(0, 1, 3, 2, 4).contiguous()
        return keys, values, attention_mask


class TorchCheckpointHeadRunner(HeadRunner):
    def __init__(
        self,
        manifest: DraftFreeKVManifest,
        target_worker,
        *,
        window_size: int,
    ):
        self.manifest = manifest
        self.target_worker = target_worker
        self.device = target_worker.device
        self.window_size = int(window_size)
        self.head = self._load_head()
        self.compiled_head = self._maybe_compile_head()

    def _maybe_compile_head(self):
        if not _env_flag("DFK_COMPILE_HEAD"):
            return None
        mode = os.environ.get("DFK_COMPILE_HEAD_MODE", "reduce-overhead")
        dynamic = _env_flag("DFK_COMPILE_HEAD_DYNAMIC", default=True)
        fullgraph = _env_flag("DFK_COMPILE_HEAD_FULLGRAPH")
        skip_dynamic_cudagraphs = _env_flag(
            "DFK_COMPILE_HEAD_SKIP_DYNAMIC_CUDAGRAPHS",
            default=True,
        )
        if skip_dynamic_cudagraphs:
            try:
                import torch._inductor.config as inductor_config

                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
            except Exception:
                logger.exception(
                    "Failed to set DRAFT_FREE_KV torch.compile cudagraph guards."
                )
        try:
            compiled = torch.compile(
                self.head,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )
        except Exception:
            logger.exception(
                "Failed to initialize torch.compile for DRAFT_FREE_KV head; using eager head."
            )
            return None
        logger.info(
            "Enabled torch.compile for DRAFT_FREE_KV head. mode=%s dynamic=%s fullgraph=%s skip_dynamic_cudagraphs=%s",
            mode,
            dynamic,
            fullgraph,
            skip_dynamic_cudagraphs,
        )
        return compiled

    def _load_head(self):
        model = getattr(self.target_worker.model_runner, "model", None)
        if model is None:
            raise RuntimeError("target worker does not expose model_runner.model")

        if self._is_specforge_checkpoint():
            return self._load_specforge_head(model)
        return self._load_legacy_head(model)

    def _is_specforge_checkpoint(self) -> bool:
        if self.manifest.config_path is None:
            return False
        try:
            with self.manifest.config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        except OSError:
            return False
        return config.get("model_type") == "draftfreekv"

    def _target_lm_head(self, model):
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("DRAFT_FREE_KV target model does not expose lm_head")
        return lm_head

    def _target_embed_tokens(self, model):
        for path in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            module = model
            for name in path.split("."):
                module = getattr(module, name, None)
                if module is None:
                    break
            if module is not None:
                return module
        if hasattr(model, "get_input_embeddings"):
            embed_tokens = model.get_input_embeddings()
            if embed_tokens is not None:
                return embed_tokens
        raise RuntimeError("DRAFT_FREE_KV target model does not expose input embeddings")

    def _target_final_norm(self, model):
        for path in ("model.norm", "norm", "transformer.ln_f"):
            module = model
            for name in path.split("."):
                module = getattr(module, name, None)
                if module is None:
                    break
            if module is not None:
                return module
        return None

    def _load_specforge_head(self, model):
        try:
            from specforge.modeling.draft import DraftFreeKVModel
        except Exception as exc:
            raise RuntimeError(
                "DRAFT_FREE_KV SpecForge checkpoints require specforge on PYTHONPATH."
            ) from exc
        fixed_cost_audit = _env_flag("DFK_FIXED_COST_AUDIT")
        loaded_specforge_path = Path(inspect.getfile(DraftFreeKVModel)).resolve()

        draft_model = DraftFreeKVModel.from_pretrained(
            str(self.manifest.path),
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)
        if fixed_cost_audit:
            boot_id = os.environ.get("LONGSPARK_BOOT_ID", "")
            if len(boot_id) != 32 or any(
                character not in "0123456789abcdef" for character in boot_id
            ):
                raise RuntimeError(
                    "fixed-cost Draft loading requires the canonical "
                    "LONGSPARK_BOOT_ID"
                )
            query_mode = str(getattr(draft_model.config, "global_query_mode", ""))
            query_correlation = float(
                getattr(
                    draft_model.config, "global_query_equicorrelation", float("nan")
                )
            )
            if query_mode != "fixed_equicorrelated" or query_correlation != 0.4:
                raise RuntimeError(
                    "fixed-cost serving requires a validated fixed-equicorrelated "
                    f"Global query checkpoint, got mode={query_mode!r}, "
                    f"correlation={query_correlation!r}"
                )
            try:
                manifest_data = json.loads(
                    (self.manifest.path / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "fixed-cost serving cannot read its Draft manifest"
                ) from exc
            state_dict_sha256 = manifest_data.get("state_dict_sha256")
            if (
                not isinstance(state_dict_sha256, str)
                or len(state_dict_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in state_dict_sha256
                )
            ):
                raise RuntimeError(
                    "fixed-cost Draft manifest must bind its weights with SHA-256"
                )
            # This marker is deliberately emitted only after from_pretrained has
            # validated the frozen query tensor's normalized Gram matrix.
            logger.info(
                "DFK_SOURCE_ATTESTATION boot_id=%s state_dict_sha256=%s "
                "specforge_draftfreekv=%s query_geometry_validated=1",
                boot_id,
                state_dict_sha256,
                loaded_specforge_path,
            )
        if getattr(draft_model.config, "global16_raw256_enabled", False):
            restored_rope_layers = _restore_dfk_parallel_qwen_inv_freq(
                draft_model
            )
            shared_rope_layers = _verify_global16_shared_rope(draft_model)
            replaced_rmsnorms = _install_global16_sglang_rmsnorm(draft_model)
            logger.info(
                "Global16 serving restored %d RoPE buffers, verified %d "
                "shared-RoPE Draft layers, and installed %d exact "
                "hidden-state SGLang RMSNorm modules",
                restored_rope_layers,
                shared_rope_layers,
                replaced_rmsnorms,
            )
        native_global16_executor = None
        if (
            getattr(draft_model.config, "global16_raw256_enabled", False)
            and _env_flag("DFK_GLOBAL16_NATIVE_EXECUTOR")
        ):
            from sglang.srt.speculative.global16_native_draft_executor import (
                build_global16_native_draft_executor,
            )

            native_global16_executor = build_global16_native_draft_executor(
                draft_model
            )
            logger.info(
                "Enabled native SGLang Global16 Draft executor for %d Qwen layers",
                len(native_global16_executor.decoder_layers),
            )
        # Legacy heads use a gathered K/V window.  Global16 + Raw256 owns its
        # fixed direct Raw256 page-table view and must never inherit that
        # override as a Python K/V gather limit.
        if not getattr(draft_model.config, "global16_raw256_enabled", False):
            draft_model.config.target_kv_window_size = self.window_size
        target_embed_tokens = self._target_embed_tokens(model)
        target_lm_head = self._target_lm_head(model)
        target_final_norm = self._target_final_norm(model)

        class SpecForgeDraftFreeKVHead(nn.Module):
            def __init__(
                self,
                draft_model,
                target_embed_tokens,
                target_lm_head,
                target_final_norm,
                native_global16_executor,
            ):
                super().__init__()
                self.draft_model = draft_model
                self.target_embed_tokens = target_embed_tokens
                self.target_lm_head = target_lm_head
                self.target_final_norm = target_final_norm
                self.native_global16_executor = native_global16_executor
                (
                    self.requires_current_hidden_conditioning,
                    self.requires_hidden_conditioning,
                    self.requires_current_token_conditioning,
                    self.requires_current_position_conditioning,
                ) = _infer_specforge_conditioning_requirements(
                    draft_model,
                )
                if getattr(draft_model.config, "prediction_head_type", "target_plus_lora") == "draft_vocab":
                    draft_ids = torch.arange(
                        draft_model.d2t.numel(),
                        dtype=draft_model.d2t.dtype,
                        device=draft_model.d2t.device,
                    )
                    self.draft_to_target = draft_model.d2t + draft_ids
                else:
                    self.draft_to_target = None

            def forward(
                self,
                keys=None,
                values=None,
                attention_mask=None,
                conditioning_hidden_states=None,
                current_token_ids=None,
                current_position_ids=None,
                global16_paged_context=None,
                return_proposal_ids=True,
                apply_markov_correction=True,
            ):
                if self.native_global16_executor is not None:
                    if global16_paged_context is None:
                        raise RuntimeError(
                            "the native Global16 Draft executor is serving-only "
                            "and requires a proposal-only paged context"
                        )
                    return self.native_global16_executor(
                        conditioning_hidden_states=conditioning_hidden_states,
                        current_token_ids=current_token_ids,
                        current_position_ids=current_position_ids,
                        target_final_norm=self.target_final_norm,
                        target_embed_tokens=self.target_embed_tokens,
                        target_lm_head=self.target_lm_head,
                        global16_paged_context=global16_paged_context,
                        return_proposal_ids=return_proposal_ids,
                        apply_markov_correction=apply_markov_correction,
                    )
                kwargs = _filter_supported_forward_kwargs(
                    self.draft_model,
                    {
                        "keys": keys,
                        "values": values,
                        "attention_mask": attention_mask,
                        "current_token_ids": current_token_ids,
                        "current_position_ids": current_position_ids,
                        "current_hidden_states": (
                            conditioning_hidden_states
                            if self.requires_current_hidden_conditioning
                            else None
                        ),
                        "target_final_norm": self.target_final_norm,
                        "target_embed_tokens": self.target_embed_tokens,
                        # load_manifest gates target-lm-head-projection to SpecForge
                        # DraftFreeKV configs; projection is delegated to this model.
                        "target_lm_head": self.target_lm_head,
                        "global16_paged_context": global16_paged_context,
                    },
                )
                return self.draft_model(**kwargs)

        head = SpecForgeDraftFreeKVHead(
            draft_model,
            target_embed_tokens,
            target_lm_head,
            target_final_norm,
            native_global16_executor,
        ).to(device=self.device)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad = False
        logger.info("Loaded SpecForge DRAFT_FREE_KV head from %s", self.manifest.path)
        return head

    def _load_legacy_head(self, model):
        try:
            from draft_free_sd.head_factory import build_head_from_checkpoint
        except Exception as exc:
            raise RuntimeError(
                "DRAFT_FREE_KV legacy checkpoints require draft_free_sd on PYTHONPATH."
            ) from exc

        checkpoint = torch.load(self.manifest.state_dict_path, map_location="cpu")

        class ModelWrapper:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def get_output_embeddings(self):
                return getattr(self._wrapped, "lm_head", None)

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        class Adapter:
            pass

        adapter = Adapter()
        adapter.model = ModelWrapper(model)
        checkpoint.setdefault("head_type", self.manifest.head_type)
        checkpoint.setdefault("block_size", self.manifest.block_size)
        checkpoint.setdefault("config", {
            "window_size": self.manifest.window_size,
            "selected_layers": list(self.manifest.selected_layers),
            "selected_layer_policy": self.manifest.selected_layer_policy,
        })
        head = build_head_from_checkpoint(checkpoint, device=self.device, adapter=adapter)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad = False
        logger.info("Loaded legacy DRAFT_FREE_KV head from %s", self.manifest.path)
        return head

    @torch.inference_mode()
    def sampling_base_logits(
        self,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        conditioning_hidden_states: torch.Tensor | None = None,
        current_token_ids: torch.Tensor | None = None,
        current_position_ids: torch.Tensor | None = None,
        global16_paged_context=None,
    ):
        """Run only the deterministic trunk and vocabulary projection.

        This boundary may be captured without consuming sampling RNG or
        executing the greedy Markov head.
        """

        native_executor = getattr(
            self.head,
            "native_global16_executor",
            None,
        )
        if native_executor is None or global16_paged_context is None:
            raise RuntimeError(
                "LongSpark temperature sampling currently requires the native "
                "Global16 serving path"
            )
        if current_token_ids is None or current_position_ids is None:
            raise RuntimeError(
                "LongSpark temperature sampling requires current tokens and positions"
            )
        if (
            getattr(self.head, "requires_hidden_conditioning", False)
            and conditioning_hidden_states is None
        ):
            raise RuntimeError(
                "LongSpark temperature sampling requires cached target hidden states"
            )

        return self.head(
            keys,
            values,
            attention_mask,
            conditioning_hidden_states=conditioning_hidden_states,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            global16_paged_context=global16_paged_context,
            return_proposal_ids=False,
            apply_markov_correction=False,
        )

    @torch.inference_mode()
    def propose_for_sampling(
        self,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        block_size: int,
        sampling_info,
        conditioning_hidden_states: torch.Tensor | None = None,
        current_token_ids: torch.Tensor | None = None,
        current_position_ids: torch.Tensor | None = None,
        global16_paged_context=None,
        base_logits: torch.Tensor | None = None,
    ):
        """Return sampled ids and their actual q, optionally from graph logits.

        The sampler mutates the logits in place. Graph output is borrowed for
        this round through acceptance; the next replay overwrites every row
        with fresh base logits before applying any Markov correction.
        """

        if base_logits is None:
            base_logits = self.sampling_base_logits(
                keys,
                values,
                attention_mask,
                conditioning_hidden_states=conditioning_hidden_states,
                current_token_ids=current_token_ids,
                current_position_ids=current_position_ids,
                global16_paged_context=global16_paged_context,
            )
        native_executor = self.head.native_global16_executor
        proposal_slots = int(block_size) - 1
        if (
            base_logits.ndim != 3
            or base_logits.shape[0] != current_token_ids.shape[0]
            or base_logits.shape[1] < proposal_slots
        ):
            raise RuntimeError(
                "native Global16 sampling must return [batch, seven, vocabulary] logits"
            )
        base_logits = base_logits[:, :proposal_slots, :]
        candidate_token_ids = native_executor.proposal_candidate_token_ids
        return (
            *sample_markov_draft_block(
                base_logits,
                current_token_ids=current_token_ids,
                markov_head=getattr(
                    self.head.draft_model,
                    "parallel_markov_head",
                    None,
                ),
                sampling_info=sampling_info,
                candidate_token_ids=candidate_token_ids,
            ),
            candidate_token_ids,
        )

    @torch.inference_mode()
    def propose(
        self,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        *,
        block_size: int,
        conditioning_hidden_states: torch.Tensor | None = None,
        current_token_ids: torch.Tensor | None = None,
        current_position_ids: torch.Tensor | None = None,
        global16_paged_context=None,
    ):
        requires_hidden = getattr(self.head, "requires_hidden_conditioning", False)
        requires_current = getattr(self.head, "requires_current_token_conditioning", False)
        requires_position = getattr(
            self.head,
            "requires_current_position_conditioning",
            False,
        )
        if requires_hidden and conditioning_hidden_states is None:
            raise RuntimeError("DRAFT_FREE_KV hidden-conditioned head requires cached target hidden states")
        if requires_current and current_token_ids is None:
            raise RuntimeError("DRAFT_FREE_KV current-token-conditioned head requires current_token_ids")
        if requires_position and current_position_ids is None:
            raise RuntimeError(
                "DRAFT_FREE_KV Qwen head requires current_position_ids"
            )
        # The paged context carries cache pointers and variable page metadata;
        # it deliberately bypasses the legacy compiled gathered-K/V graph.
        head = self.head if global16_paged_context is not None else (self.compiled_head or self.head)
        try:
            if requires_hidden or requires_current or requires_position:
                logits = head(
                    keys,
                    values,
                    attention_mask,
                    conditioning_hidden_states=conditioning_hidden_states,
                    current_token_ids=current_token_ids,
                    current_position_ids=current_position_ids,
                    global16_paged_context=global16_paged_context,
                )
            else:
                logits = head(
                    keys,
                    values,
                    attention_mask,
                    global16_paged_context=global16_paged_context,
                )
        except Exception:
            if self.compiled_head is None:
                raise
            logger.exception(
                "DRAFT_FREE_KV compiled head failed at runtime; falling back to eager head."
            )
            self.compiled_head = None
            if requires_hidden or requires_current or requires_position:
                logits = self.head(
                    keys,
                    values,
                    attention_mask,
                    conditioning_hidden_states=conditioning_hidden_states,
                    current_token_ids=current_token_ids,
                    current_position_ids=current_position_ids,
                    global16_paged_context=global16_paged_context,
                )
            else:
                logits = self.head(
                    keys,
                    values,
                    attention_mask,
                    global16_paged_context=global16_paged_context,
                )
        # The existing native Global16 serving executor already projects the
        # full target vocabulary and returns greedy target token ids [bs, 7].
        # Keep that proposal carrier intact; the legacy head path below returns
        # logits [bs, block, vocab] and still owns its argmax/mapping logic.
        if getattr(self.head, "native_global16_executor", None) is not None:
            if logits.ndim != 2:
                raise RuntimeError(
                    "native Global16 Draft executor must return proposal ids [bs, 7]"
                )
            return logits.to(torch.int64)

        draft_ids = torch.argmax(logits[:, :block_size, :], dim=-1).to(torch.int64)
        draft_to_target = getattr(self.head, "draft_to_target", None)
        if draft_to_target is None:
            return draft_ids
        flat = draft_ids.reshape(-1).to(device=draft_to_target.device)
        target_ids = draft_to_target.index_select(0, flat).view_as(draft_ids)
        return target_ids.to(device=logits.device, dtype=torch.int64)
