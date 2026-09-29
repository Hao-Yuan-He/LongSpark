"""Native SGLang execution path for the Global16 LongSpark Draft decoder.

The checkpoint's five Qwen layers are stored as ordinary PyTorch ``nn.Linear``
modules because that is the training representation.  Serving does not need to
keep that representation once the checkpoint is loaded: this adapter maps a
whole decoder layer to SGLang's TP-aware linear, residual-normalization, and
activation primitives.  The Global16/Raw256 attention read itself remains the
existing direct-cache implementation, so this module changes the Draft
execution boundary without rebuilding a Target-KV bank.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.speculative.global16_markov_precision import (
    Float8LinearProjection,
    install_bfloat16_markov_projection,
    install_float8_input_feedforwards,
    install_float8_vocab_projection,
    install_native_vocab_gemm,
)
from sglang.srt.speculative.global16_markov_fused import (
    install_fused_markov_sampler,
)

logger = logging.getLogger(__name__)


class _ShapePreservingSGLangRMSNorm(SGLangRMSNorm):
    """Run the rank-2 CUDA kernel without changing Draft block dimensions."""

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_residual_addition: torch.Tensor | None = None,
    ):
        if residual is not None or x.ndim <= 2:
            return super().forward_cuda(x, residual, post_residual_addition)
        original_shape = x.shape
        output = super().forward_cuda(
            x.reshape(-1, original_shape[-1]),
            None,
            post_residual_addition,
        )
        return output.reshape(original_shape)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int = 0) -> int:
    return int(os.getenv(name, str(default)).strip())


def _fused_qk_norm_rope_op():
    from sgl_kernel import fused_qk_norm_rope

    return fused_qk_norm_rope


def _copy_weight(destination: torch.Tensor, source: torch.Tensor, *, name: str) -> None:
    if destination.shape != source.shape:
        raise ValueError(
            f"Global16 native executor {name} shape mismatch: "
            f"expected {tuple(destination.shape)}, got {tuple(source.shape)}"
        )
    with torch.no_grad():
        destination.copy_(source.detach())


def _require_weight(module: nn.Module, *, name: str) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(
            f"Global16 native executor requires a rank-2 weight for {name}"
        )
    if getattr(module, "bias", None) is not None:
        raise ValueError(
            f"Global16 native executor does not support a bias for {name}"
        )
    return weight


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Global16NativeQwenAttention(nn.Module):
    """SGLang-linear version of one Global16/Raw256 Draft attention layer."""

    def __init__(
        self,
        source_attention: nn.Module,
        *,
        use_sglang_qk_rmsnorm: bool = False,
        use_fused_qk_norm_rope: bool = False,
        use_float8_qkv: bool = False,
        use_float8_output_projection: bool = False,
        use_fused_float8_quantization: bool = False,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        q_weight = _require_weight(source_attention.q_proj, name="q_proj")
        k_weight = _require_weight(source_attention.k_proj, name="k_proj")
        v_weight = _require_weight(source_attention.v_proj, name="v_proj")
        o_weight = _require_weight(source_attention.o_proj, name="o_proj")

        self.num_attention_heads = int(source_attention.num_attention_heads)
        self.num_key_value_heads = int(source_attention.num_key_value_heads)
        self.head_dim = int(source_attention.head_dim)
        self.num_key_value_groups = int(source_attention.num_key_value_groups)
        self.scaling = float(source_attention.scaling)
        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        if q_weight.shape[0] != self.q_size or k_weight.shape[0] != self.kv_size:
            raise ValueError("Global16 native executor Q/K projection shapes are invalid")
        if v_weight.shape[0] != self.kv_size:
            raise ValueError("Global16 native executor V projection shape is invalid")

        dtype = q_weight.dtype
        device = q_weight.device
        self.qkv_proj = QKVParallelLinear(
            hidden_size=int(q_weight.shape[1]),
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            params_dtype=dtype,
            tp_rank=0,
            tp_size=1,
        ).to(device=device)
        self.o_proj = RowParallelLinear(
            input_size=self.q_size,
            output_size=int(o_weight.shape[0]),
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            tp_rank=0,
            tp_size=1,
        ).to(device=device)
        _copy_weight(
            self.qkv_proj.weight,
            torch.cat((q_weight, k_weight, v_weight), dim=0),
            name="qkv_proj",
        )
        _copy_weight(self.o_proj.weight, o_weight, name="o_proj")
        if use_float8_qkv and not use_fused_float8_quantization:
            raise RuntimeError(
                "FP8 Q/K/V projection requires fused FP8 input quantization"
            )
        self.float8_qkv = (
            Float8LinearProjection(
                self.qkv_proj.weight,
                use_fused_quantization=use_fused_float8_quantization,
            )
            if use_float8_qkv
            else None
        )
        if use_float8_output_projection and not use_fused_float8_quantization:
            raise RuntimeError(
                "FP8 output projection requires fused FP8 input quantization"
            )
        self.float8_output_projection = (
            Float8LinearProjection(
                self.o_proj.weight,
                use_fused_quantization=use_fused_float8_quantization,
            )
            if use_float8_output_projection
            else None
        )

        if use_sglang_qk_rmsnorm:
            self.q_norm = self._convert_qk_norm(source_attention.q_norm, name="q_norm")
            self.k_norm = self._convert_qk_norm(source_attention.k_norm, name="k_norm")
        else:
            self.q_norm = source_attention.q_norm
            self.k_norm = source_attention.k_norm
        self.fused_qk_norm_rope_enabled = bool(use_fused_qk_norm_rope)
        self.rope_theta = float(rope_theta)
        q_eps = getattr(source_attention.q_norm, "eps", None)
        k_eps = getattr(source_attention.k_norm, "eps", None)
        if self.fused_qk_norm_rope_enabled:
            if self.head_dim != 128 or q_eps is None or q_eps != k_eps:
                raise ValueError(
                    "Global16 fused Q/K norm-plus-RoPE requires matching "
                    "128-wide RMSNorm modules"
                )
            self.qk_norm_eps = float(q_eps)
        else:
            self.qk_norm_eps = 0.0
        inv_freq = getattr(source_attention, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor):
            raise ValueError("Global16 native executor attention has no RoPE frequencies")
        self.register_buffer("inv_freq", inv_freq.detach(), persistent=False)
        self._dfk_global16_shared_rope_verified = True

    @staticmethod
    def _convert_qk_norm(source_norm: nn.Module, *, name: str) -> nn.Module:
        weight = getattr(source_norm, "weight", None)
        eps = getattr(source_norm, "eps", None)
        if not isinstance(weight, nn.Parameter) or weight.ndim != 1:
            raise ValueError(f"Global16 native executor {name} has no vector weight")
        if weight.numel() != 128 or eps is None:
            raise ValueError(
                f"Global16 native executor {name} does not match the 128-wide contract"
            )
        replacement = _ShapePreservingSGLangRMSNorm(
            int(weight.numel()),
            eps=float(eps),
            cast_x_before_out_mul=True,
            weight_dtype=weight.dtype,
        ).to(device=weight.device)
        replacement.weight = weight
        replacement.train(source_norm.training)
        return replacement

    def _position_embeddings(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = (
            position_ids.to(device=self.inv_freq.device, dtype=torch.float32)[..., None]
            * self.inv_freq[None, None, :]
        )
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)

    @staticmethod
    def _apply_rotary(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        return hidden_states * cos + _rotate_half(hidden_states) * sin

    def forward_global16_raw256(
        self,
        hidden_states: torch.Tensor,
        *,
        input_norm: Optional[nn.Module] = None,
        position_ids: torch.Tensor,
        global16_context,
        layer_index: int,
    ) -> torch.Tensor:
        batch_size, block_size, _ = hidden_states.shape
        if self.float8_qkv is None:
            if input_norm is not None:
                raise RuntimeError(
                    "BF16 Q/K/V projection received an unexpected input norm"
                )
            qkv, _ = self.qkv_proj(hidden_states)
        else:
            norm_weight = getattr(input_norm, "weight", None)
            variance_epsilon = getattr(input_norm, "variance_epsilon", None)
            if not isinstance(norm_weight, torch.Tensor) or variance_epsilon is None:
                raise RuntimeError(
                    "fused Q/K/V RMSNorm requires a weight-backed RMSNorm"
                )
            qkv = self.float8_qkv.project_rmsnorm(
                hidden_states,
                norm_weight=norm_weight,
                variance_epsilon=float(variance_epsilon),
            )
        use_fused_qk_norm_rope = (
            self.fused_qk_norm_rope_enabled
            and qkv.is_cuda
            and qkv.dtype == torch.bfloat16
        )
        if use_fused_qk_norm_rope:
            flat_positions = position_ids.reshape(-1).to(
                device=qkv.device,
                dtype=torch.int32,
            ).contiguous()
            _fused_qk_norm_rope_op()(
                qkv.view(-1, qkv.shape[-1]),
                self.num_attention_heads,
                self.num_key_value_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.qk_norm_eps,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rope_theta,
                True,
                flat_positions,
                1.0,
                0.0,
                0.0,
                1.0,
            )
        query, local_key, local_value = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query = query.view(
            batch_size,
            block_size,
            self.num_attention_heads,
            self.head_dim,
        )
        local_key = local_key.view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        if not use_fused_qk_norm_rope:
            query = self.q_norm(query)
            local_key = self.k_norm(local_key)
        query = query.transpose(1, 2)
        local_key = local_key.transpose(1, 2)
        local_value = local_value.view(
            batch_size,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        prepare_shared_inputs = getattr(
            global16_context,
            "prepare_draft_layer_inputs",
            None,
        )
        if callable(prepare_shared_inputs):
            cos, sin, local_attention_mask = prepare_shared_inputs(
                attention=self,
                position_ids=position_ids,
                dtype=query.dtype,
                batch_size=batch_size,
                block_size=block_size,
            )
        else:
            cos, sin = self._position_embeddings(
                position_ids,
                dtype=query.dtype,
            )
            local_attention_mask = torch.ones(
                (batch_size, block_size, block_size),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if not use_fused_qk_norm_rope:
            query = self._apply_rotary(query, cos, sin)
            local_key = self._apply_rotary(local_key, cos, sin)
        attended = global16_context.attend(
            layer_index=layer_index,
            query=query,
            local_keys=local_key,
            local_values=local_value,
            local_attention_mask=local_attention_mask,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            block_size,
            self.q_size,
        )
        if self.float8_output_projection is None:
            output, _ = self.o_proj(attended)
        else:
            output = self.float8_output_projection(attended)
        return output


class Global16NativeQwenDecoderLayer(nn.Module):
    """A full Draft layer using SGLang GEMMs and selectable SwiGLU."""

    _PROFILE_MARKS = (
        "start",
        "input_norm",
        "attention",
        "attention_residual",
        "gate_up",
        "activation",
        "down",
        "end",
    )

    def __init__(
        self,
        source_layer: nn.Module,
        *,
        use_fused_silu: bool = False,
        use_sglang_qk_rmsnorm: bool = False,
        use_fused_qk_norm_rope: bool = False,
        use_float8_qkv: bool = False,
        use_float8_output_projection: bool = False,
        use_float8_gate_up: bool = False,
        use_float8_down_fused_activation: bool = False,
        use_fused_float8_quantization: bool = False,
        use_fused_rmsnorm_float8_quantization: bool = False,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.self_attn = Global16NativeQwenAttention(
            source_layer.self_attn,
            use_sglang_qk_rmsnorm=use_sglang_qk_rmsnorm,
            use_fused_qk_norm_rope=use_fused_qk_norm_rope,
            use_float8_qkv=use_float8_qkv,
            use_float8_output_projection=use_float8_output_projection,
            use_fused_float8_quantization=use_fused_float8_quantization,
            rope_theta=rope_theta,
        )
        gate_weight = _require_weight(source_layer.mlp.gate_proj, name="gate_proj")
        up_weight = _require_weight(source_layer.mlp.up_proj, name="up_proj")
        down_weight = _require_weight(source_layer.mlp.down_proj, name="down_proj")
        if gate_weight.shape != up_weight.shape:
            raise ValueError("Global16 native executor gate/up projection shapes differ")
        intermediate_size, hidden_size = gate_weight.shape
        if down_weight.shape != (hidden_size, intermediate_size):
            raise ValueError("Global16 native executor down projection shape is invalid")

        dtype = gate_weight.dtype
        device = gate_weight.device
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            params_dtype=dtype,
            tp_rank=0,
            tp_size=1,
        ).to(device=device)
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=dtype,
            tp_rank=0,
            tp_size=1,
        ).to(device=device)
        _copy_weight(
            self.gate_up_proj.weight,
            torch.cat((gate_weight, up_weight), dim=0),
            name="gate_up_proj",
        )
        _copy_weight(self.down_proj.weight, down_weight, name="down_proj")
        self.float8_gate_up = (
            Float8LinearProjection(
                self.gate_up_proj.weight,
                use_fused_quantization=use_fused_float8_quantization,
            )
            if use_float8_gate_up
            else None
        )
        if use_float8_down_fused_activation and not use_fused_float8_quantization:
            raise RuntimeError(
                "fused activation plus FP8 down projection requires fused "
                "FP8 input quantization"
            )
        self.float8_down_fused_activation = (
            Float8LinearProjection(
                self.down_proj.weight,
                use_fused_quantization=use_fused_float8_quantization,
            )
            if use_float8_down_fused_activation
            else None
        )
        self.fused_rmsnorm_float8_quantization = bool(
            use_fused_rmsnorm_float8_quantization
        )
        if self.fused_rmsnorm_float8_quantization and self.float8_gate_up is None:
            raise RuntimeError(
                "fused RMSNorm plus FP8 quantization requires FP8 gate/up"
            )
        # These full-hidden norms are installed as exact SGLang RMSNorms by
        # DraftFreeKV's serving loader before this executor is constructed.
        self.input_layernorm = source_layer.input_layernorm
        self.post_attention_layernorm = source_layer.post_attention_layernorm
        self.fused_residual_norm = _env_flag("DFK_GLOBAL16_FUSED_RESIDUAL_NORM")
        self.fused_silu = SiluAndMul() if use_fused_silu else None
        if self.float8_down_fused_activation is not None and self.fused_silu is not None:
            raise RuntimeError(
                "fused activation plus FP8 down projection is incompatible "
                "with the standalone fused activation"
            )
        self._profile_active = False
        self._profile_events: dict[str, torch.cuda.Event] | None = None

    def initialize_profile_events(self) -> None:
        device = self.gate_up_proj.weight.device
        with torch.cuda.device(device):
            self._profile_events = {
                name: torch.cuda.Event(enable_timing=True)
                for name in self._PROFILE_MARKS
            }
            for event in self._profile_events.values():
                event.record()

    def set_profile_active(self, active: bool) -> None:
        self._profile_active = bool(active)

    def profile_timings(self) -> dict[str, float]:
        if self._profile_events is None:
            raise RuntimeError("Global16 native layer profiling is not initialized")
        events = self._profile_events
        return {
            "input_norm_ms": events["start"].elapsed_time(events["input_norm"]),
            "attention_ms": events["input_norm"].elapsed_time(events["attention"]),
            "attention_residual_ms": events["attention"].elapsed_time(
                events["attention_residual"]
            ),
            "gate_up_ms": events["attention_residual"].elapsed_time(
                events["gate_up"]
            ),
            "activation_ms": events["gate_up"].elapsed_time(events["activation"]),
            "down_ms": events["activation"].elapsed_time(events["down"]),
            "output_residual_ms": events["down"].elapsed_time(events["end"]),
            "total_ms": events["start"].elapsed_time(events["end"]),
        }

    @staticmethod
    def _apply_hidden_norm(norm: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Keep SGLang's rank-2 RMSNorm kernels shape-transparent to Draft."""

        normalized = norm(hidden_states)
        if isinstance(normalized, tuple):
            raise RuntimeError("Global16 Draft RMSNorm unexpectedly returned residual state")
        if normalized.shape == hidden_states.shape:
            return normalized
        if normalized.numel() != hidden_states.numel():
            raise RuntimeError(
                "Global16 Draft RMSNorm changed the hidden-state element count: "
                f"input={tuple(hidden_states.shape)} output={tuple(normalized.shape)}"
            )
        return normalized.reshape_as(hidden_states)

    def forward_global16_raw256(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        global16_context,
        layer_index: int,
    ) -> torch.Tensor:
        profile_events = self._profile_events if self._profile_active else None
        if profile_events is not None:
            profile_events["start"].record()
        residual = hidden_states
        if self.self_attn.float8_qkv is None:
            attention_input = self._apply_hidden_norm(
                self.input_layernorm,
                hidden_states,
            )
            input_norm = None
        else:
            attention_input = hidden_states
            input_norm = self.input_layernorm
        if profile_events is not None:
            profile_events["input_norm"].record()
        hidden_states = self.self_attn.forward_global16_raw256(
            attention_input,
            input_norm=input_norm,
            position_ids=position_ids,
            global16_context=global16_context,
            layer_index=layer_index,
        )
        if profile_events is not None:
            profile_events["attention"].record()
        fused_mlp_input = None
        if self.fused_residual_norm and self.float8_gate_up is None:
            from sglang.srt.speculative.global16_fused_residual_norm import (
                can_fuse_residual_norm, fused_residual_norm_hf,
            )
            if can_fuse_residual_norm(hidden_states, residual, self.post_attention_layernorm):
                hidden_states, fused_mlp_input = fused_residual_norm_hf(
                    hidden_states, residual, self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.variance_epsilon,
                )
            else:
                hidden_states = residual + hidden_states
        else:
            hidden_states = residual + hidden_states
        if profile_events is not None:
            profile_events["attention_residual"].record()
        if self.fused_rmsnorm_float8_quantization:
            norm_weight = getattr(
                self.post_attention_layernorm,
                "weight",
                None,
            )
            variance_epsilon = getattr(
                self.post_attention_layernorm,
                "variance_epsilon",
                None,
            )
            if not isinstance(norm_weight, torch.Tensor) or variance_epsilon is None:
                raise RuntimeError(
                    "fused gate/up RMSNorm requires a weight-backed RMSNorm"
                )
            gate_up = self.float8_gate_up.project_rmsnorm(
                hidden_states,
                norm_weight=norm_weight,
                variance_epsilon=float(variance_epsilon),
            )
        else:
            mlp_input = (
                fused_mlp_input if fused_mlp_input is not None else
                self._apply_hidden_norm(self.post_attention_layernorm, hidden_states)
            )
        if self.float8_gate_up is None:
            gate_up, _ = self.gate_up_proj(mlp_input)
        elif not self.fused_rmsnorm_float8_quantization:
            gate_up = self.float8_gate_up(mlp_input)
        if profile_events is not None:
            profile_events["gate_up"].record()
        if self.float8_down_fused_activation is not None:
            if profile_events is not None:
                profile_events["activation"].record()
            mlp = self.float8_down_fused_activation.project_silu_and_mul(gate_up)
        else:
            if self.fused_silu is None:
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
            else:
                activated = self.fused_silu(gate_up)
            if profile_events is not None:
                profile_events["activation"].record()
            mlp, _ = self.down_proj(activated)
        if profile_events is not None:
            profile_events["down"].record()
        output = hidden_states + mlp
        if profile_events is not None:
            profile_events["end"].record()
        return output


class Global16NativeDraftExecutor(nn.Module):
    """Run the serving-only Global16 branch with native SGLang decoder layers."""

    def __init__(self, draft_model: nn.Module) -> None:
        super().__init__()
        source_layers = getattr(draft_model, "parallel_qwen_layers", None)
        if not isinstance(source_layers, nn.ModuleList) or not source_layers:
            raise ValueError("Global16 native executor requires Qwen Draft layers")
        if not bool(getattr(draft_model.config, "global16_raw256_enabled", False)):
            raise ValueError("Global16 native executor requires the Global16/Raw256 contract")
        if int(getattr(draft_model.config, "block_size", 0)) != 7:
            raise ValueError("Global16 native executor requires the Local7 proposal contract")
        self.fused_silu_enabled = _env_flag("DFK_GLOBAL16_FUSED_SILU")
        self.qk_rmsnorm_enabled = _env_flag("DFK_GLOBAL16_QK_SGLANG_RMSNORM")
        self.fused_qk_norm_rope_enabled = _env_flag(
            "DFK_GLOBAL16_FUSED_QK_NORM_ROPE"
        )
        self.disable_markov_correction = _env_flag(
            "DFK_GLOBAL16_DISABLE_MARKOV_CORRECTION"
        )
        self.float8_vocab_projection_requested = _env_flag(
            "DFK_GLOBAL16_FLOAT8_VOCAB_PROJECTION"
        )
        self.float8_vocab_projection_enabled = False
        self.float8_gate_up_enabled = _env_flag(
            "DFK_GLOBAL16_FLOAT8_GATE_UP"
        )
        self.float8_qkv_enabled = _env_flag("DFK_GLOBAL16_FLOAT8_QKV")
        self.float8_down_fused_activation_enabled = _env_flag(
            "DFK_GLOBAL16_FLOAT8_DOWN_FUSED_ACTIVATION"
        )
        self.float8_output_projection_enabled = _env_flag(
            "DFK_GLOBAL16_FLOAT8_OUTPUT_PROJECTION"
        )
        self.float8_input_feedforwards_enabled = _env_flag(
            "DFK_GLOBAL16_FLOAT8_INPUT_FEEDFORWARDS"
        )
        proposal_vocab_prefix_size = _env_int(
            "DFK_GLOBAL16_PROPOSAL_VOCAB_PREFIX_SIZE"
        )
        proposal_vocab_extra_start = _env_int(
            "DFK_GLOBAL16_PROPOSAL_VOCAB_EXTRA_START"
        )
        proposal_vocab_extra_end = _env_int(
            "DFK_GLOBAL16_PROPOSAL_VOCAB_EXTRA_END"
        )
        self.proposal_candidate_token_ids = None
        if proposal_vocab_prefix_size > 0:
            vocab_size = int(getattr(draft_model.config, "vocab_size", 0))
            if not 0 < proposal_vocab_prefix_size <= vocab_size:
                raise ValueError(
                    "Global16 proposal vocabulary prefix must be within the "
                    "checkpoint vocabulary"
                )
            if not (
                0
                <= proposal_vocab_extra_start
                <= proposal_vocab_extra_end
                <= vocab_size
            ):
                raise ValueError(
                    "Global16 proposal vocabulary extra range is invalid"
                )
            markov_weight = getattr(
                getattr(
                    getattr(draft_model, "parallel_markov_head", None),
                    "markov_w2",
                    None,
                ),
                "weight",
                None,
            )
            if (
                not isinstance(markov_weight, torch.Tensor)
                or markov_weight.shape[0] != vocab_size
            ):
                raise ValueError(
                    "Global16 proposal vocabulary requires a matching Markov head"
                )
            prefix_ids = torch.arange(
                proposal_vocab_prefix_size,
                dtype=torch.long,
                device=markov_weight.device,
            )
            extra_start = max(
                proposal_vocab_prefix_size,
                proposal_vocab_extra_start,
            )
            if extra_start < proposal_vocab_extra_end:
                extra_ids = torch.arange(
                    extra_start,
                    proposal_vocab_extra_end,
                    dtype=torch.long,
                    device=markov_weight.device,
                )
                self.proposal_candidate_token_ids = torch.cat(
                    (prefix_ids, extra_ids)
                )
            else:
                self.proposal_candidate_token_ids = prefix_ids
        self.fused_float8_quantization_enabled = _env_flag(
            "DFK_GLOBAL16_FUSED_FLOAT8_QUANTIZATION"
        )
        self.fused_rmsnorm_float8_quantization_enabled = _env_flag(
            "DFK_GLOBAL16_FUSED_RMSNORM_FLOAT8_QUANTIZATION"
        )
        if self.fused_rmsnorm_float8_quantization_enabled and not (
            self.fused_float8_quantization_enabled
            and self.float8_vocab_projection_requested
            and self.float8_gate_up_enabled
        ):
            raise RuntimeError(
                "fused RMSNorm plus FP8 quantization requires fused FP8 "
                "vocabulary and gate/up projections"
            )
        if self.float8_qkv_enabled and not (
            self.fused_float8_quantization_enabled
            and self.fused_rmsnorm_float8_quantization_enabled
        ):
            raise RuntimeError(
                "FP8 Q/K/V projection requires fused RMSNorm plus FP8 "
                "input quantization"
            )
        if self.float8_input_feedforwards_enabled and not (
            self.fused_float8_quantization_enabled
        ):
            raise RuntimeError(
                "FP8 input feed-forwards require fused FP8 input quantization"
            )
        if self.proposal_candidate_token_ids is not None and not (
            self.float8_vocab_projection_requested
            and _env_flag("DFK_GLOBAL16_MARKOV_FUSED_ARGMAX")
        ):
            raise RuntimeError(
                "Global16 proposal vocabulary reduction requires the FP8 "
                "vocabulary projection and fused Markov sampler"
            )
        if self.qk_rmsnorm_enabled and self.fused_qk_norm_rope_enabled:
            raise RuntimeError(
                "Global16 Q/K RMSNorm replacement and fused norm-plus-RoPE "
                "are mutually exclusive"
            )
        rope_theta = float(
            getattr(draft_model.config, "qwen_rope_theta", 1_000_000.0)
        )
        self.decoder_layers = nn.ModuleList(
            [
                Global16NativeQwenDecoderLayer(
                    layer,
                    use_fused_silu=self.fused_silu_enabled,
                    use_sglang_qk_rmsnorm=self.qk_rmsnorm_enabled,
                    use_fused_qk_norm_rope=self.fused_qk_norm_rope_enabled,
                    use_float8_qkv=self.float8_qkv_enabled,
                    use_float8_output_projection=(
                        self.float8_output_projection_enabled
                    ),
                    use_float8_gate_up=self.float8_gate_up_enabled,
                    use_float8_down_fused_activation=(
                        self.float8_down_fused_activation_enabled
                    ),
                    use_fused_float8_quantization=(
                        self.fused_float8_quantization_enabled
                    ),
                    use_fused_rmsnorm_float8_quantization=(
                        self.fused_rmsnorm_float8_quantization_enabled
                    ),
                    rope_theta=rope_theta,
                )
                for layer in source_layers
            ]
        )
        self.native_profile_max_rounds = int(
            os.getenv("DFK_GLOBAL16_NATIVE_PROFILE_MAX_ROUNDS", "0")
        )
        self.native_profile_skip_rounds = int(
            os.getenv("DFK_GLOBAL16_NATIVE_PROFILE_SKIP_ROUNDS", "0")
        )
        if self.native_profile_max_rounds < 0 or self.native_profile_skip_rounds < 0:
            raise ValueError("Global16 native profile bounds must be non-negative")
        self._native_profile_step = 0
        self._native_profile_captured = 0
        self._native_profile_start: torch.cuda.Event | None = None
        self._native_profile_end: torch.cuda.Event | None = None
        if self.native_profile_max_rounds:
            profile_device = self.decoder_layers[0].gate_up_proj.weight.device
            with torch.cuda.device(profile_device):
                self._native_profile_start = torch.cuda.Event(enable_timing=True)
                self._native_profile_end = torch.cuda.Event(enable_timing=True)
                self._native_profile_start.record()
                self._native_profile_end.record()
                for layer in self.decoder_layers:
                    layer.initialize_profile_events()
                torch.cuda.synchronize(profile_device)
            logger.info(
                "Enabled bounded Global16 native profile: skip=%d capture=%d",
                self.native_profile_skip_rounds,
                self.native_profile_max_rounds,
            )
        if self.fused_silu_enabled:
            logger.info(
                "Enabled Global16 fused SiLU-and-multiply in %d Draft layers",
                len(self.decoder_layers),
            )
        if self.qk_rmsnorm_enabled:
            logger.info(
                "Enabled Global16 SGLang query/key RMSNorm in %d Draft layers",
                len(self.decoder_layers),
            )
        if self.fused_qk_norm_rope_enabled:
            logger.info(
                "Enabled Global16 fused query/key RMSNorm plus RoPE in %d "
                "Draft layers",
                len(self.decoder_layers),
            )
        if self.float8_gate_up_enabled:
            logger.info(
                "Enabled Global16 FP8 gate/up projection in %d Draft layers",
                len(self.decoder_layers),
            )
        if self.float8_qkv_enabled:
            logger.info(
                "Enabled fused Global16 RMSNorm plus FP8 Q/K/V projection "
                "in %d Draft layers",
                len(self.decoder_layers),
            )
        if self.float8_down_fused_activation_enabled:
            logger.info(
                "Enabled fused Global16 activation plus FP8 down projection "
                "in %d Draft layers",
                len(self.decoder_layers),
            )
        if self.float8_output_projection_enabled:
            logger.info(
                "Enabled Global16 FP8 attention output projection in %d "
                "Draft layers",
                len(self.decoder_layers),
            )
        if self.float8_input_feedforwards_enabled:
            if not install_float8_input_feedforwards(
                draft_model,
                use_fused_quantization=self.fused_float8_quantization_enabled,
            ):
                raise RuntimeError(
                    "Global16 FP8 input feed-forwards were requested but the "
                    "checkpoint modules are unsupported"
                )
            logger.info(
                "Enabled Global16 FP8 state-encoder and shifted-input "
                "feed-forwards"
            )
        if self.fused_float8_quantization_enabled:
            if not (
                self.float8_vocab_projection_requested
                or self.float8_gate_up_enabled
                or self.float8_qkv_enabled
                or self.float8_down_fused_activation_enabled
                or self.float8_output_projection_enabled
                or self.float8_input_feedforwards_enabled
            ):
                raise RuntimeError(
                    "fused FP8 input quantization requires an FP8 projection"
                )
            logger.info(
                "Enabled fused Global16 FP8 input quantization with reusable workspaces"
            )
        if self.fused_rmsnorm_float8_quantization_enabled:
            logger.info(
                "Enabled fused Global16 RMSNorm plus FP8 input quantization"
            )
        if self.disable_markov_correction:
            logger.info(
                "Disabled Global16 previous-token Markov correction for the "
                "proposal speed/acceptance upper-bound probe"
            )
        if self.proposal_candidate_token_ids is not None:
            logger.info(
                "Enabled Global16 proposal vocabulary reduction to %d tokens",
                self.proposal_candidate_token_ids.numel(),
            )
        if len(self.decoder_layers) != len(getattr(draft_model.config, "target_layer_ids", ())):
            raise ValueError("Global16 native executor decoder count does not match selected layers")
        # Keep this reference out of the module tree: ``draft_model`` already
        # owns the common input, Markov, and projection weights.
        object.__setattr__(self, "_draft_model", draft_model)
        self.markov_bfloat16_enabled = False
        if _env_flag("DFK_GLOBAL16_MARKOV_BFLOAT16"):
            self.markov_bfloat16_enabled = install_bfloat16_markov_projection(
                draft_model
            )
            if not self.markov_bfloat16_enabled:
                raise RuntimeError(
                    "Global16 BF16 Markov projection was requested but the "
                    "checkpoint projection interface is unsupported"
                )
            logger.info(
                "Enabled Global16 BF16 vocabulary logits for Markov proposal sampling"
            )
        self.native_vocab_gemm_enabled = False
        if _env_flag("DFK_GLOBAL16_NATIVE_VOCAB_GEMM"):
            if not self.markov_bfloat16_enabled:
                raise RuntimeError(
                    "Global16 native vocabulary GEMM requires BF16 Markov logits"
                )
            self.native_vocab_gemm_enabled = install_native_vocab_gemm(
                draft_model,
                compare=_env_flag(
                    "DFK_GLOBAL16_NATIVE_VOCAB_GEMM_COMPARE"
                ),
            )
            if not self.native_vocab_gemm_enabled:
                raise RuntimeError(
                    "Global16 native vocabulary GEMM was requested but the "
                    "Target or Markov projection interface is unsupported"
                )
            logger.info(
                "Enabled native CUDA GEMM for Global16 Target and Markov "
                "full-vocabulary projections"
            )
        self.markov_fused_argmax_enabled = False
        if _env_flag("DFK_GLOBAL16_MARKOV_FUSED_ARGMAX"):
            if not self.markov_bfloat16_enabled:
                raise RuntimeError(
                    "Global16 fused Markov argmax requires BF16 vocabulary logits"
                )
            max_batch_size = int(
                os.getenv("DFK_GLOBAL16_MARKOV_MAX_BATCH", "0")
            )
            self.markov_fused_argmax_enabled = install_fused_markov_sampler(
                draft_model,
                max_batch_size=max_batch_size,
                fixed_candidate_token_ids=self.proposal_candidate_token_ids,
            )
            if not self.markov_fused_argmax_enabled:
                raise RuntimeError(
                    "Global16 fused Markov argmax was requested but the "
                    "checkpoint Markov interface is unsupported"
                )
            logger.info(
                "Enabled Global16 fused Markov score addition and argmax "
                "with max_batch_size=%d",
                max_batch_size,
            )

    @torch.inference_mode()
    def forward(
        self,
        *,
        conditioning_hidden_states: Optional[torch.Tensor],
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        target_final_norm: Optional[nn.Module],
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        global16_paged_context,
        apply_markov_correction: bool = True,
        return_proposal_ids: bool = True,
    ) -> torch.Tensor:
        if (
            self.float8_vocab_projection_requested
            and not self.float8_vocab_projection_enabled
        ):
            self.float8_vocab_projection_enabled = install_float8_vocab_projection(
                self._draft_model,
                target_lm_head=target_lm_head,
                use_fused_quantization=self.fused_float8_quantization_enabled,
                use_fused_rmsnorm_quantization=(
                    self.fused_rmsnorm_float8_quantization_enabled
                ),
                fixed_candidate_token_ids=self.proposal_candidate_token_ids,
            )
            if not self.float8_vocab_projection_enabled:
                raise RuntimeError(
                    "Global16 FP8 vocabulary projection was requested but the "
                    "Target head or device is unsupported"
                )
            logger.info(
                "Enabled Global16 proposal-only FP8 Target vocabulary projection"
            )
        self._native_profile_step += 1
        profile_active = (
            self.native_profile_max_rounds > 0
            and self._native_profile_step > self.native_profile_skip_rounds
            and self._native_profile_captured < self.native_profile_max_rounds
        )
        if profile_active:
            assert self._native_profile_start is not None
            self._native_profile_start.record()
            for layer in self.decoder_layers:
                layer.set_profile_active(True)
        result = self._draft_model._forward_global16_raw256(
            keys=None,
            values=None,
            attention_mask=None,
            kv_source_indices=None,
            anchor_attention_mask=None,
            current_hidden_states=conditioning_hidden_states,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            teacher_forcing_token_ids=None,
            target_final_norm=target_final_norm,
            target_embed_tokens=target_embed_tokens,
            target_lm_head=target_lm_head,
            candidate_token_ids=self.proposal_candidate_token_ids,
            apply_markov_correction=(
                apply_markov_correction and not self.disable_markov_correction
            ),
            global16_paged_context=global16_paged_context,
            return_proposal_ids=return_proposal_ids,
            decoder_layers=self.decoder_layers,
        )
        if profile_active:
            assert self._native_profile_start is not None
            assert self._native_profile_end is not None
            self._native_profile_end.record()
            self._native_profile_end.synchronize()
            layer_timings = [
                layer.profile_timings() for layer in self.decoder_layers
            ]
            for layer in self.decoder_layers:
                layer.set_profile_active(False)
            summed = {
                key: sum(layer[key] for layer in layer_timings)
                for key in layer_timings[0]
            }
            proposal_total_ms = self._native_profile_start.elapsed_time(
                self._native_profile_end
            )
            self._native_profile_captured += 1
            logger.info(
                "DFK_GLOBAL16_NATIVE_PROFILE %s",
                json.dumps(
                    {
                        "step": self._native_profile_step,
                        "captured": self._native_profile_captured,
                        "batch_size": int(current_token_ids.shape[0]),
                        "proposal_total_ms": proposal_total_ms,
                        "decoder_sums_ms": summed,
                        "proposal_outside_decoders_ms": (
                            proposal_total_ms - summed["total_ms"]
                        ),
                        "layers": layer_timings,
                    },
                    sort_keys=True,
                ),
            )
        return result


def build_global16_native_draft_executor(
    draft_model: nn.Module,
) -> Global16NativeDraftExecutor:
    """Build the native executor and release the unused PyTorch decoder bank."""

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError(
            "Global16 native Draft executor currently requires TP1; "
            "its packed execution weights are intentionally single-GPU"
        )
    executor = Global16NativeDraftExecutor(draft_model)
    # The native executor owns all decoder weights now.  Releasing the training
    # layout is essential: retaining both would consume cache capacity without
    # providing a fallback in the graph-enabled serving path.
    draft_model.parallel_qwen_layers = nn.ModuleList()
    return executor


__all__ = [
    "Global16NativeDraftExecutor",
    "build_global16_native_draft_executor",
]
