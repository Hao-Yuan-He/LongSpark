"""Serving-only lower-precision vocabulary projection for LongSpark proposals."""

from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


def _native_linear_out(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run native ``aten::mm.out`` without the global ``aten::mm`` override.

    LongSpark enables batch-invariant ``aten::mm`` process-wide.  For the two
    fixed full-vocabulary projections validated by the Global16 serving path,
    the native CUDA kernel is both bit-identical and materially faster.  The
    ``out`` overload has its own dispatch entry, so it remains native while the
    ordinary ``mm`` overload stays batch-invariant for every decoder GEMM.
    """

    if hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError("native vocabulary projection dimensions do not match")
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    output = torch.empty(
        (flattened.shape[0], weight.shape[0]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    torch.ops.aten.mm.out(flattened, weight.t(), out=output)
    return output.view(*hidden_states.shape[:-1], weight.shape[0])


def _can_use_native_vocab_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    return (
        not torch.is_grad_enabled()
        and hidden_states.is_cuda
        and weight.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and bias is None
    )


def _assert_vocab_gemm_exact(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    name: str,
) -> None:
    if torch.equal(reference, actual):
        return
    max_abs = (reference.float() - actual.float()).abs().max().item()
    raise RuntimeError(
        f"Global16 {name} native GEMM changed BF16 output (max_abs={max_abs})"
    )


def install_bfloat16_markov_projection(draft_model: nn.Module) -> bool:
    """Keep full-vocabulary proposal logits in BF16 before Markov sampling."""

    original = getattr(draft_model, "_project_logits", None)
    prepare_hidden = getattr(draft_model, "_prepare_logits_hidden", None)
    if not callable(original) or not callable(prepare_hidden):
        return False

    def _project_logits_bfloat16(
        self,
        hidden_states: torch.Tensor,
        *,
        target_final_norm: Optional[nn.Module],
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        weight = getattr(target_lm_head, "weight", None)
        if (
            candidate_token_ids is not None
            or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
        ):
            return original(
                hidden_states,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
        hidden_states = self._prepare_logits_hidden(
            hidden_states,
            target_final_norm=target_final_norm,
        )
        bias = getattr(target_lm_head, "bias", None)
        bias = None if bias is None else bias.detach()
        projected_hidden = hidden_states.to(dtype=weight.dtype)
        if (
            getattr(self, "_dfk_native_vocab_gemm_enabled", False)
            and _can_use_native_vocab_gemm(projected_hidden, weight, bias)
        ):
            native_output = _native_linear_out(
                projected_hidden,
                weight.detach(),
            )
            if getattr(self, "_dfk_native_vocab_gemm_compare", False):
                reference = F.linear(
                    projected_hidden,
                    weight.detach(),
                    bias,
                )
                _assert_vocab_gemm_exact(
                    reference,
                    native_output,
                    name="Target vocabulary",
                )
            return native_output
        return F.linear(
            projected_hidden,
            weight.detach(),
            bias,
        )

    draft_model._project_logits = MethodType(
        _project_logits_bfloat16,
        draft_model,
    )
    return True


def install_native_vocab_gemm(
    draft_model: nn.Module,
    *,
    compare: bool = False,
) -> bool:
    """Bypass the global persistent GEMM only for LongSpark vocab projections."""

    project_logits = getattr(draft_model, "_project_logits", None)
    markov_head = getattr(draft_model, "parallel_markov_head", None)
    compute_bias = getattr(markov_head, "compute_bias", None)
    markov_w1 = getattr(markov_head, "markov_w1", None)
    markov_w2 = getattr(markov_head, "markov_w2", None)
    markov_weight = getattr(markov_w2, "weight", None)
    if (
        not callable(project_logits)
        or not callable(compute_bias)
        or not callable(markov_w1)
        or not isinstance(markov_weight, torch.Tensor)
        or getattr(markov_w2, "bias", None) is not None
    ):
        return False

    def _compute_bias_native(
        self,
        previous_token_ids: torch.Tensor,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if candidate_token_ids is not None:
            return compute_bias(
                previous_token_ids,
                candidate_token_ids=candidate_token_ids,
            )
        latent = self.markov_w1(previous_token_ids.long())
        weight = self.markov_w2.weight.detach()
        if not _can_use_native_vocab_gemm(latent, weight, None):
            return compute_bias(previous_token_ids, candidate_token_ids=None)
        native_output = _native_linear_out(latent, weight)
        if compare:
            reference = compute_bias(
                previous_token_ids,
                candidate_token_ids=None,
            )
            _assert_vocab_gemm_exact(
                reference,
                native_output,
                name="Markov vocabulary",
            )
        return native_output

    markov_head.compute_bias = MethodType(_compute_bias_native, markov_head)
    draft_model._dfk_native_vocab_gemm_enabled = True
    draft_model._dfk_native_vocab_gemm_compare = bool(compare)
    return True


def install_native_target_verify_vocab_gemm(
    target_model: nn.Module,
    *,
    compare: bool = False,
) -> bool:
    """Bypass the persistent GEMM only for LongSpark Target-verify logits.

    Batch-invariant serving overrides the ordinary ``aten::mm`` dispatch
    process-wide.  Installing this adapter on the LongSpark worker's Target
    ``LogitsProcessor`` keeps every decoder GEMM and every non-verify forward
    on that established path while routing the fixed-width verify LM head
    through native ``aten::mm.out``.
    """

    logits_processor = getattr(target_model, "logits_processor", None)
    if logits_processor is None:
        return False
    if getattr(
        logits_processor,
        "_dfk_native_target_verify_vocab_gemm_installed",
        False,
    ):
        return True

    original_get_logits = getattr(logits_processor, "_get_logits", None)
    original_compute_lm_head = getattr(
        logits_processor,
        "_compute_lm_head",
        None,
    )
    if not callable(original_get_logits) or not callable(original_compute_lm_head):
        return False

    def _compute_lm_head_native(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = getattr(lm_head, "weight", None)
        is_lora = hasattr(lm_head, "set_lora") and hasattr(
            lm_head,
            "apply_lora",
        )
        if (
            getattr(
                self,
                "_dfk_native_target_verify_vocab_gemm_active",
                False,
            )
            and not getattr(self, "use_fp32_lm_head", False)
            and not is_lora
            and isinstance(weight, torch.Tensor)
            and _can_use_native_vocab_gemm(
                hidden_states,
                weight,
                embedding_bias,
            )
        ):
            native_output = _native_linear_out(hidden_states, weight)
            if compare:
                reference = original_compute_lm_head(
                    hidden_states,
                    lm_head,
                    embedding_bias,
                )
                _assert_vocab_gemm_exact(
                    reference,
                    native_output,
                    name="Target verify vocabulary",
                )
            return native_output
        return original_compute_lm_head(
            hidden_states,
            lm_head,
            embedding_bias,
        )

    def _get_logits_target_verify(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        logits_metadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        forward_mode = getattr(logits_metadata, "forward_mode", None)
        is_target_verify = getattr(forward_mode, "is_target_verify", None)
        previous = getattr(
            self,
            "_dfk_native_target_verify_vocab_gemm_active",
            False,
        )
        self._dfk_native_target_verify_vocab_gemm_active = bool(
            callable(is_target_verify) and is_target_verify()
        )
        try:
            return original_get_logits(
                hidden_states,
                lm_head,
                logits_metadata,
                embedding_bias,
            )
        finally:
            self._dfk_native_target_verify_vocab_gemm_active = previous

    logits_processor._compute_lm_head = MethodType(
        _compute_lm_head_native,
        logits_processor,
    )
    logits_processor._get_logits = MethodType(
        _get_logits_target_verify,
        logits_processor,
    )
    logits_processor._dfk_native_target_verify_vocab_gemm_active = False
    logits_processor._dfk_native_target_verify_vocab_gemm_installed = True
    return True


@triton.jit
def _rowwise_float8_quantize_kernel(
    input_ptr,
    output_ptr,
    scale_inverse_ptr,
    columns: tl.constexpr,
    FLOAT8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    values = tl.load(
        input_ptr + row * columns + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.where(absmax > 0.0, FLOAT8_MAX / absmax, 1.0)
    quantized = tl.maximum(
        tl.minimum(values * scale, FLOAT8_MAX),
        -FLOAT8_MAX,
    )
    tl.store(
        output_ptr + row * columns + offsets,
        quantized,
        mask=valid,
    )
    tl.store(scale_inverse_ptr + row, 1.0 / scale)


@triton.jit
def _rowwise_rmsnorm_float8_quantize_kernel(
    input_ptr,
    norm_weight_ptr,
    output_ptr,
    scale_inverse_ptr,
    columns: tl.constexpr,
    EPSILON: tl.constexpr,
    FLOAT8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    values = tl.load(
        input_ptr + row * columns + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        norm_weight_ptr + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / columns
    normalized = (
        values * tl.rsqrt(variance + EPSILON) * weights
    ).to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(normalized), axis=0)
    scale = tl.where(absmax > 0.0, FLOAT8_MAX / absmax, 1.0)
    quantized = tl.maximum(
        tl.minimum(normalized * scale, FLOAT8_MAX),
        -FLOAT8_MAX,
    )
    tl.store(
        output_ptr + row * columns + offsets,
        quantized,
        mask=valid,
    )
    tl.store(scale_inverse_ptr + row, 1.0 / scale)


@triton.jit
def _rowwise_silu_and_mul_float8_quantize_kernel(
    gate_up_ptr,
    output_ptr,
    scale_inverse_ptr,
    columns: tl.constexpr,
    FLOAT8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    row_start = row * columns * 2
    gate = tl.load(
        gate_up_ptr + row_start + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + row_start + columns + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    absmax = tl.max(tl.abs(activated), axis=0)
    scale = tl.where(absmax > 0.0, FLOAT8_MAX / absmax, 1.0)
    quantized = tl.maximum(
        tl.minimum(activated * scale, FLOAT8_MAX),
        -FLOAT8_MAX,
    )
    tl.store(
        output_ptr + row * columns + offsets,
        quantized,
        mask=valid,
    )
    tl.store(scale_inverse_ptr + row, 1.0 / scale)


@triton.jit
def _rowwise_silu_float8_quantize_kernel(
    input_ptr,
    output_ptr,
    scale_inverse_ptr,
    columns: tl.constexpr,
    FLOAT8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < columns
    values = tl.load(
        input_ptr + row * columns + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    activated = values * tl.sigmoid(values)
    absmax = tl.max(tl.abs(activated), axis=0)
    scale = tl.where(absmax > 0.0, FLOAT8_MAX / absmax, 1.0)
    quantized = tl.maximum(
        tl.minimum(activated * scale, FLOAT8_MAX),
        -FLOAT8_MAX,
    )
    tl.store(
        output_ptr + row * columns + offsets,
        quantized,
        mask=valid,
    )
    tl.store(scale_inverse_ptr + row, 1.0 / scale)


class Float8LinearProjection:
    """A row-scaled FP8 copy of one serving-only linear projection."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        chunk_rows: int = 16384,
        use_fused_quantization: bool = False,
    ) -> None:
        if not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError("FP8 vocabulary projection requires a CUDA BF16 matrix")
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("this PyTorch build does not provide scaled FP8 matmul")
        if torch.cuda.get_device_capability(weight.device) < (9, 0):
            raise RuntimeError("FP8 vocabulary projection requires compute capability 9.0")
        if chunk_rows <= 0:
            raise ValueError("FP8 vocabulary projection chunk_rows must be positive")

        self.source_weight = weight
        self.float8_dtype = torch.float8_e4m3fn
        self.float8_max = float(torch.finfo(self.float8_dtype).max)
        self.use_fused_quantization = bool(use_fused_quantization)
        self._quantization_workspaces: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.weight = torch.empty(
            weight.shape,
            dtype=self.float8_dtype,
            device=weight.device,
        )
        self.weight_scale_inverse = torch.empty(
            (1, weight.shape[0]),
            dtype=torch.float32,
            device=weight.device,
        )
        with torch.no_grad():
            for start in range(0, weight.shape[0], int(chunk_rows)):
                end = min(start + int(chunk_rows), weight.shape[0])
                source = weight[start:end]
                absmax = source.abs().amax(dim=1, keepdim=True).float()
                scale = torch.where(
                    absmax > 0,
                    self.float8_max / absmax,
                    torch.ones_like(absmax),
                )
                self.weight[start:end].copy_(
                    (source.float() * scale)
                    .clamp(-self.float8_max, self.float8_max)
                    .to(self.float8_dtype)
                )
                self.weight_scale_inverse[:, start:end].copy_(
                    scale.reciprocal().transpose(0, 1)
                )

    def matches(self, weight: torch.Tensor) -> bool:
        return (
            weight is self.source_weight
            or (
                weight.device == self.source_weight.device
                and weight.shape == self.source_weight.shape
                and weight.data_ptr() == self.source_weight.data_ptr()
            )
        )

    def _workspace_for(
        self,
        flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._workspace_for_shape(
            rows=int(flat.shape[0]),
            columns=int(flat.shape[1]),
            device=flat.device,
        )

    def _workspace_for_shape(
        self,
        *,
        rows: int,
        columns: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace_key = (rows, columns)
        workspace = self._quantization_workspaces.get(workspace_key)
        if workspace is None:
            workspace = (
                torch.empty(
                    (rows, columns),
                    dtype=self.float8_dtype,
                    device=device,
                ),
                torch.empty(
                    (rows, 1),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._quantization_workspaces[workspace_key] = workspace
        return workspace

    def _project_quantized(
        self,
        quantized: torch.Tensor,
        scale_inverse: torch.Tensor,
        *,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        output = torch._scaled_mm(
            quantized,
            self.weight.transpose(0, 1),
            scale_inverse,
            self.weight_scale_inverse,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return output.reshape(*original_shape[:-1], self.weight.shape[0])

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        if self.use_fused_quantization:
            quantized, scale_inverse = self._workspace_for(flat)
            block = triton.next_power_of_2(int(flat.shape[1]))
            _rowwise_float8_quantize_kernel[(flat.shape[0],)](
                flat,
                quantized,
                scale_inverse,
                columns=int(flat.shape[1]),
                FLOAT8_MAX=self.float8_max,
                BLOCK=block,
                num_warps=8,
            )
        else:
            absmax = flat.abs().amax(dim=1, keepdim=True).float()
            scale = torch.where(
                absmax > 0,
                self.float8_max / absmax,
                torch.ones_like(absmax),
            )
            quantized = (
                (flat.float() * scale)
                .clamp(-self.float8_max, self.float8_max)
                .to(self.float8_dtype)
            )
            scale_inverse = scale.reciprocal()
        return self._project_quantized(
            quantized,
            scale_inverse,
            original_shape=original_shape,
        )

    def project_rmsnorm(
        self,
        hidden_states: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        variance_epsilon: float,
    ) -> torch.Tensor:
        """Fuse one BF16 RMSNorm with row-wise FP8 input quantization."""

        if not self.use_fused_quantization:
            raise RuntimeError(
                "fused RMSNorm requires fused FP8 input quantization"
            )
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or not isinstance(norm_weight, torch.Tensor)
            or not norm_weight.is_cuda
            or norm_weight.dtype != torch.bfloat16
            or norm_weight.ndim != 1
            or norm_weight.numel() != hidden_states.shape[-1]
        ):
            raise ValueError(
                "fused RMSNorm requires matching CUDA BF16 input and weight"
            )
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        quantized, scale_inverse = self._workspace_for(flat)
        block = triton.next_power_of_2(int(flat.shape[1]))
        _rowwise_rmsnorm_float8_quantize_kernel[(flat.shape[0],)](
            flat,
            norm_weight,
            quantized,
            scale_inverse,
            columns=int(flat.shape[1]),
            EPSILON=float(variance_epsilon),
            FLOAT8_MAX=self.float8_max,
            BLOCK=block,
            num_warps=8,
        )
        return self._project_quantized(
            quantized,
            scale_inverse,
            original_shape=original_shape,
        )

    def project_silu_and_mul(
        self,
        gate_up: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse SwiGLU activation with row-wise FP8 input quantization."""

        if not self.use_fused_quantization:
            raise RuntimeError(
                "fused SiLU-and-multiply requires fused FP8 input quantization"
            )
        columns = int(self.weight.shape[1])
        if (
            not gate_up.is_cuda
            or gate_up.dtype != torch.bfloat16
            or gate_up.shape[-1] != columns * 2
        ):
            raise ValueError(
                "fused SiLU-and-multiply requires a matching CUDA BF16 gate/up tensor"
            )
        rows = gate_up.numel() // gate_up.shape[-1]
        flat_gate_up = gate_up.reshape(rows, columns * 2).contiguous()
        quantized, scale_inverse = self._workspace_for_shape(
            rows=rows,
            columns=columns,
            device=gate_up.device,
        )
        block = triton.next_power_of_2(columns)
        _rowwise_silu_and_mul_float8_quantize_kernel[(rows,)](
            flat_gate_up,
            quantized,
            scale_inverse,
            columns=columns,
            FLOAT8_MAX=self.float8_max,
            BLOCK=block,
            num_warps=8,
        )
        activated_shape = torch.Size((*gate_up.shape[:-1], columns))
        return self._project_quantized(
            quantized,
            scale_inverse,
            original_shape=activated_shape,
        )

    def project_silu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse SiLU activation with row-wise FP8 input quantization."""

        if not self.use_fused_quantization:
            raise RuntimeError("fused SiLU requires fused FP8 input quantization")
        columns = int(self.weight.shape[1])
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[-1] != columns
        ):
            raise ValueError(
                "fused SiLU requires a matching CUDA BF16 input tensor"
            )
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, columns).contiguous()
        quantized, scale_inverse = self._workspace_for(flat)
        block = triton.next_power_of_2(columns)
        _rowwise_silu_float8_quantize_kernel[(flat.shape[0],)](
            flat,
            quantized,
            scale_inverse,
            columns=columns,
            FLOAT8_MAX=self.float8_max,
            BLOCK=block,
            num_warps=8,
        )
        return self._project_quantized(
            quantized,
            scale_inverse,
            original_shape=original_shape,
        )


def install_float8_vocab_projection(
    draft_model: nn.Module,
    *,
    target_lm_head: nn.Module,
    use_fused_quantization: bool = False,
    use_fused_rmsnorm_quantization: bool = False,
    fixed_candidate_token_ids: Optional[torch.Tensor] = None,
) -> bool:
    """Use an FP8 Target-head copy only for LongSpark proposal logits."""

    original = getattr(draft_model, "_project_logits", None)
    prepare_hidden = getattr(draft_model, "_prepare_logits_hidden", None)
    weight = getattr(target_lm_head, "weight", None)
    bias = getattr(target_lm_head, "bias", None)
    if (
        not callable(original)
        or not callable(prepare_hidden)
        or not isinstance(weight, torch.Tensor)
        or bias is not None
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
    ):
        return False

    if fixed_candidate_token_ids is not None:
        if (
            fixed_candidate_token_ids.ndim != 1
            or fixed_candidate_token_ids.dtype != torch.long
            or fixed_candidate_token_ids.device != weight.device
            or fixed_candidate_token_ids.numel() == 0
            or int(fixed_candidate_token_ids[0]) < 0
            or int(fixed_candidate_token_ids[-1]) >= weight.shape[0]
        ):
            return False
        projection_source = weight.detach().index_select(
            0,
            fixed_candidate_token_ids,
        )
    else:
        projection_source = weight.detach()

    projection = Float8LinearProjection(
        projection_source,
        use_fused_quantization=use_fused_quantization,
    )
    # The selected BF16 rows are only needed while creating the FP8 copy.
    # Keep the original Target weight solely to validate later calls.
    projection.source_weight = weight

    def _matches_fixed_candidates(
        candidate_token_ids: Optional[torch.Tensor],
    ) -> bool:
        if fixed_candidate_token_ids is None:
            return candidate_token_ids is None
        return (
            candidate_token_ids is not None
            and candidate_token_ids.device == fixed_candidate_token_ids.device
            and candidate_token_ids.shape == fixed_candidate_token_ids.shape
            and candidate_token_ids.data_ptr()
            == fixed_candidate_token_ids.data_ptr()
        )

    def _project_logits_float8(
        self,
        hidden_states: torch.Tensor,
        *,
        target_final_norm: Optional[nn.Module],
        target_lm_head: nn.Module,
        candidate_token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        current_weight = getattr(target_lm_head, "weight", None)
        if (
            not _matches_fixed_candidates(candidate_token_ids)
            or not isinstance(current_weight, torch.Tensor)
            or not projection.matches(current_weight)
            or not hidden_states.is_cuda
            or hidden_states.shape[-1] != projection.weight.shape[1]
        ):
            return original(
                hidden_states,
                target_final_norm=target_final_norm,
                target_lm_head=target_lm_head,
                candidate_token_ids=candidate_token_ids,
            )
        if use_fused_rmsnorm_quantization:
            norm_weight = getattr(target_final_norm, "weight", None)
            variance_epsilon = getattr(
                target_final_norm,
                "variance_epsilon",
                None,
            )
            if (
                not isinstance(norm_weight, torch.Tensor)
                or variance_epsilon is None
            ):
                raise RuntimeError(
                    "fused FP8 vocabulary RMSNorm requires a weight-backed "
                    "Target RMSNorm"
                )
            return projection.project_rmsnorm(
                hidden_states.to(dtype=torch.bfloat16),
                norm_weight=norm_weight.detach(),
                variance_epsilon=float(variance_epsilon),
            )
        prepared = self._prepare_logits_hidden(
            hidden_states,
            target_final_norm=target_final_norm,
        ).to(dtype=torch.bfloat16)
        return projection(prepared)

    draft_model._project_logits = MethodType(
        _project_logits_float8,
        draft_model,
    )
    object.__setattr__(draft_model, "_dfk_float8_vocab_projection", projection)
    return True


def install_float8_input_feedforwards(
    draft_model: nn.Module,
    *,
    use_fused_quantization: bool = False,
) -> bool:
    """Replace the two serving input feed-forwards with fused FP8 paths."""

    if not use_fused_quantization:
        raise RuntimeError(
            "FP8 input feed-forwards require fused FP8 input quantization"
        )
    state_encoder = getattr(draft_model, "draft_state_encoder", None)
    shifted_input = getattr(draft_model, "parallel_shifted_input", None)
    modules = (
        getattr(state_encoder, "proj", None),
        getattr(shifted_input, "proj", None),
    )
    projections = []
    for module in modules:
        net = getattr(module, "net", None)
        if (
            not isinstance(module, nn.Module)
            or not isinstance(net, nn.Sequential)
            or len(net) != 3
            or not isinstance(net[0], nn.Linear)
            or not isinstance(net[1], nn.SiLU)
            or not isinstance(net[2], nn.Linear)
            or net[0].bias is not None
            or net[2].bias is not None
        ):
            return False
        first = Float8LinearProjection(
            net[0].weight,
            use_fused_quantization=True,
        )
        second = Float8LinearProjection(
            net[2].weight,
            use_fused_quantization=True,
        )

        def _forward_float8(
            self,
            hidden_states: torch.Tensor,
            *,
            first_projection: Float8LinearProjection = first,
            second_projection: Float8LinearProjection = second,
        ) -> torch.Tensor:
            return second_projection.project_silu(
                first_projection(hidden_states)
            )

        module.forward = MethodType(_forward_float8, module)
        projections.append((first, second))
    object.__setattr__(
        draft_model,
        "_dfk_float8_input_feedforwards",
        tuple(projections),
    )
    return True


__all__ = [
    "Float8LinearProjection",
    "install_bfloat16_markov_projection",
    "install_float8_input_feedforwards",
    "install_float8_vocab_projection",
    "install_native_target_verify_vocab_gemm",
    "install_native_vocab_gemm",
]
