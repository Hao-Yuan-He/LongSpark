"""Residual addition and HF RMSNorm with the existing BF16 rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_hf(x, residual, weight, summed, output, D: tl.constexpr, EPS: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    a = tl.load(x + row * D + col, col < D, 0).to(tl.float32)
    b = tl.load(residual + row * D + col, col < D, 0).to(tl.float32)
    # The unfused path first materializes residual+x in BF16, then normalizes.
    value = (a + b).to(x.dtype.element_ty)
    full = value.to(tl.float32)
    inv = tl.rsqrt(tl.sum(full * full, 0) / D + EPS)
    normalized = (full * inv).to(x.dtype.element_ty)
    w = tl.load(weight + col, col < D, 0).to(tl.float32)
    result = normalized.to(tl.float32) * w
    tl.store(summed + row * D + col, value, col < D)
    tl.store(output + row * D + col, result, col < D)


def can_fuse_residual_norm(x, residual, norm):
    weight = getattr(norm, "weight", None)
    return (
        x.is_cuda and x.dtype == torch.bfloat16 and x.shape == residual.shape
        and residual.device == x.device and residual.dtype == x.dtype
        and x.is_contiguous() and residual.is_contiguous()
        and x.shape[-1] == 4096 and isinstance(weight, torch.Tensor)
        and weight.shape == (4096,) and weight.dtype == x.dtype
        and weight.device == x.device and weight.is_contiguous()
        and getattr(norm, "variance_epsilon", None) is not None
        and bool(getattr(norm, "cast_x_before_out_mul", False))
    )


def fused_residual_norm_hf(x, residual, weight, eps):
    """Return fresh (residual sum, normalized sum), without mutating either input."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape == residual.shape
            and x.dtype == residual.dtype == weight.dtype
            and x.device == residual.device == weight.device
            and x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
            and x.shape[-1] == 4096 and weight.shape == (4096,)):
        raise ValueError("Fused HF residual norm requires contiguous BF16 hidden_size=4096")
    summed, output = torch.empty_like(x), torch.empty_like(x)
    if x.numel():
        _add_rmsnorm_hf[(x.numel() // 4096,)](
            x, residual, weight, summed, output, 4096, float(eps), 4096,
            num_warps=4,
        )
    return summed, output
