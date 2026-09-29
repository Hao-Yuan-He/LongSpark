from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


_BLOCK_V = 1024
_IDX_SENTINEL = tl.constexpr(2147483647)


@dataclass(frozen=True)
class _SampleStepWorkspace:
    batch_size: int
    vocab_size: int
    device: torch.device
    n_tiles: int
    block_tiles: int
    tile_max: torch.Tensor
    tile_sum: torch.Tensor
    tile_best_key: torch.Tensor
    tile_best_num: torch.Tensor
    tile_best_idx: torch.Tensor
    tile_valid: torch.Tensor
    row_valid: torch.Tensor
    exp_noise: torch.Tensor


def _canonical_cuda_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        raise ValueError("sample step workspace requires a CUDA device")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def create_sample_step_workspace(
    *, batch: int, vocab: int, device: torch.device
) -> _SampleStepWorkspace:
    """Allocate scratch owned by one sequential proposal block."""
    if batch < 0:
        raise ValueError(f"batch must be non-negative, got {batch}")
    if vocab <= 0:
        raise ValueError(f"vocab must be positive, got {vocab}")
    device = _canonical_cuda_device(torch.device(device))
    n_tiles = triton.cdiv(vocab, _BLOCK_V)
    block_tiles = triton.next_power_of_2(n_tiles)
    tile_shape = (batch, n_tiles)
    return _SampleStepWorkspace(
        batch_size=batch,
        vocab_size=vocab,
        device=device,
        n_tiles=n_tiles,
        block_tiles=block_tiles,
        tile_max=torch.empty(tile_shape, dtype=torch.float32, device=device),
        tile_sum=torch.empty(tile_shape, dtype=torch.float32, device=device),
        tile_best_key=torch.empty(tile_shape, dtype=torch.float32, device=device),
        tile_best_num=torch.empty(tile_shape, dtype=torch.float32, device=device),
        tile_best_idx=torch.empty(tile_shape, dtype=torch.int32, device=device),
        tile_valid=torch.empty(tile_shape, dtype=torch.bool, device=device),
        row_valid=torch.empty((batch,), dtype=torch.bool, device=device),
        exp_noise=torch.empty((batch, vocab), dtype=torch.float32, device=device),
    )


@triton.jit
def _sample_partial_kernel(
    logits_ptr,
    temperatures_ptr,
    noise_ptr,
    tile_max_ptr,
    tile_sum_ptr,
    tile_best_key_ptr,
    tile_best_num_ptr,
    tile_best_idx_ptr,
    tile_valid_ptr,
    vocab,
    logits_row_stride,
    n_tiles,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < vocab

    raw_logits = tl.load(
        logits_ptr + row * logits_row_stride + offs,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    temperature = tl.load(temperatures_ptr + row).to(tl.float32)
    noise = tl.load(
        noise_ptr + row * vocab + offs,
        mask=mask,
        other=1.0,
    ).to(tl.float32)

    logits_finite = tl.abs(raw_logits) < float("inf")
    # -inf is the usual masked-logit value and is a valid zero-probability
    # entry.  NaN and +inf make the probability row invalid like softmax.
    logits_valid = (raw_logits == raw_logits) & (raw_logits != float("inf"))
    temp_valid = (temperature > 0.0) & (temperature < float("inf"))
    noise_valid = (noise > 0.0) & (noise < float("inf"))
    scaled = tl.where(logits_finite, raw_logits / temperature, float("-inf"))
    scaled_finite = tl.abs(scaled) < float("inf")
    scaled_valid = (scaled == scaled) & (scaled != float("inf"))
    lane_valid = logits_valid & scaled_valid & tl.where(scaled_finite, noise_valid, True)
    tile_valid = temp_valid & (
        tl.min(tl.where(mask, lane_valid, True).to(tl.int32), axis=0) != 0
    )

    tile_max = tl.max(tl.where(mask, scaled, float("-inf")), axis=0)
    tile_has_finite = tl.abs(tile_max) < float("inf")
    safe_tile_max = tl.where(tile_has_finite, tile_max, 0.0)

    # Guard the subtraction for an all -inf tile.  Its mass must be zero.
    exp_shifted = tl.exp(scaled - safe_tile_max)
    exp_shifted = tl.where(mask & tile_has_finite & logits_finite, exp_shifted, 0.0)
    tile_sum = tl.sum(exp_shifted, axis=0)

    keys = exp_shifted / noise
    keys = tl.where(mask & tile_has_finite & noise_valid, keys, -1.0)
    tile_best_key = tl.max(keys, axis=0)
    best_idx_candidates = tl.where(keys == tile_best_key, offs, _IDX_SENTINEL)
    tile_best_idx = tl.min(best_idx_candidates, axis=0)
    tile_best_num = tl.sum(tl.where(offs == tile_best_idx, exp_shifted, 0.0), axis=0)

    tile_offset = row * n_tiles + tile
    tl.store(tile_max_ptr + tile_offset, tile_max)
    tl.store(tile_sum_ptr + tile_offset, tile_sum)
    tl.store(tile_best_key_ptr + tile_offset, tile_best_key)
    tl.store(tile_best_num_ptr + tile_offset, tile_best_num)
    tl.store(tile_best_idx_ptr + tile_offset, tile_best_idx)
    tl.store(tile_valid_ptr + tile_offset, tile_valid)


@triton.jit
def _sample_combine_kernel(
    tile_max_ptr,
    tile_sum_ptr,
    tile_best_key_ptr,
    tile_best_num_ptr,
    tile_best_idx_ptr,
    tile_valid_ptr,
    row_valid_ptr,
    next_tokens_ptr,
    sampled_q_ptr,
    n_tiles,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile_offs = tl.arange(0, BLOCK_TILES)
    tile_mask = tile_offs < n_tiles

    tile_max = tl.load(
        tile_max_ptr + row * n_tiles + tile_offs,
        mask=tile_mask,
        other=float("-inf"),
    )
    tile_sum = tl.load(
        tile_sum_ptr + row * n_tiles + tile_offs,
        mask=tile_mask,
        other=0.0,
    )
    tile_keys = tl.load(
        tile_best_key_ptr + row * n_tiles + tile_offs,
        mask=tile_mask,
        other=-1.0,
    )
    tile_nums = tl.load(
        tile_best_num_ptr + row * n_tiles + tile_offs,
        mask=tile_mask,
        other=0.0,
    )
    tile_idxs = tl.load(
        tile_best_idx_ptr + row * n_tiles + tile_offs,
        mask=tile_mask,
        other=_IDX_SENTINEL,
    )
    tile_valid = tl.load(
        tile_valid_ptr + row * n_tiles + tile_offs, mask=tile_mask, other=0
    )

    global_max = tl.max(tile_max, axis=0)
    global_max_finite = tl.abs(global_max) < float("inf")
    safe_max = tl.where(global_max_finite, global_max, 0.0)
    finite_tile = tile_mask & (tl.abs(tile_max) < float("inf"))
    rescale = tl.where(finite_tile, tl.exp(tile_max - safe_max), 0.0)
    denominator = tl.sum(tl.where(finite_tile, tile_sum * rescale, 0.0), axis=0)

    rescaled_keys = tl.where(finite_tile, tile_keys * rescale, -1.0)
    best_key = tl.max(rescaled_keys, axis=0)
    best_idx_candidates = tl.where(
        rescaled_keys == best_key, tile_idxs, _IDX_SENTINEL
    )
    selected_idx = tl.min(best_idx_candidates, axis=0)
    selected_num = tl.sum(
        tl.where(tile_idxs == selected_idx, tile_nums * rescale, 0.0), axis=0
    )

    all_tiles_valid = (
        tl.min(tl.where(tile_mask, tile_valid != 0, True).to(tl.int32), axis=0) != 0
    )
    valid_row = all_tiles_valid & global_max_finite
    valid_row = valid_row & (tl.abs(denominator) < float("inf"))
    valid_row = valid_row & (denominator > 0.0)
    token = tl.where(valid_row, selected_idx, 0).to(tl.int64)
    q = tl.where(valid_row, selected_num / denominator, float("nan"))
    tl.store(next_tokens_ptr + row, token)
    tl.store(sampled_q_ptr + row, q)
    tl.store(row_valid_ptr + row, valid_row)


def _check_inputs(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    exp_noise: torch.Tensor | None,
) -> tuple[int, int]:
    if step_logits.ndim != 2:
        raise ValueError(f"step_logits must be rank-2, got shape {tuple(step_logits.shape)}")
    if step_logits.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"step_logits must be float32 or bfloat16, got {step_logits.dtype}"
        )
    if not step_logits.is_cuda:
        raise ValueError("sample_step_tokens_with_q requires CUDA step_logits")
    if step_logits.stride(1) != 1:
        raise ValueError("step_logits last dimension must have unit stride")

    n, vocab = step_logits.shape
    if vocab <= 0:
        raise ValueError("step_logits vocabulary dimension must be positive")
    if temperatures.ndim != 1 or temperatures.shape[0] != n:
        raise ValueError(
            f"temperatures must have shape [{n}], got {tuple(temperatures.shape)}"
        )
    if exp_noise is not None:
        if exp_noise.ndim != 2 or tuple(exp_noise.shape) != (n, vocab):
            raise ValueError(
                f"exp_noise must have shape [{n}, {vocab}], got {tuple(exp_noise.shape)}"
            )
        if exp_noise.dtype != torch.float32:
            raise TypeError(f"exp_noise must be float32, got {exp_noise.dtype}")
        if not exp_noise.is_cuda:
            raise ValueError("exp_noise must be on CUDA")
        if exp_noise.device != step_logits.device:
            raise ValueError("exp_noise must be on the step_logits device")
    return n, vocab


def _validate_workspace(
    workspace: _SampleStepWorkspace,
    *,
    batch: int,
    vocab: int,
    device: torch.device,
) -> None:
    if not isinstance(workspace, _SampleStepWorkspace):
        raise TypeError("workspace must come from create_sample_step_workspace")
    if workspace.batch_size != batch:
        raise ValueError(
            f"workspace batch {workspace.batch_size} does not match input batch {batch}"
        )
    if workspace.vocab_size != vocab:
        raise ValueError(
            f"workspace vocab {workspace.vocab_size} does not match input vocab {vocab}"
        )
    if workspace.device != device:
        raise ValueError(
            f"workspace device {workspace.device} does not match input device {device}"
        )


def sample_step_tokens_with_q(
    *,
    step_logits: torch.Tensor,
    temperatures: torch.Tensor,
    exp_noise: torch.Tensor | None = None,
    workspace: _SampleStepWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each row by the exponential race and return its selected q.

    ``step_logits`` may be a strided 2-D view as long as its vocabulary
    dimension has unit stride.  The helper keeps all intermediate state at
    per-tile size; it never materializes a ``[N, V]`` probability tensor.
    Pass a workspace created by ``create_sample_step_workspace`` to reuse
    scratch across sequential proposal steps.  The private workspace is owned
    by one proposal block and must be used sequentially on one CUDA stream;
    token and q outputs remain fresh on every call.

    Invalid rows (NaN or +inf logits, an all--inf row, invalid temperatures,
    or invalid injected noise) are reported with ``torch._assert_async`` after
    the reduction kernels.  Isolated -inf logits are valid masked entries when
    another finite logit exists.  This intentionally avoids a host
    synchronization; the CUDA context may be poisoned after an invalid device
    assertion, as is normal for asynchronous CUDA assertions.
    """
    n, vocab = _check_inputs(
        step_logits=step_logits, temperatures=temperatures, exp_noise=exp_noise
    )
    device = step_logits.device
    if workspace is not None:
        _validate_workspace(workspace, batch=n, vocab=vocab, device=device)
    if n == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )

    temperatures = temperatures.to(device=device, dtype=torch.float32).contiguous()
    if workspace is None:
        n_tiles = triton.cdiv(vocab, _BLOCK_V)
        block_tiles = triton.next_power_of_2(n_tiles)
        tile_shape = (n, n_tiles)
        tile_max = torch.empty(tile_shape, dtype=torch.float32, device=device)
        tile_sum = torch.empty(tile_shape, dtype=torch.float32, device=device)
        tile_best_key = torch.empty(tile_shape, dtype=torch.float32, device=device)
        tile_best_num = torch.empty(tile_shape, dtype=torch.float32, device=device)
        tile_best_idx = torch.empty(tile_shape, dtype=torch.int32, device=device)
        tile_valid = torch.empty(tile_shape, dtype=torch.bool, device=device)
        row_valid = torch.empty((n,), dtype=torch.bool, device=device)
        if exp_noise is None:
            exp_noise = torch.empty(
                (n, vocab), device=device, dtype=torch.float32
            ).exponential_(1)
    else:
        n_tiles = workspace.n_tiles
        block_tiles = workspace.block_tiles
        tile_max = workspace.tile_max
        tile_sum = workspace.tile_sum
        tile_best_key = workspace.tile_best_key
        tile_best_num = workspace.tile_best_num
        tile_best_idx = workspace.tile_best_idx
        tile_valid = workspace.tile_valid
        row_valid = workspace.row_valid
        if exp_noise is None:
            exp_noise = workspace.exp_noise
            exp_noise.exponential_(1)

    if exp_noise is not None:
        exp_noise = exp_noise.contiguous()
    next_tokens = torch.empty((n,), dtype=torch.int64, device=device)
    sampled_q = torch.empty((n,), dtype=torch.float32, device=device)

    _sample_partial_kernel[(n, n_tiles)](
        step_logits,
        temperatures,
        exp_noise,
        tile_max,
        tile_sum,
        tile_best_key,
        tile_best_num,
        tile_best_idx,
        tile_valid,
        vocab,
        step_logits.stride(0),
        n_tiles,
        BLOCK_V=_BLOCK_V,
    )
    _sample_combine_kernel[(n,)](
        tile_max,
        tile_sum,
        tile_best_key,
        tile_best_num,
        tile_best_idx,
        tile_valid,
        row_valid,
        next_tokens,
        sampled_q,
        n_tiles,
        BLOCK_TILES=block_tiles,
    )

    torch._assert_async(row_valid.all())
    return next_tokens, sampled_q


__all__ = ["create_sample_step_workspace", "sample_step_tokens_with_q"]
