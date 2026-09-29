"""Device-side result layout for the fixed-width DraftFreeKV verifier."""

import torch
import triton
import triton.language as tl


@triton.jit
def _finalize_verify_outputs(
    candidates, accepted, bonus, prefix, output, committed, next_lengths,
    candidate_stride_0: tl.constexpr, candidate_stride_1: tl.constexpr,
    accepted_stride: tl.constexpr, bonus_stride: tl.constexpr,
    prefix_stride: tl.constexpr, width: tl.constexpr, count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < count
    row, col = offsets // width, offsets % width
    length = tl.load(accepted + row * accepted_stride, live, 0).to(tl.int32)
    target_bonus = tl.load(bonus + row * bonus_stride, live, 0)
    draft = tl.load(
        candidates + row * candidate_stride_0 + (col + 1) * candidate_stride_1,
        live & (col < width - 1), 0,
    )
    tl.store(output + offsets, tl.where(col == length, target_bonus, draft), live)
    first = live & (col == 0)
    previous_length = tl.load(prefix + row * prefix_stride, first, 0)
    tl.store(committed + row, length + 1, first)
    tl.store(next_lengths + row, previous_length + length + 1, first)


def finalize_verify_outputs(*, candidates, accept_lens, bonus, prefix_lens):
    """Build scheduler outputs without touching request maps or model state.

    Outputs stay freshly allocated: the scheduler may retain them after the
    worker returns. The bonus tensor already is the next current token.
    """
    if candidates.ndim != 2 or candidates.shape[1] < 2:
        raise ValueError("verification candidates must have shape [batch, width>=2]")
    batch, width = candidates.shape
    for name, value in (("accept_lens", accept_lens), ("bonus", bonus),
                        ("prefix_lens", prefix_lens)):
        if value.shape != (batch,) or value.device != candidates.device:
            raise ValueError(f"{name} must have one device row per request")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must have an integer dtype")
    output = torch.empty((batch, width), device=candidates.device, dtype=torch.int64)
    committed = torch.empty((batch,), device=candidates.device, dtype=torch.int32)
    next_lengths = torch.empty((batch,), device=prefix_lens.device, dtype=prefix_lens.dtype)
    if batch:
        _finalize_verify_outputs[(triton.cdiv(batch * width, 256),)](
            candidates, accept_lens, bonus, prefix_lens, output, committed,
            next_lengths, *candidates.stride(), accept_lens.stride(0),
            bonus.stride(0), prefix_lens.stride(0), width, batch * width,
            BLOCK=256,
        )
    return output, committed, next_lengths
