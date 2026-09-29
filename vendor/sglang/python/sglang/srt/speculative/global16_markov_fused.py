"""Fused score addition and top-token selection for LongSpark Markov steps."""

from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _add_argmax_stage1(
    base_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_id_ptr,
    vocab_size,
    base_stride_b,
    bias_stride_b,
    chunks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_index = tl.program_id(0)
    chunk_index = tl.program_id(1)
    offsets = chunk_index * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < vocab_size
    base = tl.load(
        base_ptr + batch_index * base_stride_b + offsets,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + batch_index * bias_stride_b + offsets,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    scores = base + bias
    best_value = tl.max(scores, axis=0)
    best_id = tl.min(
        tl.where(scores == best_value, offsets, vocab_size),
        axis=0,
    )
    output_offset = batch_index * chunks + chunk_index
    tl.store(partial_value_ptr + output_offset, best_value)
    tl.store(partial_id_ptr + output_offset, best_id)


@triton.jit
def _add_argmax_stage2(
    partial_value_ptr,
    partial_id_ptr,
    output_ptr,
    vocab_size,
    chunks: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    batch_index = tl.program_id(0)
    chunk_offsets = tl.arange(0, REDUCE_BLOCK)
    valid = chunk_offsets < chunks
    input_offsets = batch_index * chunks + chunk_offsets
    values = tl.load(
        partial_value_ptr + input_offsets,
        mask=valid,
        other=float("-inf"),
    )
    token_ids = tl.load(
        partial_id_ptr + input_offsets,
        mask=valid,
        other=vocab_size,
    )
    best_value = tl.max(values, axis=0)
    best_id = tl.min(
        tl.where(values == best_value, token_ids, vocab_size),
        axis=0,
    )
    tl.store(output_ptr + batch_index, best_id)


def fused_add_argmax(
    base: torch.Tensor,
    bias: torch.Tensor,
    *,
    partial_values: torch.Tensor,
    partial_ids: torch.Tensor,
    output: torch.Tensor,
    block_size: int = 8192,
) -> torch.Tensor:
    """Return argmax(base + bias) without materializing the summed scores."""

    if base.ndim != 2 or bias.shape != base.shape:
        raise ValueError("base and bias must have the same [batch, vocab] shape")
    if not base.is_cuda or not bias.is_cuda:
        raise ValueError("fused LongSpark Markov argmax requires CUDA tensors")
    if base.device != bias.device:
        raise ValueError("base and bias must be on the same CUDA device")
    if base.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported base dtype: {base.dtype}")
    if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported bias dtype: {bias.dtype}")
    batch_size, vocab_size = base.shape
    chunks = triton.cdiv(vocab_size, block_size)
    if partial_values.shape[0] < batch_size or partial_values.shape[1] < chunks:
        raise ValueError("partial-value workspace is too small")
    if partial_ids.shape[0] < batch_size or partial_ids.shape[1] < chunks:
        raise ValueError("partial-id workspace is too small")
    if output.numel() < batch_size or output.dtype != torch.int64:
        raise ValueError("output workspace must provide one int64 row per request")
    reduce_block = triton.next_power_of_2(chunks)
    _add_argmax_stage1[(batch_size, chunks)](
        base,
        bias,
        partial_values,
        partial_ids,
        vocab_size,
        base.stride(0),
        bias.stride(0),
        chunks=chunks,
        BLOCK=block_size,
    )
    _add_argmax_stage2[(batch_size,)](
        partial_values,
        partial_ids,
        output,
        vocab_size,
        chunks=chunks,
        REDUCE_BLOCK=reduce_block,
    )
    return output[:batch_size]


class _FusedMarkovSampler:
    def __init__(
        self,
        *,
        max_batch_size: int,
        max_steps: int,
        vocab_size: int,
        device: torch.device,
        block_size: int = 8192,
        candidate_token_ids: Optional[torch.Tensor] = None,
        candidate_weight: Optional[torch.Tensor] = None,
    ) -> None:
        if max_batch_size <= 0 or max_steps <= 0 or vocab_size <= 0:
            raise ValueError("fused Markov workspace dimensions must be positive")
        self.max_batch_size = int(max_batch_size)
        self.max_steps = int(max_steps)
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.candidate_token_ids = candidate_token_ids
        self.candidate_weight = candidate_weight
        chunks = triton.cdiv(self.vocab_size, self.block_size)
        self.partial_values = torch.empty(
            self.max_batch_size,
            chunks,
            dtype=torch.float32,
            device=device,
        )
        self.partial_ids = torch.empty(
            self.max_batch_size,
            chunks,
            dtype=torch.int32,
            device=device,
        )
        self.step_output = torch.empty(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        self.sampled = torch.empty(
            self.max_batch_size,
            self.max_steps,
            dtype=torch.int64,
            device=device,
        )

    def sample(
        self,
        head: nn.Module,
        base_logits: torch.Tensor,
        *,
        first_previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, steps, vocab_size = base_logits.shape
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Markov batch {batch_size} exceeds workspace {self.max_batch_size}"
            )
        if steps > self.max_steps or vocab_size != self.vocab_size:
            raise ValueError("Markov proposal shape exceeds the registered workspace")
        previous = first_previous_token_ids.long()
        sampled = self.sampled[:batch_size, :steps]
        for step in range(steps):
            if self.candidate_weight is None:
                bias = head.compute_bias(previous, None)
            else:
                latent = head.markov_w1(previous.long())
                bias = F.linear(latent, self.candidate_weight)
            next_indices = fused_add_argmax(
                base_logits[:, step, :],
                bias,
                partial_values=self.partial_values,
                partial_ids=self.partial_ids,
                output=self.step_output,
                block_size=self.block_size,
            )
            if self.candidate_token_ids is None:
                next_tokens = next_indices
            else:
                next_tokens = self.candidate_token_ids.index_select(
                    0,
                    next_indices,
                )
            sampled[:, step].copy_(next_tokens)
            previous = next_tokens
        return sampled

    def matches_candidate_token_ids(
        self,
        candidate_token_ids: Optional[torch.Tensor],
    ) -> bool:
        if self.candidate_token_ids is None:
            return candidate_token_ids is None
        return (
            candidate_token_ids is not None
            and candidate_token_ids.device == self.candidate_token_ids.device
            and candidate_token_ids.shape == self.candidate_token_ids.shape
            and candidate_token_ids.data_ptr()
            == self.candidate_token_ids.data_ptr()
        )


def install_fused_markov_sampler(
    draft_model: nn.Module,
    *,
    max_batch_size: int,
    block_size: int = 8192,
    fixed_candidate_token_ids: Optional[torch.Tensor] = None,
) -> bool:
    """Install the fused greedy sampler on one serving model."""

    head = getattr(draft_model, "parallel_markov_head", None)
    original = getattr(head, "sample_block_token_ids", None)
    weight = getattr(getattr(head, "markov_w2", None), "weight", None)
    if not callable(original) or not isinstance(weight, torch.Tensor):
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
        candidate_weight = weight.detach().index_select(
            0,
            fixed_candidate_token_ids,
        )
        sampler_vocab_size = int(fixed_candidate_token_ids.numel())
    else:
        candidate_weight = None
        sampler_vocab_size = int(weight.shape[0])
    max_steps = int(getattr(draft_model.config, "block_size", 0))
    sampler = _FusedMarkovSampler(
        max_batch_size=max_batch_size,
        max_steps=max_steps,
        vocab_size=sampler_vocab_size,
        device=weight.device,
        block_size=block_size,
        candidate_token_ids=fixed_candidate_token_ids,
        candidate_weight=candidate_weight,
    )

    def _sample_block_token_ids_fused(
        self,
        base_logits: torch.Tensor,
        *,
        first_previous_token_ids: torch.Tensor,
        temperature: float = 0.0,
        candidate_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            temperature > 0
            or not sampler.matches_candidate_token_ids(candidate_token_ids)
            or not base_logits.is_cuda
            or base_logits.ndim != 3
            or base_logits.shape[-1] != sampler.vocab_size
        ):
            return original(
                base_logits,
                first_previous_token_ids=first_previous_token_ids,
                temperature=temperature,
                candidate_token_ids=candidate_token_ids,
            )
        return sampler.sample(
            self,
            base_logits,
            first_previous_token_ids=first_previous_token_ids,
        )

    head.sample_block_token_ids = MethodType(
        _sample_block_token_ids_fused,
        head,
    )
    object.__setattr__(head, "_dfk_fused_markov_sampler", sampler)
    return True


__all__ = ["fused_add_argmax", "install_fused_markov_sampler"]
