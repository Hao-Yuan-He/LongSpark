from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    _get_or_create_chain_verify_buffers,
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.dspark_components.kernels.cap_correct_len import (
    CapCorrectLen,
)
from sglang.srt.speculative.dspark_components.kernels.softmax_temp import SoftmaxTemp
from sglang.srt.speculative.reject_sampling import chain_speculative_sampling_triton

_KERNEL_IMPL = envs.SGLANG_DSPARK_KERNEL_ACCEPT_SAMPLING.get()
_RESIDUAL_MASS_EPS = 1e-8


class AcceptSampling:
    @classmethod
    def execute_single_q(cls, **kwargs):
        if _KERNEL_IMPL == "torch":
            return accept_sampling_single_q(**kwargs)
        return accept_sampling_single_q_triton(**kwargs)

    @classmethod
    def execute_single_q_torch(cls, **kwargs):
        return accept_sampling_single_q(**kwargs)

    @classmethod
    def execute(
        cls, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _KERNEL_IMPL == "torch":
            return cls.torch(*args, **kwargs)
        return cls.triton(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    @classmethod
    def triton(
        cls,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        draft_input: DFlashDraftInputV2,
        gamma: int,
        verify_num_draft_tokens: int,
        cutoff_verify_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return accept_sampling_triton(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )


def _accept_sampling_core(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        target_probs = SoftmaxTemp.execute(
            logits=target_logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=verify_num_draft_tokens,
        ).view(bs, verify_num_draft_tokens, -1)
    else:
        target_probs = build_dflash_verify_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_token_num=verify_num_draft_tokens,
            bs=bs,
            max_top_k=draft_input.max_top_k,
            uniform_top_k_value=draft_input.uniform_top_k_value,
        )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(
        bs=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    chain_speculative_sampling_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    correct_len = accept_token_num
    if cutoff_verify_lens is not None:
        correct_len, cap_trim_lens = CapCorrectLen.execute(
            correct_len=correct_len, verify_lens=cutoff_verify_lens
        )
    else:
        cap_trim_lens = torch.zeros_like(correct_len)
    return correct_len, cap_trim_lens, accept_index, predicts


def accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = candidates.shape[0]
    device = candidates.device
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus, cap_trim_lens


def _sample_rows_from_probs(
    probs: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    """Inverse-CDF sample with the same uniform convention as the Triton path."""
    cdf = probs.cumsum(dim=-1)
    row_sums = cdf[:, -1]
    matches = cdf > uniforms[:, None] * row_sums[:, None]
    sampled = matches.to(torch.int32).argmax(dim=-1)
    # A valid probability row has positive mass.  The final-token fallback
    # only guards against round-off in the inverse CDF; callers handle a
    # degenerate speculative residual by substituting the Target row first.
    return torch.where(
        matches.any(dim=-1) & (row_sums > 0),
        sampled,
        torch.full_like(sampled, probs.shape[-1] - 1),
    ).to(torch.int64)


def _validate_single_q_inputs(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    draft_temperatures: torch.Tensor,
    sampling_info,
    gamma: int,
    verify_num_draft_tokens: int,
) -> tuple[int, int]:
    if candidates.ndim != 2:
        raise ValueError("single-q acceptance candidates must be two-dimensional")
    bs, candidate_rows = candidates.shape
    if candidate_rows != verify_num_draft_tokens or gamma != candidate_rows - 1:
        raise ValueError("single-q acceptance requires gamma = verify width - 1")
    if target_logits.ndim != 2 or target_logits.shape[0] != bs * candidate_rows:
        raise ValueError("single-q acceptance target logits have an invalid shape")
    if corrected_logits.ndim != 3 or corrected_logits.shape[:2] != (bs, gamma):
        raise ValueError(
            "single-q acceptance corrected logits must be [batch, gamma, vocab]"
        )
    if draft_token_probs.ndim != 2 or draft_token_probs.shape != (bs, gamma):
        raise ValueError("single-q acceptance probabilities must be [batch, gamma]")
    if corrected_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("single-q acceptance logits use different vocabularies")
    if draft_temperatures.numel() != bs:
        raise ValueError(
            "single-q acceptance temperatures must have one value per request"
        )
    if draft_token_probs.dtype != torch.float32:
        raise ValueError("single-q acceptance probabilities must be FP32")
    tensors = {
        "target logits": target_logits,
        "corrected logits": corrected_logits,
        "draft probabilities": draft_token_probs,
        "draft temperatures": draft_temperatures,
        "target temperatures": sampling_info.temperatures,
    }
    mismatched = [
        name for name, tensor in tensors.items() if tensor.device != candidates.device
    ]
    if mismatched:
        raise ValueError(
            "single-q acceptance inputs must share a device; mismatched: "
            + ", ".join(mismatched)
        )
    return bs, candidate_rows


def accept_sampling_single_q(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    draft_temperatures: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept full-vocabulary samples while retaining only sampled q values.

    Proposal sampling already computed the exact q used for every sampled
    token.  The rejection loop therefore needs only those scalars.  The
    production Triton kernel reconstructs a full q row only for requests that
    reject; the torch reference keeps a fixed [batch, vocab] row shape and
    masks q for accepted requests.  Accepted requests sample their bonus
    directly from Target.
    """
    del draft_input
    if cutoff_verify_lens is not None:
        raise ValueError("single-q acceptance does not support cutoff layouts")
    bs, _ = _validate_single_q_inputs(
        candidates=candidates,
        target_logits=target_logits,
        corrected_logits=corrected_logits,
        draft_token_probs=draft_token_probs,
        draft_temperatures=draft_temperatures,
        sampling_info=sampling_info,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
    )

    target_probs = SoftmaxTemp.execute(
        logits=target_logits,
        temperatures=sampling_info.temperatures,
        rows_per_request=verify_num_draft_tokens,
    ).view(bs, verify_num_draft_tokens, -1)

    # Preserve the existing random event order: gamma uniforms for the
    # sequential checks, then one final uniform for every request.
    uniform_samples = torch.rand(
        (bs, gamma), dtype=torch.float32, device=candidates.device
    )
    uniform_samples_final = torch.rand(
        (bs,), dtype=torch.float32, device=candidates.device
    )
    draft_candidates = candidates[:, 1:].to(torch.long)
    target_candidate_probs = (
        target_probs[:, :gamma].gather(2, draft_candidates[:, :, None]).squeeze(-1)
    )
    accepted = uniform_samples * draft_token_probs < target_candidate_probs
    accepted_prefix = torch.cumprod(accepted.to(torch.int32), dim=1)
    correct_len = accepted_prefix.sum(dim=1).to(torch.int32)
    rejected = correct_len < gamma

    # Reconstruct one q row per request at the first rejection position.  For
    # all-accepted requests this row is masked away and the Target bonus row
    # is used.  The shape is [batch, vocab], independent of prefix length and
    # seven times smaller than the old [batch, gamma, vocab] draft cube.
    row_ids = torch.arange(bs, device=candidates.device)
    q_positions = correct_len.clamp(max=gamma - 1).to(torch.long)
    q_logits = corrected_logits[row_ids, q_positions, :]
    draft_temperatures = draft_temperatures.reshape(-1).to(
        device=q_logits.device, dtype=torch.float32
    )
    reconstructed_q = torch.softmax(
        q_logits.float() / draft_temperatures[:, None],
        dim=-1,
    )
    selected_target = target_probs[row_ids, correct_len.to(torch.long), :]
    residual = (selected_target - reconstructed_q).clamp_min(0.0)
    residual_mass = residual.sum(dim=-1)
    use_residual = rejected & (residual_mass > _RESIDUAL_MASS_EPS)
    # Match the paper/evaluator's lossless boundary: numerical cancellation
    # can leave (p-q)+ with negligible mass, in which case sampling an
    # arbitrary token would not preserve Target.  Fall back to the verified
    # Target row instead.
    bonus_probs = torch.where(use_residual[:, None], residual, selected_target)
    bonus = _sample_rows_from_probs(bonus_probs, uniform_samples_final)

    return correct_len, bonus, torch.zeros_like(correct_len)


@triton.jit
def _accept_sampling_single_q_kernel(
    candidates_ptr,
    target_probs_ptr,
    draft_token_probs_ptr,
    corrected_logits_ptr,
    temperatures_ptr,
    uniform_samples_ptr,
    uniform_samples_final_ptr,
    correct_len_ptr,
    bonus_ptr,
    candidate_b,
    candidate_s,
    target_b,
    target_s,
    target_v,
    draft_q_b,
    draft_q_s,
    corrected_b,
    corrected_s,
    corrected_v,
    uniform_b,
    uniform_s,
    VOCAB_SIZE: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_V: tl.constexpr,
    RESIDUAL_MASS_EPS: tl.constexpr,
):
    request = tl.program_id(0)
    candidates_base = candidates_ptr + request * candidate_b
    target_base = target_probs_ptr + request * target_b
    draft_q_base = draft_token_probs_ptr + request * draft_q_b
    corrected_base = corrected_logits_ptr + request * corrected_b
    uniform_base = uniform_samples_ptr + request * uniform_b
    temperature = tl.load(temperatures_ptr + request).to(tl.float32)

    accepted_len = 0
    rejected = 0
    for step in range(GAMMA):
        candidate = tl.load(candidates_base + (step + 1) * candidate_s).to(tl.int64)
        p = tl.load(target_base + step * target_s + candidate * target_v).to(tl.float32)
        q = tl.load(draft_q_base + step * draft_q_s).to(tl.float32)
        coin = tl.load(uniform_base + step * uniform_s).to(tl.float32)
        accept = coin * q < p
        if rejected == 0:
            if accept:
                accepted_len += 1
            else:
                rejected = 1

    target_row = target_base + accepted_len * target_s
    final_coin = tl.load(uniform_samples_final_ptr + request).to(tl.float32)
    row_max = 0.0
    row_sum = 0.0
    if rejected == 1:
        corrected_row = corrected_base + accepted_len * corrected_s
        row_max = -float("inf")
        for v0 in range(0, VOCAB_SIZE, BLOCK_V):
            offsets = v0 + tl.arange(0, BLOCK_V)
            mask = offsets < VOCAB_SIZE
            logits = tl.load(
                corrected_row + offsets * corrected_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            row_max = tl.maximum(row_max, tl.max(logits / temperature, axis=0))
        for v0 in range(0, VOCAB_SIZE, BLOCK_V):
            offsets = v0 + tl.arange(0, BLOCK_V)
            mask = offsets < VOCAB_SIZE
            logits = tl.load(
                corrected_row + offsets * corrected_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            scaled = logits / temperature
            row_sum += tl.sum(tl.where(mask, tl.exp(scaled - row_max), 0.0))

    norm_sum = 0.0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        offsets = v0 + tl.arange(0, BLOCK_V)
        mask = offsets < VOCAB_SIZE
        p = tl.load(target_row + offsets * target_v, mask=mask, other=0.0).to(
            tl.float32
        )
        if rejected == 1:
            corrected_row = corrected_base + accepted_len * corrected_s
            logits = tl.load(
                corrected_row + offsets * corrected_v,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            q = tl.where(mask, tl.exp(logits / temperature - row_max) / row_sum, 0.0)
            p = tl.maximum(p - q, 0.0)
        norm_sum += tl.sum(tl.where(mask, p, 0.0))

    use_residual = rejected == 1
    if use_residual:
        if norm_sum <= RESIDUAL_MASS_EPS:
            use_residual = norm_sum > RESIDUAL_MASS_EPS
            # Degenerate residuals are rare.  Only in that boundary case pay
            # for a separate Target normalization pass; all-accepted rows
            # already accumulated Target mass in ``norm_sum`` above.
            norm_sum = 0.0
            for v0 in range(0, VOCAB_SIZE, BLOCK_V):
                offsets = v0 + tl.arange(0, BLOCK_V)
                mask = offsets < VOCAB_SIZE
                p = tl.load(
                    target_row + offsets * target_v, mask=mask, other=0.0
                ).to(tl.float32)
                norm_sum += tl.sum(tl.where(mask, p, 0.0))
    threshold = final_coin * norm_sum
    cumulative = 0.0
    bonus = VOCAB_SIZE - 1
    found = 0
    for v0 in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            offsets = v0 + tl.arange(0, BLOCK_V)
            mask = offsets < VOCAB_SIZE
            p = tl.load(target_row + offsets * target_v, mask=mask, other=0.0).to(
                tl.float32
            )
            if use_residual:
                corrected_row = corrected_base + accepted_len * corrected_s
                logits = tl.load(
                    corrected_row + offsets * corrected_v,
                    mask=mask,
                    other=-float("inf"),
                ).to(tl.float32)
                q = tl.where(
                    mask, tl.exp(logits / temperature - row_max) / row_sum, 0.0
                )
                p = tl.maximum(p - q, 0.0)
            block_cumulative = tl.cumsum(tl.where(mask, p, 0.0), axis=0)
            total_cumulative = cumulative + block_cumulative
            matches = total_cumulative > threshold
            if tl.max(matches, axis=0):
                bonus = v0 + tl.argmax(matches.to(tl.int32), axis=0)
                found = 1
            cumulative += tl.sum(tl.where(mask, p, 0.0))

    tl.store(correct_len_ptr + request, accepted_len)
    tl.store(bonus_ptr + request, bonus)


def accept_sampling_single_q_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    draft_temperatures: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cutoff_verify_lens is not None:
        raise ValueError("single-q acceptance does not support cutoff layouts")
    bs, _ = _validate_single_q_inputs(
        candidates=candidates,
        target_logits=target_logits,
        corrected_logits=corrected_logits,
        draft_token_probs=draft_token_probs,
        draft_temperatures=draft_temperatures,
        sampling_info=sampling_info,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
    )
    del draft_input

    target_probs = SoftmaxTemp.execute(
        logits=target_logits,
        temperatures=sampling_info.temperatures,
        rows_per_request=verify_num_draft_tokens,
    ).view(bs, verify_num_draft_tokens, -1)
    uniform_samples = torch.rand(
        (bs, gamma), dtype=torch.float32, device=candidates.device
    )
    uniform_samples_final = torch.rand(
        (bs,), dtype=torch.float32, device=candidates.device
    )
    candidates = candidates.contiguous()
    target_probs = target_probs.contiguous()
    draft_token_probs = draft_token_probs.contiguous()
    corrected_logits = corrected_logits.contiguous()
    temperatures = (
        draft_temperatures.reshape(-1)
        .to(device=candidates.device, dtype=torch.float32)
        .contiguous()
    )
    correct_len = torch.empty(bs, dtype=torch.int32, device=candidates.device)
    bonus = torch.empty(bs, dtype=torch.int64, device=candidates.device)
    _accept_sampling_single_q_kernel[(bs,)](
        candidates,
        target_probs,
        draft_token_probs,
        corrected_logits,
        temperatures,
        uniform_samples,
        uniform_samples_final,
        correct_len,
        bonus,
        candidates.stride(0),
        candidates.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_token_probs.stride(0),
        draft_token_probs.stride(1),
        corrected_logits.stride(0),
        corrected_logits.stride(1),
        corrected_logits.stride(2),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        GAMMA=gamma,
        VOCAB_SIZE=target_probs.shape[-1],
        BLOCK_V=4096,
        RESIDUAL_MASS_EPS=_RESIDUAL_MASS_EPS,
        num_warps=8,
    )
    return correct_len, bonus, torch.zeros_like(correct_len)


@triton.jit
def _gather_two_level_bonus_kernel(
    accept_index_ptr,
    predicts_ptr,
    correct_len_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
    accept_pos = tl.load(accept_index_ptr + offs * cols + cl, mask=mask, other=0).to(
        tl.int64
    )
    bonus = tl.load(predicts_ptr + accept_pos, mask=mask, other=0)
    tl.store(out_ptr + offs, bonus.to(tl.int64), mask=mask)


def gather_two_level_bonus_triton(
    *,
    accept_index: torch.Tensor,
    predicts: torch.Tensor,
    correct_len: torch.Tensor,
) -> torch.Tensor:
    bs, cols = accept_index.shape
    accept_index = accept_index.contiguous()
    predicts = predicts.contiguous()
    correct_len = correct_len.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=accept_index.device)
    BLOCK = 256
    grid = (triton.cdiv(bs, BLOCK),)
    _gather_two_level_bonus_kernel[grid](
        accept_index, predicts, correct_len, out, cols, bs, BLOCK=BLOCK
    )
    return out


def accept_sampling_triton(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct_len, cap_trim_lens, accept_index, predicts = _accept_sampling_core(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    bonus = gather_two_level_bonus_triton(
        accept_index=accept_index, predicts=predicts, correct_len=correct_len
    )
    return correct_len, bonus, cap_trim_lens
