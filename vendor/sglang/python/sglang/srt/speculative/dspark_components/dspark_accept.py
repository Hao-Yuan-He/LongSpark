from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.accept_greedy import AcceptGreedy
from sglang.srt.speculative.dspark_components.kernels.accept_sampling import (
    AcceptSampling,
)
from sglang.srt.speculative.dspark_components.kernels.mixed_accept_select import (
    SelectMixedAccept,
)
from sglang.srt.speculative.dspark_components.kernels.softmax_temp import SoftmaxTemp
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


def accept_scheduled_greedy_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Commit every draft token in the scheduled verify window.

    This is intentionally separate from ``AcceptGreedy``: the target forward is
    still used to produce the bonus token, but target/draft token equality is not
    consulted for the scheduled draft prefix.
    """
    bs = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        bs, verify_num_draft_tokens
    )
    correct_len = torch.full(
        (bs,),
        verify_num_draft_tokens - 1,
        dtype=torch.int32,
        device=target_predict.device,
    )
    if cutoff_verify_lens is not None:
        correct_len = (cutoff_verify_lens.to(torch.int32) - 1).clamp(
            min=0, max=verify_num_draft_tokens - 1
        )
    row_ids = torch.arange(bs, device=target_predict.device)
    bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
    return correct_len, bonus, torch.zeros_like(correct_len)


def accept_draft_tokens(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_block: DraftBlockResult,
    sampling_info,
    draft_input: DFlashDraftInputV2,
    gamma: int,
    verify_num_draft_tokens: int,
    cutoff_layout: Optional[RaggedVerifyLayout] = None,
    nonstrict_verify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    greedy_mask = draft_block.greedy_mask
    cutoff_verify_lens = None if cutoff_layout is None else cutoff_layout.verify_lens
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
    if nonstrict_verify:
        if not all_greedy:
            raise ValueError(
                "DSpark nonstrict verify currently supports greedy sampling only."
            )
        return accept_scheduled_greedy_tokens(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    if all_greedy:
        return AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    bs, gamma_rows, vocab = draft_block.corrected_logits.shape
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    pure_temperature_scaling = (
        not getattr(sampling_info, "has_custom_logit_processor", False)
        and getattr(sampling_info, "acc_linear_penalties", None) is None
        and not (penalizer is not None and penalizer.is_required)
        and getattr(sampling_info, "vocab_mask", None) is None
        and getattr(sampling_info, "logit_bias", None) is None
    )
    if not sampling_info.is_any_greedy:
        if (
            draft_block.draft_token_probs is not None
            and cutoff_verify_lens is None
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_top_p_sampling
            and not getattr(sampling_info, "need_min_p_sampling", False)
            and pure_temperature_scaling
        ):
            return AcceptSampling.execute_single_q(
                candidates=candidates,
                target_logits=target_logits,
                corrected_logits=draft_block.corrected_logits,
                draft_token_probs=draft_block.draft_token_probs,
                draft_temperatures=draft_block.temperatures,
                sampling_info=sampling_info,
                draft_input=draft_input,
                gamma=gamma,
                verify_num_draft_tokens=verify_num_draft_tokens,
            )
        draft_probs = SoftmaxTemp.execute(
            logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
            temperatures=draft_block.temperatures,
            rows_per_request=gamma_rows,
        ).view(bs, gamma_rows, vocab)
        return AcceptSampling.execute(
            candidates=candidates,
            target_logits=target_logits,
            draft_probs=draft_probs,
            sampling_info=sampling_info,
            draft_input=draft_input,
            gamma=gamma,
            verify_num_draft_tokens=verify_num_draft_tokens,
            cutoff_verify_lens=cutoff_verify_lens,
        )
    draft_probs = SoftmaxTemp.execute(
        logits=draft_block.corrected_logits.reshape(bs * gamma_rows, vocab),
        temperatures=draft_block.temperatures,
        rows_per_request=gamma_rows,
    ).view(bs, gamma_rows, vocab)
    greedy_len, greedy_bonus, greedy_trim = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    sampling_len, sampling_bonus, sampling_trim = AcceptSampling.execute(
        candidates=candidates,
        target_logits=target_logits,
        draft_probs=draft_probs,
        sampling_info=sampling_info,
        draft_input=draft_input,
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        cutoff_verify_lens=cutoff_verify_lens,
    )
    selected = SelectMixedAccept.execute(
        greedy_mask=greedy_mask,
        greedy_len=greedy_len,
        greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim,
        sampling_len=sampling_len,
        sampling_bonus=sampling_bonus,
        sampling_trim=sampling_trim,
    )
    return selected.correct_len, selected.bonus, selected.cap_trim_lens
