from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens import (
    SampleStepTokens,
)


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:

    def __init__(self, *, model, gamma, max_bs, device, confidence_fn=None, out=None):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        if out is not None:
            assert out.shape == (int(max_bs) * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (int(max_bs) * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((int(max_bs), self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.gamma
        base_logits, confidence_tap = self.model.compute_base_logits(hidden_states)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.gamma)[:, 0]
        draft_tokens, _ = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=hidden_states.view(bs, self.gamma, -1),
            sampler=greedy_step_sampler,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=hidden_states.view(bs, self.gamma, -1),
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()
    fast_sampling_with_q = envs.SGLANG_DSPARK_FAST_SAMPLING_WITH_Q.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    # The q carrier is valid only for pure temperature sampling.  In
    # particular, do not use it for mixed greedy batches or filtered logits;
    # those continue through the established DSpark sampler below.
    any_greedy = getattr(sampling_info, "is_any_greedy", None)
    pure_temperature = (
        any_sampling
        and any_greedy is False
        and not getattr(sampling_info, "need_top_k_sampling", False)
        and not getattr(sampling_info, "need_top_p_sampling", False)
        and not getattr(sampling_info, "need_min_p_sampling", False)
        and not getattr(sampling_info, "has_custom_logit_processor", False)
        and getattr(sampling_info, "acc_linear_penalties", None) is None
        and not (
            getattr(sampling_info, "penalizer_orchestrator", None) is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
        and getattr(sampling_info, "vocab_mask", None) is None
        and getattr(sampling_info, "logit_bias", None) is None
    )
    full_vocab = (
        markov_head is not None
        and getattr(markov_head, "vocab_size", None) == int(base_logits.shape[-1])
        and not getattr(markov_head, "_opt_markov_w2_tp_shard", False)
    )
    sampled_token_probs = None
    q_workspace = None
    q_sampling_enabled = False
    if (
        fast_sampling_with_q
        and pure_temperature
        and full_vocab
        and base_logits.is_cuda
        and base_logits.dtype in (torch.float32, torch.bfloat16)
        and base_logits.stride(-1) == 1
    ):
        from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens_with_q import (
            create_sample_step_workspace,
            sample_step_tokens_with_q,
        )

        q_sampler = sample_step_tokens_with_q
        q_workspace = create_sample_step_workspace(
            batch=bs,
            vocab=int(base_logits.shape[-1]),
            device=base_logits.device,
        )
        sampled_token_probs = []
        q_sampling_enabled = True
    else:
        q_sampler = None

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            nonlocal q_sampling_enabled
            q_layout_valid = (
                step_logits.ndim == 2
                and step_logits.shape == (bs, int(base_logits.shape[-1]))
                and step_logits.dtype in (torch.float32, torch.bfloat16)
                and step_logits.is_cuda
                and step_logits.stride(-1) == 1
            )
            if q_sampler is not None and q_sampling_enabled and q_layout_valid:
                next_tokens, sampled_q = q_sampler(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    workspace=q_workspace,
                )
                sampled_token_probs.append(sampled_q)
                return next_tokens
            if q_sampling_enabled and not q_layout_valid:
                # A changed Markov implementation must not leave a partial q
                # carrier that acceptance could mistake for a complete block.
                q_sampling_enabled = False
                sampled_token_probs.clear()
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    # This changes only the explicit Vanilla/full-vocabulary/q fast path.  A
    # filtered, mixed, non-Vanilla, grad-enabled, or fallback sampler keeps the
    # established out-of-place apply_step_logits semantics.
    reuse_corrected_carrier = False
    if envs.SGLANG_DSPARK_INPLACE_MARKOV_CARRIER.get():
        # Keep this import out of the default path and avoid coupling module
        # initialization to models.dspark's class construction.
        from sglang.srt.models.dspark import VanillaMarkov

        reuse_corrected_carrier = bool(
            type(markov_head) is VanillaMarkov
            and not torch.is_grad_enabled()
            and pure_temperature
            and full_vocab
            and q_sampling_enabled
            and base_logits.ndim == 3
            and base_logits.is_contiguous()
        )
    if reuse_corrected_carrier:
        draft_tokens, corrected_logits = markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
            reuse_corrected_carrier=True,
        )
    else:
        draft_tokens, corrected_logits = markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
        )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
        draft_token_probs=(
            None
            if sampled_token_probs is None or not q_sampling_enabled
            else torch.stack(sampled_token_probs, dim=1)
        ),
    )
