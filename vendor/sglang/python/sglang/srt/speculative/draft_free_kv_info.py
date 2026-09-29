from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_free_kv_dispatch import finalize_verify_outputs
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.draft_free_kv_utils import (
    build_greedy_verify_out_tokens,
    compute_greedy_accept_len_and_bonus,
)
from sglang.srt.speculative.dspark_components.dspark_accept import (
    accept_draft_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_extend_cache_locs_func,
)


@triton.jit
def _assign_fixed_verify_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    commit_lens,
    verify_cache_locs,
    pool_len: tl.constexpr,
    verify_width: tl.constexpr,
    block: tl.constexpr,
):
    """Commit each request directly from its fixed-width verify cache row."""

    request = tl.program_id(axis=0)
    offsets = tl.arange(0, block)
    commit_len = tl.load(commit_lens + request).to(tl.int32)
    start = tl.load(start_offset + request)
    request_slot = tl.load(req_pool_indices + request)
    cache_locs = tl.load(
        verify_cache_locs + request * verify_width + offsets,
        mask=offsets < commit_len,
        other=0,
    )
    tl.store(
        req_to_token + request_slot * pool_len + start + offsets,
        cache_locs,
        mask=offsets < commit_len,
    )


def _assign_fixed_verify_cache_locs_func(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    commit_lens: torch.Tensor,
    verify_cache_locs: torch.Tensor,
    batch_size: int,
    verify_width: int,
) -> None:
    if verify_cache_locs.shape != (batch_size, verify_width):
        raise ValueError("fixed verify cache locations have the wrong shape")
    block = triton.next_power_of_2(verify_width)
    _assign_fixed_verify_cache_locs[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        commit_lens,
        verify_cache_locs,
        req_to_token.shape[1],
        verify_width=verify_width,
        block=block,
    )


@dataclass
class DraftFreeKVVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    custom_mask: torch.Tensor | None = None
    draft_block: DraftBlockResult | None = None
    draft_input: DFlashDraftInputV2 | None = None
    use_fused_outputs: bool = False

    def __post_init__(self):
        super().__init__(SpecInputType.DRAFT_FREE_KV_VERIFY)
        self.accept_lens = None
        self.verify_cache_locs = None
        self.out_tokens = None
        self.new_seq_lens = None

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return
        if page_size != 1:
            raise NotImplementedError("DRAFT_FREE_KV P0 currently supports page_size=1 only")

        # DFlashDraftInputV2.prepare_for_decode has already reserved two
        # fixed-width blocks in the shared request-to-token table.  The
        # verify block must use that reservation rather than allocate a
        # second block: manual allocation leaves the V2 high-water mark and
        # the extra slots with different owners, which leaks a block at
        # request teardown.
        batch.input_ids = self.draft_token
        bs = batch.batch_size()
        end_offset = batch.seq_lens + self.draft_token_num
        batch.out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            end_offset=end_offset,
            batch_size=bs,
            draft_token_num=self.draft_token_num,
            device=batch.device,
        )
        self.custom_mask = (
            self._build_causal_mask(batch) if build_custom_mask else None
        )

    def _build_causal_mask(self, batch: ScheduleBatch):
        chunks = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len = int(prefix_len)
            kv_len = prefix_len + q_len
            k_idx = torch.arange(kv_len, device=batch.device, dtype=torch.int32).unsqueeze(0)
            chunks.append((k_idx <= (prefix_len + q_idx)).flatten())
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.bool, device=batch.device)

    def generate_attn_arg_prefill(self, req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token):
        device = req_pool_indices.device
        bs = len(req_pool_indices)
        q_len = int(self.draft_token_num)
        qo_indptr = torch.arange(0, (bs + 1) * q_len, step=q_len, dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + q_len
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indices = torch.empty(paged_kernel_lens_sum + bs * q_len, dtype=torch.int32, device=device)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token, req_pool_indices, paged_kernel_lens, cum_kv_seq_len,
            None, kv_indices, req_to_token.size(1)
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: "DraftFreeKVVerifyInput"):
        pass

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        sampling_info=None,
    ):
        if page_size != 1:
            raise NotImplementedError("DRAFT_FREE_KV P0 currently supports page_size=1 only")

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        candidates = self.draft_token.view(bs, self.draft_token_num)
        if sampling_info is None or sampling_info.is_all_greedy:
            target_predict = torch.argmax(
                logits_output.next_token_logits,
                dim=-1,
            ).view(bs, self.draft_token_num)
            accepted_lens, bonus = compute_greedy_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
        else:
            if self.draft_block is None or self.draft_block.corrected_logits is None:
                raise RuntimeError(
                    "LongSpark temperature sampling is missing draft probabilities"
                )
            if self.draft_input is None:
                raise RuntimeError(
                    "LongSpark temperature sampling is missing scheduler metadata"
                )
            expected_shape = (
                bs,
                self.draft_token_num - 1,
                logits_output.next_token_logits.shape[-1],
            )
            if self.draft_block.corrected_logits.shape != expected_shape:
                raise RuntimeError(
                    "LongSpark temperature sampling requires full-vocabulary draft "
                    f"logits; expected {expected_shape}, got "
                    f"{tuple(self.draft_block.corrected_logits.shape)}"
                )
            if (
                self.draft_block.draft_token_probs is not None
                and self.draft_block.draft_token_probs.shape
                != expected_shape[:2]
            ):
                raise RuntimeError(
                    "LongSpark temperature sampling requires one draft probability "
                    f"per sampled token; expected {expected_shape[:2]}, got "
                    f"{tuple(self.draft_block.draft_token_probs.shape)}"
                )
            if self.draft_block.draft_token_probs is not None and (
                self.draft_block.draft_token_probs.dtype != torch.float32
                or self.draft_block.draft_token_probs.device
                != self.draft_block.corrected_logits.device
            ):
                raise RuntimeError(
                    "LongSpark sampled draft probabilities must be FP32 on the "
                    "same device as corrected logits"
                )
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=self.draft_token_num,
            )
            accepted_lens, bonus, _ = accept_draft_tokens(
                candidates=candidates,
                target_logits=logits_output.next_token_logits,
                draft_block=self.draft_block,
                sampling_info=sampling_info,
                draft_input=self.draft_input,
                gamma=self.draft_token_num - 1,
                verify_num_draft_tokens=self.draft_token_num,
            )
        if self.use_fused_outputs:
            out_tokens, commit_lens, self.new_seq_lens = finalize_verify_outputs(
                candidates=candidates,
                accept_lens=accepted_lens,
                bonus=bonus,
                prefix_lens=batch.seq_lens,
            )
        else:
            out_tokens = build_greedy_verify_out_tokens(
                candidates=candidates,
                accept_lens=accepted_lens,
                bonus=bonus,
            )
            commit_lens = accepted_lens.to(device=device, dtype=torch.int32) + 1

        # The official V2 scheduler owns output, grammar, EOS, request metrics,
        # and sequence-length accounting.  This verifier only commits the
        # accepted Target-KV rows and returns the fixed-width result layout.
        self.accept_lens = commit_lens
        self.out_tokens = out_tokens

        verify_cache_locs = batch.out_cache_loc.view(bs, self.draft_token_num)
        self.verify_cache_locs = verify_cache_locs

        # V2 owns the full reservation through req.kv_allocated_len.  Rejected
        # rows remain in that reserved range and are either reused next round
        # or released once by release_kv_cache() when the request completes.
        # Freeing them here would make the request-side release free the same
        # physical slots a second time.
        _assign_fixed_verify_cache_locs_func(
            req_pool_indices=batch.req_pool_indices,
            req_to_token=batch.req_to_token_pool.req_to_token,
            start_offset=batch.seq_lens,
            commit_lens=commit_lens,
            verify_cache_locs=verify_cache_locs,
            batch_size=bs,
            verify_width=self.draft_token_num,
        )

        # The fixed view above remains owned by ``verify_cache_locs`` for the
        # Global16 accepted-delta merge.  It must not masquerade as a compact
        # decode allocation after this point.
        batch.out_cache_loc = None
        if self.use_fused_outputs:
            next_token_ids = bonus
        else:
            actual_last = (commit_lens - 1).to(torch.long)[:, None]
            next_token_ids = out_tokens.gather(1, actual_last).squeeze(1)
        return (
            logits_output,
            next_token_ids,
            0,
        )
