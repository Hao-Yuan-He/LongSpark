"""Opt-in, shared FA3 settings for fixed-width LongSpark/DSpark verification."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class VerifyFA3Options:
    reuse_metadata: bool = False
    num_splits: int | None = None
    pack_gqa: bool | None = None

    @classmethod
    def from_environment(cls):
        reuse = os.getenv("SGLANG_SPEC_VERIFY_FA3_REUSE_METADATA", "0")
        splits = os.getenv("SGLANG_SPEC_VERIFY_FA3_NUM_SPLITS")
        pack = os.getenv("SGLANG_SPEC_VERIFY_FA3_PACK_GQA", "auto")
        if reuse not in {"0", "1"} or pack not in {"auto", "0", "1"}:
            raise ValueError("FA3 verify metadata expects 0/1; pack_gqa expects auto/0/1")
        splits = None if splits is None else int(splits)
        if splits is not None and splits not in {0, 1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("Unsupported FA3 verify split count")
        return cls(reuse == "1", splits, None if pack == "auto" else pack == "1")

    @property
    def requested(self):
        return self.reuse_metadata or self.num_splits is not None or self.pack_gqa is not None


def prepare_verify_scheduler_metadata(backend, metadata):
    """Match the forward kernel's static allocation bound and live GPU lengths.

    The page-table extent must match forward's split-workspace bound, including
    graph padding. Using only the longest live request can underallocate splits.
    Keep each captured metadata tensor's address stable across subsequent rounds.
    """
    options = backend.verify_fa3_options
    if not backend.verify_fa3_eligible or not options.reuse_metadata:
        return
    plan = backend._get_scheduler_metadata(
        batch_size=metadata.cache_seqlens_int32.numel(),
        max_seqlen_q=backend.speculative_num_draft_tokens,
        max_seqlen_k=metadata.page_table.shape[1] * backend.page_size,
        num_heads=backend.num_attention_heads,
        num_heads_k=backend.num_kv_heads,
        headdim=backend.head_dim,
        cache_seqlens=metadata.cache_seqlens_int32,
        qkv_dtype=backend.kv_cache_dtype,
        cu_seqlens_q=metadata.cu_seqlens_q,
        page_size=backend.page_size,
        causal=True,
        has_softcap=False,
        num_splits=(backend.num_splits if options.num_splits is None else options.num_splits),
        pack_gqa=options.pack_gqa,
    )
    if metadata.scheduler_metadata is None:
        metadata.scheduler_metadata = plan
    else:
        if metadata.scheduler_metadata.shape != plan.shape:
            raise RuntimeError("FA3 verify scheduler metadata changed captured shape")
        metadata.scheduler_metadata.copy_(plan)
