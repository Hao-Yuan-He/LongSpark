from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_free_kv_info import DraftFreeKVVerifyInput
from sglang.srt.speculative.draft_free_kv_mask_policy import (
    resolve_draft_free_kv_verify_mask_policy,
)
from sglang.srt.speculative.draft_free_kv_runner import TorchCheckpointHeadRunner, TorchKVReader
from sglang.srt.speculative.dspark_components.dspark_info import DraftBlockResult
from sglang.srt.speculative.global16_raw256 import (
    RAW256_ROWS,
    Global16Raw256PagedDraftContext,
    Global16Raw256VerifyContext,
    select_global16_prefill_rows,
    validate_global16_raw256_runtime_contract,
)
from sglang.srt.speculative.global16_proposal_cuda_graph import (
    Global16ProposalCudaGraphRunner,
    global16_proposal_cuda_graph_enabled,
    resolve_global16_proposal_graph_batch_sizes,
)
from sglang.srt.speculative.global16_markov_precision import (
    install_native_target_verify_vocab_gemm,
)
from sglang.srt.speculative.draft_free_kv_utils import (
    DraftFreeKVRuntimeConfig,
    load_manifest,
    reshape_selected_target_hidden_states,
    resolve_selected_layers,
    resolve_window_size,
)
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_kernel.flash_attn import flash_attn_with_kvcache

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def resolve_extend_hidden_row_lengths(batch) -> list[int]:
    """Return the row layout emitted by the actual radix-prefix extend."""

    lengths = getattr(batch, "extend_lens", None)
    if lengths is None:
        lengths = getattr(batch, "extend_seq_lens_cpu", None)
    if lengths is None:
        lengths = getattr(batch, "extend_seq_lens", None)
    if lengths is None:
        lengths = getattr(batch, "seq_lens_cpu", None)
    if lengths is None:
        raise RuntimeError("DRAFT_FREE_KV extend batch has no row lengths")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.detach().cpu().tolist()
    return [int(length) for length in lengths]


def build_multistep_verify_tokens(
    head_runner: TorchCheckpointHeadRunner,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    block_size: int,
    num_steps: int,
    current_token_ids: torch.Tensor,
    current_position_ids: Optional[torch.Tensor] = None,
    conditioning_hidden_states: Optional[torch.Tensor] = None,
    global16_paged_context=None,
) -> torch.Tensor:
    if block_size <= 1:
        raise RuntimeError("DRAFT_FREE_KV requires block_size > 1")
    if num_steps <= 0:
        raise RuntimeError("DRAFT_FREE_KV requires speculative_num_steps >= 1")

    current = current_token_ids
    proposal_chunks = []
    for step_index in range(num_steps):
        step_position_ids = None
        if current_position_ids is not None:
            step_position_ids = current_position_ids + step_index * (block_size - 1)
        proposals = head_runner.propose(
            keys,
            values,
            attention_mask,
            block_size=block_size,
            conditioning_hidden_states=conditioning_hidden_states,
            current_token_ids=current,
            current_position_ids=step_position_ids,
            global16_paged_context=global16_paged_context,
        )
        if proposals.ndim != 2:
            raise RuntimeError(
                f"DRAFT_FREE_KV proposals must be [bs, block], got {tuple(proposals.shape)}"
            )
        if proposals.shape[1] < block_size - 1:
            raise RuntimeError("DRAFT_FREE_KV head returned too few proposal tokens")
        step_tokens = proposals[:, : block_size - 1].to(
            device=current_token_ids.device,
            dtype=torch.int64,
        )
        proposal_chunks.append(step_tokens)
        current = step_tokens[:, -1]

    return torch.cat([current_token_ids[:, None], *proposal_chunks], dim=1)


class DraftFreeKVWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = target_worker.device
        self.manifest = load_manifest(server_args.speculative_draft_model_path)
        self.window_size = resolve_window_size(server_args.draft_free_window_size, self.manifest)
        self.selected_layers = resolve_selected_layers(server_args.draft_free_selected_layers, self.manifest)
        self.runtime_config = DraftFreeKVRuntimeConfig.from_server_args(server_args, self.manifest)
        self.block_size = self.runtime_config.block_size
        self.speculative_num_draft_tokens = self.block_size
        self.speculative_num_steps = self.runtime_config.num_steps
        self.verify_token_num = self.runtime_config.verify_token_num
        self._fixed_width_linear_verify = (
            self.speculative_num_steps == 1
            and self.verify_token_num == self.block_size
        )
        self._verify_mask_policy_logged = False
        self.head_runner = TorchCheckpointHeadRunner(
            self.manifest,
            target_worker,
            window_size=self.window_size,
        )
        draft_model = getattr(self.head_runner.head, "draft_model", None)
        draft_config = getattr(draft_model, "config", None)
        self.uses_global16_raw256 = bool(
            getattr(draft_config, "global16_raw256_enabled", False)
        )
        self._dfk_fused_verify_outputs = _env_flag(
            "DFK_FUSED_VERIFY_OUTPUTS", default=self.uses_global16_raw256
        )
        self._dfk_fast_metadata = self.uses_global16_raw256 and _env_flag(
            "DFK_GLOBAL16_FAST_METADATA", default=True
        )
        self._dfk_verify_position_offsets = (
            torch.arange(self.verify_token_num, dtype=torch.int64, device=self.device)[None, :]
            if self._dfk_fast_metadata else None
        )
        logger.info(
            "DFK_DISPATCH fused_verify_outputs=%d fast_metadata=%d",
            int(self._dfk_fused_verify_outputs), int(self._dfk_fast_metadata),
        )
        self.native_target_verify_vocab_gemm_enabled = False
        if self.uses_global16_raw256 and _env_flag(
            "DFK_GLOBAL16_NATIVE_TARGET_VERIFY_VOCAB_GEMM"
        ):
            self.native_target_verify_vocab_gemm_enabled = (
                install_native_target_verify_vocab_gemm(
                    self.model_runner.model,
                    compare=_env_flag(
                        "DFK_GLOBAL16_NATIVE_TARGET_VERIFY_VOCAB_GEMM_COMPARE"
                    ),
                )
            )
            if not self.native_target_verify_vocab_gemm_enabled:
                raise RuntimeError(
                    "Global16 native Target-verify vocabulary GEMM was requested "
                    "but the Target logits processor is unsupported"
                )
            logger.info(
                "Enabled LongSpark-only native CUDA GEMM for Target-verify "
                "full-vocabulary logits"
            )
        # The official DSpark scheduler constructs a speculative worker before
        # its normal init_memory_pools() phase.  LongSpark's request-slot
        # state is sized from that pool, so materialize the shared target pool
        # here after the draft checkpoint has loaded.  The scheduler sees the
        # initialized target pool later and does not allocate it again.
        if self.model_runner.req_to_token_pool is None:
            target_worker.alloc_memory_pool()
        if self.uses_global16_raw256:
            validate_global16_raw256_runtime_contract(
                page_size=self.page_size,
                selected_layer_ids=self.selected_layers,
                configured_layer_ids=draft_config.target_layer_ids,
                raw_window_size=self.window_size,
                configured_raw_rows=draft_config.raw_target_rows,
                model_local_rows=draft_config.block_size,
                # The runtime calls this verifier width `block_size`, while
                # the Qwen head emits `block_size - 1` Local7 proposals.
                runtime_verify_width=self.block_size,
                verify_token_num=self.verify_token_num,
                tp_size=int(getattr(self.model_runner, "tp_size", 1)),
            )
            self.kv_reader = None
            reference = self._global16_reference()
            learned_query = reference.global_query_generator.learned_query
            # Follow the request pool's addressable rows.  Some SGLang
            # versions reserve row 0 for graph padding and expose active
            # request slots as 1..size.
            capacity = int(
                self.model_runner.req_to_token_pool.req_to_token.shape[0]
            )
            self._global16_query_template = learned_query.detach()
            self._global16_state_output_pool = torch.empty(
                capacity,
                *learned_query.shape,
                dtype=torch.float32,
                device=self.device,
            )
            self._global16_state_lse_pool = torch.empty(
                capacity,
                learned_query.shape[0],
                learned_query.shape[1],
                learned_query.shape[2],
                dtype=torch.float32,
                device=self.device,
            )
            self._global16_score_workspace = torch.empty(
                capacity,
                len(self.selected_layers),
                learned_query.shape[1],
                learned_query.shape[2],
                self.verify_token_num,
                dtype=torch.float32,
                device=self.device,
            )
            self._global16_accept_offset_workspace = torch.empty(
                capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._global16_request_slot_workspace = torch.empty(
                capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._global16_request_slot_long_workspace = torch.empty(
                capacity,
                dtype=torch.int64,
                device=self.device,
            )
            self._global16_raw_page_table_workspace = torch.empty(
                capacity,
                RAW256_ROWS,
                dtype=torch.int32,
                device=self.device,
            )
            self._global16_raw_length_workspace = torch.empty(
                capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._global16_next_generation = 1
            self._global16_slot_owners: dict[int, tuple[str, int]] = {}
            self._global16_layer_objects: dict[int, object] = {}
            self._global16_position_cache_layers = 0
            self._global16_accepted_merge_layers = 0
            self._global16_verify_commits = 0
            self._global16_max_verify_batch = 0
            self._global16_fused_draft_merge_layers = 0
            self._global16_fused_all_attention_layers = 0
            self._global16_fallback_draft_merge_layers = 0
            self._global16_shared_draft_input_layers = 0
            self._global16_shared_draft_input_builds = 0
            self._global16_target_graph_contexts: dict[
                int, Global16Raw256VerifyContext
            ] = {}
            self._global16_target_graph_replays = 0
            self.model_runner.global16_raw256_cuda_graph_context_factory = (
                self._new_global16_target_graph_capture_context
            )
            self._global16_fixed_cost_audit = _env_flag("DFK_FIXED_COST_AUDIT")
            self._global16_fixed_cost_prefill_requests = 0
            self._global16_fixed_cost_request_layer_inits = 0
            self._global16_fixed_cost_prefix_rows = 0
            self._global16_fixed_cost_query_key_pairs = 0
            self._global16_fixed_cost_update_ev_start = None
            self._global16_fixed_cost_update_ev_end = None
            self._global16_fixed_cost_post_ev_start = None
            self._global16_fixed_cost_post_ev_end = None
            # Parity tracing is deliberately separate from the fixed-cost
            # profiler: copying proposal IDs to the host is useful for a
            # correctness probe but would contaminate latency evidence.
            self._global16_parity_trace = _env_flag("DFK_PARITY_TRACE")
        else:
            self.kv_reader = TorchKVReader(target_worker)
        self.uses_hidden_conditioning = bool(
            getattr(self.head_runner.head, "requires_hidden_conditioning", False)
        )
        request_state_capacity = int(
            self.model_runner.req_to_token_pool.req_to_token.shape[0]
        )
        self._dfk_current_token_pool = torch.empty(
            request_state_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        self._dfk_current_token_valid = torch.zeros(
            request_state_capacity,
            dtype=torch.bool,
            device=self.device,
        )
        self._dfk_hidden_state_pool = None
        self._dfk_hidden_state_valid = None
        if self.uses_hidden_conditioning:
            draft_model = getattr(self.head_runner.head, "draft_model", None)
            draft_config = getattr(draft_model, "config", None)
            hidden_size = int(
                getattr(
                    draft_config,
                    "hidden_size",
                    getattr(self.model_runner.model_config, "hidden_size", 0),
                )
            )
            if hidden_size <= 0:
                raise RuntimeError(
                    "DRAFT_FREE_KV hidden-conditioned head has no hidden size"
                )
            self._dfk_hidden_state_pool = torch.empty(
                request_state_capacity,
                len(self.selected_layers),
                hidden_size,
                dtype=self.model_runner.model_config.dtype,
                device=self.device,
            )
            self._dfk_hidden_state_valid = torch.zeros(
                request_state_capacity,
                dtype=torch.bool,
                device=self.device,
            )
        self._global16_proposal_graph = None
        self._global16_sampling_graph = None
        if (
            self.uses_global16_raw256
            and global16_proposal_cuda_graph_enabled(server_args)
        ):
            if self.speculative_num_steps != 1:
                raise NotImplementedError(
                    "Global16 proposal CUDA graph currently requires one "
                    "speculative step"
                )
            if torch.device(self.device).type != "cuda":
                raise RuntimeError(
                    "Global16 proposal CUDA graph requires a CUDA device"
                )
            graph_batch_sizes = resolve_global16_proposal_graph_batch_sizes(
                server_args,
                capacity=self._global16_state_output_pool.shape[0],
            )
            graph_kwargs = dict(
                device=self.device,
                capture_batch_sizes=graph_batch_sizes,
                context_factory=self._new_global16_proposal_graph_context,
                selected_layer_count=len(self.selected_layers),
                conditioning_tail_shape=(
                    (
                        len(self.selected_layers),
                        int(self.model_runner.model_config.hidden_size),
                    )
                    if self.uses_hidden_conditioning
                    else None
                ),
                conditioning_dtype=self.model_runner.model_config.dtype,
            )
            self._global16_proposal_graph = Global16ProposalCudaGraphRunner(
                proposal_fn=self._run_global16_proposal_graph, **graph_kwargs
            )
            if _env_flag("DFK_GLOBAL16_SAMPLING_CUDA_GRAPH", default=True):
                # Separate outputs/pools keep greedy and sampling captures
                # independent when temperatures alternate in one service.
                self._global16_sampling_graph = Global16ProposalCudaGraphRunner(
                    proposal_fn=self._run_global16_sampling_graph, **graph_kwargs
                )
        logger.info(
            "DFK_PROPOSAL_GRAPH greedy=%d sampling_trunk=%d",
            self._global16_proposal_graph is not None,
            self._global16_sampling_graph is not None,
        )
        self.spec_timing = getattr(server_args, "spec_timing", False)
        self._spec_timing_warmup_steps = int(
            os.environ.get("SPEC_TIMING_WARMUP_STEPS", "5")
        )
        self._spec_timing_max_rounds = int(
            os.environ.get("SPEC_TIMING_MAX_ROUNDS", "0")
        )
        self._spec_timing_step_counter = 0
        self.spec_timing_detail = self.spec_timing and (
            os.environ.get("DFK_SPEC_TIMING_DETAIL", "").lower()
            in {"1", "true", "yes", "on"}
        )
        if self.spec_timing and self.tp_rank == 0:
            logger.info(
                "DRAFT_FREE_KV spec_timing enabled (warmup_steps=%d)",
                self._spec_timing_warmup_steps,
            )
        if self.spec_timing_detail and self.tp_rank == 0:
            logger.info("DRAFT_FREE_KV detailed spec_timing enabled.")
        if self.spec_timing:
            with torch.cuda.device(self.device):
                self._spec_draft_ev_start = torch.cuda.Event(enable_timing=True)
                self._spec_draft_ev_end = torch.cuda.Event(enable_timing=True)
                self._spec_verify_ev_start = torch.cuda.Event(enable_timing=True)
                self._spec_verify_ev_end = torch.cuda.Event(enable_timing=True)
                self._spec_cycle_ev_start = torch.cuda.Event(enable_timing=True)
                self._spec_cycle_ev_end = torch.cuda.Event(enable_timing=True)
                self._spec_detail_events = (
                    {
                        key: torch.cuda.Event(enable_timing=True)
                        for key in (
                            "start",
                            "kv_read_ms",
                            "hidden_condition_ms",
                            "current_token_ms",
                            "head_propose_ms",
                            "positions_ms",
                            "spec_info_ms",
                            "prepare_verify_ms",
                        )
                    }
                    if self.spec_timing_detail
                    else None
                )
        if (
            self.uses_global16_raw256
            and self._global16_fixed_cost_audit
            and self.tp_rank == 0
        ):
            with torch.cuda.device(self.device):
                self._global16_fixed_cost_update_ev_start = torch.cuda.Event(
                    enable_timing=True
                )
                self._global16_fixed_cost_update_ev_end = torch.cuda.Event(
                    enable_timing=True
                )
                self._global16_fixed_cost_post_ev_start = torch.cuda.Event(
                    enable_timing=True
                )
                self._global16_fixed_cost_post_ev_end = torch.cuda.Event(
                    enable_timing=True
                )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DRAFT_FREE_KV worker. checkpoint=%s block_size=%d steps=%d verify_token_num=%d window_size=%d selected_layers=%s",
                self.manifest.path,
                self.block_size,
                self.speculative_num_steps,
                self.verify_token_num,
                self.window_size,
                self.selected_layers,
            )
            if self.uses_global16_raw256:
                state_bytes = (
                    self._global16_state_output_pool.numel()
                    * self._global16_state_output_pool.element_size()
                    + self._global16_state_lse_pool.numel()
                    * self._global16_state_lse_pool.element_size()
                )
                logger.info(
                    "Global16 incremental request-slot state active: "
                    "capacity=%d state_bytes=%d selected_layers=%d",
                    self._global16_state_output_pool.shape[0],
                    state_bytes,
                    len(self.selected_layers),
                )
                if self._global16_proposal_graph is not None:
                    graph_batch_sizes = (
                        self._global16_proposal_graph.capture_batch_sizes
                    )
                    logger.info(
                        "Global16 proposal CUDA graph enabled: "
                        "batch_tiers=%d min_bs=%d max_bs=%d",
                        len(graph_batch_sizes),
                        min(graph_batch_sizes),
                        max(graph_batch_sizes),
                    )
                if self.uses_hidden_conditioning:
                    logger.info(
                        "DRAFT_FREE_KV request-slot hidden-state pool active: "
                    "capacity=%d selected_layers=%d",
                    request_state_capacity,
                        len(self.selected_layers),
                    )

    def __getattr__(self, name):
        if name == "target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # The target pool is deliberately shared and was allocated during
        # construction above.  Accept the official scheduler's lifecycle
        # callback without replacing any of its pool objects.
        del memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        if self.model_runner.req_to_token_pool is None:
            self.target_worker.alloc_memory_pool()

    def on_verify_complete_cpu(self, num_correct_drafts_per_req_cpu, batch_size):
        """V2 scheduler callback; LongSpark has no adaptive proposal controller."""

        del num_correct_drafts_per_req_cpu, batch_size

    def clear_cache_pool(self):
        if self.uses_global16_raw256:
            proposal_graph_stats = (
                self._global16_proposal_graph.stats()
                if self._global16_proposal_graph is not None
                else {
                    "captures": 0,
                    "replays": 0,
                    "fallbacks": 0,
                    "captured_batch_sizes": (),
                }
            )
            if self.tp_rank == 0:
                logger.info(
                    "Global16 incremental stats: "
                    "position_cache_layers=%d accepted_merge_layers=%d "
                    "verify_commits=%d max_verify_batch=%d "
                    "fused_draft_merge_layers=%d "
                    "fused_all_attention_layers=%d "
                    "fallback_draft_merge_layers=%d "
                    "shared_draft_input_layers=%d "
                    "shared_draft_input_builds=%d "
                    "proposal_graph_captures=%d "
                    "proposal_graph_replays=%d "
                    "proposal_graph_fallbacks=%d "
                    "proposal_graph_batch_sizes=%s "
                    "full_prefix_rows_after_prefill=0 target_kv_copy_bytes=0 "
                    "prefill_prefix_rows=%d "
                    "fixed_cost_prefill_requests=%d "
                    "fixed_cost_request_layer_inits=%d "
                    "fixed_cost_prefix_rows=%d fixed_cost_query_key_pairs=%d "
                    "postverify_delta_fa_calls=0",
                    self._global16_position_cache_layers,
                    self._global16_accepted_merge_layers,
                    self._global16_verify_commits,
                    self._global16_max_verify_batch,
                    self._global16_fused_draft_merge_layers,
                    self._global16_fused_all_attention_layers,
                    self._global16_fallback_draft_merge_layers,
                    self._global16_shared_draft_input_layers,
                    self._global16_shared_draft_input_builds,
                    proposal_graph_stats["captures"],
                    proposal_graph_stats["replays"],
                    proposal_graph_stats["fallbacks"],
                    proposal_graph_stats["captured_batch_sizes"],
                    self._global16_fixed_cost_prefix_rows,
                    self._global16_fixed_cost_prefill_requests,
                    self._global16_fixed_cost_request_layer_inits,
                    self._global16_fixed_cost_prefix_rows,
                    self._global16_fixed_cost_query_key_pairs,
                )
            self._global16_position_cache_layers = 0
            self._global16_accepted_merge_layers = 0
            self._global16_verify_commits = 0
            self._global16_max_verify_batch = 0
            self._global16_fused_draft_merge_layers = 0
            self._global16_fused_all_attention_layers = 0
            self._global16_fallback_draft_merge_layers = 0
            self._global16_shared_draft_input_layers = 0
            self._global16_shared_draft_input_builds = 0
            self._global16_target_graph_replays = 0
            self._global16_fixed_cost_prefill_requests = 0
            self._global16_fixed_cost_request_layer_inits = 0
            self._global16_fixed_cost_prefix_rows = 0
            self._global16_fixed_cost_query_key_pairs = 0
            self._global16_slot_owners.clear()
            self._global16_layer_objects.clear()
            if self._global16_proposal_graph is not None:
                self._global16_proposal_graph.reset_stats()
            if self._global16_sampling_graph is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "Global16 sampling trunk graph stats: %s",
                        self._global16_sampling_graph.stats(),
                    )
                self._global16_sampling_graph.reset_stats()
        self._dfk_current_token_valid.zero_()
        if self._dfk_hidden_state_valid is not None:
            self._dfk_hidden_state_valid.zero_()
        return None

    def _should_build_verify_custom_mask(self) -> bool:
        attention_backend = getattr(self.model_runner, "attn_backend", None)
        if attention_backend is None:
            if not self._verify_mask_policy_logged and self.tp_rank == 0:
                logger.warning(
                    "DRAFT_FREE_KV target verify mask policy: "
                    "backend=unavailable build_custom_mask=1"
                )
                self._verify_mask_policy_logged = True
            return True

        backend_name, build_custom_mask = (
            resolve_draft_free_kv_verify_mask_policy(
                attention_backend,
                topk=int(self.server_args.speculative_eagle_topk or 0),
                fixed_width_linear=self._fixed_width_linear_verify,
                cuda_graph_enabled=not bool(self.server_args.disable_cuda_graph),
                overlap_enabled=not bool(
                    self.server_args.disable_overlap_schedule
                ),
            )
        )
        if not self._verify_mask_policy_logged and self.tp_rank == 0:
            logger.info(
                "DRAFT_FREE_KV target verify mask policy: "
                "backend=%s build_custom_mask=%d fixed_width_linear=%d "
                "topk=%d cuda_graph=%d overlap=%d",
                backend_name,
                int(build_custom_mask),
                int(self._fixed_width_linear_verify),
                int(self.server_args.speculative_eagle_topk or 0),
                int(not bool(self.server_args.disable_cuda_graph)),
                int(not bool(self.server_args.disable_overlap_schedule)),
            )
            self._verify_mask_policy_logged = True
        return build_custom_mask

    def _current_tokens(
        self, batch: ScheduleBatch, *, request_slots: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        slots = request_slots
        if slots is None:
            slots = batch.req_pool_indices.to(device=self.device, dtype=torch.long)
        if slots.shape != (batch.batch_size(),):
            raise RuntimeError("DRAFT_FREE_KV current-token slots do not match batch")
        return self._dfk_current_token_pool.index_select(0, slots)

    def _cache_current_tokens(
        self,
        batch: ScheduleBatch,
        token_ids: Optional[torch.Tensor],
        *,
        request_slots: Optional[torch.Tensor] = None,
    ) -> None:
        if token_ids is None:
            return
        if token_ids.shape != (batch.batch_size(),):
            raise RuntimeError(
                "DRAFT_FREE_KV current token output must have one row per request"
            )
        slots = request_slots
        if slots is None:
            slots = batch.req_pool_indices.to(device=self.device, dtype=torch.long)
        self._dfk_current_token_pool.index_copy_(
            0,
            slots,
            token_ids.to(device=self.device, dtype=torch.int64),
        )
        self._dfk_current_token_valid.index_fill_(0, slots, True)

    def _global16_reference(self):
        if not self.uses_global16_raw256:
            raise RuntimeError("Global16 runtime is not enabled for this checkpoint")
        reference = getattr(
            getattr(self.head_runner.head, "draft_model", None),
            "global16_raw256_reference",
            None,
        )
        if reference is None:
            raise RuntimeError("Global16 checkpoint is missing its shared reference module")
        return reference

    def _new_global16_verify_context(
        self,
        batch: ScheduleBatch,
    ) -> Optional[Global16Raw256VerifyContext]:
        active_rows = select_global16_prefill_rows(
            batch.reqs, getattr(batch, "chunked_req", None)
        )
        if not active_rows:
            return None
        active_rows_tensor = torch.tensor(
            active_rows, device=self.device, dtype=torch.long
        )
        global_queries = self._global16_queries(len(active_rows))
        confirmed_prefix_lens = batch.seq_lens.index_select(0, active_rows_tensor)
        request_slots = self._global16_slots_for_batch(
            batch,
            require_committed=False,
        ).index_select(0, active_rows_tensor)
        return Global16Raw256VerifyContext(
            selected_layer_ids=self.selected_layers,
            global_queries=global_queries,
            confirmed_prefix_lens=confirmed_prefix_lens.detach().clone(),
            request_slots=request_slots,
            source_batch_rows=active_rows_tensor,
            source_batch_rows_host=active_rows,
            collect_fixed_cost_audit=self._global16_fixed_cost_audit,
        )

    def _global16_request_key(self, req) -> tuple[str, int]:
        generation = getattr(req, "_draft_free_kv_global16_generation", None)
        if generation is None:
            generation = self._global16_next_generation
            self._global16_next_generation += 1
            setattr(req, "_draft_free_kv_global16_generation", generation)
        return str(req.rid), int(generation)

    def _global16_queries(self, batch_size: int) -> torch.Tensor:
        return self._global16_query_template[None].expand(
            int(batch_size),
            -1,
            -1,
            -1,
            -1,
        )

    def _global16_slots_for_batch(
        self,
        batch: ScheduleBatch,
        *,
        require_committed: bool,
    ) -> torch.Tensor:
        batch_size = batch.batch_size()
        if batch.req_pool_indices.shape != (batch_size,):
            raise RuntimeError("Global16 request-slot metadata does not match the batch")
        slots = self._global16_request_slot_workspace[:batch_size]
        long_slots = self._global16_request_slot_long_workspace[:batch_size]
        slots.copy_(batch.req_pool_indices, non_blocking=True)
        long_slots.copy_(batch.req_pool_indices, non_blocking=True)
        for req in batch.reqs:
            slot = int(req.req_pool_idx)
            if slot < 0 or slot >= self._global16_state_output_pool.shape[0]:
                raise RuntimeError(
                    f"Global16 request {req.rid} has invalid pool slot {slot}"
                )
            key = self._global16_request_key(req)
            if require_committed and self._global16_slot_owners.get(slot) != key:
                raise RuntimeError(
                    "Global16 request generation has no committed memory state for "
                    f"request {req.rid}; prefill must complete first"
                )
        return slots

    def _new_global16_incremental_context(
        self,
        batch: ScheduleBatch,
    ) -> Global16Raw256VerifyContext:
        batch_size = batch.batch_size()
        request_slots = self._global16_slots_for_batch(
            batch,
            require_committed=True,
        )
        context = Global16Raw256VerifyContext.from_prior_pool(
            selected_layer_ids=self.selected_layers,
            global_queries=self._global16_queries(batch_size),
            state_output_pool=self._global16_state_output_pool,
            state_lse_pool=self._global16_state_lse_pool,
            request_slots=request_slots,
            layer_objects=self._global16_layer_objects,
            score_workspace=self._global16_score_workspace[:batch_size],
            accept_offset_workspace=self._global16_accept_offset_workspace[:batch_size],
        )
        context.collect_fixed_cost_audit = bool(
            self._global16_fixed_cost_audit and self.tp_rank == 0
        )
        return context

    def _new_global16_target_graph_capture_context(
        self,
        *,
        batch_size: int,
    ) -> Global16Raw256VerifyContext:
        batch_size = int(batch_size)
        context = Global16Raw256VerifyContext.from_prior_pool(
            selected_layer_ids=self.selected_layers,
            global_queries=self._global16_queries(batch_size),
            state_output_pool=self._global16_state_output_pool,
            state_lse_pool=self._global16_state_lse_pool,
            request_slots=self._global16_request_slot_workspace[:batch_size],
            layer_objects={},
            score_workspace=self._global16_score_workspace[:batch_size],
            accept_offset_workspace=self._global16_accept_offset_workspace[
                :batch_size
            ],
        )
        context.cuda_graph_capture_context = True
        self._global16_target_graph_contexts[batch_size] = context
        return context

    def _mark_global16_target_graph_scores(
        self,
        *,
        context: Global16Raw256VerifyContext,
        batch_size: int,
    ) -> None:
        graph_tier = next(
            (
                tier
                for tier in sorted(self._global16_target_graph_contexts)
                if tier >= int(batch_size)
            ),
            None,
        )
        capture_context = (
            None
            if graph_tier is None
            else self._global16_target_graph_contexts[graph_tier]
        )
        missing = (
            list(self.selected_layers)
            if capture_context is None
            else [
                layer_id
                for layer_id in self.selected_layers
                if layer_id not in capture_context.position_evidence
            ]
        )
        if missing:
            raise RuntimeError(
                "Global16 Target CUDA graph did not capture position scores "
                f"for layers {missing}"
            )
        context.mark_position_scores_captured_by_cuda_graph()
        self._global16_target_graph_replays += 1

    def _remember_global16_rows(
        self,
        batch: ScheduleBatch,
        context: Global16Raw256VerifyContext,
    ) -> None:
        if not context.committed:
            raise RuntimeError("only committed Global16 state may be retained for drafting")
        context.require_base_state()
        source_rows = context.source_batch_rows
        source_rows_host = context.source_batch_rows_host
        if not context.state_pool_backed:
            active_keys = []
            for source_row in source_rows_host:
                req = batch.reqs[source_row]
                slot = int(req.req_pool_idx)
                key = self._global16_request_key(req)
                if self._global16_slot_owners.get(slot) == key:
                    raise RuntimeError(
                        "Global16 request generation received duplicate prefill base state: "
                        f"request {req.rid}"
                    )
                active_keys.append((slot, key))
            slots = self._global16_request_slot_long_workspace[
                : batch.batch_size()
            ]
            active_slots = slots.index_select(0, source_rows)
            for layer_id, state in context.layer_states.items():
                layer_index = context._layer_index[layer_id]
                self._global16_state_output_pool[:, layer_index].index_copy_(
                    0,
                    active_slots,
                    state.output,
                )
                self._global16_state_lse_pool[:, layer_index].index_copy_(
                    0,
                    active_slots,
                    state.logsumexp,
                )
            for slot, key in active_keys:
                self._global16_slot_owners[slot] = key
        for source_row in source_rows_host:
            req = batch.reqs[source_row]
            slot = int(req.req_pool_idx)
            key = self._global16_request_key(req)
            if req.finished():
                if self._global16_slot_owners.get(slot) == key:
                    self._global16_slot_owners.pop(slot, None)
                continue
            if context.state_pool_backed:
                self._global16_slot_owners[slot] = key
        self._global16_layer_objects = dict(context.layer_objects)

    def _log_global16_prefill_audit(
        self, context: Global16Raw256VerifyContext
    ) -> None:
        if not self._global16_fixed_cost_audit or self.tp_rank != 0:
            return
        requests, request_layer_inits, prefix_rows, query_key_pairs = (
            context.fixed_cost_audit_totals()
        )
        self._global16_fixed_cost_prefill_requests += requests
        self._global16_fixed_cost_request_layer_inits += request_layer_inits
        self._global16_fixed_cost_prefix_rows += prefix_rows
        self._global16_fixed_cost_query_key_pairs += query_key_pairs
        logger.info(
            "DFK_FIXED_COST phase=prefill_summary_init requests=%d "
            "request_layer_inits=%d prefix_rows=%d query_key_pairs=%d",
            requests,
            request_layer_inits,
            prefix_rows,
            query_key_pairs,
        )

    def _paged_global16_draft_context(
        self,
        batch: ScheduleBatch,
    ) -> Global16Raw256PagedDraftContext:
        batch_size = batch.batch_size()
        request_slots = self._global16_slots_for_batch(
            batch,
            require_committed=True,
        )
        committed = Global16Raw256VerifyContext.from_committed_pool(
            selected_layer_ids=self.selected_layers,
            global_queries=self._global16_queries(batch_size),
            state_output_pool=self._global16_state_output_pool,
            state_lse_pool=self._global16_state_lse_pool,
            request_slots=request_slots,
            layer_objects=self._global16_layer_objects,
        )
        return Global16Raw256PagedDraftContext(
            verify_context=committed,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            page_size=self.page_size,
            flash_attn_with_kvcache=flash_attn_with_kvcache,
            num_splits=int(getattr(self.model_runner.attn_backend, "num_splits", 0)),
            host_seq_lens=batch.seq_lens_cpu,
            page_table_workspace=self._global16_raw_page_table_workspace[
                :batch_size
            ],
            raw_length_workspace=self._global16_raw_length_workspace[
                :batch_size
            ],
            # Positions come directly from the scheduler's non-negative
            # sequence lengths. The Draft model still checks shape, dtype,
            # and device without converting a CUDA predicate to a host bool.
            position_ids_are_scheduler_owned=True,
        )

    def _new_global16_proposal_graph_context(
        self,
        *,
        request_slots: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table_workspace: torch.Tensor,
        raw_length_workspace: torch.Tensor,
        host_seq_lens,
    ) -> Global16Raw256PagedDraftContext:
        graph_batch_size = int(request_slots.shape[0])
        committed = Global16Raw256VerifyContext.from_committed_pool(
            selected_layer_ids=self.selected_layers,
            global_queries=self._global16_queries(graph_batch_size),
            state_output_pool=self._global16_state_output_pool,
            state_lse_pool=self._global16_state_lse_pool,
            request_slots=request_slots,
            layer_objects=self._global16_layer_objects,
        )
        return Global16Raw256PagedDraftContext(
            verify_context=committed,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            page_size=self.page_size,
            flash_attn_with_kvcache=flash_attn_with_kvcache,
            num_splits=int(
                getattr(self.model_runner.attn_backend, "num_splits", 0)
            ),
            host_seq_lens=host_seq_lens,
            page_table_workspace=page_table_workspace,
            raw_length_workspace=raw_length_workspace,
            cuda_graph_static_inputs=True,
        )

    def _run_global16_sampling_graph(
        self,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        conditioning_hidden_states: Optional[torch.Tensor],
        paged_context: Global16Raw256PagedDraftContext,
    ) -> torch.Tensor:
        return self.head_runner.sampling_base_logits(
            None,
            None,
            None,
            conditioning_hidden_states=conditioning_hidden_states,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            global16_paged_context=paged_context,
        )

    def _run_global16_proposal_graph(
        self,
        current_token_ids: torch.Tensor,
        current_position_ids: torch.Tensor,
        conditioning_hidden_states: Optional[torch.Tensor],
        paged_context: Global16Raw256PagedDraftContext,
    ) -> torch.Tensor:
        return build_multistep_verify_tokens(
            self.head_runner,
            None,
            None,
            None,
            block_size=self.block_size,
            num_steps=1,
            conditioning_hidden_states=conditioning_hidden_states,
            current_token_ids=current_token_ids,
            current_position_ids=current_position_ids,
            global16_paged_context=paged_context,
        )

    def _detail_profile_mark(
        self,
        timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] | None,
        key: str,
        start: torch.cuda.Event | None,
    ) -> torch.cuda.Event | None:
        if timings is None:
            return start
        end = self._spec_detail_events[key]
        end.record()
        timings[key] = (start, end)
        return end

    def _log_global16_proposal_trace(
        self,
        batch: ScheduleBatch,
        verify_tokens: torch.Tensor,
    ) -> None:
        """Emit exact per-request Draft proposals for an exclusive parity run."""

        if not getattr(self, "_global16_parity_trace", False) or self.tp_rank != 0:
            return
        batch_size = batch.batch_size()
        if verify_tokens.ndim != 2 or verify_tokens.shape[0] != batch_size:
            raise RuntimeError("LongSpark parity trace received malformed verify tokens")
        if verify_tokens.shape[1] <= 1:
            raise RuntimeError("LongSpark parity trace contains no Draft proposals")
        seq_lens = batch.seq_lens_cpu
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.detach().cpu().tolist()
        else:
            seq_lens = list(seq_lens)
        if len(seq_lens) != batch_size or len(batch.reqs) != batch_size:
            raise RuntimeError("LongSpark parity trace metadata does not match the batch")
        proposal_rows = verify_tokens[:, 1:].detach().cpu().tolist()
        for row, (req, seq_len, proposal_ids) in enumerate(
            zip(batch.reqs, seq_lens, proposal_rows)
        ):
            logger.info(
                "DFK_PARITY phase=draft_proposal row=%d request_slot=%d "
                "seq_len=%d proposal_ids=%s",
                row,
                int(req.req_pool_idx),
                int(seq_len),
                ",".join(str(int(token)) for token in proposal_ids),
            )

    def _log_global16_verify_trace(
        self,
        batch: ScheduleBatch,
        accept_lens: torch.Tensor,
    ) -> None:
        """Complete each parity round with its exact Target acceptance result."""

        if not getattr(self, "_global16_parity_trace", False) or self.tp_rank != 0:
            return
        batch_size = batch.batch_size()
        if accept_lens.shape != (batch_size,):
            raise RuntimeError("LongSpark parity accept lengths do not match the batch")
        seq_lens = batch.seq_lens_cpu
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.detach().cpu().tolist()
        else:
            seq_lens = list(seq_lens)
        accepted = accept_lens.detach().cpu().tolist()
        if len(seq_lens) != batch_size or len(batch.reqs) != batch_size:
            raise RuntimeError("LongSpark parity verify metadata does not match the batch")
        for row, (req, seq_len, accept_len) in enumerate(
            zip(batch.reqs, seq_lens, accepted)
        ):
            logger.info(
                "DFK_PARITY phase=verify_result row=%d request_slot=%d "
                "seq_len=%d accept_len=%d",
                row,
                int(req.req_pool_idx),
                int(seq_len),
                int(accept_len),
            )

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return None
        draft_input = batch.spec_info
        sampling_info = batch.sampling_info
        sampling_enabled = (
            sampling_info is not None and not sampling_info.is_all_greedy
        )
        timings = {} if self.spec_timing_detail else None
        if timings is not None:
            mark_start = self._spec_detail_events["start"]
            mark_start.record()
        else:
            mark_start = None
        global16_paged_context = None
        request_slots = None
        if self.uses_global16_raw256:
            # This object owns only page-table metadata and direct cache
            # references.  In particular, it never calls TorchKVReader.read.
            global16_paged_context = self._paged_global16_draft_context(batch)
            if self._dfk_fast_metadata:
                # The paged context validated this round's request generations
                # and populated both slot workspaces once.
                request_slots = self._global16_request_slot_long_workspace[
                    :batch.batch_size()
                ]
            keys = values = attention_mask = None
        else:
            keys, values, attention_mask = self.kv_reader.read(
                batch, selected_layers=self.selected_layers, window_size=self.window_size
            )
        mark_start = self._detail_profile_mark(timings, "kv_read_ms", mark_start)
        conditioning_hidden_states = None
        if self.uses_hidden_conditioning:
            if self._dfk_hidden_state_pool is None:
                raise RuntimeError(
                    "DRAFT_FREE_KV hidden-conditioned state pool is unavailable"
                )
            conditioning_hidden_states = self._dfk_hidden_state_pool.index_select(
                0,
                request_slots if request_slots is not None else
                batch.req_pool_indices.to(device=self.device, dtype=torch.long),
            )
        mark_start = self._detail_profile_mark(
            timings,
            "hidden_condition_ms",
            mark_start,
        )
        current = self._current_tokens(batch, request_slots=request_slots)
        mark_start = self._detail_profile_mark(timings, "current_token_ms", mark_start)
        current_position_ids = batch.seq_lens.to(
            device=batch.device,
            dtype=torch.long,
        )
        proposal_graph_used = False
        verify_tokens = None
        draft_block = None
        if (
            not sampling_enabled
            and
            self._global16_proposal_graph is not None
            and global16_paged_context is not None
        ):
            verify_tokens = self._global16_proposal_graph.try_propose(
                current_token_ids=current,
                current_position_ids=current_position_ids,
                conditioning_hidden_states=conditioning_hidden_states,
                paged_context=global16_paged_context,
            )
            proposal_graph_used = verify_tokens is not None
        if verify_tokens is None:
            if sampling_enabled:
                if self.speculative_num_steps != 1:
                    raise RuntimeError(
                        "LongSpark temperature sampling currently requires one "
                        "seven-token proposal block"
                    )
                base_logits = None
                if (
                    self._global16_sampling_graph is not None
                    and global16_paged_context is not None
                ):
                    base_logits = self._global16_sampling_graph.try_propose(
                        current_token_ids=current,
                        current_position_ids=current_position_ids,
                        conditioning_hidden_states=conditioning_hidden_states,
                        paged_context=global16_paged_context,
                    )
                    proposal_graph_used = base_logits is not None
                (
                    proposal_tokens,
                    corrected_logits,
                    temperatures,
                    greedy_mask,
                    draft_token_probs,
                    candidate_token_ids,
                ) = self.head_runner.propose_for_sampling(
                    keys,
                    values,
                    attention_mask,
                    block_size=self.block_size,
                    sampling_info=sampling_info,
                    base_logits=base_logits,
                    conditioning_hidden_states=conditioning_hidden_states,
                    current_token_ids=current,
                    current_position_ids=current_position_ids,
                    global16_paged_context=global16_paged_context,
                )
                if candidate_token_ids is not None:
                    raise RuntimeError(
                        "LongSpark temperature sampling requires the full draft "
                        "vocabulary; set the proposal vocabulary prefix size to zero"
                    )
                verify_tokens = torch.cat(
                    (current[:, None], proposal_tokens),
                    dim=1,
                )
                draft_block = DraftBlockResult(
                    draft_tokens=proposal_tokens,
                    corrected_logits=corrected_logits,
                    greedy_mask=greedy_mask,
                    temperatures=temperatures,
                    draft_token_probs=draft_token_probs,
                )
            else:
                verify_tokens = build_multistep_verify_tokens(
                    self.head_runner,
                    keys,
                    values,
                    attention_mask,
                    block_size=self.block_size,
                    num_steps=self.speculative_num_steps,
                    conditioning_hidden_states=conditioning_hidden_states,
                    current_token_ids=current,
                    current_position_ids=current_position_ids,
                    global16_paged_context=global16_paged_context,
                )
        if self.uses_global16_raw256:
            self._log_global16_proposal_trace(batch, verify_tokens)
        if global16_paged_context is not None:
            fused_layers = (
                len(self.selected_layers)
                if proposal_graph_used
                else int(global16_paged_context.fused_merge_layers)
            )
            self._global16_fused_draft_merge_layers += fused_layers
            self._global16_fused_all_attention_layers += (
                0
                if proposal_graph_used
                else int(global16_paged_context.fused_all_attention_layers)
            )
            self._global16_fallback_draft_merge_layers += (
                len(self.selected_layers) - fused_layers
            )
            self._global16_shared_draft_input_layers += (
                len(self.selected_layers)
                if proposal_graph_used
                else int(global16_paged_context.shared_draft_input_layers)
            )
            self._global16_shared_draft_input_builds += (
                1
                if proposal_graph_used
                else int(global16_paged_context.shared_draft_input_builds)
            )
        mark_start = self._detail_profile_mark(timings, "head_propose_ms", mark_start)
        verify_token_num = int(verify_tokens.shape[1])
        position_offsets = self._dfk_verify_position_offsets
        if position_offsets is None or position_offsets.shape[1] != verify_token_num:
            position_offsets = torch.arange(
                verify_token_num, dtype=torch.int64, device=batch.device
            )[None, :]
        positions = (
            batch.seq_lens[:, None]
            + position_offsets
        ).reshape(-1)
        mark_start = self._detail_profile_mark(timings, "positions_ms", mark_start)
        batch.spec_algorithm = SpeculativeAlgorithm.DRAFT_FREE_KV
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = DraftFreeKVVerifyInput(
            draft_token=verify_tokens.reshape(-1),
            positions=positions,
            draft_token_num=verify_token_num,
            draft_block=draft_block,
            draft_input=draft_input if sampling_enabled else None,
            use_fused_outputs=self._dfk_fused_verify_outputs,
        )
        if self.uses_hidden_conditioning:
            batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        if self.uses_global16_raw256:
            # Restore the prior output/LSE.  Each selected Target layer will
            # cache only this round's eight post-RoPE K scores; no historical
            # Global query is appended to verify FA3.
            if self._dfk_fast_metadata:
                # Proposal has consumed this committed pool view. Transfer it
                # within the same round, before requests can be filtered or
                # merged. The next round validates its slots anew.
                context = global16_paged_context.verify_context.begin_incremental_verify(
                    score_workspace=self._global16_score_workspace[:batch.batch_size()],
                    accept_offset_workspace=self._global16_accept_offset_workspace[
                        :batch.batch_size()
                    ],
                )
                context.collect_fixed_cost_audit = bool(
                    self._global16_fixed_cost_audit and self.tp_rank == 0
                )
                batch.global16_raw256_context = context
            else:
                batch.global16_raw256_context = self._new_global16_incremental_context(
                    batch
                )
        mark_start = self._detail_profile_mark(timings, "spec_info_ms", mark_start)
        batch.spec_info.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=self._should_build_verify_custom_mask(),
        )
        mark_start = self._detail_profile_mark(
            timings, "prepare_verify_ms", mark_start
        )
        if timings is not None:
            mark_start.synchronize()
            resolved_timings = {
                key: start.elapsed_time(end)
                for key, (start, end) in timings.items()
            }
            resolved_timings["batch_size"] = float(batch.batch_size())
            resolved_timings["verify_token_num"] = float(verify_token_num)
            return resolved_timings
        return None

    def _cache_target_hidden_states(
        self,
        batch: ScheduleBatch,
        hidden_states: Optional[torch.Tensor],
        *,
        last_hidden_states: Optional[torch.Tensor] = None,
        accept_lens: Optional[torch.Tensor] = None,
        draft_token_num: Optional[int] = None,
        request_slots: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.uses_hidden_conditioning or hidden_states is None:
            return

        draft_model = getattr(self.head_runner.head, "draft_model", None)
        draft_config = getattr(draft_model, "config", None)
        if draft_config is not None:
            hidden_size = int(
                getattr(
                    draft_config,
                    "hidden_size",
                    getattr(self.model_runner.model_config, "hidden_size", 0),
                )
            )
            hidden_states = reshape_selected_target_hidden_states(
                hidden_states,
                last_hidden_states,
                selected_layers=len(self.selected_layers),
                hidden_size=hidden_size,
            )

        if self._dfk_hidden_state_pool is None:
            raise RuntimeError(
                "DRAFT_FREE_KV hidden-conditioned state pool is unavailable"
            )

        def cache_rows(rows: torch.Tensor) -> None:
            if rows.shape[0] != batch.batch_size():
                raise RuntimeError(
                    "DRAFT_FREE_KV hidden-state rows do not match request slots"
                )
            slots = request_slots
            if slots is None:
                slots = batch.req_pool_indices.to(device=self.device, dtype=torch.long)
            self._dfk_hidden_state_pool.index_copy_(
                0,
                slots,
                rows.detach().to(dtype=self._dfk_hidden_state_pool.dtype),
            )
            assert self._dfk_hidden_state_valid is not None
            self._dfk_hidden_state_valid.index_fill_(0, slots, True)

        if accept_lens is not None:
            tokens_per_req = int(draft_token_num or self.speculative_num_draft_tokens)
            if accept_lens.shape != (batch.batch_size(),):
                raise RuntimeError(
                    "DRAFT_FREE_KV accepted lengths do not match hidden rows"
                )
            rows = torch.arange(batch.batch_size(), device=self.device)
            rows = rows * tokens_per_req + accept_lens.to(
                device=self.device,
                dtype=torch.long,
            ) - 1
            cache_rows(hidden_states.index_select(0, rows))
            return

        if batch.forward_mode.is_decode():
            if hidden_states.shape[0] != len(batch.reqs):
                raise RuntimeError(
                    "DRAFT_FREE_KV decode hidden rows must match request count"
                )
            cache_rows(hidden_states)
            return

        if batch.forward_mode.is_extend():
            lengths = resolve_extend_hidden_row_lengths(batch)
            if hidden_states.shape[0] == len(batch.reqs):
                cache_rows(hidden_states)
                return
            if sum(lengths) != hidden_states.shape[0]:
                raise RuntimeError(
                    "DRAFT_FREE_KV extend hidden rows do not match token lengths: "
                    f"rows={hidden_states.shape[0]} lengths={lengths}"
                )
            end = 0
            selected_rows = []
            for length in lengths:
                end += length
                if length > 0:
                    selected_rows.append(end - 1)
            if len(selected_rows) != batch.batch_size():
                raise RuntimeError(
                    "DRAFT_FREE_KV extend contains an empty request"
                )
            cache_rows(
                hidden_states.index_select(
                    0,
                    torch.tensor(
                        selected_rows,
                        device=self.device,
                        dtype=torch.long,
                    ),
                )
            )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        time_draft = (
            self.spec_timing
            and not batch.forward_mode.is_extend()
            and not batch.forward_mode.is_idle()
            and (
                self._spec_timing_max_rounds <= 0
                or self._spec_timing_step_counter
                < self._spec_timing_warmup_steps + self._spec_timing_max_rounds
            )
        )
        if time_draft:
            self._spec_cycle_ev_start.record()
            self._spec_draft_ev_start.record()
        draft_timings = self._prepare_for_speculative_decoding(batch)
        if time_draft and batch.forward_mode.is_target_verify():
            self._spec_draft_ev_end.record()
            self._spec_draft_ev_end.synchronize()
            self._spec_timing_step_counter += 1
            if self._spec_timing_step_counter > self._spec_timing_warmup_steps:
                draft_ms = self._spec_draft_ev_start.elapsed_time(
                    self._spec_draft_ev_end
                )
                if self.tp_rank == 0:
                    logger.info(
                        "SPEC_TIMING arm=longspark phase=draft "
                        "round=%d bs=%d ms=%.3f",
                        self._spec_timing_step_counter
                        - self._spec_timing_warmup_steps,
                        batch.batch_size(),
                        draft_ms,
                    )
                if self.spec_timing_detail and self.tp_rank == 0 and draft_timings:
                    logger.info(
                        "DFK_PROFILE_DRAFT bs=%d verify_tokens=%d "
                        "kv_read_ms=%.3f hidden_condition_ms=%.3f current_token_ms=%.3f "
                        "head_propose_ms=%.3f positions_ms=%.3f spec_info_ms=%.3f "
                        "prepare_verify_ms=%.3f draft_event_ms=%.3f",
                        int(draft_timings["batch_size"]),
                        int(draft_timings["verify_token_num"]),
                        draft_timings.get("kv_read_ms", 0.0),
                        draft_timings.get("hidden_condition_ms", 0.0),
                        draft_timings.get("current_token_ms", 0.0),
                        draft_timings.get("head_propose_ms", 0.0),
                        draft_timings.get("positions_ms", 0.0),
                        draft_timings.get("spec_info_ms", 0.0),
                        draft_timings.get("prepare_verify_ms", 0.0),
                        draft_ms,
                    )
        if (
            self.uses_global16_raw256
            and batch.forward_mode.is_extend_without_speculative()
        ):
            # Initial/prefill Target rows and Global rows share the same FA3
            # read.  The entire prefill is already confirmed, so no delta is
            # needed before it becomes the request's first draft state.
            batch.global16_raw256_context = self._new_global16_verify_context(batch)
        # The official DSpark suite dispatches ScheduleBatch directly to the
        # target worker; unlike the older LongSpark integration, it has no
        # wrapper conversion API.
        model_worker_batch = batch
        if self.uses_hidden_conditioning and not model_worker_batch.forward_mode.is_target_verify():
            # FULL is needed so multi-request prefill can select the last token
            # of every request for every captured target layer.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        if model_worker_batch.forward_mode.is_target_verify():
            if time_draft:
                self._spec_verify_ev_start.record()
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch, is_verify=True)
            if time_draft:
                self._spec_verify_ev_end.record()
                self._spec_verify_ev_end.synchronize()
                if self._spec_timing_step_counter > self._spec_timing_warmup_steps:
                    verify_ms = self._spec_verify_ev_start.elapsed_time(
                        self._spec_verify_ev_end
                    )
                else:
                    verify_ms = 0.0
                if self.tp_rank == 0 and verify_ms > 0.0:
                    logger.info(
                        "SPEC_TIMING arm=longspark phase=target_verify "
                        "round=%d bs=%d ms=%.3f",
                        self._spec_timing_step_counter
                        - self._spec_timing_warmup_steps,
                        batch.batch_size(),
                        verify_ms,
                    )
                if (
                    self.spec_timing_detail
                    and self.tp_rank == 0
                    and self._spec_timing_step_counter > self._spec_timing_warmup_steps
                ):
                    logger.info(
                        "DFK_PROFILE_VERIFY bs=%d target_verify_ms=%.3f cuda_graph=%s",
                        batch.batch_size(),
                        verify_ms,
                        batch_result.can_run_cuda_graph,
                    )
            verify_input: DraftFreeKVVerifyInput = model_worker_batch.spec_info
            logits_output, bonus_token_ids, _ = verify_input.verify(
                batch,
                batch_result.logits_output,
                self.page_size,
                sampling_info=batch.sampling_info,
            )
            fixed_cost_audit_payload = None
            if self.uses_global16_raw256:
                self._log_global16_verify_trace(batch, verify_input.accept_lens)
            if self.uses_global16_raw256:
                context = model_worker_batch.global16_raw256_context
                if not isinstance(context, Global16Raw256VerifyContext):
                    raise RuntimeError("Target verify returned without a Global16 transaction")
                if batch_result.can_run_cuda_graph:
                    self._mark_global16_target_graph_scores(
                        context=context,
                        batch_size=batch.batch_size(),
                    )
                # The verifier retains a fixed [batch, 8] view for the
                # incremental merge while releasing rejected rows separately.
                if self._global16_fixed_cost_audit and self.tp_rank == 0:
                    self._global16_fixed_cost_update_ev_start.record()
                context.commit_accepted_delta(
                    accepted_cache_locs=verify_input.verify_cache_locs,
                    accept_lens=verify_input.accept_lens,
                    token_to_kv_pool=self.model_runner.token_to_kv_pool,
                    page_size=self.page_size,
                )
                score_update_ms = None
                merge_update_ms = None
                if self._global16_fixed_cost_audit and self.tp_rank == 0:
                    self._global16_fixed_cost_update_ev_end.record()
                    self._global16_fixed_cost_update_ev_end.synchronize()
                    merge_update_ms = self._global16_fixed_cost_update_ev_start.elapsed_time(
                        self._global16_fixed_cost_update_ev_end
                    )
                    score_update_ms = context.fixed_cost_score_update_ms()
                cached_layers = len(context.position_evidence)
                self._global16_position_cache_layers += cached_layers
                self._global16_accepted_merge_layers += len(
                    context.selected_layer_ids
                )
                self._global16_verify_commits += 1
                self._global16_max_verify_batch = max(
                    self._global16_max_verify_batch,
                    batch.batch_size(),
                )
                if self._global16_fixed_cost_audit and self.tp_rank == 0:
                    accept_lens_cpu = verify_input.accept_lens.detach().cpu()
                    max_new_rows = int(accept_lens_cpu.max().item())
                    new_kv_rows = int(accept_lens_cpu.sum().item()) * len(
                        context.selected_layer_ids
                    )
                    state_bytes = (
                        self._global16_state_output_pool.numel()
                        * self._global16_state_output_pool.element_size()
                        + self._global16_state_lse_pool.numel()
                        * self._global16_state_lse_pool.element_size()
                    )
                    prefix_scan_invocations = len(context.joint_evidence)
                    historical_global_kv_rows = sum(
                        evidence.prefix_rows
                        for evidence in context.joint_evidence.values()
                    )
                    fixed_cost_audit_payload = {
                        "requests": batch.batch_size(),
                        "request_layer_updates": batch.batch_size()
                        * len(context.selected_layer_ids),
                        "new_kv_rows": new_kv_rows,
                        "max_new_rows": max_new_rows,
                        "prefix_scan_invocations": prefix_scan_invocations,
                        "historical_global_kv_rows": historical_global_kv_rows,
                        "state_bytes": state_bytes,
                        "score_update_ms": score_update_ms,
                        "merge_update_ms": merge_update_ms,
                    }
                    # The persistent drafter boundary/current-token caches are
                    # part of the post-verification context update.  Time them
                    # separately so the fixed-cost total cannot omit this tail.
                    self._global16_fixed_cost_post_ev_start.record()
                self._remember_global16_rows(batch, context)
                batch.global16_raw256_context = None
            request_slots = (
                self._global16_request_slot_long_workspace[:batch.batch_size()]
                if self._dfk_fast_metadata else None
            )
            self._cache_current_tokens(
                batch, bonus_token_ids, request_slots=request_slots
            )
            self._cache_target_hidden_states(
                batch,
                batch_result.logits_output.hidden_states,
                last_hidden_states=(
                    batch_result.logits_output.last_hidden_states
                    if batch_result.logits_output.last_hidden_states is not None
                    else batch_result.logits_output.teacher_hidden_states
                ),
                accept_lens=verify_input.accept_lens,
                draft_token_num=verify_input.draft_token_num,
                request_slots=request_slots,
            )
            if fixed_cost_audit_payload is not None:
                self._global16_fixed_cost_post_ev_end.record()
                self._global16_fixed_cost_post_ev_end.synchronize()
                post_state_update_ms = (
                    self._global16_fixed_cost_post_ev_start.elapsed_time(
                        self._global16_fixed_cost_post_ev_end
                    )
                )
                update_ms = (
                    fixed_cost_audit_payload["score_update_ms"]
                    + fixed_cost_audit_payload["merge_update_ms"]
                    + post_state_update_ms
                )
                logger.info(
                    "DFK_FIXED_COST phase=incremental_update requests=%d "
                    "request_layer_updates=%d new_kv_rows=%d "
                    "max_new_kv_rows_per_request=%d prefix_scan_invocations=%d "
                    "historical_global_kv_rows=%d persistent_state_bytes=%d "
                    "score_update_ms=%.6f merge_update_ms=%.6f "
                    "post_state_update_ms=%.6f update_ms=%.6f",
                    fixed_cost_audit_payload["requests"],
                    fixed_cost_audit_payload["request_layer_updates"],
                    fixed_cost_audit_payload["new_kv_rows"],
                    fixed_cost_audit_payload["max_new_rows"],
                    fixed_cost_audit_payload["prefix_scan_invocations"],
                    fixed_cost_audit_payload["historical_global_kv_rows"],
                    fixed_cost_audit_payload["state_bytes"],
                    fixed_cost_audit_payload["score_update_ms"],
                    fixed_cost_audit_payload["merge_update_ms"],
                    post_state_update_ms,
                    update_ms,
                )
            new_seq_lens = verify_input.new_seq_lens
            if new_seq_lens is None:
                new_seq_lens = batch.seq_lens + verify_input.accept_lens.to(
                    batch.seq_lens.dtype
                )
            next_draft_input = make_draft_input_v2(
                bonus_tokens=bonus_token_ids,
                new_seq_lens=new_seq_lens,
            )
            if time_draft:
                self._spec_cycle_ev_end.record()
                self._spec_cycle_ev_end.synchronize()
                if self._spec_timing_step_counter > self._spec_timing_warmup_steps:
                    cycle_ms = self._spec_cycle_ev_start.elapsed_time(
                        self._spec_cycle_ev_end
                    )
                    if self.tp_rank == 0:
                        logger.info(
                            "SPEC_TIMING arm=longspark phase=cycle "
                            "round=%d bs=%d ms=%.3f",
                            self._spec_timing_step_counter
                            - self._spec_timing_warmup_steps,
                            batch.batch_size(),
                            cycle_ms,
                        )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_input.out_tokens.reshape(-1),
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
                accept_lens=verify_input.accept_lens,
                next_draft_input=next_draft_input,
                speculative_num_draft_tokens=verify_input.draft_token_num,
                new_seq_lens=new_seq_lens,
            )
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output = batch_result.logits_output
        self._cache_current_tokens(batch, batch_result.next_token_ids)
        self._cache_target_hidden_states(
            batch,
            None if logits_output is None else logits_output.hidden_states,
            last_hidden_states=(
                None
                if logits_output is None
                else (
                    logits_output.last_hidden_states
                    if logits_output.last_hidden_states is not None
                    else logits_output.teacher_hidden_states
                )
            ),
        )
        if self.uses_global16_raw256 and model_worker_batch.global16_raw256_context is not None:
            context = model_worker_batch.global16_raw256_context
            if not isinstance(context, Global16Raw256VerifyContext):
                raise RuntimeError("Target prefill returned without a Global16 transaction")
            context.commit_base_only()
            self._remember_global16_rows(batch, context)
            self._log_global16_prefill_audit(context)
            batch.global16_raw256_context = None
        batch_result.next_draft_input = make_draft_input_v2(
            bonus_tokens=batch_result.next_token_ids,
            new_seq_lens=batch.seq_lens,
        )
        batch_result.new_seq_lens = batch.seq_lens
        batch_result.speculative_num_draft_tokens = self.block_size
        return batch_result
