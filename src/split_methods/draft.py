"""Single-device draft adapters. No target transformer is loaded here."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .config import GIB
from .target import available_port


def ring_page_table(slots, lengths, window=256):
    """Chronological physical row IDs for each request's bounded KV ring."""
    if slots.ndim != 1 or lengths.shape != slots.shape:
        raise ValueError("ring metadata requires one slot and length per request")
    raw_lengths = lengths.clamp_max(window).int()
    offsets = torch.arange(window, device=slots.device)
    pages = slots[:, None] * window + (lengths[:, None] - raw_lengths[:, None] + offsets).remainder(window)
    return pages.int(), raw_lengths


def target_modules(path, device):
    """Load only the three target modules previously shared by pointer."""
    from safetensors import safe_open
    from sglang.srt.layers.layernorm import RMSNorm

    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    index = root / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    names = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
    weights = {}
    for name in names:
        if name == "lm_head.weight" and config.get("tie_word_embeddings"):
            weights[name] = weights[names[0]]
            continue
        with safe_open(str(root / weight_map.get(name, "model.safetensors")), framework="pt", device="cpu") as sf:
            weights[name] = sf.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    embedding = nn.Embedding.from_pretrained(weights[names[0]], freeze=True)
    head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False, device="meta")
    head.weight = nn.Parameter(weights[names[1]], requires_grad=False)
    head.org_vocab_size = config["vocab_size"]
    head.tp_size = 1
    norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_(weights[names[2]])
    return embedding, head, norm


def initialize_single_rank(model_path, device):
    """Native SGLang layers require a TP group, even when its size is one."""
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config

    args = ServerArgs(model_path=model_path, tp_size=1, dtype="bfloat16", disable_cuda_graph=True)
    set_global_server_args_for_scheduler(args)
    initialize_moe_config(args)
    initialize_fp8_gemm_config(args)
    initialize_fp4_gemm_config(args)
    init_distributed_environment(world_size=1, rank=0, local_rank=device.index,
                                 distributed_init_method=f"tcp://127.0.0.1:{available_port()}", backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_dp_attention(server_args=args, model_config=ModelConfig.from_server_args(args))


class LongSparkDraft:
    def __init__(self, config, *, model_path, draft_path, target_ready=None):
        from specforge.modeling.draft import DraftFreeKVModel
        from sglang.srt.speculative.draft_free_kv_runner import (
            _restore_dfk_parallel_qwen_inv_freq, _verify_global16_shared_rope,
            _install_global16_sglang_rmsnorm,
        )
        from sglang.srt.speculative.global16_native_draft_executor import build_global16_native_draft_executor

        self.device = torch.device("cuda", config.draft_device)
        torch.cuda.set_device(self.device)
        initialize_single_rank(model_path, self.device)
        self.embedding, self.head, self.norm = target_modules(model_path, self.device)
        self.model = DraftFreeKVModel.from_pretrained(draft_path, torch_dtype=torch.bfloat16).to(self.device).eval()
        _restore_dfk_parallel_qwen_inv_freq(self.model)
        from .position import configure_split_longspark, configure_native
        self.position_audit = configure_split_longspark(self.model, target_ready, config.position_variant)
        _verify_global16_shared_rope(self.model)
        _install_global16_sglang_rmsnorm(self.model)
        self.executor = build_global16_native_draft_executor(self.model)
        self.native_position_audit = configure_native(self.executor.decoder_layers, config.position_variant)
        self.layer_ids = list(self.model.config.target_layer_ids)
        self.query = self.model.global16_raw256_reference.global_query_generator.learned_query.detach()
        layers, heads, slots, dim = self.query.shape
        kv_heads = int(self.model.config.num_key_value_heads)
        hidden = int(self.model.config.hidden_size)
        per_request = layers * 256 * kv_heads * dim * 2 * 2
        per_request += layers * heads * slots * (dim + 1) * 4
        per_request += layers * hidden * 2
        self.capacity = min(config.max_running_requests, int(config.draft_state_gib * GIB) // per_request)
        if self.capacity < 1:
            raise ValueError("draft state budget cannot fit one LongSpark request")
        n = self.capacity
        self.keys = torch.empty(layers, n * 256, kv_heads, dim, dtype=torch.bfloat16, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.output = torch.empty(n, layers, heads, slots, dim, dtype=torch.float32, device=self.device)
        self.lse = torch.empty(n, layers, heads, slots, dtype=torch.float32, device=self.device)
        self.hidden = torch.empty(n, layers, hidden, dtype=torch.bfloat16, device=self.device)
        self.allocated_state_bytes = sum(t.numel() * t.element_size() for t in
                                        (self.keys, self.values, self.output, self.lse, self.hidden))
        assert self.allocated_state_bytes <= config.draft_state_gib * GIB
        self.free_slots = list(range(n))
        self.requests = {}
        self.layer_objects = {
            i: SimpleNamespace(tp_k_head_num=kv_heads, tp_v_head_num=kv_heads,
                               head_dim=dim, v_head_dim=dim, scaling=dim**-.5, logit_cap=0.)
            for i in self.layer_ids
        }
        self.graph = None
        if config.cuda_graph:
            from sglang.srt.speculative.global16_proposal_cuda_graph import Global16ProposalCudaGraphRunner
            self.graph = Global16ProposalCudaGraphRunner(
                proposal_fn=self._base_logits, device=self.device,
                capture_batch_sizes=sorted({n for n in config.graph_batch_sizes if n <= self.capacity} | {self.capacity}),
                context_factory=self._graph_context, selected_layer_count=layers,
                conditioning_tail_shape=(layers, hidden))

    def _base_logits(self, tokens, positions, hidden, context):
        return self.executor(conditioning_hidden_states=hidden, current_token_ids=tokens,
                             current_position_ids=positions, target_final_norm=self.norm,
                             target_embed_tokens=self.embedding, target_lm_head=self.head,
                             global16_paged_context=context, apply_markov_correction=False,
                             return_proposal_ids=False)

    def _graph_context(self, *, request_slots, req_pool_indices, seq_lens,
                       page_table_workspace, raw_length_workspace, host_seq_lens):
        return self._context(request_slots, seq_lens, page_table_workspace,
                             raw_length_workspace, static=True)

    def _context(self, slots, lengths, page_table, raw_lengths, *, static=False):
        from sglang.srt.speculative.global16_raw256 import Global16Raw256VerifyContext, Global16Raw256PagedDraftContext
        from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
        context = Global16Raw256VerifyContext.from_committed_pool(
            selected_layer_ids=self.layer_ids, global_queries=self.query[None].expand(len(slots), -1, -1, -1, -1),
            state_output_pool=self.output, state_lse_pool=self.lse,
            request_slots=slots, layer_objects=self.layer_objects)

        class RingContext(Global16Raw256PagedDraftContext):
            def _build_raw256_page_table(this, *, host_seq_lens):
                # Always capture the full ring extent; raw_lengths masks short
                # prefixes, and later replays can cross the 256-token boundary.
                this._max_raw_rows = 256
                return page_table, raw_lengths

        return RingContext(verify_context=context, req_pool_indices=slots, seq_lens=lengths,
                            req_to_token_pool=None, token_to_kv_pool=self, page_size=1,
                            flash_attn_with_kvcache=flash_attn_with_kvcache,
                            host_seq_lens=[256] * len(slots), cuda_graph_static_inputs=static,
                            position_ids_are_scheduler_owned=True)

    def graph_stats(self):
        return self.graph.stats() if self.graph else dict(replays=0, captured_batch_sizes=[], fallbacks=0)

    def get_kv_buffer(self, layer_id):
        index = self.layer_ids.index(layer_id)
        return self.keys[index], self.values[index]

    @torch.inference_mode()
    def update(self, packets, *, prefill=False):
        if self.device.type == "cuda" and not prefill:
            return self._update_delta(packets)
        from .state_export import unpack_longspark_states
        packets = unpack_longspark_states(packets)
        from sglang.srt.speculative.global16_raw256_incremental import (
            reference_position_scores, reference_merge_accepted_positions,
        )

        for packet in packets:
            rid = packet["id"]
            if prefill:
                if rid in self.requests or not self.free_slots:
                    raise RuntimeError("LongSpark request slot unavailable or already owned")
                slot = self.free_slots.pop()
                self.requests[rid] = [slot, packet["length"]]
                self.output[slot].copy_(packet["global_output"])
                self.lse[slot].copy_(packet["global_lse"])
            else:
                slot, previous = self.requests[rid]
                if previous + packet["commit_len"] != packet["length"]:
                    raise RuntimeError("out-of-order LongSpark state commit")
                count = packet["commit_len"]
                for layer in range(len(self.layer_ids)):
                    k = torch.zeros(1, 8, self.keys.shape[-2], self.keys.shape[-1],
                                    dtype=self.keys.dtype, device=self.device)
                    v = torch.zeros_like(k)
                    k[0, :count].copy_(packet["keys"][layer])
                    v[0, :count].copy_(packet["values"][layer])
                    scores = reference_position_scores(global_query=self.query[layer][None], new_keys=k,
                                                        softmax_scale=self.query.shape[-1]**-.5)
                    output, lse = reference_merge_accepted_positions(
                        state_output=self.output[slot, layer][None],
                        state_lse=self.lse[slot, layer][None], position_scores=scores, accepted_values=v,
                        accept_lens=torch.tensor([count], dtype=torch.int32, device=self.device))
                    self.output[slot, layer].copy_(output[0])
                    self.lse[slot, layer].copy_(lse[0])
            self.requests[rid][1] = packet["length"]
            count = packet["keys"].shape[1]
            positions = torch.arange(packet["length"] - count, packet["length"], device=self.device)
            locs = slot * 256 + positions.remainder(256)
            self.keys[:, locs] = packet["keys"]
            self.values[:, locs] = packet["values"]
            self.hidden[slot].copy_(packet["last_hidden"])

    def _update_delta(self, packets):
        from sglang.srt.speculative.global16_raw256_incremental import cache_position_scores, merge_accepted_positions_
        batched = packets if isinstance(packets, dict) else None
        if batched is not None:
            from .state_export import state_metadata
            packets = state_metadata(batched)
        if not packets:
            return
        slots, offsets, ring_locs, counts = [], [], [], []
        total = 0
        for packet in packets:
            slot, previous = self.requests[packet["id"]]
            count = packet["commit_len"]
            if previous + count != packet["length"] or not 1 <= count <= 8:
                raise RuntimeError("out-of-order LongSpark state commit")
            slots.append(slot)
            offsets.append(total)
            counts.append(count)
            ring_locs.extend(slot * 256 + pos % 256 for pos in range(previous, previous + count))
            total += count
            self.requests[packet["id"]][1] = packet["length"]
        if batched is None:
            keys = torch.cat([p["keys"] for p in packets], dim=1)
            values = torch.cat([p["values"] for p in packets], dim=1)
            hidden = torch.stack([p["last_hidden"] for p in packets])
        else:
            keys, values, hidden = batched["keys"], batched["values"], batched["last_hidden"]
            if batched["export_counts"] != counts or keys.shape[1] != total or values.shape != keys.shape:
                raise ValueError("decode batch must contain exactly the committed KV rows")
        slots = torch.tensor(slots, dtype=torch.int32, device=self.device)
        lens = torch.tensor(counts, dtype=torch.int32, device=self.device)
        locs = (torch.tensor(offsets, dtype=torch.int64, device=self.device)[:, None]
                + torch.arange(8, device=self.device)).clamp_max(total - 1)
        scores = torch.empty(len(packets), self.query.shape[1], 16, 8, dtype=torch.float32, device=self.device)
        for layer in range(len(self.layer_ids)):
            new_keys = keys[layer][locs]
            cache_position_scores(layer_id=self.layer_ids[layer],
                global_query=self.query[layer][None].expand(len(packets), -1, -1, -1),
                new_keys=new_keys, output=scores, softmax_scale=self.query.shape[-1]**-.5)
            merge_accepted_positions_(state_output=self.output[:, layer], state_lse=self.lse[:, layer],
                position_scores=scores, value_cache=values[layer][:, None], accepted_cache_locs=locs,
                accept_lens=lens, accept_offsets=None, state_slots=slots)
        ring_locs = torch.tensor(ring_locs, dtype=torch.int64, device=self.device)
        self.keys[:, ring_locs] = keys
        self.values[:, ring_locs] = values
        self.hidden[slots] = hidden

    @torch.inference_mode()
    def propose(self, ids, anchors, temperature):
        from sglang.srt.speculative.global16_raw256 import (
            Global16Raw256VerifyContext, Global16Raw256PagedDraftContext,
        )
        from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache

        device = self.device
        slots = torch.tensor([self.requests[i][0] for i in ids], dtype=torch.int32, device=device)
        lengths = torch.tensor([self.requests[i][1] for i in ids], dtype=torch.int32, device=device)
        page_table, raw_lengths = ring_page_table(slots, lengths)
        paged = self._context(slots, lengths, page_table, raw_lengths)
        anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=device)
        if self.graph:
            base_logits = self.graph.try_propose(current_token_ids=anchor_tensor,
                current_position_ids=lengths.long(), conditioning_hidden_states=self.hidden[slots], paged_context=paged)
            if base_logits is None:
                raise RuntimeError("LongSpark graph unexpectedly fell back to eager")
        else:
            base_logits = self._base_logits(anchor_tensor, lengths.long(), self.hidden[slots], paged)
        tokens, corrected = self.model.sample_parallel_tokens(
            base_logits, first_previous_token_ids=anchor_tensor, temperature=temperature)
        q = None if temperature == 0 else torch.softmax(corrected.float() / temperature, -1).gather(
            -1, tokens[..., None]).squeeze(-1)
        return tokens, corrected, q

    def release(self, ids):
        for rid in ids:
            slot, _ = self.requests.pop(rid)
            self.free_slots.append(slot)
