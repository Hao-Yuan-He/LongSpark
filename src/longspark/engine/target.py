"""SGLang target-only TP processes for a serial split evaluation.

TP is confined to these processes. The draft process never joins their process
group, never loads a target transformer, and never sees target cache addresses.
Uses native TARGET_VERIFY CUDA Graphs when enabled and eager
EXTEND for prefill. It is not the SGLang HTTP scheduler.
"""

from array import array
import json
from pathlib import Path
import socket
import secrets
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import GIB, kv_bytes_per_token
from .sampling import decide_acceptance, finish_sampling
from .transport import TensorChannel


def _update_fill_ids(req, committed_tokens, block, *, incremental):
    """Retain the committed prefix and replace the previous proposal tail.

    PagedTarget commits only a prefix of each block and releases the Req before
    reusing its ID. Thus the cached prefix stays valid across verify/decode
    rounds, including rollback to an earlier committed prefix. Prefill rebuilds
    the cache, as do fresh/short caches and aliases of the original input.
    """
    fill = req.full_untruncated_fill_ids
    prefix = len(committed_tokens)
    if incremental and fill is not req.origin_input_ids and len(fill) >= prefix:
        del fill[prefix:]
        fill.extend(block)
    else:
        req.full_untruncated_fill_ids = array("q", committed_tokens + list(block))


def available_port():
    # bind(0) selects a Linux ephemeral client port. Another target/draft
    # connection can claim it between the probe and TCPStore.listen(). Keep
    # rendezvous outside that range and probe all local IPv4 addresses.
    for _ in range(100):
        port = 16000 + secrets.randbelow(14000)
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free split rendezvous port in 16000..29999")


class PagedTarget:
    def __init__(self, config, *, model_path, method, draft_path, rank, port, seed):
        from sglang.benchmark.one_batch import load_model
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.server_args import ServerArgs
        from types import SimpleNamespace

        self.config, self.rank, self.method = config, rank, method
        from .position import model_override, audit_runner
        self.hf = json.loads((Path(model_path) / "config.json").read_text())
        self.draft_config = {} if method == 'vanilla' else json.loads((Path(draft_path) / "config.json").read_text())
        self.layer_ids = [] if method == 'vanilla' else list(self.draft_config["target_layer_ids"])
        cell = kv_bytes_per_token(
            layers=self.hf["num_hidden_layers"], kv_heads=self.hf["num_key_value_heads"],
            head_dim=self.hf["head_dim"], tp=config.target_tp,
        )
        max_tokens = int(config.target_kv_gib_per_rank * GIB) // cell - 1
        if max_tokens < 8:
            raise ValueError("target KV budget cannot hold one verify block")
        server_args = ServerArgs(
            model_path=model_path, dtype="bfloat16", tp_size=config.target_tp,
            mem_fraction_static=0.80,
            max_total_tokens=max_tokens,
            max_running_requests=config.max_running_requests,
            context_length=config.context_length, page_size=1, attention_backend="fa3",
            json_model_override_args=json.dumps(model_override(config.position_variant)),
            disable_cuda_graph=True, disable_radix_cache=True,
            disable_overlap_schedule=True, random_seed=seed,
            enable_deterministic_inference=config.deterministic_inference,
            chunked_prefill_size=config.prefill_chunk_size, log_level="warning",
        )
        initialize_moe_config(server_args)
        initialize_fp8_gemm_config(server_args)
        initialize_fp4_gemm_config(server_args)
        wrapper, _ = load_model(server_args, SimpleNamespace(nccl_port=port), config.target_devices[rank], rank)
        self.runner = wrapper.torch_runner
        self.position_audit = audit_runner(self.runner, config.position_variant)
        self.device = torch.device("cuda", config.target_devices[rank])
        torch.manual_seed(seed)
        if method != 'vanilla':
            self.runner.model.set_dflash_layers_to_capture(self.layer_ids)
        # LongSpark verifies a linear eight-token target block. Use the native
        # DSPARK metadata contract without loading a draft inside the target
        # group. Selected hidden states are graph outputs, not Python hook
        # references (hooks do not execute during replay).
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        server_args.speculative_algorithm = None if method == 'vanilla' else "DSPARK"
        server_args.speculative_num_steps = 1
        server_args.speculative_num_draft_tokens = 8
        server_args.speculative_eagle_topk = 1
        self.runner.spec_algorithm = SpeculativeAlgorithm.NONE if method == 'vanilla' else SpeculativeAlgorithm.DSPARK
        self.runner.init_attention_backends()
        self.graph_replays = self.graph_fallbacks = 0
        if config.cuda_graph:
            from .graphs import enable_native_graphs
            enable_native_graphs(self.runner, config, hidden=method != 'vanilla')
        self.requests = {}
        self.tokens = {}
        self.kv_locs = {}
        self.global_query = None
        if method == "longspark":
            from safetensors import safe_open
            with safe_open(str(Path(draft_path) / "model.safetensors"), framework="pt", device="cpu") as sf:
                query = sf.get_tensor("global16_raw256_reference.global_query_generator.learned_query")
            self.global_query = query.chunk(config.target_tp, dim=1)[rank].contiguous().to(self.device)
        self.total_forward_seconds = 0.0

    def pool_info(self):
        actual = 0
        for layer in range(self.hf["num_hidden_layers"]):
            actual += sum(t.numel() * t.element_size() for t in self.runner.token_to_kv_pool.get_kv_buffer(layer))
        if actual > self.config.target_kv_gib_per_rank * GIB:
            raise RuntimeError("actual target KV tensors exceed per-rank cache budget")
        return dict(ready=True, rank=self.rank, max_tokens=self.runner.max_total_num_tokens,
                    kv_allocated_bytes=actual, logical_device=self.device.index,
                    position_audit=self.position_audit,
                    deterministic_inference=self.runner.server_args.enable_deterministic_inference,
                    attention_num_splits=getattr(self.runner.attn_backend, 'num_splits', None))

    def _gather_heads(self, tensor, dim):
        if self.config.target_tp == 1:
            return tensor
        chunks = [torch.empty_like(tensor) for _ in range(self.config.target_tp)]
        dist.all_gather(chunks, tensor.contiguous(), group=self.runner.tp_group.device_group)
        return torch.cat(chunks, dim=dim)

    @torch.inference_mode()
    def forward(self, ids, blocks, *, verify=False, decode=False):
        from sglang.benchmark.one_batch import TreeCacheNamespace
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, CaptureHiddenMode, ForwardMode
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.layers.logits_processor import LogitsMetadata

        reqs, starts = [], []
        for rid, block in zip(ids, blocks, strict=True):
            if rid not in self.requests:
                req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", block),
                          sampling_params=SamplingParams(temperature=0, max_new_tokens=8192))
                self.requests[rid] = req
                self.tokens[rid] = []
                self.kv_locs[rid] = torch.empty(0, dtype=torch.int64, device=self.device)
            req = self.requests[rid]
            prefix = len(self.tokens[rid])
            if prefix + len(block) > self.config.context_length:
                raise ValueError("target context limit exceeded")
            _update_fill_ids(req, self.tokens[rid], block, incremental=verify or decode)
            req.prefix_indices = self.kv_locs[rid]
            req.logprob_start_len = -1
            req.set_extend_range(prefix, prefix + len(block))
            reqs.append(req)
            starts.append(prefix)
        runner = self.runner
        cache = TreeCacheNamespace(page_size=1, device=runner.device,
                                   token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator)
        batch = ScheduleBatch.init_new(
            reqs=reqs, req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=cache, model_config=runner.model_config, enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(self.device)
            batch.prefill_input_ids_cpu = None
        batch.capture_hidden_mode = CaptureHiddenMode.NULL if self.method == 'vanilla' else CaptureHiddenMode.FULL
        fb = ForwardBatch.init_new(batch, runner)
        if decode:
            if self.method != 'vanilla' or any(len(b) != 1 for b in blocks):
                raise ValueError('vanilla decode requires one token per request')
            fb.forward_mode = ForwardMode.DECODE
        if verify:
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            fb.forward_mode = ForwardMode.TARGET_VERIFY
            fb.spec_algorithm = SpeculativeAlgorithm.DSPARK
            fb.spec_info = DFlashVerifyInput(draft_token=fb.input_ids, positions=fb.positions,
                                            draft_token_num=8, capture_hidden_mode=CaptureHiddenMode.FULL)
            fb.seq_lens = torch.tensor(starts, dtype=torch.int32, device=self.device)
            fb.seq_lens_cpu = torch.tensor(starts, dtype=torch.int32)
            fb.seq_lens_sum = sum(starts)
        torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        out = runner.forward(fb)
        result = out.logits_output
        if verify or decode:
            self.graph_replays += int(out.can_run_graph)
            self.graph_fallbacks += int(not out.can_run_graph)
        torch.cuda.synchronize(self.device)
        self.total_forward_seconds += time.perf_counter() - tic
        aux_hidden = None if self.method == 'vanilla' else result.hidden_states.reshape(-1, len(self.layer_ids), self.hf["hidden_size"])
        return batch, result.next_token_logits, aux_hidden, starts

    def _kv_rows(self, locations):
        if not getattr(self.config, "fuse_kv_export", False):
            keys, values = [], []
            for layer_id in self.layer_ids:
                k, v = self.runner.token_to_kv_pool.get_kv_buffer(layer_id)
                keys.append(k[locations])
                values.append(v[locations])
            return self._gather_heads(torch.stack(keys), 2), self._gather_heads(torch.stack(values), 2)
        from .state_export import gather_selected_kv
        if not hasattr(self, "_kv_export_workspace"):
            self._kv_export_workspace = {}
        return gather_selected_kv(
            self.runner.token_to_kv_pool, self.layer_ids, locations,
            tp=self.config.target_tp, rank=self.rank,
            group=self.runner.tp_group.device_group if self.config.target_tp > 1 else None,
            workspace=self._kv_export_workspace)

    def _prefill_summaries(self, ids):
        from sglang.srt.speculative.global16_raw256 import Global16Raw256VerifyContext
        from sglang.srt.layers.attention.flashattention_backend import flash_attn_with_kvcache

        query = self.global_query[None].expand(len(ids), -1, -1, -1, -1)
        lengths = [len(self.tokens[rid]) for rid in ids]
        context = Global16Raw256VerifyContext(
            selected_layer_ids=self.layer_ids, global_queries=query,
            confirmed_prefix_lens=torch.tensor(lengths, dtype=torch.int32, device=self.device),
        )
        req_slots = torch.tensor([self.requests[rid].req_pool_idx for rid in ids], device=self.device)
        page_table = self.runner.req_to_token_pool.req_to_token[req_slots, :max(lengths)]
        for layer_id in self.layer_ids:
            layer = self.runner.model.model.layers[layer_id].self_attn.attn
            k, v = self.runner.token_to_kv_pool.get_kv_buffer(layer_id)
            context.run_prefill_global_attention(
                layer_id=layer_id, layer=layer, key_cache=k[:, None], value_cache=v[:, None],
                page_table=page_table, cache_seqlens=context.confirmed_prefix_lens,
                flash_attn_with_kvcache=flash_attn_with_kvcache,
                softmax_scale=layer.scaling, softcap=layer.logit_cap,
            )
        output = torch.stack([context.layer_states[i].output for i in self.layer_ids], dim=1)
        lse = torch.stack([context.layer_states[i].logsumexp for i in self.layer_ids], dim=1)
        return self._gather_heads(output, 2), self._gather_heads(lse, 2)

    @torch.inference_mode()
    def commit(self, ids, blocks, batch, hidden, lengths, *, prefill, export_state=True):
        packets = []
        all_export_locs, export_counts = [], []
        offset = 0
        for row, (rid, block, length) in enumerate(zip(ids, blocks, lengths, strict=True)):
            count = len(block)
            if not 1 <= length <= count:
                raise ValueError("invalid committed prefix length")
            locs = batch.out_cache_loc[offset:offset + count]
            self.kv_locs[rid] = torch.cat((self.kv_locs[rid], locs[:length]))
            if length < count:
                self.runner.token_to_kv_pool_allocator.free(locs[length:])
            self.tokens[rid].extend(block[:length])
            self.requests[rid].kv_committed_len = len(self.tokens[rid])
            self.requests[rid].kv_allocated_len = len(self.tokens[rid])
            if not export_state:
                offset += count
                continue
            packet = {"id": rid, "length": len(self.tokens[rid]), "commit_len": length,
                      "last_hidden": hidden[offset + length - 1].contiguous()}
            export_locs = self.kv_locs[rid][-256:] if prefill else locs[:length]
            all_export_locs.append(export_locs)
            export_counts.append(len(export_locs))
            packets.append(packet)
            offset += count
        if all_export_locs:
            keys, values = self._kv_rows(torch.cat(all_export_locs))
            global_output = global_lse = None
            if prefill:
                global_output, global_lse = self._prefill_summaries(ids)
            # All TP ranks participate in collectives, but only rank zero owns
            # the tensor channel. Do not rebuild full KV states on other ranks.
            if self.rank != 0:
                return []
            from .state_export import batch_longspark_states
            return batch_longspark_states(packets, export_counts, keys, values,
                                          global_output=global_output, global_lse=global_lse)
        return packets

    def release(self, ids):
        for rid in ids:
            self.runner.token_to_kv_pool_allocator.free(self.kv_locs.pop(rid))
            self.runner.req_to_token_pool.free(self.requests.pop(rid))
            self.tokens.pop(rid)

    @torch.inference_mode()
    def greedy_reference(self, ids, prompts, max_new_tokens, eos_ids):
        """Independent one-token AR baseline, without draft or summary updates."""
        output = {rid: [] for rid in ids}
        active, blocks = list(ids), prompts
        initial = True
        while active:
            if initial:
                # Match the production chunking without consulting any draft.
                first_logits = []
                for rid, prompt in zip(active, blocks, strict=True):
                    for start in range(0, len(prompt), self.config.prefill_chunk_size):
                        piece = prompt[start:start + self.config.prefill_chunk_size]
                        batch, last_logits, hidden, _ = self.forward([rid], [piece])
                        self.commit([rid], [piece], batch, hidden, [len(piece)],
                                    prefill=False, export_state=False)
                    first_logits.append(last_logits)
                logits = torch.cat(first_logits)
                initial = False
            else:
                batch, logits, hidden, _ = self.forward(active, blocks)
                self.commit(active, blocks, batch, hidden, [len(b) for b in blocks],
                            prefill=False, export_state=False)
            next_tokens = logits.argmax(-1).tolist()
            retained, next_blocks, finished = [], [], []
            for rid, token in zip(active, next_tokens, strict=True):
                output[rid].append(token)
                if token in eos_ids or len(output[rid]) >= max_new_tokens:
                    finished.append(rid)
                else:
                    retained.append(rid)
                    next_blocks.append([token])
            self.release(finished)
            active, blocks = retained, next_blocks
        return [output[rid] for rid in ids]

    @torch.inference_mode()
    def greedy_path_check(self, ids, prompts, outputs):
        """Independent one-token target replay on the actual output prefix.

        Equal highest logits are valid greedy choices even when argmax's index
        tie-break differs. No tolerance is applied to a lower-scoring token.
        """
        checks = []
        for rid, prompt, tokens in zip(ids, prompts, outputs, strict=True):
            ties, violations = [], []
            for start in range(0, len(prompt), self.config.prefill_chunk_size):
                piece = prompt[start:start+self.config.prefill_chunk_size]
                batch, logits, hidden, _ = self.forward([rid], [piece])
                self.commit([rid], [piece], batch, hidden, [len(piece)], prefill=False, export_state=False)
            for i, token in enumerate(tokens):
                top, chosen = logits[0].max().item(), logits[0, token].item()
                best = logits[0].argmax().item()
                if chosen < top:
                    violations.append(dict(position=i,token=token,argmax=best,
                                           chosen_logit=chosen,top_logit=top,gap=top-chosen))
                elif token != best:
                    ties.append(dict(position=i,token=token,argmax=best,logit=top))
                if i+1 < len(tokens):
                    block = [token]
                    batch, logits, hidden, _ = self.forward([rid], [block])
                    self.commit([rid], [block], batch, hidden, [1], prefill=False, export_state=False)
            self.release([rid])
            checks.append(dict(id=rid,checked_tokens=len(tokens),passed=not violations,
                               exact_logit_ties=ties,violations=violations))
        return checks


def target_process_main(rank, config, model_path, method, draft_path, port, controls, wire, seed):
    # Each rank has its own command pipe. Tensor transport exists only on rank 0.
    control = controls[rank]
    target = channel = None
    closing = False
    try:
        torch.set_num_threads(4)
        target = PagedTarget(config, model_path=model_path, method=method, draft_path=draft_path,
                             rank=rank, port=port, seed=seed)
        channel = TensorChannel(wire, device=target.device, mode=config.transport) if rank == 0 and method != 'vanilla' else None
        control.send(target.pool_info())
        while True:
            command = control.recv()
            op = command["op"]
            if op == "close":
                closing = True
                break
            if op == "release":
                target.release(command["ids"])
                control.send({"released": True})
                continue
            if op == "stats":
                from .graphs import native_graph_stats
                stats = native_graph_stats(target.runner, target.graph_replays, target.graph_fallbacks)
                stats["forward_seconds"] = target.total_forward_seconds
                stats["transport"] = channel.report() if channel else None
                control.send(stats)
                continue
            if op == "reset_measurement":
                if target.requests:
                    raise RuntimeError("measurement reset requires an empty target")
                torch.manual_seed(command["seed"])
                target.graph_replays = target.graph_fallbacks = 0
                target.total_forward_seconds = 0.
                if channel:
                    from .transport import TransferStats
                    channel.stats = TransferStats()
                control.send({"reset": True})
                continue
            if op == "greedy_reference":
                output_ids = target.greedy_reference(command["ids"], command["blocks"],
                                                     command["max_new_tokens"], command["eos_ids"])
                control.send({"output_ids": output_ids})
                continue
            if op == "greedy_path_check":
                checks = target.greedy_path_check(command["ids"], command["blocks"], command["outputs"])
                control.send({"checks": checks})
                continue
            if op == "prefill_logits":
                ids = [f"__prefill_logits_{i}" for i in range(len(command["blocks"]))]
                blocks = command["blocks"]
                batch, logits, hidden, _ = target.forward(ids, blocks)
                values, indices = logits.float().topk(8, dim=-1)
                target.commit(ids, blocks, batch, hidden, [len(b) for b in blocks], prefill=True, export_state=False)
                target.release(ids)
                control.send({"top_values": values.tolist(), "top_ids": indices.tolist()})
                continue
            if op == "audit_next":
                ids = [f"__longctx_next_audit_{i}" for i in range(command.get("batch_size", 1))]
                prompt = command["prefix"]
                for rid in ids:
                    for start in range(0, len(prompt), config.prefill_chunk_size):
                        piece = prompt[start:start + config.prefill_chunk_size]
                        batch, logits, hidden, _ = target.forward([rid], [piece])
                        target.commit([rid], [piece], batch, hidden, [len(piece)], prefill=False, export_state=False)
                block = command["block"]
                saved = target.runner.decode_cuda_graph_runner
                if command["mode"] == "verify_eager":
                    target.runner.decode_cuda_graph_runner = None
                before_replays = target.graph_replays
                try:
                    batch, logits, hidden, _ = target.forward(ids, [block]*len(ids), verify=command["mode"] != "ar")
                    target.commit(ids, [block]*len(ids), batch, hidden, [len(block)]*len(ids), prefill=False, export_state=False)
                    values, indices = logits[0].float().topk(8)
                    selected = logits[0, command["candidate_ids"]].float()
                    result = dict(top_values=values.tolist(), top_ids=indices.tolist(),
                                  candidate_logits=selected.tolist(), mode=command["mode"],
                                  batch_size=len(ids), graph_replays=target.graph_replays-before_replays)
                finally:
                    target.runner.decode_cuda_graph_runner = saved
                    target.release(ids)
                control.send(result)
                continue
            if op not in ("prefill", "verify", "decode"):
                raise ValueError(f"unknown split target operation: {op}")
            ids, blocks = command["ids"], command["blocks"]
            batch, logits, hidden, _ = target.forward(ids, blocks, verify=op == "verify", decode=op == 'decode')
            temp = command["temperature"]
            decision_data = [None]
            if rank == 0:
                if op in ("prefill", "decode"):
                    final_chunk = command.get("final_chunk", True)
                    anchors = command.get("preserved_anchors")
                    if anchors is None:
                        anchors = [None] * len(ids)
                    fresh = [i for i, value in enumerate(anchors) if value is None] if final_chunk else []
                    bonus = torch.tensor([0 if v is None else v for v in anchors], device=target.device)
                    if fresh:
                        selected = logits[fresh]
                        bonus[fresh] = selected.argmax(-1) if temp == 0 else torch.multinomial(
                            torch.softmax(selected.float() / temp, -1), 1).squeeze(-1)
                    lens = [len(block) for block in blocks]
                else:
                    logits = logits.view(len(ids), 8, -1)
                    candidates = torch.tensor(blocks, dtype=torch.long, device=target.device)[:, 1:]
                    if temp == 0:
                        truth = logits.argmax(-1)
                        accept = (truth[:, :7] == candidates).long().cumprod(1).sum(1)
                        bonus = truth[torch.arange(len(ids), device=target.device), accept]
                    else:
                        sampled_q = channel.receive()["sampled_q"]
                        decision = decide_acceptance(
                            candidates=candidates, sampled_q=sampled_q,
                            target_probs=torch.softmax(logits.float() / temp, -1),
                            uniforms=torch.rand(len(ids), 7, device=target.device),
                        )
                        final_uniforms = torch.rand(len(ids), device=target.device)
                        channel.send({"rejected_rows": decision.rejected_rows,
                                      "positions": decision.rejection_positions})
                        q_rows = channel.receive()["q_rows"]
                        bonus = finish_sampling(decision, q_rows, uniforms=final_uniforms)
                        accept = decision.lengths
                    lens = (accept + 1).tolist()
                decision_data[0] = (lens, bonus.tolist())
            if config.target_tp > 1:
                dist.broadcast_object_list(decision_data, src=0, group=target.runner.tp_group.cpu_group)
            lengths, bonus_list = decision_data[0]
            export_state = method != 'vanilla' and (op == "verify" or command.get("final_chunk", True))
            packets = target.commit(ids, blocks, batch, hidden, lengths, prefill=op == "prefill",
                                    export_state=export_state)
            if rank == 0 and channel is not None:
                channel.send({"states": packets, "bonus": bonus_list, "commit_lengths": lengths})
            control.send({"done": True, "states": [], "bonus": bonus_list, "commit_lengths": lengths}
                         if method == 'vanilla' else {"done": True})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            control.send({"error": error})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        from .lifecycle import release_native_graphs, cleanup_parallel_runtime, trace_shutdown
        trace_shutdown(f"target_rank{rank}:begin", begin=True)
        if target is not None:
            torch.cuda.synchronize(target.device)
            release_native_graphs(target.runner)
        if channel is not None:
            trace_shutdown(f"target_rank{rank}:channel_close")
            channel.close()
        if dist.is_initialized():
            cleanup_parallel_runtime()
        if closing:
            control.send({"closed": True, "forward_seconds": target.total_forward_seconds})
        control.close()
        trace_shutdown(f"target_rank{rank}:done")


class TargetClient:
    def __init__(self, config, *, model_path, method, draft_path, seed=1234):
        self.config = config
        self.method = method
        if (method == 'vanilla') != (config.draft_device is None):
            raise ValueError('vanilla must have no draft GPU; speculative methods require one')
        self.prefill_consumer = None
        ctx = mp.get_context("spawn")
        pairs = [ctx.Pipe() for _ in range(config.target_tp)]
        parents, children = zip(*pairs)
        self.controls = parents
        local_wire, target_wire = ctx.Pipe()
        self.channel = None if method == 'vanilla' else TensorChannel(local_wire, device=f"cuda:{config.draft_device}", mode=config.transport)
        port = available_port()
        self.processes = []
        for rank in range(config.target_tp):
            process = ctx.Process(target=target_process_main,
                args=(rank, config, model_path, method, draft_path, port, children, target_wire, seed))
            process.start()
            self.processes.append(process)
        try:
            self.ready = self._responses()
        except BaseException:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            raise

    def _responses(self, timeout=600):
        responses = []
        for pipe, process in zip(self.controls, self.processes):
            deadline = time.monotonic() + timeout
            while not pipe.poll(1):
                if not process.is_alive():
                    raise RuntimeError(f"target process {process.pid} exited: {process.exitcode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("target RPC timed out")
            response = pipe.recv()
            if "error" in response:
                raise RuntimeError(response["error"])
            responses.append(response)
        return responses

    def _send(self, command):
        for pipe in self.controls:
            pipe.send(command)

    def prefill(self, ids, blocks, temperature, *, preserved_anchors=None):
        if self.prefill_consumer is None and self.method != 'vanilla':
            raise RuntimeError("chunked prefill requires an incremental draft-state consumer")
        if preserved_anchors is None:
            preserved_anchors = [None] * len(ids)
        bonus = []
        for rid, block, anchor in zip(ids, blocks, preserved_anchors, strict=True):
            for start in range(0, len(block), self.config.prefill_chunk_size):
                piece = block[start:start + self.config.prefill_chunk_size]
                final = start + len(piece) == len(block)
                self._send(dict(op="prefill", ids=[rid], blocks=[piece], temperature=temperature,
                                preserved_anchors=[anchor], final_chunk=final))
                if self.channel is None:
                    payload = self._responses()[0]
                else:
                    payload = self.channel.receive()
                    self._responses()
                if payload["states"]:
                    self.prefill_consumer(payload["states"])
                if final:
                    bonus.append(payload["bonus"][0])
        return dict(states=[], bonus=bonus, commit_lengths=[len(b) for b in blocks],
                    states_already_consumed=True)

    def decode(self, ids, anchors, temperature):
        if self.method != 'vanilla':
            raise ValueError('decode without a draft is reserved for vanilla')
        self._send(dict(op='decode', ids=ids, blocks=[[x] for x in anchors], temperature=temperature))
        return self._responses()[0]

    def verify(self, ids, blocks, temperature, *, draft_logits=None, sampled_q=None):
        self._send(dict(op="verify", ids=ids, blocks=blocks, temperature=temperature))
        if temperature > 0:
            if draft_logits is None or sampled_q is None:
                raise ValueError("T>0 requires corrected logits and actual sampled q values")
            self.channel.send({"sampled_q": sampled_q})
            request = self.channel.receive()
            rows = draft_logits[request["rejected_rows"], request["positions"]]
            self.channel.send({"q_rows": torch.softmax(rows.float() / temperature, -1)})
        payload = self.channel.receive()
        self._responses()
        return payload

    def release(self, ids):
        self._send(dict(op="release", ids=ids))
        self._responses()

    def graph_stats(self):
        self._send(dict(op="stats"))
        return self._responses()

    def reset_measurement(self, seed):
        from .transport import TransferStats
        self._send(dict(op="reset_measurement", seed=seed))
        self._responses()
        if self.channel is not None:
            self.channel.stats = TransferStats()

    def prefill_logits(self, blocks):
        self._send(dict(op="prefill_logits", blocks=blocks))
        return self._responses()[0]

    def greedy_reference(self, ids, blocks, max_new_tokens, eos_ids):
        self._send(dict(op="greedy_reference", ids=ids, blocks=blocks,
                        max_new_tokens=max_new_tokens, eos_ids=list(eos_ids)))
        responses = self._responses()
        if any(r["output_ids"] != responses[0]["output_ids"] for r in responses):
            raise RuntimeError("TP ranks produced inconsistent greedy reference tokens")
        return responses[0]["output_ids"]

    def greedy_path_check(self, ids, blocks, outputs):
        self._send(dict(op="greedy_path_check",ids=ids,blocks=blocks,outputs=outputs))
        responses = self._responses()
        if any(r["checks"] != responses[0]["checks"] for r in responses):
            raise RuntimeError("TP ranks disagree on independent greedy-path logits")
        return responses[0]["checks"]

    def close(self, *, graceful=True):
        failure = None
        try:
            if graceful and all(p.is_alive() for p in self.processes):
                self._send(dict(op="close"))
                self._responses(timeout=60)
        except (BrokenPipeError, EOFError, RuntimeError, TimeoutError) as exc:
            failure = exc
        finally:
            for process in self.processes:
                if not graceful and process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if graceful and process.exitcode != 0 and failure is None:
                    failure = RuntimeError(f"target {process.pid} shutdown exit={process.exitcode}")
            if self.channel is not None:
                self.channel.close()
            for pipe in self.controls:
                pipe.close()
        if graceful and failure is not None:
            raise RuntimeError("target shutdown did not complete cleanly") from failure
