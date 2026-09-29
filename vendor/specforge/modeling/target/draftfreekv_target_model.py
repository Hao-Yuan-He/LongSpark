from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group
from specforge.modeling.target.sglang_backend import SGLangRunner
from specforge.modeling.target.sglang_backend.runner_lifecycle import (
    build_direct_forward_batch,
    prepare_direct_extend_request,
)
from specforge.training_attribution import profile_range, record_source_token_counts


@dataclass
class DraftFreeKVTargetOutput:
    keys: torch.Tensor
    values: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    current_token_ids: Optional[torch.Tensor] = None
    current_position_ids: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    teacher_hidden_states: Optional[torch.Tensor] = None
    teacher_hidden_states_are_post_norm: bool = False
    current_hidden_states: Optional[torch.Tensor] = None
    selected_layer_hidden_states: Optional[torch.Tensor] = None
    memory_hidden_states: Optional[torch.Tensor] = None
    memory_query_mask: Optional[torch.Tensor] = None
    projected_memory_refinements: Optional[torch.Tensor] = None
    # Training-only packed layout. ``keys``/``values`` remain at source-sequence
    # granularity while these tensors map each sampled anchor back to its source
    # row and describe that anchor's visible target prefix.
    kv_source_indices: Optional[torch.Tensor] = None
    anchor_attention_mask: Optional[torch.Tensor] = None


def _get_key_value(pair) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(pair, "key") and hasattr(pair, "value"):
        return pair.key, pair.value
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return pair[0], pair[1]
    raise TypeError("past_key_values entries must expose key/value tensors")


def stack_selected_kv(
    past_key_values,
    target_layer_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack selected target KV layers as [batch, layers, heads, tokens, head_dim]."""

    if not target_layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    keys = []
    values = []
    for layer_id in target_layer_ids:
        key, value = _get_key_value(past_key_values[layer_id])
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("KV tensors must have shape [batch, heads, tokens, head_dim]")
        if key.shape != value.shape:
            raise ValueError("key and value tensors must have the same shape")
        keys.append(key)
        values.append(value)
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)


def _token_rows_from_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[list[int]]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    mask = attention_mask.bool()
    rows = []
    for row_ids, row_mask in zip(input_ids, mask):
        row = row_ids[row_mask].detach().cpu().tolist()
        if not row:
            raise ValueError("each DraftFreeKV context must contain at least one token")
        rows.append(row)
    return rows


def _pad_local_kv_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        raise ValueError("rows must not be empty")
    max_seq = max(row.shape[0] for row in rows)
    first = rows[0]
    padded = first.new_zeros((len(rows), max_seq, *first.shape[1:]))
    for row_index, row in enumerate(rows):
        padded[row_index, : row.shape[0]] = row
    return padded


def _pad_token_rows(
    token_rows: list[list[int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_rows:
        raise ValueError("token_rows must not be empty")
    max_seq = max(len(row) for row in token_rows)
    input_ids = torch.zeros((len(token_rows), max_seq), dtype=torch.long, device=device)
    attention_mask = torch.zeros(
        (len(token_rows), max_seq), dtype=torch.bool, device=device
    )
    for row_index, row in enumerate(token_rows):
        row_tensor = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[row_index, : row_tensor.numel()] = row_tensor
        attention_mask[row_index, : row_tensor.numel()] = True
    return input_ids, attention_mask


def _gather_tp_head_shards(local: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return local
    shards = [torch.empty_like(local) for _ in range(tp_size)]
    with profile_range("tp_gather"):
        dist.all_gather(shards, local, group=tp_group)
        return torch.cat(shards, dim=2)


class DraftFreeKVTargetModel(ABC):
    """Target backend contract for DraftFreeKV training."""

    def __init__(self):
        self.target_layer_ids: list[int] = []
        self.capture_selected_layer_hidden = False

    def set_target_layer_ids(self, target_layer_ids: list[int]) -> None:
        if not target_layer_ids:
            raise ValueError("target_layer_ids must not be empty")
        self.target_layer_ids = list(target_layer_ids)

    def set_selected_layer_hidden_capture(self, enabled: bool) -> None:
        self.capture_selected_layer_hidden = bool(enabled)

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Initialize the target backend."""

    @abstractmethod
    def generate_draftfreekv_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> DraftFreeKVTargetOutput:
        """Return selected target KV features for DraftFreeKV training."""


class HFDraftFreeKVTargetModel(DraftFreeKVTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        self.model.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDraftFreeKVTargetModel":
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if device:
            model = model.to(device)
        return cls(model)

    @torch.no_grad()
    def generate_draftfreekv_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> DraftFreeKVTargetOutput:
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids must be set before generation")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        keys, values = stack_selected_kv(outputs.past_key_values, self.target_layer_ids)
        selected_layer_hidden_states = None
        if self.capture_selected_layer_hidden:
            layer_hiddens = []
            for layer_id in self.target_layer_ids:
                hidden_index = min(layer_id + 1, len(outputs.hidden_states) - 1)
                layer_hiddens.append(outputs.hidden_states[hidden_index])
            selected_layer_hidden_states = torch.stack(layer_hiddens, dim=2)
        return DraftFreeKVTargetOutput(
            keys=keys,
            values=values,
            attention_mask=attention_mask.bool(),
            input_ids=input_ids,
            loss_mask=loss_mask,
            logits=getattr(outputs, "logits", None),
            hidden_states=outputs.hidden_states[-1],
            selected_layer_hidden_states=selected_layer_hidden_states,
        )


class SGLangDraftFreeKVTargetModel(DraftFreeKVTargetModel):
    def __init__(self, model_runner: SGLangRunner, page_size: int = 1):
        super().__init__()
        self.model_runner = model_runner
        self.page_size = page_size
        self.sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        cache_params = CacheInitParams(
            disable=True,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.page_size,
        )
        # Training still clears request/token pools every batch. Reusing the
        # disabled tree cache avoids rebuilding SGLang request scaffolding while
        # preserving independent full-sequence prefill semantics.
        self.training_tree_cache = RadixCache(cache_params)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDraftFreeKVTargetModel":
        tp_group = get_tp_group()
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            disable_cuda_graph=True,
            skip_tokenizer_init=True,
            tp_size=tp_size,
            pp_size=1,
            device=device or "cuda",
            **kwargs,
        )
        model_config = ModelConfig.from_server_args(server_args)
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner=model_runner, page_size=server_args.page_size)

    @torch.no_grad()
    def generate_draftfreekv_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> DraftFreeKVTargetOutput:
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids must be set before generation")
        with profile_range("target_request_prep"):
            token_rows = _token_rows_from_attention_mask(input_ids, attention_mask)
        record_source_token_counts(token_rows, padded_tokens=input_ids.shape[1])
        (
            batch,
            hidden_states,
            selected_layer_hidden_states,
            teacher_hidden_states_are_post_norm,
        ) = self._extend(token_rows)
        keys, values = self._extract_selected_kv(batch)
        compact_input_ids, compact_attention_mask = _pad_token_rows(
            token_rows,
            device=input_ids.device,
        )
        self._clear_request_memory()
        return DraftFreeKVTargetOutput(
            keys=keys,
            values=values,
            attention_mask=compact_attention_mask,
            input_ids=compact_input_ids,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            selected_layer_hidden_states=selected_layer_hidden_states,
            teacher_hidden_states_are_post_norm=teacher_hidden_states_are_post_norm,
        )

    @torch.no_grad()
    def _extend(
        self,
        token_rows: list[list[int]],
    ) -> tuple[
        Any,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        bool,
    ]:
        with profile_range("target_request_prep"):
            reqs = []
            for row_index, tokens in enumerate(token_rows):
                req = Req(
                    rid=str(row_index),
                    origin_input_text="",
                    origin_input_ids=tokens,
                    sampling_params=self.sampling_params,
                )
                prepare_direct_extend_request(
                    req,
                    tree_cache=self.training_tree_cache,
                )
                reqs.append(req)

            batch = ScheduleBatch.init_new(
                reqs=reqs,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                tree_cache=self.training_tree_cache,
                model_config=self.model_runner.model_config,
                enable_overlap=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
            )
            batch.prepare_for_extend()

            if require_mlp_sync(self.model_runner.server_args):
                Scheduler.prepare_mlp_sync_batch_raw(
                    batch,
                    dp_size=self.model_runner.server_args.dp_size,
                    attn_tp_size=1,
                    tp_group=self.model_runner.tp_group,
                    get_idle_batch=None,
                    disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    speculative_num_draft_tokens=None,
                    require_mlp_tp_gather=require_mlp_tp_gather(
                        self.model_runner.server_args
                    ),
                    disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                    offload_tags=set(),
                )

            if self.capture_selected_layer_hidden:
                self.model_runner.model.set_dflash_layers_to_capture(self.target_layer_ids)
            forward_batch = build_direct_forward_batch(
                batch,
                self.model_runner,
                forward_batch_cls=ForwardBatch,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
        with profile_range("target_prefill"):
            logits_output = self.model_runner.forward(forward_batch).logits_output
        with profile_range("hidden_reshape_repack"):
            hidden_states = None
            last_hidden = getattr(logits_output, "last_hidden_states", None)
            if last_hidden is None:
                last_hidden = getattr(logits_output, "teacher_hidden_states", None)
            output_hidden = getattr(logits_output, "hidden_states", None)
            raw_hidden = last_hidden if last_hidden is not None else output_hidden
            teacher_hidden_states_are_post_norm = last_hidden is not None
            if raw_hidden is not None:
                hidden_rows = torch.split(
                    raw_hidden, [len(row) for row in token_rows], dim=0
                )
                hidden_states = _pad_local_kv_rows(
                    [row.contiguous() for row in hidden_rows]
                )
            selected_layer_hidden_states = None
            aux_hidden = getattr(logits_output, "aux_hidden_states", None)
            selected_raw_hidden = aux_hidden if aux_hidden is not None else output_hidden
            if self.capture_selected_layer_hidden:
                if selected_raw_hidden is None:
                    raise RuntimeError(
                        "SGLang selected-layer hidden capture returned no hidden states. "
                        "Expected logits_output.aux_hidden_states or logits_output.hidden_states."
                    )
                if last_hidden is None:
                    raise RuntimeError(
                        "SGLang selected-layer hidden capture requires last_hidden_states "
                        "or teacher_hidden_states for the final selected layer fallback."
                    )
                selected_layers = len(self.target_layer_ids)
                selected_raw_hidden = self._reshape_selected_layer_hidden(
                    selected_raw_hidden,
                    last_hidden,
                    selected_layers,
                )
                aux_rows = torch.split(
                    selected_raw_hidden,
                    [len(row) for row in token_rows],
                    dim=0,
                )
                selected_layer_hidden_states = _pad_local_kv_rows(
                    [row.contiguous() for row in aux_rows]
                )
        return (
            batch,
            hidden_states,
            selected_layer_hidden_states,
            teacher_hidden_states_are_post_norm,
        )

    @staticmethod
    def _reshape_selected_layer_hidden(
        aux_hidden: torch.Tensor,
        last_hidden: Optional[torch.Tensor],
        selected_layers: int,
    ) -> torch.Tensor:
        if aux_hidden.ndim == 3:
            if aux_hidden.shape[1] == selected_layers:
                return aux_hidden.contiguous()
            if aux_hidden.shape[1] == selected_layers - 1 and last_hidden is not None:
                return torch.cat([aux_hidden, last_hidden[:, None, :]], dim=1).contiguous()
            raise RuntimeError(
                "SGLang aux hidden states must have selected_layers or one fewer "
                f"layers, got shape={tuple(aux_hidden.shape)} selected_layers={selected_layers}"
            )
        if aux_hidden.ndim != 2:
            raise RuntimeError(
                "SGLang aux hidden states must be rank 2 or 3, got "
                f"shape={tuple(aux_hidden.shape)}"
            )
        if (
            selected_layers > 1
            and aux_hidden.shape[-1] % (selected_layers - 1) == 0
            and last_hidden is not None
        ):
            hidden_size = aux_hidden.shape[-1] // (selected_layers - 1)
            if last_hidden.shape[-1] == hidden_size:
                # Prefer the explicit final-hidden fallback when it is
                # shape-consistent.  For two selected layers, a single
                # captured H-wide auxiliary tensor is also divisible by two;
                # interpreting it as two H/2 layers silently breaks HF parity.
                aux_hidden = aux_hidden.view(
                    aux_hidden.shape[0],
                    selected_layers - 1,
                    hidden_size,
                )
                return torch.cat([aux_hidden, last_hidden[:, None, :]], dim=1)
        if aux_hidden.shape[-1] % selected_layers == 0:
            hidden_size = aux_hidden.shape[-1] // selected_layers
            return aux_hidden.view(aux_hidden.shape[0], selected_layers, hidden_size)
        raise RuntimeError(
            "SGLang flattened aux hidden width is not divisible by selected layer "
            f"count or one fewer layer: shape={tuple(aux_hidden.shape)} "
            f"selected_layers={selected_layers}"
        )

    def _extract_selected_kv(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        with profile_range("local_kv_extraction"):
            token_to_kv = self.model_runner.token_to_kv_pool_allocator.get_kvcache()
            req_map = self.model_runner.req_to_token_pool.req_to_token
            req_indices = batch.req_pool_indices.detach().cpu().tolist()
            seq_lens = batch.seq_lens.detach().cpu().tolist()
        selected_keys = []
        selected_values = []
        for layer_index in self.target_layer_ids:
            with profile_range("local_kv_extraction"):
                key_buffer = token_to_kv.get_key_buffer(layer_index)
                value_buffer = token_to_kv.get_value_buffer(layer_index)
                local_keys = []
                local_values = []
                for req_pool_idx, seq_len in zip(req_indices, seq_lens):
                    loc = req_map[req_pool_idx, :seq_len].to(key_buffer.device)
                    local_keys.append(key_buffer.index_select(0, loc))
                    local_values.append(value_buffer.index_select(0, loc))
                key = _pad_local_kv_rows(local_keys)
                value = _pad_local_kv_rows(local_values)
            key = _gather_tp_head_shards(key)
            value = _gather_tp_head_shards(value)
            with profile_range("local_kv_extraction"):
                selected_keys.append(key.transpose(1, 2).contiguous())
                selected_values.append(value.transpose(1, 2).contiguous())
        with profile_range("local_kv_extraction"):
            return torch.stack(selected_keys, dim=1), torch.stack(selected_values, dim=1)

    def _clear_request_memory(self) -> None:
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()


def get_draftfreekv_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "hf",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DraftFreeKVTargetModel:
    if backend == "hf":
        return HFDraftFreeKVTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    if backend == "sglang":
        return SGLangDraftFreeKVTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    raise ValueError(f"Invalid DraftFreeKV target backend: {backend}")


__all__ = [
    "DraftFreeKVTargetModel",
    "DraftFreeKVTargetOutput",
    "HFDraftFreeKVTargetModel",
    "SGLangDraftFreeKVTargetModel",
    "get_draftfreekv_target_model",
    "stack_selected_kv",
]
