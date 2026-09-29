from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch


DRAFT_FREE_ALGORITHM = "DRAFT_FREE_KV"
REQUIRED_HEAD_TYPE = "query-lora-lm-head"
HIDDEN_CONDITIONED_HEAD_TYPE = "hidden-conditioned-query-lora-lm-head"
TARGET_LM_HEAD_PROJECTION_HEAD_TYPE = "target-lm-head-projection"
SUPPORTED_LEGACY_HEAD_TYPES = {REQUIRED_HEAD_TYPE, HIDDEN_CONDITIONED_HEAD_TYPE}
SUPPORTED_HEAD_TYPES = {
    *SUPPORTED_LEGACY_HEAD_TYPES,
    TARGET_LM_HEAD_PROJECTION_HEAD_TYPE,
}
REQUIRED_BLOCK_ATTENTION = "causal"
PARALLEL_BLOCK_ATTENTION = "bidirectional"
PARALLEL_SHIFTED_DRAFT_MODES = {
    "parallel_shifted",
    "parallel_shifted_qwen",
}
TARGET_PLUS_LORA_DRAFT_LM_HEAD = "unified-target-plus-lora"
EAGLE3_STYLE_DRAFT_VOCAB_HEAD = "eagle3-style-draft-vocab"
SUPPORTED_DRAFT_LM_HEADS = {TARGET_PLUS_LORA_DRAFT_LM_HEAD, EAGLE3_STYLE_DRAFT_VOCAB_HEAD}


def _allow_nondeterministic_inference() -> bool:
    """Return the explicit throughput-only opt-in used by LongSpark.

    Correctness and exact-output checks keep deterministic inference as their
    default.  Production-style throughput runs may opt out because the
    batch-invariant operators add work that is outside the maintained serving
    profile.
    """

    value = os.environ.get(
        "SGLANG_DRAFT_FREE_KV_ALLOW_NONDETERMINISTIC",
        "0",
    )
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DraftFreeKVManifest:
    path: Path
    schema_version: int
    head_type: str
    block_size: int
    proposal_block_size: int
    verify_block_size: int
    draft_mode: str
    window_size: int
    selected_layers: tuple[int, ...]
    selected_layer_policy: str
    state_dict_path: Path
    config_path: Optional[Path]
    target_model_id: Optional[str]
    target_revision: Optional[str]
    target_arch_signature: Optional[str]
    draft_lm_head: Optional[str]
    draft_vocab_size: Optional[int]


@dataclass(frozen=True)
class DraftFreeKVRuntimeConfig:
    block_size: int
    num_steps: int
    verify_token_num: int
    draft_slots_per_verify: int

    @classmethod
    def from_values(cls, *, block_size: int, num_steps: int) -> "DraftFreeKVRuntimeConfig":
        block_size = int(block_size)
        num_steps = int(num_steps)
        if block_size <= 1:
            raise ValueError("DRAFT_FREE_KV requires block_size > 1")
        if num_steps < 1:
            raise ValueError("DRAFT_FREE_KV requires speculative_num_steps >= 1")
        draft_slots_per_verify = num_steps * (block_size - 1)
        return cls(
            block_size=block_size,
            num_steps=num_steps,
            verify_token_num=1 + draft_slots_per_verify,
            draft_slots_per_verify=draft_slots_per_verify,
        )

    @classmethod
    def from_server_args(
        cls,
        server_args,
        manifest: Optional[DraftFreeKVManifest] = None,
    ) -> "DraftFreeKVRuntimeConfig":
        block_size = getattr(server_args, "draft_free_block_size", None)
        if block_size is None:
            block_size = getattr(server_args, "speculative_num_draft_tokens", None)
        if block_size is None and manifest is not None:
            block_size = manifest.block_size
        if block_size is None:
            raise ValueError("DRAFT_FREE_KV requires a block size")
        return cls.from_values(
            block_size=int(block_size),
            num_steps=int(getattr(server_args, "speculative_num_steps", None) or 1),
        )


def _as_int_tuple(values: Iterable[object], *, field_name: str) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> DraftFreeKVManifest:
    root = Path(path)
    if root.is_file() and root.suffix == ".pt":
        raise ValueError(
            "Legacy .pt draft-free checkpoints must be exported to a directory "
            "with manifest.json/model.pt/config.json before SGLang serving."
        )
    if not root.is_dir():
        raise ValueError(f"Draft-free checkpoint path must be a directory: {root}")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing draft-free manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    config_path = None
    config_data = None
    config_file = data.get("config_file")
    if config_file:
        candidate_config_path = root / config_file
        if candidate_config_path.is_file():
            config_path = candidate_config_path
            with config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)

    head_type = data.get("head_type")
    if head_type not in SUPPORTED_HEAD_TYPES:
        raise ValueError(
            f"DRAFT_FREE_KV requires one of head_type={sorted(SUPPORTED_HEAD_TYPES)!r}, got {head_type!r}"
        )
    is_specforge_target_projection = head_type == TARGET_LM_HEAD_PROJECTION_HEAD_TYPE
    if is_specforge_target_projection:
        if config_data is None or config_data.get("model_type") != "draftfreekv":
            raise ValueError(
                "DRAFT_FREE_KV head_type='target-lm-head-projection' is supported only "
                "for SpecForge DraftFreeKV checkpoints with config_file model_type='draftfreekv'"
            )
    draft_mode = str(
        (config_data or {}).get("draft_mode", data.get("draft_mode", "markov"))
    ).lower()
    expected_attention = (
        PARALLEL_BLOCK_ATTENTION
        if draft_mode in PARALLEL_SHIFTED_DRAFT_MODES
        else REQUIRED_BLOCK_ATTENTION
    )
    if data.get("draft_block_attention") != expected_attention:
        raise ValueError(
            "DRAFT_FREE_KV draft_block_attention must be "
            f"{expected_attention!r} for draft_mode={draft_mode!r}"
        )
    draft_lm_head = data.get("draft_lm_head")
    if not is_specforge_target_projection and draft_lm_head not in SUPPORTED_DRAFT_LM_HEADS:
        raise ValueError(
            f"DRAFT_FREE_KV requires draft_lm_head in {sorted(SUPPORTED_DRAFT_LM_HEADS)!r}, got {draft_lm_head!r}"
        )

    state_dict_path = root / data.get("state_dict", "model.pt")
    if not state_dict_path.is_file():
        raise ValueError(f"Missing draft-free state dict: {state_dict_path}")
    expected_sha = data.get("state_dict_sha256")
    if expected_sha and _sha256(state_dict_path) != expected_sha:
        raise ValueError(f"state_dict_sha256 mismatch for {state_dict_path}")

    selected_layers = data.get("selected_layers")
    if selected_layers is None:
        selected_layers = data.get("config", {}).get("selected_layers")
    window_size = data.get("window_size")
    if window_size is None:
        window_size = data.get("config", {}).get("window_size")
    if selected_layers is None or window_size is None:
        raise ValueError("Draft-free manifest must include selected_layers and window_size")

    block_size = int(data["block_size"])
    proposal_block_size = int(
        data.get("proposal_block_size", max(1, block_size - 1))
    )
    verify_block_size = int(
        data.get("verify_block_size", 1 + proposal_block_size)
    )
    if proposal_block_size <= 0:
        raise ValueError("proposal_block_size must be positive")
    if verify_block_size != proposal_block_size + 1:
        # Old Markov manifests wrote advisory proposal/verify fields that did
        # not match their runtime block_size. Preserve those checkpoints while
        # making the new parallel schema strict and unambiguous.
        if draft_mode in PARALLEL_SHIFTED_DRAFT_MODES:
            raise ValueError(
                "parallel shifted verify_block_size must equal proposal_block_size + 1"
            )

    return DraftFreeKVManifest(
        path=root,
        schema_version=int(data.get("schema_version", 1)),
        head_type=str(head_type),
        block_size=block_size,
        proposal_block_size=proposal_block_size,
        verify_block_size=verify_block_size,
        draft_mode=draft_mode,
        window_size=int(window_size),
        selected_layers=_as_int_tuple(selected_layers, field_name="selected_layers"),
        selected_layer_policy=str(data.get("selected_layer_policy", data.get("config", {}).get("selected_layer_policy", "checkpoint"))),
        state_dict_path=state_dict_path,
        config_path=config_path,
        target_model_id=data.get("target_model_id"),
        target_revision=data.get("target_revision"),
        target_arch_signature=data.get("target_arch_signature"),
        draft_lm_head=str(draft_lm_head) if draft_lm_head is not None else None,
        draft_vocab_size=(
            int(data["draft_vocab_size"])
            if data.get("draft_vocab_size") is not None
            else None
        ),
    )


def resolve_window_size(arg_value: Optional[str], manifest: DraftFreeKVManifest) -> int:
    if arg_value in (None, "", "checkpoint"):
        return manifest.window_size
    value = int(arg_value)
    if value <= 0:
        raise ValueError(f"draft-free window size must be positive, got {value}")
    return value


def resolve_selected_layers(arg_value: Optional[str], manifest: DraftFreeKVManifest) -> tuple[int, ...]:
    if arg_value in (None, "", "checkpoint"):
        return manifest.selected_layers
    return _as_int_tuple(arg_value.split(","), field_name="draft-free selected layers")


def handle_draft_free_kv_server_args(server_args) -> None:
    """Apply the current LongSpark serving contract to the official suite.

    This is intentionally limited to translating the official suite's
    algorithm hook into the existing Global16+Raw256 adapter contract.  It
    does not enable any new scheduling or model optimizations.
    """

    server_args.speculative_algorithm = DRAFT_FREE_ALGORITHM
    if getattr(server_args, "speculative_draft_model_revision", None) is None:
        server_args.speculative_draft_model_revision = "main"
    if not str(getattr(server_args, "device", "")).startswith("cuda"):
        raise ValueError("DRAFT_FREE_KV only supports CUDA devices")
    deterministic = getattr(server_args, "enable_deterministic_inference", False)
    if not deterministic and not _allow_nondeterministic_inference():
        raise ValueError(
            "DRAFT_FREE_KV nondeterministic inference requires "
            "SGLANG_DRAFT_FREE_KV_ALLOW_NONDETERMINISTIC=1"
        )
    if int(getattr(server_args, "pp_size", 1)) != 1:
        raise ValueError("DRAFT_FREE_KV only supports pp_size == 1")
    if int(getattr(server_args, "page_size", 1)) != 1:
        raise ValueError("DRAFT_FREE_KV only supports page_size == 1")
    if not getattr(server_args, "speculative_draft_model_path", None):
        raise ValueError("DRAFT_FREE_KV requires --speculative-draft-model-path")
    if getattr(server_args, "draft_free_head_runner", "torch") != "torch":
        raise ValueError("DRAFT_FREE_KV only supports --draft-free-head-runner torch")
    if getattr(server_args, "draft_free_kv_reader", "torch") != "torch":
        raise ValueError("DRAFT_FREE_KV only supports --draft-free-kv-reader torch")
    if getattr(server_args, "speculative_num_steps", None) is None:
        server_args.speculative_num_steps = 1
    if int(server_args.speculative_num_steps) != 1:
        raise ValueError("DRAFT_FREE_KV only supports speculative_num_steps == 1")
    if getattr(server_args, "speculative_eagle_topk", None) not in (None, 1):
        raise ValueError("DRAFT_FREE_KV only supports speculative_eagle_topk == 1")

    manifest = load_manifest(server_args.speculative_draft_model_path)
    runtime = DraftFreeKVRuntimeConfig.from_server_args(server_args, manifest)
    if runtime.verify_token_num != manifest.verify_block_size:
        raise ValueError(
            "DRAFT_FREE_KV runtime verify width does not match checkpoint: "
            f"runtime={runtime.verify_token_num}, checkpoint={manifest.verify_block_size}"
        )
    resolve_window_size(getattr(server_args, "draft_free_window_size", None), manifest)
    resolve_selected_layers(
        getattr(server_args, "draft_free_selected_layers", None), manifest
    )
    server_args.speculative_eagle_topk = 1
    server_args.draft_free_block_size = runtime.block_size
    server_args.speculative_num_draft_tokens = runtime.block_size
    server_args.disable_overlap_schedule = True
    if hasattr(server_args, "enable_mixed_chunk"):
        server_args.enable_mixed_chunk = False


def reshape_selected_target_hidden_states(
    aux_hidden_states: torch.Tensor,
    last_hidden_states: Optional[torch.Tensor],
    *,
    selected_layers: int,
    hidden_size: int,
) -> torch.Tensor:
    """Restore SGLang's flattened aux capture to ``[tokens, layers, hidden]``."""

    if aux_hidden_states.ndim == 3:
        if aux_hidden_states.shape[1:] != (selected_layers, hidden_size):
            raise RuntimeError(
                "Unexpected DRAFT_FREE_KV selected hidden shape: "
                f"{tuple(aux_hidden_states.shape)}"
            )
        return aux_hidden_states
    if aux_hidden_states.ndim != 2:
        raise RuntimeError(
            "DRAFT_FREE_KV selected target hidden must be rank 2 or 3, got "
            f"{tuple(aux_hidden_states.shape)}"
        )
    token_count, width = aux_hidden_states.shape
    if width == selected_layers * hidden_size:
        return aux_hidden_states.view(token_count, selected_layers, hidden_size)
    if width == (selected_layers - 1) * hidden_size:
        if last_hidden_states is None:
            raise RuntimeError(
                "DRAFT_FREE_KV selected target hidden is missing the final layer"
            )
        if last_hidden_states.shape != (token_count, hidden_size):
            raise RuntimeError(
                "DRAFT_FREE_KV final hidden does not align with aux capture: "
                f"aux={tuple(aux_hidden_states.shape)} "
                f"last={tuple(last_hidden_states.shape)}"
            )
        aux = aux_hidden_states.view(
            token_count,
            selected_layers - 1,
            hidden_size,
        )
        return torch.cat([aux, last_hidden_states[:, None, :]], dim=1)
    raise RuntimeError(
        "DRAFT_FREE_KV flattened selected hidden width mismatch: "
        f"width={width} selected_layers={selected_layers} hidden_size={hidden_size}"
    )


def compute_greedy_accept_len_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep greedy verification on device and return its target bonus token.

    ``candidates[:, 0]`` is the already committed current token.  A draft token
    is accepted only while every preceding candidate agrees with Target.  The
    returned length therefore excludes both the current token and the Target
    bonus token, matching the existing DRAFT_FREE_KV contract.
    """

    if candidates.ndim != 2 or target_predict.shape != candidates.shape:
        raise ValueError(
            "candidates and target_predict must have the same rank-2 shape"
        )
    if candidates.shape[1] < 2:
        raise ValueError("DRAFT_FREE_KV verification requires at least two rows")

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_lens = matches.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
    bonus = target_predict.gather(1, accept_lens.to(torch.long)[:, None]).squeeze(1)
    return accept_lens, bonus.to(torch.int64)


def build_greedy_verify_out_tokens(
    *,
    candidates: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed-width committed-token buffer without a host round-trip."""

    if candidates.ndim != 2 or accept_lens.shape != (candidates.shape[0],):
        raise ValueError("DRAFT_FREE_KV verify tensors have incompatible shapes")
    if bonus.shape != (candidates.shape[0],):
        raise ValueError("DRAFT_FREE_KV bonus must have one token per request")

    batch_size, verify_width = candidates.shape
    out_tokens = torch.empty(
        (batch_size, verify_width),
        dtype=torch.int64,
        device=candidates.device,
    )
    out_tokens[:, :-1].copy_(candidates[:, 1:].to(torch.int64))
    out_tokens[:, -1].zero_()
    out_tokens.scatter_(1, accept_lens.to(torch.long)[:, None], bonus[:, None])
    return out_tokens


def compute_greedy_accept_lengths(candidates, target_predict):
    """Compatibility wrapper for legacy callers that still need host integers."""

    accept_lens, _ = compute_greedy_accept_len_and_bonus(
        candidates=candidates,
        target_predict=target_predict,
    )
    return accept_lens.cpu().tolist()
