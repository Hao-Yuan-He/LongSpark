"""Device placement and cache budgets for a split target/draft run."""

from dataclasses import dataclass
import math


GIB = 1024**3


@dataclass(frozen=True)
class SplitConfig:
    target_tp: int = 2
    target_devices: tuple[int, ...] = (0, 1)
    draft_device: int | None = 2
    # CUDA P2P is the default; on NVLink-connected GPUs CUDA selects that path.
    # `host` is an explicit CPU-staged control, NOT a PCIe-P2P simulation.
    transport: str = "p2p"
    target_kv_gib_per_rank: float = 16.0
    draft_state_gib: float = 8.0
    max_running_requests: int = 128
    cuda_graph: bool = False
    # Correctness runs opt into batch-invariant target kernels; benchmark
    # defaults remain unchanged.
    deterministic_inference: bool = False
    # Experimental fused gather is off: it saved little in the microbenchmark
    # and the combined v1 candidate regressed on a long-input C8 workload.
    fuse_kv_export: bool = False
    context_length: int = 140288
    prefill_chunk_size: int = 8192
    position_variant: str = "fixed_ntk"
    eos_token_ids: tuple[int, ...] = (151645, 151643)

    @property
    def generation_context_limit(self):
        # Native service capacity includes eight transient verify rows;
        # these do not enlarge the public prompt+response token budget.
        return self.context_length - 8

    @property
    def graph_batch_sizes(self):
        return sorted({n for n in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
                       if n <= self.max_running_requests} | {self.max_running_requests})

    def __post_init__(self):
        contexts = {'native': 40968, 'fixed_ntk': 140288}
        if self.position_variant not in contexts or self.context_length != contexts[self.position_variant]:
            raise ValueError('select native/40968 (40960 + 8 scratch) or fixed_ntk/140288')
        if not 1 <= self.prefill_chunk_size <= 16384:
            raise ValueError("prefill chunk size must be in 1..16384")
        if self.target_tp not in (1, 2, 4):
            raise ValueError("initially supported target TP sizes are 1, 2, and 4")
        if len(self.target_devices) != self.target_tp:
            raise ValueError("target_devices must contain exactly target_tp devices")
        devices = (*self.target_devices,) if self.draft_device is None else (*self.target_devices, self.draft_device)
        if min(devices) < 0 or len(set(devices)) != len(devices):
            raise ValueError("target ranks and draft require distinct nonnegative GPU IDs")
        if self.transport not in ("p2p", "auto", "host"):
            raise ValueError("transport must be p2p, auto, or host")
        budgets = (self.target_kv_gib_per_rank,) if self.draft_device is None else (self.target_kv_gib_per_rank, self.draft_state_gib)
        for budget in budgets:
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("cache budgets must be finite and positive")
        if not 1 <= self.max_running_requests <= 128:
            raise ValueError("this experiment supports concurrency from 1 to 128")


def kv_bytes_per_token(*, layers, kv_heads, head_dim, element_bytes=2, tp=1):
    if min(layers, kv_heads, head_dim, element_bytes, tp) <= 0:
        raise ValueError("KV dimensions and TP must be positive")
    if kv_heads % tp:
        raise ValueError("this runtime requires even KV-head sharding (no replication)")
    return layers * 2 * (kv_heads // tp) * head_dim * element_bytes

