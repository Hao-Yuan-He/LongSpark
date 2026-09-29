"""CPU-only slot bookkeeping, NOT a draft model or a CUDA context."""
import torch


class VanillaSlots:
    is_vanilla = True

    def __init__(self, config, *, target_ready):
        if config.draft_device is not None:
            raise ValueError('vanilla must not reserve a draft GPU')
        self.capacity = config.max_running_requests
        self.device = torch.device('cpu')
        self.allocated_state_bytes = 0
        self.position_audit = target_ready[0]['position_audit']

    def update(self, packets, *, prefill=False):
        if packets:
            raise RuntimeError('vanilla target must not export draft states')

    def release(self, ids):
        pass

    def graph_stats(self):
        return dict(enabled=False, captured_batch_sizes=[], replays=0, fallbacks=0,
                    note='no draft model, no draft GPU')
