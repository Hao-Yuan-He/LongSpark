"""Opt-in measured BF16 cuBLASLt choices for Qwen3 Target verification.

Build jit_kernel/csrc/target_verify_gemm.cpp with cuBLASLt and supply the
registered library/config paths. Selection never searches during inference.
All unsupported models, shapes, quantization and forward modes use SGLang.
"""

import ctypes as ct
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class _Runtime:
    def __init__(self, config_path, library_path):
        self.config = json.loads(Path(config_path).read_text())
        assert self.config['version'] == 1
        assert hashlib.sha256(Path(library_path).read_bytes()).hexdigest() == self.config['library_sha256']
        self.lib = ct.CDLL(library_path)
        ptr = ct.c_void_p
        self.lib.make_plan.argtypes = [ct.c_int]*4 + [ct.c_size_t,ct.POINTER(ct.c_int)]
        self.lib.make_plan.restype = ptr
        self.lib.matmul.argtypes = [ptr,ct.c_int,ptr,ptr,ptr,ptr,ct.c_size_t,ptr]
        self.lib.matmul.restype = ct.c_int
        self.lib.export_algorithm.argtypes = [ptr,ct.c_int,ptr,ct.c_size_t]
        self.lib.export_algorithm.restype = ct.c_size_t
        self.lib.release_plan.argtypes = [ptr]
        self.lib.lt_version.restype = ct.c_size_t
        assert self.lib.lt_version() == self.config['lt_version']
        self.entries = {(e['m'],e['n'],e['k']):e for e in self.config['entries']}
        assert len(self.entries) == len(self.config['entries'])
        self.plans = {}

    def apply(self, x, weight):
        shape = (x.shape[0],weight.shape[0],x.shape[1])
        entry = self.entries.get(shape)
        if entry is None:
            return None
        key = (x.device.index,shape)
        cached = self.plans.get(key)
        if cached is None:
            # Normal graph warmup creates descriptors and the workspace. An
            # unexpected first call inside capture must take the native path.
            if torch.cuda.is_current_stream_capturing():
                return None
            props = torch.cuda.get_device_properties(x.device)
            if ([props.major,props.minor] != self.config['compute_capability']
                    or props.multi_processor_count != self.config['multiprocessor_count']):
                return None
            workspace = torch.empty(self.config['workspace_bytes'],device=x.device,dtype=torch.uint8)
            count = ct.c_int()
            plan = self.lib.make_plan(*shape,16,workspace.numel(),ct.byref(count))
            if not plan:
                raise RuntimeError(f'Cannot create registered Target GEMM plan {shape}')
            blob = ct.create_string_buffer(256)
            length = self.lib.export_algorithm(plan,entry['candidate'],blob,256)
            if not 0 < length <= 256 or blob.raw[:length].hex() != entry['algorithm_hex']:
                self.lib.release_plan(plan)
                raise RuntimeError(f'Registered Target GEMM algorithm changed for {shape}')
            cached = self.plans[key] = (plan,workspace)
            logger.info('Target verify GEMM selected shape=%s algorithm_sha256=%s',
                        shape,hashlib.sha256(blob.raw[:length]).hexdigest())
        plan,workspace = cached
        y = torch.empty((shape[0],shape[1]),device=x.device,dtype=x.dtype)
        code = self.lib.matmul(plan,entry['candidate'],x.data_ptr(),weight.data_ptr(),y.data_ptr(),
                               workspace.data_ptr(),workspace.numel(),torch.cuda.current_stream(x.device).cuda_stream)
        if code:
            raise RuntimeError(f'Target GEMM failed with cuBLASLt status {code}')
        return y


@lru_cache(maxsize=1)
def _runtime(config_path, library_path):
    return _Runtime(config_path, library_path)


def target_verify_gemm(x, layer, forward_batch, config_path, library_path):
    if (not config_path or forward_batch is None
            or not forward_batch.forward_mode.is_target_verify()
            or not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous()):
        return None
    from sglang.srt.server_args import get_global_server_args
    args = get_global_server_args()
    if (args.tp_size != 1 or args.speculative_num_draft_tokens != 8
            or args.speculative_algorithm not in ('DSPARK','DRAFT_FREE_KV')
            or args.enable_deterministic_inference or args.rl_on_policy_target is not None
            or not args.disable_overlap_schedule
            or type(layer.quant_method).__name__ != 'UnquantizedLinearMethod'
            or layer.bias is not None):
        return None
    weight = layer.weight
    if (weight.dtype != torch.bfloat16 or weight.device != x.device
            or weight.ndim != 2 or not weight.is_contiguous() or weight.shape[1] != x.shape[1]):
        return None
    return _runtime(config_path,library_path).apply(x,weight)
