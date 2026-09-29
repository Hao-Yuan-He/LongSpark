"""GPU selection checks and the environment passed to worker processes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .paths import SRC_DIR, VENDOR_PATHS

# Kernel selections read by the vendored SGLang runtime.
KERNEL_FLAGS = dict(
    SGLANG_USE_SGL_FA3_KERNEL='1',
    DFK_GLOBAL16_FUSED_QK_NORM_ROPE='0',
    SGLANG_DSPARK_FUSED_QK_NORM_ROPE='0',
    DFK_GLOBAL16_QK_SGLANG_RMSNORM='1',
    DFK_GLOBAL16_FUSED_SILU='1',
    DFK_GLOBAL16_DRAFT_FUSED_ALL_ATTENTION='1',
    DFK_GLOBAL16_FAST_METADATA='1',
    DFK_GLOBAL16_MARKOV_BFLOAT16='1',
)


def worker_environment(gpus):
    """Environment for a worker that sees exactly `gpus`, with CUDA 12.8 on PATH."""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
        PYTHONPATH=os.pathsep.join(map(str, [SRC_DIR, *VENDOR_PATHS])),
        PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
        **KERNEL_FLAGS,
    )
    cuda = Path(env.get('CUDA_HOME', '/usr/local/cuda'))
    if not (cuda / 'bin/nvcc').is_file() or not (cuda / 'include/cuda.h').is_file():
        raise RuntimeError('Set CUDA_HOME to a CUDA 12.8 toolkit (nvcc and include/cuda.h required).')
    version = subprocess.check_output([str(cuda / 'bin/nvcc'), '--version'], text=True)
    if 'release 12.8' not in version:
        raise RuntimeError('This runtime requires CUDA 12.8')
    env['CUDA_HOME'] = str(cuda)
    env['PATH'] = os.pathsep.join([str(cuda / 'bin'), str(Path(sys.executable).parent), env['PATH']])
    if shutil.which('ninja', path=env['PATH']) is None:
        raise RuntimeError('ninja is required for kernel compilation; activate the configured Conda environment.')
    env['LD_LIBRARY_PATH'] = str(cuda / 'lib64') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def require_idle_gpus(gpus):
    """Refuse to start on GPUs that already hold more than 64 MiB."""
    output = subprocess.check_output(
        ['nvidia-smi', '--id=' + ','.join(map(str, gpus)),
         '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    used = [int(x.strip()) for x in output.splitlines()]
    if len(used) != len(gpus) or any(x > 64 for x in used):
        raise RuntimeError(f'Selected GPUs are occupied: {used} MiB. No external tasks will be stopped.')
