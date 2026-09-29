"""Process-local patches applied to the engine inside every worker process.

Target TP ranks are started with the `spawn` method, so each child must apply
the same patches again; `patched_target_process` does that before running the
normal target entry point.
"""
import os


def use_gqa5_adapter(size, method):
    """Enable the GQA-5 attention adapter (14B LongSpark only) for this process and its children."""
    os.environ['LONGSPARK_GQA5'] = '1' if (size, method) == ('14B', 'longspark') else '0'


def apply_runtime_patches():
    from . import registered_ipc
    registered_ipc.install()
    if os.environ.get('LONGSPARK_GQA5') == '1':
        from . import gqa5_attention
        gqa5_attention.install()


def patched_target_process(*args):
    apply_runtime_patches()
    from ..engine.target import target_process_main
    return target_process_main(*args)


def install_target_patches():
    """Apply the patches here and make spawned target ranks apply them too."""
    from ..engine import target
    apply_runtime_patches()
    target.target_process_main = patched_target_process
