"""Release graph/IPC owners while CUDA and native TP groups are still alive."""

import gc
import faulthandler
import json
import os
import torch


def trace_shutdown(stage, *, begin=False):
    """Opt-in diagnostics outside measured forwards, including native crashes."""
    if os.environ.get("SPLIT_TRACE_SHUTDOWN") != "1":
        return
    if begin:
        # Native dependencies can replace handlers installed at interpreter
        # startup. Reinstall only after the final measured workload is done.
        faulthandler.disable()
        faulthandler.enable(all_threads=True)
    print(json.dumps(dict(phase="shutdown", pid=os.getpid(), stage=stage)), flush=True)


def release_native_graphs(runner):
    if runner is None:
        return
    for name in ("decode_cuda_graph_runner", "prefill_cuda_graph_runner"):
        graph = getattr(runner, name, None)
        if graph is not None:
            trace_shutdown(f"native_graph:{name}:begin")
            backend = getattr(graph, "backend", None)
            if backend is not None:
                backend.cleanup()
            setattr(runner, name, None)
            trace_shutdown(f"native_graph:{name}:end")


def release_draft_graphs(draft):
    """Explicitly break callback cycles instead of interpreter-finalizer order."""
    if draft is None:
        return
    trace_shutdown("draft_graphs:synchronize")
    if draft.device.type == 'cuda':
        torch.cuda.synchronize(draft.device)
    graph = getattr(draft, "graph", None)
    if graph is not None:
        for state in graph._states.values():
            trace_shutdown("longspark_graph:reset")
            state.graph.reset()
        graph._states.clear()
        graph.proposal_fn = graph.context_factory = graph._inputs = None
        draft.graph = None
    for name in ('logits_graph', 'hidden_projection_graph'):
        graph = getattr(draft, name, None)
        if graph is not None:
            for _, captured, _ in graph.states.values():
                captured.reset()
            graph.states.clear()
            setattr(draft, name, None)
    release_native_graphs(getattr(draft, "runner", None))
    trace_shutdown("draft_graphs:done")


def cleanup_parallel_runtime():
    # torch.destroy_process_group() alone leaves SGLang's GroupCoordinator
    # objects (custom collectives and shared-memory queues) in module globals.
    # GroupCoordinator.destroy() currently destroys cpu_group before dropping
    # ca_comm. CustomAllReduceV2.close() needs that group for its IPC-release
    # barrier, so close custom collectives first while every rank is alive.
    from sglang.srt.distributed import parallel_state

    trace_shutdown("parallel:gc")
    gc.collect()
    groups = [reference() for _, reference in sorted(parallel_state._groups.items())]
    close_custom_collectives(groups)
    trace_shutdown("parallel:native_cleanup")
    parallel_state.cleanup_dist_env_and_memory()
    trace_shutdown("parallel:done")


def close_custom_collectives(groups):
    """Close each custom communicator once, before native groups are destroyed.

    Only our dedicated target/draft processes use this adapter. Do not patch
    the shared library or disable custom collectives during measured forwards.
    The upstream V2 close is not idempotent, so disarm its later destructor
    after successful release; failures remain visible and stop the run.
    """
    closed = set()
    for group in groups:
        comm = getattr(group, "ca_comm", None)
        if comm is None:
            continue
        if id(comm) not in closed:
            trace_shutdown(f"custom_collective:{type(comm).__name__}:close")
            comm.close()
            trace_shutdown(f"custom_collective:{type(comm).__name__}:closed")
            # V2 otherwise calls obj.free(group) again from __del__. Its VMM
            # manager must also not be asked to release the same maps twice.
            comm.disabled = True
            if hasattr(comm, "_vmm_graph_input_manager"):
                del comm._vmm_graph_input_manager
            closed.add(id(comm))
        group.ca_comm = None
