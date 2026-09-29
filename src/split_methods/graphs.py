"""Opt-in native SGLang graph setup, after independent cache allocation."""


def enable_native_graphs(runner, config, *, hidden=False):
    args = runner.server_args
    args.disable_cuda_graph = False
    args.enable_return_hidden_states = hidden
    args.cuda_graph_config.decode.backend = "full"
    args.cuda_graph_config.decode.bs = config.graph_batch_sizes
    args.cuda_graph_config.decode.max_bs = config.max_running_requests
    args.cuda_graph_config.prefill.backend = "disabled"
    runner.init_cuda_graphs()
    graph = runner.decode_cuda_graph_runner
    if graph is None or not hasattr(graph, "capture_bs"):
        raise RuntimeError("CUDA Graph requested but native capture was not installed")


def native_graph_stats(runner, replays, fallbacks):
    graph = runner.decode_cuda_graph_runner
    return dict(captured_batch_sizes=list(getattr(graph, "capture_bs", ())),
                replays=replays, fallbacks=fallbacks)
