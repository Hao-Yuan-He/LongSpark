"""Lifecycle glue for SGLang model runners constructed without a Scheduler."""

from array import array


def complete_direct_runner_initialization(runner) -> None:
    """Finish the split SGLang runtime lifecycle for a directly built runner.

    Recent SGLang versions leave memory-pool allocation, attention-backend
    setup, and eager-runner construction to ``Scheduler``. SpecForge creates a
    ``ModelRunner`` directly for offline teacher forwards, so it must perform
    those same phases itself. Attribute guards keep this safe for already
    initialized runners.
    """

    if (
        getattr(runner, "req_to_token_pool", None) is None
        or getattr(runner, "token_to_kv_pool_allocator", None) is None
    ):
        runner.alloc_memory_pool()

    if (
        getattr(runner, "req_to_token_pool", None) is None
        or getattr(runner, "token_to_kv_pool_allocator", None) is None
    ):
        raise RuntimeError("SGLang direct runner did not allocate its memory pools")

    if getattr(runner, "attn_backend", None) is None:
        runner.init_attention_backends()

    if getattr(runner, "eager_runner", None) is None:
        runner.init_cuda_graphs()


def prepare_direct_extend_request(request, *, tree_cache) -> None:
    """Populate the current SGLang request fields for a full eager prefill."""

    if not (
        isinstance(request.origin_input_ids, array)
        and request.origin_input_ids.typecode == "q"
    ):
        request.origin_input_ids = array("q", request.origin_input_ids)
    unpadded = getattr(request, "origin_input_ids_unpadded", None)
    if unpadded is not None and not (
        isinstance(unpadded, array) and unpadded.typecode == "q"
    ):
        request.origin_input_ids_unpadded = array("q", unpadded)

    request.init_next_round_input(tree_cache=tree_cache)
    prefix_len = len(request.prefix_indices)
    full_len = len(request.full_untruncated_fill_ids)
    request.set_extend_range(prefix_len, full_len)
    request.extend_input_len = full_len - prefix_len


def build_direct_forward_batch(
    batch,
    model_runner,
    *,
    forward_batch_cls,
    capture_hidden_mode,
):
    """Build a ForwardBatch through SGLang's current direct conversion path."""

    if getattr(batch, "input_ids", None) is None:
        staged_input_ids = getattr(batch, "prefill_input_ids_cpu", None)
        if staged_input_ids is None:
            raise RuntimeError("SGLang direct prefill has no staged input IDs")
        batch.input_ids = staged_input_ids.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None

    batch.capture_hidden_mode = capture_hidden_mode
    return forward_batch_cls.init_new(batch, model_runner)
