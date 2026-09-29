"""Per-request latency added to steady-state summaries (reporting only; no scheduling changes)."""
from ..engine.steady_metrics import distribution


def request_latency(rows, *, begin, end):
    """Completed-request TPOT, with explicit selection and censoring counts.

    Completion-selected requests can have started before the window. The
    separate fully-contained statistic requires the FIRST TOKEN, not request
    submission, to be inside the window. Neither is an uncensored full-cohort
    estimator. Pending/unfinished requests NEVER receive a fabricated TPOT.
    """
    if not 0 <= begin < end:
        raise ValueError('invalid observation window')
    observed = [r for r in rows if r['submitted_seconds'] < end and
                (r.get('completion_seconds') is None or r['completion_seconds'] >= begin)]
    completed = [r for r in observed if r.get('completion_seconds') is not None
                 and begin <= r['completion_seconds'] < end]
    unfinished = [r for r in observed if r.get('completion_seconds') is None
                  or r['completion_seconds'] >= end]
    tpots, contained, ttfts, e2es = [], [], [], []
    single_token = left_crossing = 0
    for row in completed:
        first, last = row['first_token_seconds'], row['completion_seconds']
        if first is None or not row['submitted_seconds'] <= first <= last:
            raise ValueError('invalid completed-request timestamps')
        n = row['output_tokens']
        if n < 1:
            raise ValueError('completed request must emit at least one token')
        ttfts.append(1000 * (first - row['submitted_seconds']))
        e2es.append(1000 * (last - row['submitted_seconds']))
        left_crossing += int(first < begin)
        if n == 1:
            single_token += 1
            continue
        # Match the existing seconds-to-ms evaluation order bit for bit.
        tpot = 1000 * ((last - first) / (n - 1))
        tpots.append(tpot)
        if first >= begin:
            contained.append(tpot)
    return dict(
        schema_version=1,
        selection='completion_seconds in [window_begin, window_end); request-equal mean',
        tpot_definition='1000 * (last_token_seconds - first_token_seconds) / (output_tokens - 1)',
        timestamp_source='coordinator emission after target response and draft-state commit; not HTTP/SSE arrival',
        tpot_ms=distribution(tpots),
        fully_contained_tpot_ms=distribution(contained),
        completed_request_ttft_ms=distribution(ttfts),
        completed_request_e2e_ms=distribution(e2es),
        observed_requests=len(observed), completed_requests=len(completed),
        tpot_eligible_requests=len(tpots), single_token_completions_excluded=single_token,
        completed_first_token_before_window=left_crossing,
        unfinished_at_window_end=len(unfinished),
        unfinished_before_first_token=sum(r.get('first_token_seconds') is None or
            r['first_token_seconds'] >= end for r in unfinished),
        completion_fraction=len(completed) / len(observed) if observed else None,
        completed_only_selection_bias_possible=bool(unfinished),
        limitation='No final drain; unfinished requests excluded from request TPOT, retained in original window metric. Not a full-cohort TPOT estimate.',
    )


def extend_summary(original, rows, *, begin, end):
    if 'request_latency' in original:
        raise ValueError('request latency extension installed twice')
    return dict(original, request_latency=request_latency(rows, begin=begin, end=end))


def install_request_latency_metrics():
    """Wrap the CPU reporting callback, which runs only after timed execution."""
    from ..engine import steady_benchmark as steady
    original = steady.summarize_window
    if getattr(original, '_request_latency_extension', False):
        return

    def augmented(rows, events, stages, occupancy, *, begin, end, concurrency):
        legacy = original(rows, events, stages, occupancy,
                          begin=begin, end=end, concurrency=concurrency)
        return extend_summary(legacy, rows, begin=begin, end=end)

    augmented._request_latency_extension = True
    steady.summarize_window = augmented
