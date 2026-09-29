"""CPU-only, explicitly window-clipped metrics for closed-loop serving.

The window aggregate is decoding request-seconds / post-first output tokens.
It includes boundary-crossing requests and waits AFTER their first token. It
is NOT wall seconds / total tokens, nor the unweighted mean request TPOT.
Token emissions are timestamped as bursts, as in speculative streaming.
"""

from collections import Counter
import math


def overlap(begin, end, left, right):
    return max(0., min(end, right) - max(begin, left))


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, p50=None, p90=None, p95=None, p99=None)

    def quantile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return dict(n=len(values), mean=sum(values) / len(values),
                **{f"p{int(100*q)}": quantile(q) for q in (.5, .9, .95, .99)})


def request_tpot(row):
    first, last = row.get("first_token_seconds"), row.get("completion_seconds")
    n = row["output_tokens"]
    return (last - first) / (n - 1) if first is not None and last is not None and n > 1 else None


def summarize_window(rows, events, stages, occupancy, *, begin, end, concurrency):
    if not 0 <= begin < end:
        raise ValueError("window must have positive duration and nonnegative start")
    duration = end - begin
    tokens = postfirst = rounds = accepted = raw_accepted = 0
    histogram, generated_by_source = Counter(), Counter()
    for event in events:
        if not begin <= event["time"] < end:
            continue
        if event["phase"] in ("verify", "decode"):
            histogram[len(event["emissions"])] += 1
        for emission in event["emissions"]:
            n = emission["tokens"]
            tokens += n
            generated_by_source[emission["source_id"]] += n
            if event["phase"] in ("verify", "decode"):
                postfirst += n
                rounds += 1
                accepted += emission["accepted_emitted"]
                raw_accepted += emission["accepted_raw"]
    decoding_seconds = 0.
    completed, contained, ttfts = [], [], []
    censored_decoding = boundary_crossing = 0
    finish_reasons = Counter()
    output_lengths = []
    for row in rows:
        first, last = row.get("first_token_seconds"), row.get("completion_seconds")
        if first is None or first >= end:
            continue
        finish = last if last is not None else end
        amount = overlap(first, finish, begin, end)
        decoding_seconds += amount
        if amount > 0:
            boundary_crossing += int(first < begin or last is None or last >= end)
            censored_decoding += int(last is None or last >= end)
        if begin <= first < end:
            ttfts.append(1000 * (first - row["submitted_seconds"]))
        if last is not None and begin <= last < end:
            finish_reasons[row["finish_reason"]] += 1
            output_lengths.append(row["output_tokens"])
            tpot = request_tpot(row)
            if tpot is not None:
                completed.append(1000 * tpot)
                if first >= begin:
                    contained.append(1000 * tpot)
    resident_seconds, prefirst_wait_seconds = 0., 0.
    time_by_resident = Counter()
    if any(b["time"] < a["time"] for a, b in zip(occupancy, occupancy[1:])):
        raise ValueError("occupancy transitions must be chronological")
    for i, point in enumerate(occupancy):
        finish = occupancy[i + 1]["time"] if i + 1 < len(occupancy) else end
        amount = overlap(point["time"], finish, begin, end)
        resident_seconds += amount * point["resident"]
        prefirst_wait_seconds += amount * point["pending_prefirst"]
        time_by_resident[point["resident"]] += amount
    stage_seconds = Counter()
    for stage in stages:
        stage_seconds[stage["name"]] += overlap(stage["begin"], stage["end"], begin, end)
    accounted = sum(stage_seconds.values())
    return dict(
        begin_seconds=begin, end_seconds=end, duration_seconds=duration,
        output_tokens=tokens, postfirst_output_tokens=postfirst,
        output_tokens_per_second=tokens / duration,
        decode_token_throughput=postfirst / duration,
        decoding_request_seconds=decoding_seconds,
        window_decode_ms_per_token=1000 * decoding_seconds / postfirst if postfirst else None,
        mean_postfirst_outstanding_requests=decoding_seconds / duration,
        completed_in_window_tpot_ms=distribution(completed),
        fully_contained_request_tpot_ms=distribution(contained),
        first_token_in_window_ttft_ms=distribution(ttfts),
        completed_in_window_output_tokens=distribution(output_lengths),
        completed_requests=sum(finish_reasons.values()), finish_reasons=dict(finish_reasons),
        boundary_crossing_decoding_requests=boundary_crossing,
        censored_decoding_requests=censored_decoding,
        mean_resident_requests=resident_seconds / duration,
        mean_pending_prefirst_requests=prefirst_wait_seconds / duration,
        fraction_time_resident_at_client_concurrency=time_by_resident[concurrency] / duration,
        time_seconds_by_resident_requests=dict(sorted(time_by_resident.items())),
        actual_verify_batch_histogram=dict(sorted(histogram.items())),
        mean_verify_batch=sum(k * n for k, n in histogram.items()) / sum(histogram.values()) if histogram else None,
        verify_request_rounds=rounds,
        actual_advance_per_verify=postfirst / rounds if rounds else None,
        accepted_emitted_rate=accepted / (7 * rounds) if rounds else None,
        raw_advance_per_verify=1 + raw_accepted / rounds if rounds else None,
        stage_seconds=dict(stage_seconds), uninstrumented_seconds=duration-accounted,
        output_tokens_by_source_id=dict(generated_by_source),
    )
