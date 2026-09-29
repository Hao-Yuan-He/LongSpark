# How to use LongSpark

Start with the [quick start](../README.md#quick-start). This guide walks through the full evaluation workflow, from choosing GPUs to reading a result.

## Hardware

The target and the drafter run on separate GPUs. `--gpus` lists physical device IDs: the first devices run the target, and the next one runs the drafter. Autoregressive decoding (`vanilla`) uses the target GPUs alone.

| Workload | Target GPUs | Drafter GPU | Target KV budget per GPU |
|---|---:|---:|---:|
| Short context / concurrency | 1 | 1 | 64 GiB |
| LongSpec 32K | 4 | 1 | 48 GiB |
| CodeSpan 64K / LongSWE 128K | 4 | 1 | 96 GiB |

Leave room beyond these budgets for weights, CUDA graphs, and workspace. We ran all evaluations on NVIDIA L20X GPUs with about 140 GiB each. `--smoke` shortens a run but keeps the same memory budgets.

## Configure checkpoints

`configs/models.local.json` maps each target size to local checkpoint directories:

```json
{
  "8B": {
    "target": "models/8B/target",
    "longspark": "models/8B/longspark"
  }
}
```

Only the methods you run need entries. See the [example](../configs/models.example.json) for all three sizes, or pass another file with `--models`. Paths are local, so download Hub checkpoints first; [WEIGHTS.md](WEIGHTS.md) has the download commands.

## Presets

| Script | Inputs | Target GPUs | Concurrency | Output cap | Seeds |
|---|---|---:|---:|---:|---:|
| `short_context.sh` | Eight benchmarks | 1 | 32 | 2,048 | 3 |
| `long_context.sh` | 32K / 64K / 128K workloads | 4 | 16 | 8,192 | 10 |
| `concurrency.sh` | Eight benchmarks | 1 | 8–128 | 2,048 | 3 |
| `fixed_requests.sh` | 32 fixed long-context requests | 4 | 16 | 8,192 | 3 |

Steady-state presets warm up for 30 seconds and measure for 120. `fixed_requests.sh` instead times the full request set to completion. Long-context presets use fixed NTK scaling; short-context presets use native Qwen3 RoPE.

Every script previews its plan by default. Add `--execute` to run.

## Select a workload

| Filter | Values |
|---|---|
| `--size` | `4B`, `8B`, `14B` |
| `--method` | `vanilla`, `longspark` |
| `--dataset` (short) | `gsm8k`, `math500`, `aime25`, `humaneval`, `mbpp`, `livecodebench`, `mt-bench`, `alpaca` |
| `--dataset` (long) | `longspec_32k`, `code_64k`, `longswe_128k` |
| `--temperature` | `1` (default), `0` (greedy) |
| `--concurrency` | `8`, `16`, `32`, `64`, `128` |
| `--seed` | `980426`, `980427`, `2026`; `long_context.sh` adds `980428`–`980434` |

Filters take comma-separated lists of preset values. Omitted filters expand to every value, and a full preset runs for many hours, so start narrow:

```bash
# Long-context comparison: four target GPUs and one drafter GPU.
bash scripts/long_context.sh \
  --size 8B --method vanilla,longspark --dataset longspec_32k \
  --gpus 0,1,2,3,4 --execute

# Concurrency sweep with one CUDA graph capacity for all points.
bash scripts/concurrency.sh \
  --size 8B --method vanilla,longspark --dataset math500,mbpp,alpaca \
  --graph-max-batch 128 --gpus 0,1 --execute

# Greedy decoding.
bash scripts/short_context.sh \
  --size 8B --method vanilla,longspark --dataset math500 \
  --temperature 0 --gpus 0,1 --execute
```

## Read the results

Each run writes one directory per setting:

```text
results/runs/short_context/t1/8B/math500/c32/s980426/longspark/
├── job.json                 # Expanded configuration
├── worker.log               # Runtime log
├── result.json              # Performance metrics
├── result.requests.jsonl    # Per-request outputs and timing
└── audit.json               # Checks passed by this run
```

Smoke runs go under `results/smoke/`. Use `--output` to choose another root.

```bash
python scripts/summarize.py results/runs/short_context \
  --output results/short-context-summary.json
```

| Field | Meaning |
|---|---|
| `tps` | Output tokens per second |
| `tpot` | Time per output token, in ms |
| `tau` | Tokens emitted per verification round |
| `speedup` | TPS relative to `vanilla` at the same setting |
| `seconds` | Completion time of a fixed-request run |
| `*_sd` | Standard deviation across seeds |
| `complete` | Whether every preset seed is present |

The summary averages over seeds. When all eight short-context benchmarks are complete, it also adds an `avg.` row. Smoke runs and cells that failed their audit are left out.

## Resume a run

Repeat the original command with `--resume` to skip cells that already completed.

```bash
bash scripts/short_context.sh \
  --size 8B --method vanilla,longspark --dataset math500 \
  --gpus 0,1 --execute --resume
```

A failed or partial cell stops the launcher and keeps its logs. Rerun it, or any run with changed model paths or graph capacity, into a fresh `--output` directory.

## Verify correctness

The correctness runner decodes greedily with LongSpark and with the target alone, then compares every output token, including EOS.

```bash
python scripts/check_correctness.py \
  --size 8B --dataset native --gpus 0,1 \
  --max-new-tokens 512 --output results/correctness/8B/native

python scripts/check_correctness.py \
  --size 8B --dataset longswe_128k --gpus 0,1,2,3,4 \
  --max-new-tokens 128 --output results/correctness/8B/128k
```

`--dataset` takes `native`, `longspec_32k`, `code_64k`, or `longswe_128k`. A run passes only when every request matches. The runner switches the target to deterministic kernels, so its timings are not speed measurements.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Missing `config.json` or checkpoint shard | Check the paths in `models.local.json` and finish the download. |
| Tokenizer mismatch | Use the pinned target revision in [WEIGHTS.md](WEIGHTS.md) and keep its tokenizer files unchanged. |
| CUDA toolkit error | Point `CUDA_HOME` at a CUDA 12.8 toolkit with `bin/nvcc`. |
| `ninja` not found | Activate the `longspark` conda environment. |
| Selected GPUs occupied | Choose idle devices; the launcher refuses GPUs already in use. |
| Out of memory at startup | Check device memory against the KV budgets above, plus weights and graphs. |
| Worker failed | Read that cell's `worker.log`. |
| Result directory exists | Rerun with `--resume`, or choose a fresh `--output`. |
| Missing dataset file | Place the evaluation inputs under `data/`; upstream sources are listed in [NOTICE](../NOTICE). |
