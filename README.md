<div align="center">

# LongSpark

**Efficient Speculative Decoding with a Fixed-Cost Parallel Drafter**

[Quick start](#quick-start) · [Artifacts](#artifacts) · [Usage guide](docs/HOW_TO_USE.md)

</div>

Speculative decoding makes a simple bargain: a small drafter guesses a block of tokens, the target checks all of them in one forward pass, and every accepted guess is a token the target never had to produce alone. The bargain pays as long as guessing stays cheap.

On long contexts, guessing stops being cheap. Today's strongest drafters read the entire prefix every round, so their cost grows with the very context they were meant to accelerate.

**LongSpark drafts at a fixed cost.** It reads the prefix through fixed-size views taken from the target's own verification pass, so its drafting time and state stay flat from 1K to 128K tokens.

## Highlights

- **Flat drafting cost.** Drafting time stays near 3.5 ms from 1K to 128K tokens. At 128K, the drafter state is 406× smaller than DSpark's.
- **Fastest at every scale.** 1.88×, 1.99×, and 2.13× average throughput speedups on Qwen3-4B, 8B, and 14B across math, code, and chat benchmarks.
- **Lowest TPOT at long context.** Best time-per-output-token on every long-context workload from 32K to 128K.
- **Lossless.** The target verifies every proposal, so the output distribution is the target's own.

## How it works

A drafter does not need to remember the prefix faithfully. The target verifies every proposal, and a wrong guess costs one extra round, never a wrong answer. LongSpark keeps just enough context to propose well:

- **Boundary state**: where generation is right now.
- **Recent window**: token-level detail, read directly from the target's KV cache.
- **Global summary**: the whole prefix, compressed into a fixed number of entries and updated incrementally.

The target refreshes all three views in the same pass that verifies the previous block. The drafter then proposes the next block in one parallel pass. The only full-prefix work in the loop is verification, which the target runs anyway.

## Quick start

Requirements: Linux, Python 3.11, CUDA 12.8, and two NVIDIA GPUs with peer-to-peer access.

**1. Install**

```bash
conda env create -f environment.yml
conda activate longspark

export CUDA_HOME=/path/to/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash scripts/install.sh
```

**2. Download a target/drafter pair**

```bash
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir models/8B/target

hf download Hehy/LongSpark-Qwen3-8B \
  --revision a0e73f6e9765bd9d3690150c90b77485401b92e0 \
  --local-dir models/8B/longspark
```

Then point the launcher at them in `configs/models.local.json`:

```json
{
  "8B": {
    "target": "models/8B/target",
    "longspark": "models/8B/longspark"
  }
}
```

**3. Measure the speedup**

Compare LongSpark with autoregressive decoding on MATH-500. The target runs on GPU 0 and the drafter on GPU 1:

```bash
bash scripts/short_context.sh \
  --size 8B --method vanilla,longspark --dataset math500 \
  --gpus 0,1 --execute

python scripts/summarize.py results/runs/short_context \
  --output results/math500-8b-summary.json
```

Drop `--execute` to preview the run without touching the GPUs, or add `--smoke` for a quick functional check.

## Artifacts

All model weights and the CodeSpan dataset are collected on Hugging Face:

<https://huggingface.co/collections/Hehy/longspark>

| Drafter | Target |
|---|---|
| [LongSpark-Qwen3-4B](https://huggingface.co/Hehy/LongSpark-Qwen3-4B) | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |
| [LongSpark-Qwen3-8B](https://huggingface.co/Hehy/LongSpark-Qwen3-8B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| [LongSpark-Qwen3-14B](https://huggingface.co/Hehy/LongSpark-Qwen3-14B) | [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |

Each drafter pairs with one Qwen3 target and runs through this repository's launcher. See [WEIGHTS.md](docs/WEIGHTS.md) for pinned revisions.

## Evaluation

| Script | Workload |
|---|---|
| `scripts/short_context.sh` | Eight math, code, and chat benchmarks |
| `scripts/long_context.sh` | LongSpec 32K, CodeSpan 64K, and LongSWE-Bench 128K |
| `scripts/concurrency.sh` | Throughput from concurrency 8 to 128 |
| `scripts/fixed_requests.sh` | A fixed set of long-context requests, run to completion |

Each script covers Qwen3-4B, 8B, and 14B, and compares LongSpark with autoregressive decoding. The [usage guide](docs/HOW_TO_USE.md) covers hardware, filters, metrics, and correctness checks.

## Documentation

- [Usage guide](docs/HOW_TO_USE.md): running and reading evaluations.
- [Artifacts](https://huggingface.co/collections/Hehy/longspark): model weights and the CodeSpan dataset.

## Acknowledgments

LongSpark builds on [SGLang](https://github.com/sgl-project/sglang) and [SpecForge](https://github.com/sgl-project/SpecForge).

## License

Project code and LongSpark weights are released under the [MIT License](LICENSE). Vendored code, datasets, and third-party weights keep their own terms; see [NOTICE](NOTICE).

## Citation

```bibtex
@misc{he2026longspark,
  title  = {Efficient Speculative Decoding with a Fixed-Cost Parallel Drafter},
  author = {Hao-Yuan He and Peng-Fei Liu and Si Shen and Ming Li},
  year   = {2026}
}
```
