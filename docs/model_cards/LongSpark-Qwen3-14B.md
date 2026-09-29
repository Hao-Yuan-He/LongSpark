---
license: mit
base_model: Qwen/Qwen3-14B
tags:
  - longspark
  - speculative-decoding
  - qwen3
  - custom-code
---

# LongSpark-Qwen3-14B

LongSpark parallel drafter weights for [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).

| File | Contents |
|---|---|
| `model.safetensors` | 1,965,736,192 drafter parameters in BF16 (3.93 GB). |
| `config.json` | Drafter architecture and configuration. |

The matching Qwen3 target and tokenizer are loaded separately. Use this checkpoint with the [LongSpark runtime](https://github.com/Hao-Yuan-He/LongSpark).

**License:** [MIT](LICENSE). File hashes and checkpoint provenance are recorded in [release-manifest.json](release-manifest.json).
