# LongSpark checkpoints

The three LongSpark drafters are on Hugging Face under the MIT License. Each one pairs with a specific Qwen3 target, which you download separately.

| Drafter | Target | Parameters | BF16 weights |
|---|---|---:|---:|
| [LongSpark-Qwen3-4B](https://huggingface.co/Hehy/LongSpark-Qwen3-4B) | Qwen3-4B | 642M | 1.28 GB |
| [LongSpark-Qwen3-8B](https://huggingface.co/Hehy/LongSpark-Qwen3-8B) | Qwen3-8B | 1.19B | 2.39 GB |
| [LongSpark-Qwen3-14B](https://huggingface.co/Hehy/LongSpark-Qwen3-14B) | Qwen3-14B | 1.97B | 3.93 GB |

Parameter counts cover the drafter alone. At inference, the drafter also borrows the target's embedding, output head, and final normalization.

## Download a pair

The [checkpoint manifest](../configs/longspark-checkpoints.json) pins each target and drafter to an exact revision. To download any size:

```python
import json
from huggingface_hub import snapshot_download

size = "8B"
with open("configs/longspark-checkpoints.json") as f:
    model = json.load(f)["models"][size]

snapshot_download(model["target_model"],
                  revision=model["target_candidate_revision"],
                  local_dir=f"models/{size}/target")
snapshot_download(model["hub_repo_id"], revision=model["hub_revision"],
                  local_dir=f"models/{size}/longspark")
```

| Qwen3 target | Revision |
|---|---|
| 4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| 8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| 14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |

Keep the target snapshot intact: the evaluation inputs are pre-tokenized, and the launcher checks that the target's tokenizer matches them.

## What is in a model repository

```text
config.json             # Drafter configuration
model.safetensors       # BF16 drafter weights
README.md               # Model card
LICENSE                 # MIT
NOTICE                  # Target and runtime attribution
release-manifest.json   # File hashes and matching target
```

The checkpoints use a custom architecture defined in this repository. Load them through the LongSpark launcher; generic Transformers pipelines do not support them.
