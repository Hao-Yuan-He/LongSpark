---
pretty_name: CodeSpan
license: other
license_name: mixed-upstream-licenses
license_link: LICENSE
task_categories:
  - text-generation
size_categories:
  - n<1K
tags:
  - code
  - code-completion
  - long-context
  - speculative-decoding
  - longspark
configs:
  - config_name: default
    default: true
    data_files:
      - split: test
        path: codespan_64K.jsonl
  - config_name: longspark
    data_files:
      - split: test
        path: longspark/code_64k.jsonl
---

# CodeSpan

**64K source-code continuation for long-context decoding.**

CodeSpan contains **32 examples from 17 open-source projects**, including LLVM,
GCC, Linux, and PostgreSQL. Each example is a contiguous prefix of a distinct
source file, wrapped as a continuation prompt: **65,536 Qwen3 tokens**, including
the chat template with thinking disabled. Files are neither concatenated nor
repeated. There are at most four files per project; 28 of the 32 files are C/C++.

| Configuration | Contents |
|---|---|
| `default` | `id`, `language`, formatted `prompt`, `input_ids`, and `source` with repository, revision, path, URL, license, and modification notice. |
| `longspark` | The same 32 examples in the same order, as `source_id` and `input_ids`, matching the [LongSpark](https://github.com/Hao-Yuan-He/LongSpark) evaluation inputs. |

```python
from datasets import load_dataset

data = load_dataset("Hehy/CodeSpan", split="test")
input_ids = data[0]["input_ids"]
```

Use the supplied token IDs directly; the chat template is already included.
This is a decoding workload with no reference continuations. The release
[manifest](release-manifest.json) records file hashes and tokenizer identity.

Source excerpts retain their upstream licenses. Full license texts, copyright
notices, and the per-sample inventory are in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt); retain them when redistributing
the data.
