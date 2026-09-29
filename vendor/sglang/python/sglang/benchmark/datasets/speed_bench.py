"""SPEED-Bench (nvidia/SPEED-Bench) dataset for the SGLang serving benchmark.

Reads a local throughput split prepared through NVIDIA's official SPEED-Bench
framework in JSONL or Parquet format, optionally filtering by category
(low_entropy / mixed / high_entropy) and fixing the output length.

CLI args consumed:
  --dataset-path            Path to the local JSONL or Parquet file.
  --speed-bench-category    Category filter: low_entropy | mixed | high_entropy
                            (default: all categories).
  --speed-bench-output-len  Fixed number of output tokens per request (default: 512).
  --num-prompts             Number of requests; must not exceed usable fixture rows.
"""

import json
import random
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


SPEED_BENCH_DATA_PLACEHOLDER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)


def render_speed_bench_prompt(
    tokenizer: PreTrainedTokenizerBase, prompt_text: str
) -> Tuple[str, List[int]]:
    """Render and tokenize one SPEED-Bench prompt using the fixed eval policy."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        return_dict=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, prompt_ids


@dataclass
class SpeedBenchDataset(BaseDataset):
    dataset_path: str
    category: Optional[str]
    output_len: int
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "SpeedBenchDataset":
        if not args.dataset_path:
            raise ValueError(
                "--dataset-path must point to a local SPEED-Bench .jsonl or "
                ".parquet file prepared through NVIDIA's official framework."
            )
        return cls(
            dataset_path=args.dataset_path,
            category=getattr(args, "speed_bench_category", None) or None,
            output_len=getattr(args, "speed_bench_output_len", 512),
            num_requests=args.num_prompts,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        dataset_path = Path(self.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"SPEED-Bench dataset file not found: {dataset_path}"
            )

        if dataset_path.suffix.lower() == ".jsonl":
            with dataset_path.open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
        elif dataset_path.suffix.lower() == ".parquet":
            import pandas as pd

            records = pd.read_parquet(dataset_path).to_dict(orient="records")
        else:
            raise ValueError(
                f"Unsupported SPEED-Bench dataset format: {dataset_path.suffix!r}; "
                "expected .jsonl or .parquet."
            )

        prompt_texts = []
        for row in records:
            turns = row.get("turns", [])
            tolist = getattr(turns, "tolist", None)
            if callable(tolist):
                turns = tolist()
            if not isinstance(turns, (list, tuple)) or not turns:
                continue
            prompt_text = turns[0]
            if not isinstance(prompt_text, str) or not prompt_text:
                continue
            if SPEED_BENCH_DATA_PLACEHOLDER in prompt_text:
                raise ValueError(
                    f"{dataset_path} contains the SPEED-Bench placeholder instead "
                    "of benchmark prompts. Prepare the full data through the "
                    "official NVIDIA SPEED-Bench framework and source-license flow."
                )
            if self.category and row.get("category") != self.category:
                continue
            prompt_texts.append(prompt_text)

        if not prompt_texts:
            raise ValueError(
                f"No rows found in {self.dataset_path}"
                + (f" for category={self.category}" if self.category else "")
            )
        if self.num_requests > len(prompt_texts):
            raise ValueError(
                f"SPEED-Bench requested {self.num_requests} prompts, but only "
                f"{len(prompt_texts)} usable rows are available"
                + (f" for category={self.category}" if self.category else "")
                + ". Fixture reuse is disabled for fair evaluation."
            )

        # Preserve official fixture order for a full run. A seeded global RNG
        # still controls reproducible strict subsampling for smaller runs.
        if self.num_requests == len(prompt_texts):
            selected_prompt_texts = prompt_texts
        else:
            selected_prompt_texts = random.sample(prompt_texts, self.num_requests)

        dataset_rows: List[DatasetRow] = []
        for prompt_text in selected_prompt_texts:
            prompt, prompt_ids = render_speed_bench_prompt(tokenizer, prompt_text)
            dataset_rows.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=len(prompt_ids),
                    output_len=self.output_len,
                )
            )

        return dataset_rows
