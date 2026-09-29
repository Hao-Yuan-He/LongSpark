# Vendored SpecForge (`specforge`)

This directory contains a **minimal subset** of [SpecForge](https://github.com/sgl-project/SpecForge)
containing the model definitions needed by LongSpark evaluation. The evaluation launcher
loads this tree directly; a separate SpecForge checkout is not needed.

## License

The upstream project is distributed under the **MIT License** (Copyright (c) 2025 sgl-project).
The full license text is in [`LICENSE`](LICENSE) (copied from the upstream repository).

## Files vendored from upstream

| Local path | Upstream path |
|------------|---------------|
| `utils.py` | subset of `SpecForge/specforge/utils.py` (`print_with_rank` only) |
| `distributed.py` | `SpecForge/specforge/distributed.py` |
| `modeling/draft/draftfreekv.py` | `SpecForge/specforge/modeling/draft/draftfreekv.py` |
| `modeling/draft/global16_raw256.py` | `SpecForge/specforge/modeling/draft/global16_raw256.py` |
| `modeling/draft/global16_raw256_training.py` | `SpecForge/specforge/modeling/draft/global16_raw256_training.py` |
| `modeling/draft/draftfreekv_validation.py` | `SpecForge/specforge/modeling/draft/draftfreekv_validation.py` |
| `modeling/target/draftfreekv_target_model.py` | `SpecForge/specforge/modeling/target/draftfreekv_target_model.py` |
| `modeling/target/target_utils.py` | `SpecForge/specforge/modeling/target/target_utils.py` |
| `modeling/target/sglang_backend/*` | `SpecForge/specforge/modeling/target/sglang_backend/*` |

Each `.py` file begins with a short vendoring notice pointing here.

## Updating from upstream

This snapshot includes LongSpark model extensions. Preserve the notices and
validate checkpoint compatibility before updating any model definitions.

## Runtime

The evaluation launcher adds `vendor/` to the worker import path. Training
drivers and configuration parsers are excluded; model-internal training helpers
remain where the shared model definitions import them.
