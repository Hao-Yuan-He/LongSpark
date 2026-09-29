"""Numerical check of the GQA-5 adapter against SGLang's torch reference.

Run on a GPU host: pytest tests/gpu/test_gqa5_attention.py
Set LONGSPARK_TEST_GPU to choose the device (default 0).
"""
import os

import pytest

torch = pytest.importorskip('torch')
if not torch.cuda.is_available():
    pytest.skip('requires CUDA', allow_module_level=True)

from longspark.patches.gqa5_attention import (  # noqa: E402
    cache_position_scores, merge_accepted_positions_, original)

GPU = int(os.environ.get('LONGSPARK_TEST_GPU', '0'))


@torch.inference_mode()
def test_gqa5_matches_reference():
    torch.cuda.set_device(GPU)
    device = f'cuda:{GPU}'
    torch.manual_seed(980426)
    torch.backends.cuda.matmul.allow_tf32 = False
    for batch in (1, 3, 64):
        query = torch.randn(1, 40, 16, 128, device=device, dtype=torch.bfloat16).expand(batch, -1, -1, -1)
        capacity = batch + 3
        output = torch.randn(capacity, 2, 40, 16, 128, device=device)[:, 1]
        lse = torch.randn(capacity, 2, 40, 16, device=device)[:, 1]
        slots = torch.randperm(capacity, device=device)[:batch].to(torch.int32)
        expected_output, expected_lse = output[slots].clone(), lse[slots].clone()
        for step in range(16):
            keys = torch.randn(batch, 8, 8, 128, device=device, dtype=torch.bfloat16)
            values = torch.randn_like(keys)
            scores = torch.empty(batch, 40, 16, 8, device=device)
            cache_position_scores(layer_id=0, global_query=query, new_keys=keys,
                output=scores, softmax_scale=128**-.5)
            reference = original.reference_position_scores(global_query=query, new_keys=keys, softmax_scale=128**-.5)
            torch.testing.assert_close(scores, reference, rtol=2e-5, atol=2e-5)
            lens = ((torch.arange(batch, device=device) + step) % 8 + 1).to(torch.int32)
            locs = torch.arange(batch*8, device=device).reshape(batch, 8)
            expected_output, expected_lse = original.reference_merge_accepted_positions(
                state_output=expected_output, state_lse=expected_lse, position_scores=reference,
                accepted_values=values, accept_lens=lens)
            merge_accepted_positions_(state_output=output, state_lse=lse, position_scores=scores,
                value_cache=values.reshape(batch*8, 1, 8, 128), accepted_cache_locs=locs,
                accept_lens=lens, accept_offsets=None, state_slots=slots)
            torch.testing.assert_close(output[slots], expected_output, rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(lse[slots], expected_lse, rtol=3e-5, atol=3e-5)
