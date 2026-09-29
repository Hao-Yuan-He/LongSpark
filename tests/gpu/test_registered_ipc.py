"""Round-trip stress test of the registered CUDA IPC channel between two processes.

Run on a GPU host with two peer-accessible GPUs: pytest tests/gpu/test_registered_ipc.py
LONGSPARK_TEST_GPUS=<sender>,<receiver> chooses the devices (default 0,1).
"""
import os

import pytest

torch = pytest.importorskip('torch')
if torch.cuda.device_count() < 2:
    pytest.skip('requires two CUDA devices', allow_module_level=True)

from longspark.engine.transport import TensorChannel  # noqa: E402
from longspark.patches.registered_ipc import install  # noqa: E402

SENDER, RECEIVER = map(int, os.environ.get('LONGSPARK_TEST_GPUS', '0,1').split(','))
ROUNDS = 1100


def _peer(connection, rounds, device):
    install()
    torch.cuda.set_device(device)
    channel = TensorChannel(connection, device=f'cuda:{device}')
    saved = None
    try:
        for i in range(rounds):
            packet = channel.receive()
            assert packet['i'] == i
            for tensor in packet['values']:
                assert (tensor == i % 97).all().item()
            if i == 1:
                saved = packet['values'][0]
            channel.send(dict(i=i, values=[tensor + 1 for tensor in packet['values']]))
        assert (saved == 1).all().item(), 'consumer-owned memory was overwritten'
        report = channel.report()
        channel.drop_receiver_cache()
        connection.send(('validation', report))
        assert connection.recv() == 'receiver_dropped'
    finally:
        channel.close()


def test_registered_ipc_round_trip():
    import torch.multiprocessing as mp
    install()
    ctx = mp.get_context('spawn')
    local, remote = ctx.Pipe()
    peer = ctx.Process(target=_peer, args=(remote, ROUNDS, RECEIVER))
    peer.start()
    torch.cuda.set_device(SENDER)
    device = f'cuda:{SENDER}'
    channel = TensorChannel(local, device=device)
    saved = None
    try:
        for i in range(ROUNDS):
            n = (0, 7, 255, 4096, 65536, 1048576)[i % 6]
            payload = dict(i=i, values=[torch.full((n,), i % 97, device=device, dtype=dtype)
                           for dtype in (torch.float32, torch.bfloat16, torch.int64)])
            channel.send(payload)
            answer = channel.receive()
            assert answer['i'] == i
            for tensor in answer['values']:
                assert (tensor == (i % 97) + 1).all().item()
            if i == 1:
                saved = answer['values'][0]
        assert (saved == 2).all().item(), 'consumer-owned memory was overwritten'
        tag, remote_report = local.recv()
        assert tag == 'validation'
        for report in (channel.report(), remote_report):
            stats = report['registered_ipc']
            assert stats['metadata_payloads_sent'] == ROUNDS
            assert stats['registrations_sent'] == 18, stats
            assert stats['retirements_sent'] == 15, stats
        channel.drop_receiver_cache()
        local.send('receiver_dropped')
        channel.close()
        peer.join(timeout=30)
        assert peer.exitcode == 0, f'peer failed: {peer.exitcode}'
    finally:
        if peer.is_alive():
            peer.terminate()
            peer.join(timeout=10)
