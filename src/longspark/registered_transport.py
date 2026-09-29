# Port of the validated local registered_ipc.py (SHA256 f80149c217f52e9f02f5fe4f634c0e77c64ed41739b141f4fbcfc72db4d960d1).
# Only package binding and the no-channel (vanilla) close guard differ.
"""Opt-in registered-buffer transport for the frozen, synchronous split V2.

Register each sender workspace once. Normal packets contain only a schema and
buffer IDs/lengths, never CUDA tensors. A growth retires the old remote mapping
with an explicit ACK before its producer reference is released. Consumers still
copy into owned tensors and synchronize before ACK; no borrowed storage reaches
model state. This avoids repeatedly serializing the same CUDA storage, so the
old 256-send destructor-chain workaround is unnecessary on this path.

P2P only, single in-flight message per channel, max 16 registered buffers and
512 MiB logical buffer capacity per direction. These are transport buffers, not
KV allocations. All changes are process-local; no shared library is modified.
"""
from dataclasses import asdict
import time

import torch
import importlib


MAX_BUFFERS = 16
MAX_BYTES = 512 * 1024 * 1024
_BASE_INIT = None
_PACKAGE = "split_methods"
_BOUND_PACKAGE = None
_PACK = _UNPACK = None


class RegisteredChannel:
    def __init__(self, connection, *, device, mode='p2p', timeout=600.):
        if mode != 'p2p':
            raise ValueError('registered IPC currently supports explicit P2P only')
        _BASE_INIT(self, connection, device=device, mode=mode, timeout=timeout)
        self._registered_send = {}
        self._registered_receive = {}
        self._next_buffer_id = 0
        self._registered_stats = dict(registrations_sent=0, registrations_received=0,
            retirements_sent=0, retirements_received=0, metadata_payloads_sent=0,
            metadata_payloads_received=0, registration_seconds=0., retirement_seconds=0.,
            peak_send_bytes=0, peak_receive_bytes=0)

    def _registered_expect(self, expected):
        received = self._read()
        if received != expected:
            raise RuntimeError(f'registered IPC expected {expected!r}, received {received!r}')

    def _registered_id(self, owner):
        key = (owner.device, owner.dtype)
        previous = self._registered_send.get(key)
        if previous is not None and previous[1] is owner:
            return previous[0]
        if previous is not None:
            begin = time.perf_counter()
            buffer_id, _ = previous
            self.connection.send(('retire_registered_v1', buffer_id))
            self._registered_expect(('retired_registered_v1', buffer_id))
            del self._registered_send[key]
            del previous
            self.stats.workspace_recycles += 1
            self._registered_stats['retirements_sent'] += 1
            self._registered_stats['retirement_seconds'] += time.perf_counter() - begin
        size = owner.numel() * owner.element_size()
        total = sum(t.numel() * t.element_size() for _, t in self._registered_send.values()) + size
        if len(self._registered_send) >= MAX_BUFFERS or total > MAX_BYTES:
            raise RuntimeError(f'registered sender transport capacity exceeded: {total} bytes')
        begin = time.perf_counter()
        buffer_id = self._next_buffer_id
        self._next_buffer_id += 1
        self._registered_send[key] = (buffer_id, owner)
        # This is the ONLY tensor serialization for this storage generation.
        self.connection.send(('register_buffer_v1', buffer_id, owner))
        self._registered_expect(('registered_buffer_v1', buffer_id))
        self._registered_stats['registrations_sent'] += 1
        self._registered_stats['registration_seconds'] += time.perf_counter() - begin
        self._registered_stats['peak_send_bytes'] = max(self._registered_stats['peak_send_bytes'], total)
        return buffer_id

    def send(self, payload):
        pack_tensors = _PACK
        start = time.perf_counter()
        packet, source_count = pack_tensors(payload, self._send_workspaces)
        packed_at = time.perf_counter()
        for device in {t.device for t in packet['buffers'] if t.is_cuda}:
            torch.cuda.synchronize(device)
        synced_at = time.perf_counter()
        refs, count = [], 0
        for view in packet['buffers']:
            owner = self._send_workspaces[(view.device, view.dtype)]
            refs.append((self._registered_id(owner), view.numel()))
            count += view.numel() * view.element_size()
        self.connection.send(('payload_registered_v1', self.mode,
                              dict(schema=packet['schema'], buffers=refs)))
        published_at = time.perf_counter()
        self._registered_expect(('copied_registered_v1',))
        acknowledged_at = time.perf_counter()
        self.stats.messages_sent += 1
        self.stats.tensor_bytes_sent += count
        self.stats.send_seconds += acknowledged_at - start
        self.stats.send_pack_seconds += packed_at - start
        self.stats.send_sync_seconds += synced_at - packed_at
        self.stats.send_publish_seconds += published_at - synced_at
        self.stats.send_ack_wait_seconds += acknowledged_at - published_at
        self.stats.source_tensors += source_count
        self.stats.packed_buffers += len(refs)
        self._registered_stats['metadata_payloads_sent'] += 1

    def receive(self):
        unpack_tensors = _UNPACK
        waiting_at = time.perf_counter()
        while True:
            packet = self._read()
            kind = packet[0]
            if kind == 'register_buffer_v1':
                _, buffer_id, tensor = packet
                if (buffer_id in self._registered_receive or not isinstance(tensor, torch.Tensor)
                        or tensor.ndim != 1 or not tensor.is_contiguous()):
                    raise RuntimeError('invalid or duplicate IPC registration')
                total = sum(t.numel() * t.element_size() for t in self._registered_receive.values())
                total += tensor.numel() * tensor.element_size()
                if len(self._registered_receive) >= MAX_BUFFERS or total > MAX_BYTES:
                    raise RuntimeError(f'registered receiver transport capacity exceeded: {total} bytes')
                self._registered_receive[buffer_id] = tensor
                self._registered_stats['registrations_received'] += 1
                self._registered_stats['peak_receive_bytes'] = max(
                    self._registered_stats['peak_receive_bytes'], total)
                del packet, tensor
                self.connection.send(('registered_buffer_v1', buffer_id))
                continue
            if kind == 'retire_registered_v1':
                _, buffer_id = packet
                # Prior payload was copied and synchronized before its ACK.
                # No model output or previous receive frame owns an IPC view.
                del self._registered_receive[buffer_id]
                self._registered_stats['retirements_received'] += 1
                del packet
                self.connection.send(('retired_registered_v1', buffer_id))
                continue
            if kind != 'payload_registered_v1' or packet[1] != self.mode:
                raise RuntimeError('peers disagree about registered IPC protocol')
            wire = packet[2]
            del packet
            break
        self.stats.receive_wait_seconds += time.perf_counter() - waiting_at
        begin = time.perf_counter()
        owned, count, p2p_count, host_count = [], 0, 0, 0
        try:
            for buffer_id, length in wire['buffers']:
                source = self._registered_receive[buffer_id]
                if not isinstance(length, int) or length < 0 or length > source.numel():
                    raise RuntimeError('registered payload length out of bounds')
                size = length * source.element_size()
                if source.is_cuda and self.device.type == 'cuda':
                    direct = source.device == self.device or torch.cuda.can_device_access_peer(
                        self.device.index, source.device.index)
                    if not direct:
                        raise RuntimeError('registered P2P requested but peer access unavailable')
                    p2p_count += size
                else:
                    host_count += size
                owned.append(source[:length].to(self.device, copy=True))
                count += size
                del source
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            self.connection.send(('error_registered_v1', str(exc)))
            raise
        self.connection.send(('copied_registered_v1',))
        self.stats.messages_received += 1
        self.stats.tensor_bytes_received += count
        self.stats.p2p_bytes_received += p2p_count
        self.stats.host_bytes_received += host_count
        self.stats.receive_copy_seconds += time.perf_counter() - begin
        self._registered_stats['metadata_payloads_received'] += 1
        return unpack_tensors(dict(schema=wire['schema'], buffers=owned))

    def drop_receiver_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self._registered_receive.clear()

    def close(self):
        self.drop_receiver_cache()
        self._registered_send.clear()
        self._send_workspaces.clear()
        self.connection.close()

    def report(self):
        return dict(requested_mode=self.mode, **asdict(self.stats),
            send_buffer_allocated_bytes=sum(t.numel() * t.element_size()
                                           for t in self._send_workspaces.values()),
            registered_ipc=dict(self._registered_stats,
                send_entries=len(self._registered_send), receive_entries=len(self._registered_receive),
                max_buffers=MAX_BUFFERS, max_bytes=MAX_BYTES,
                counter_scope='lifetime; TransferStats reset at each benchmark case'))


def install(package=None):
    global _BASE_INIT, _PACKAGE, _BOUND_PACKAGE, _PACK, _UNPACK
    package = package or _PACKAGE
    if _BOUND_PACKAGE is not None and package != _BOUND_PACKAGE:
        raise RuntimeError("Only one split package may be bound per process")
    _PACKAGE = package
    transport = importlib.import_module(_PACKAGE + ".transport")
    TensorChannel = transport.TensorChannel
    if getattr(TensorChannel, '_lpf_registered_ipc_installed', False):
        return
    if getattr(TensorChannel, '_lpf_mapping_cache_installed', False):
        raise RuntimeError('registered IPC and the old mapping-cache hook are mutually exclusive')
    _BOUND_PACKAGE = _PACKAGE
    _PACK, _UNPACK = transport.pack_tensors, transport.unpack_tensors
    _BASE_INIT = TensorChannel.__init__
    for name in ('__init__', '_registered_expect', '_registered_id', 'send', 'receive',
                 'report', 'close', 'drop_receiver_cache'):
        setattr(TensorChannel, name, getattr(RegisteredChannel, name))
    TensorChannel._lpf_registered_ipc_installed = True
    TargetClient = importlib.import_module(_PACKAGE + ".target").TargetClient
    original_close = TargetClient.close

    def close_client(client, *, graceful=True):
        if graceful and client.channel is not None:
            # Release imported target storage BEFORE asking target producers to
            # exit. Target close releases imported draft storage before join.
            client.channel.drop_receiver_cache()
        return original_close(client, graceful=graceful)
    TargetClient.close = close_client


def _test_peer(connection, rounds, package):
    install(package)
    transport = importlib.import_module(_PACKAGE + ".transport")
    TensorChannel = transport.TensorChannel
    torch.cuda.set_device(0)
    channel = TensorChannel(connection, device='cuda:0')
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


def self_test(rounds=1100):
    import torch.multiprocessing as mp
    install()
    transport = importlib.import_module(_PACKAGE + ".transport")
    TensorChannel = transport.TensorChannel
    ctx = mp.get_context('spawn')
    local, remote = ctx.Pipe()
    peer = ctx.Process(target=_test_peer, args=(remote, rounds, _PACKAGE))
    peer.start()
    torch.cuda.set_device(2)
    channel = TensorChannel(local, device='cuda:2')
    begin, saved = time.perf_counter(), None
    try:
        for i in range(rounds):
            n = (0, 7, 255, 4096, 65536, 1048576)[i % 6]
            payload = dict(i=i, values=[torch.full((n,), i % 97, device='cuda:2', dtype=dtype)
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
            assert stats['metadata_payloads_sent'] == rounds
            assert stats['registrations_sent'] == 18, stats
            assert stats['retirements_sent'] == 15, stats
        result = dict(passed=True, rounds=rounds, wire_payload_messages=2 * rounds,
                      seconds=time.perf_counter()-begin, sender=channel.report(), receiver=remote_report)
        channel.drop_receiver_cache()
        local.send('receiver_dropped')
        channel.close()
        peer.join(timeout=30)
        if peer.exitcode != 0:
            raise RuntimeError(f'peer failed: {peer.exitcode}')
        return result
    finally:
        if peer.is_alive():
            peer.terminate()
            peer.join(timeout=10)

