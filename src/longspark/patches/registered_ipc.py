"""Registered-buffer CUDA IPC for TensorChannel, installed in place of its methods.

Each sender workspace is registered with the peer once. Normal packets then
carry only a schema and buffer IDs/lengths, never CUDA tensors. Growing a
workspace retires the old remote mapping with an explicit ACK before the
producer releases it. Consumers copy into owned tensors and synchronize before
ACK, so no borrowed storage reaches model state, and repeated serialization of
the same CUDA storage (see transport.MAX_SEND_WORKSPACE_USES) is avoided.

P2P only, one in-flight message per channel, at most 16 registered buffers and
512 MiB per direction. These are transport buffers, not KV allocations.
"""
from dataclasses import asdict
import time

import torch

from ..engine import target, transport

MAX_BUFFERS = 16
MAX_BYTES = 512 * 1024 * 1024
_BASE_INIT = None


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
        start = time.perf_counter()
        packet, source_count = transport.pack_tensors(payload, self._send_workspaces)
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
        return transport.unpack_tensors(dict(schema=wire['schema'], buffers=owned))

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


def install():
    """Replace TensorChannel's transport methods in this process (idempotent)."""
    global _BASE_INIT
    TensorChannel = transport.TensorChannel
    if getattr(TensorChannel, '_lpf_registered_ipc_installed', False):
        return
    _BASE_INIT = TensorChannel.__init__
    for name in ('__init__', '_registered_expect', '_registered_id', 'send', 'receive',
                 'report', 'close', 'drop_receiver_cache'):
        setattr(TensorChannel, name, getattr(RegisteredChannel, name))
    TensorChannel._lpf_registered_ipc_installed = True
    original_close = target.TargetClient.close

    def close_client(client, *, graceful=True):
        if graceful and client.channel is not None:
            # Release imported target storage BEFORE asking target producers to
            # exit. Target close releases imported draft storage before join.
            client.channel.drop_receiver_cache()
        return original_close(client, graceful=graceful)
    target.TargetClient.close = close_client
