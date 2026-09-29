"""Synchronous CUDA-IPC/P2P or explicit host-staged tensor messages.

Target TP uses its own normal process group. This channel does not set NCCL
environment variables and cannot disable target collectives' NVLink transport.
The producer retains CUDA storage until the consumer copies it and acknowledges
completion; received IPC tensors must never be forwarded to another process.
"""

from dataclasses import dataclass, asdict
import time
from typing import Any

import torch


# torch Tensor IPC wraps a storage's DataPtr on every serialization. Reusing
# one serialized CUDA storage indefinitely builds a recursive destructor chain
# (CudaIPCSentData -> original_ptr) and eventually overflows the native stack.
# Rotate only after the consumer's copy ACK. Keep ordinary workspace reuse, but
# bound each generation's serialization count instead of relying on exit order.
MAX_SEND_WORKSPACE_USES = 256


@dataclass
class TransferStats:
    messages_sent: int = 0
    messages_received: int = 0
    tensor_bytes_sent: int = 0
    tensor_bytes_received: int = 0
    p2p_bytes_received: int = 0
    host_bytes_received: int = 0
    send_seconds: float = 0.0
    receive_copy_seconds: float = 0.0
    send_pack_seconds: float = 0.0
    send_sync_seconds: float = 0.0
    send_publish_seconds: float = 0.0
    send_ack_wait_seconds: float = 0.0
    receive_wait_seconds: float = 0.0
    source_tensors: int = 0
    packed_buffers: int = 0
    workspace_recycles: int = 0


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported wire value: {type(value).__name__}")


def pack_tensors(payload, workspaces):
    """Coalesce by device/dtype; retain reusable sender storage until ACK."""
    groups, group_ids = [], {}
    count = 0

    def record(tensor):
        nonlocal count
        count += 1
        key = (tensor.device, tensor.dtype)
        if key not in group_ids:
            group_ids[key] = len(groups)
            groups.append([key, [], 0])
        index = group_ids[key]
        group = groups[index]
        start = group[2]
        group[1].append(tensor.detach().reshape(-1))
        group[2] += tensor.numel()
        return ("__split_tensor_v1__", index, start, tuple(tensor.shape), tensor.numel())

    schema = _map_tensors(payload, record)
    buffers = []
    for key, tensors, size in groups:
        old = workspaces.get(key)
        if old is None or old.numel() < size:
            old = torch.empty(size, device=key[0], dtype=key[1])
            workspaces[key] = old
        view = old[:size]
        torch.cat(tensors, out=view)
        buffers.append(view)
    return dict(schema=schema, buffers=buffers), count


def unpack_tensors(packet):
    buffers = packet["buffers"]

    def walk(value):
        if isinstance(value, tuple) and len(value) == 5 and value[0] == "__split_tensor_v1__":
            _, index, start, shape, size = value
            return buffers[index][start:start + size].view(shape)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(walk(v) for v in value)
        return value

    return walk(packet["schema"])


class TensorChannel:
    def __init__(self, connection, *, device, mode="p2p", timeout=600.0):
        if mode not in ("p2p", "auto", "host"):
            raise ValueError("unsupported transport")
        self.connection = connection
        self.device = torch.device(device)
        self.mode = mode
        self.timeout = timeout
        self.stats = TransferStats()
        self._send_workspaces = {}
        self._send_workspace_uses = 0

    def _read(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError("split tensor channel timed out waiting for its peer")
        return self.connection.recv()

    def send(self, payload: Any):
        start = time.perf_counter()
        count = 0
        payload, source_count = pack_tensors(payload, self._send_workspaces)
        packed_at = time.perf_counter()
        # One producer barrier per device/message, not one per original tensor.
        for device in {t.device for t in payload["buffers"] if t.is_cuda}:
            torch.cuda.synchronize(device)
        synced_at = time.perf_counter()

        def prepare(tensor):
            nonlocal count
            tensor = tensor.detach().contiguous()
            count += tensor.numel() * tensor.element_size()
            if tensor.is_cuda:
                if self.mode == "host":
                    tensor = tensor.cpu()
            return tensor

        wire = _map_tensors(payload, prepare)
        self.connection.send(("payload", self.mode, wire))
        published_at = time.perf_counter()
        ack = self._read()
        acknowledged_at = time.perf_counter()
        if ack != ("copied",):
            raise RuntimeError(f"split peer rejected transfer: {ack}")
        # Retain `wire` until acknowledgement: CUDA IPC storage lifetime contract.
        del wire
        self.stats.messages_sent += 1
        self.stats.tensor_bytes_sent += count
        self.stats.send_seconds += time.perf_counter() - start
        self.stats.send_pack_seconds += packed_at - start
        self.stats.send_sync_seconds += synced_at - packed_at
        self.stats.send_publish_seconds += published_at - synced_at
        # This is peer-readiness/ACK wait, NOT a link-bandwidth measurement.
        self.stats.send_ack_wait_seconds += acknowledged_at - published_at
        self.stats.source_tensors += source_count
        self.stats.packed_buffers += len(payload["buffers"])
        self._send_workspace_uses += 1
        if self._send_workspace_uses >= MAX_SEND_WORKSPACE_USES:
            # `wire` has been ACKed and dropped. Any remaining local packed
            # views are released when send returns, with bounded chain depth.
            self._send_workspaces.clear()
            self._send_workspace_uses = 0
            self.stats.workspace_recycles += 1

    def receive(self):
        waiting_at = time.perf_counter()
        kind, source_mode, wire = self._read()
        self.stats.receive_wait_seconds += time.perf_counter() - waiting_at
        if kind != "payload" or source_mode != self.mode:
            raise RuntimeError("split peers disagree about protocol or transport mode")
        start = time.perf_counter()
        count = p2p_count = host_count = 0

        def materialize(tensor):
            nonlocal count, p2p_count, host_count
            size = tensor.numel() * tensor.element_size()
            count += size
            if tensor.is_cuda and self.device.type == "cuda":
                direct = tensor.device == self.device or torch.cuda.can_device_access_peer(
                    self.device.index, tensor.device.index
                )
                if not direct and self.mode == "p2p":
                    raise RuntimeError("P2P explicitly requested, but peer access is unavailable")
                if direct:
                    p2p_count += size
                    return tensor.to(self.device, copy=True)
                host_count += size
                return tensor.cpu().to(self.device)
            host_count += size
            return tensor.to(self.device, copy=True)

        try:
            result = _map_tensors(wire, materialize)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            del wire
            self.connection.send(("error", str(exc)))
            raise
        # Consumer owns every returned allocation, including same-device input.
        del wire
        self.connection.send(("copied",))
        self.stats.messages_received += 1
        self.stats.tensor_bytes_received += count
        self.stats.p2p_bytes_received += p2p_count
        self.stats.host_bytes_received += host_count
        self.stats.receive_copy_seconds += time.perf_counter() - start
        return unpack_tensors(result)

    def report(self):
        return {"requested_mode": self.mode, **asdict(self.stats),
                "send_buffer_allocated_bytes": sum(x.numel() * x.element_size() for x in self._send_workspaces.values())}

    def close(self):
        # Call only after the final transfer ACK (or after terminating a failed
        # peer). No in-flight source buffer may be destroyed before that point.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._send_workspaces.clear()
        self._send_workspace_uses = 0
        self.connection.close()
