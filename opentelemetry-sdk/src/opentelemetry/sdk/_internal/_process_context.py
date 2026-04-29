# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Process Context publisher for OTEP 4719 (experimental, Linux-only).

This module publishes process-level resource attributes via a memfd-backed
private mapping that external readers (such as the OpenTelemetry eBPF profiler)
can locate by parsing ``/proc/<pid>/maps``.

The feature is gated behind
:envvar:`OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER`; on platforms other than
Linux every public function is a no-op.

OTEP draft: https://github.com/open-telemetry/opentelemetry-specification/pull/4719
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
import threading
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_HEADER_SIZE = 32
_HEADER_VERSION = 2
_PUBLISHED_AT_OFFSET = 16
_PAYLOAD_SIZE_OFFSET = 12
_PAYLOAD_PTR_OFFSET = 24
_SIGNATURE = b"OTEL_CTX"

_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_MFD_NOEXEC_SEAL = 0x0008

_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_PRIVATE = 0x02
_MAP_ANONYMOUS = 0x20

_MADV_DONTFORK = 10

_PR_SET_VMA = 0x53564D41
_PR_SET_VMA_ANON_NAME = 0

_MAP_FAILED = (1 << 64) - 1


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint values must be non-negative")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _len_field(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _string_field(field: int, value: str) -> bytes:
    return _len_field(field, value.encode("utf-8"))


def _bytes_field(field: int, value: bytes) -> bytes:
    return _len_field(field, value)


def _bool_field(field: int, value: bool) -> bytes:
    return _tag(field, 0) + _varint(1 if value else 0)


def _int64_field(field: int, value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    return _tag(field, 0) + _varint(value)


def _double_field(field: int, value: float) -> bytes:
    return _tag(field, 1) + struct.pack("<d", value)


def _encode_any_value(value: Any) -> bytes:
    if isinstance(value, bool):
        return _bool_field(2, value)
    if isinstance(value, int):
        return _int64_field(3, value)
    if isinstance(value, float):
        return _double_field(4, value)
    if isinstance(value, str):
        return _string_field(1, value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes_field(7, bytes(value))
    if isinstance(value, (list, tuple)):
        inner = b"".join(
            _len_field(1, _encode_any_value(item)) for item in value
        )
        return _len_field(5, inner)
    raise TypeError(
        f"Unsupported attribute value type: {type(value).__name__}"
    )


def _encode_key_value(key: str, value: Any) -> bytes:
    return _string_field(1, key) + _len_field(2, _encode_any_value(value))


def encode_process_context(
    resource_attributes: Mapping[str, Any],
    extra_attributes: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """Encode a ``ProcessContext`` payload (OTEP 4719).

    ``resource_attributes`` maps to ``ProcessContext.resource.attributes``,
    ``extra_attributes`` to ``ProcessContext.attributes``.
    """
    resource_inner = b"".join(
        _len_field(1, _encode_key_value(k, v))
        for k, v in resource_attributes.items()
    )
    pieces = [_len_field(1, resource_inner)]
    if extra_attributes:
        for k, v in extra_attributes.items():
            pieces.append(_len_field(2, _encode_key_value(k, v)))
    return b"".join(pieces)


class _State:
    __slots__ = (
        "lock",
        "libc",
        "mapping_addr",
        "payload_buf",
        "name_buf",
        "attributes",
        "extra_attributes",
        "last_timestamp",
        "owner_pid",
    )

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.libc: Optional[ctypes.CDLL] = None
        self.mapping_addr: int = 0
        self.payload_buf: Optional[ctypes.Array[ctypes.c_char]] = None
        self.name_buf: Optional[ctypes.Array[ctypes.c_char]] = None
        self.attributes: Optional[dict] = None
        self.extra_attributes: Optional[dict] = None
        self.last_timestamp: int = 0
        self.owner_pid: int = 0


_state = _State()
_fork_hook_lock = threading.Lock()
_fork_hook_registered = False


def is_supported() -> bool:
    """Return True iff this platform is Linux."""
    return sys.platform.startswith("linux")


def _get_libc() -> ctypes.CDLL:
    if _state.libc is not None:
        return _state.libc
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    libc.ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]
    libc.ftruncate.restype = ctypes.c_int
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_size_t
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    libc.close.argtypes = [ctypes.c_int]
    libc.close.restype = ctypes.c_int
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    _state.libc = libc
    return libc


def _now_boottime_ns() -> int:
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    return time.monotonic_ns()


def _name_mapping(libc: ctypes.CDLL, addr: int):
    name_buf = ctypes.create_string_buffer(_SIGNATURE)
    libc.prctl(
        _PR_SET_VMA,
        _PR_SET_VMA_ANON_NAME,
        addr,
        _HEADER_SIZE,
        ctypes.addressof(name_buf),
    )
    return name_buf


def _publish_new(
    payload: bytes,
    attributes: dict,
    extra_attributes: dict,
) -> bool:
    libc = _get_libc()
    payload_buf = ctypes.create_string_buffer(payload)
    payload_addr = ctypes.addressof(payload_buf)
    payload_size = len(payload)

    fd = libc.memfd_create(
        _SIGNATURE,
        _MFD_CLOEXEC | _MFD_ALLOW_SEALING | _MFD_NOEXEC_SEAL,
    )
    if fd < 0:
        fd = libc.memfd_create(
            _SIGNATURE, _MFD_CLOEXEC | _MFD_ALLOW_SEALING
        )

    if fd >= 0:
        if libc.ftruncate(fd, _HEADER_SIZE) != 0:
            errno = ctypes.get_errno()
            libc.close(fd)
            logger.warning(
                "process context: ftruncate failed (errno=%d)", errno
            )
            return False
        addr = libc.mmap(
            0,
            _HEADER_SIZE,
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE,
            fd,
            0,
        )
        libc.close(fd)
    else:
        addr = libc.mmap(
            0,
            _HEADER_SIZE,
            _PROT_READ | _PROT_WRITE,
            _MAP_PRIVATE | _MAP_ANONYMOUS,
            -1,
            0,
        )

    if addr == _MAP_FAILED or addr == 0:
        logger.warning(
            "process context: mmap failed (errno=%d)", ctypes.get_errno()
        )
        return False

    libc.madvise(addr, _HEADER_SIZE, _MADV_DONTFORK)

    header = struct.pack(
        "<8sIIQQ",
        _SIGNATURE,
        _HEADER_VERSION,
        payload_size,
        0,
        payload_addr,
    )
    ctypes.memmove(addr, header, _HEADER_SIZE)

    timestamp = _now_boottime_ns()
    if timestamp == 0:
        timestamp = 1
    ctypes.c_uint64.from_address(addr + _PUBLISHED_AT_OFFSET).value = timestamp

    name_buf = _name_mapping(libc, addr)

    _state.mapping_addr = addr
    _state.payload_buf = payload_buf
    _state.name_buf = name_buf
    _state.attributes = attributes
    _state.extra_attributes = extra_attributes
    _state.last_timestamp = timestamp
    _state.owner_pid = os.getpid()
    return True


def _update_existing(
    payload: bytes,
    attributes: dict,
    extra_attributes: dict,
) -> bool:
    libc = _get_libc()
    addr = _state.mapping_addr
    new_buf = ctypes.create_string_buffer(payload)
    new_addr = ctypes.addressof(new_buf)
    new_size = len(payload)

    ctypes.c_uint64.from_address(addr + _PUBLISHED_AT_OFFSET).value = 0
    ctypes.c_uint32.from_address(addr + _PAYLOAD_SIZE_OFFSET).value = new_size
    ctypes.c_uint64.from_address(addr + _PAYLOAD_PTR_OFFSET).value = new_addr

    timestamp = _now_boottime_ns()
    if timestamp <= _state.last_timestamp:
        timestamp = _state.last_timestamp + 1
    ctypes.c_uint64.from_address(addr + _PUBLISHED_AT_OFFSET).value = timestamp

    name_buf = _name_mapping(libc, addr)

    _state.payload_buf = new_buf
    _state.name_buf = name_buf
    _state.attributes = attributes
    _state.extra_attributes = extra_attributes
    _state.last_timestamp = timestamp
    return True


def _ensure_fork_hook() -> None:
    global _fork_hook_registered
    if _fork_hook_registered:
        return
    with _fork_hook_lock:
        if _fork_hook_registered:
            return
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_after_fork_in_child)
        _fork_hook_registered = True


def _after_fork_in_child() -> None:
    with _state.lock:
        attrs = _state.attributes
        extra = _state.extra_attributes
        _state.mapping_addr = 0
        _state.payload_buf = None
        _state.name_buf = None
        _state.last_timestamp = 0
        _state.owner_pid = 0
    if attrs is not None:
        publish(attrs, extra)


def publish(
    attributes: Mapping[str, Any],
    extra_attributes: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Publish or update the process context.

    Returns ``True`` on success. Returns ``False`` on unsupported platforms or
    if any required system call fails. Safe to call concurrently; updates serialize
    on a module-level lock.
    """
    if not is_supported():
        return False
    attrs = dict(attributes)
    extra = dict(extra_attributes) if extra_attributes else {}
    try:
        payload = encode_process_context(attrs, extra)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("process context: failed to encode payload")
        return False
    with _state.lock:
        if _state.mapping_addr == 0:
            ok = _publish_new(payload, attrs, extra)
        else:
            ok = _update_existing(payload, attrs, extra)
    if ok:
        _ensure_fork_hook()
    return ok


def try_publish_first_wins(
    attributes: Mapping[str, Any],
    extra_attributes: Optional[Mapping[str, Any]] = None,
) -> bool:
    """First-call wins: subsequent callers with different attributes log a warning.

    Suitable for SDK provider ``__init__`` integration where multiple providers
    may be constructed with different resources.
    """
    if not is_supported():
        return False
    attrs = dict(attributes)
    extra = dict(extra_attributes) if extra_attributes else {}
    with _state.lock:
        if _state.mapping_addr != 0:
            if attrs != _state.attributes or extra != _state.extra_attributes:
                logger.warning(
                    "process context: already published; ignoring conflicting "
                    "attributes (existing=%s new=%s)",
                    sorted((_state.attributes or {}).keys()),
                    sorted(attrs.keys()),
                )
            return False
    return publish(attrs, extra)


def maybe_publish_from_resource(resource) -> None:
    """Publish a process context from an SDK ``Resource``, if the feature is
    enabled and the platform is supported.

    Reads :envvar:`OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER`. Anything other
    than ``"true"`` (case-insensitive, whitespace stripped) disables the call.
    Multiple calls with the same attributes are idempotent; calls with
    different attributes log a warning and do not update.
    """
    enabled = os.environ.get(
        "OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER", ""
    ).strip().lower()
    if enabled != "true":
        return
    if not is_supported():
        return
    try:
        attributes = dict(resource.attributes)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "process context: could not read resource attributes"
        )
        return
    try_publish_first_wins(attributes)


def drop() -> None:
    """Unmap the published context. Intended for tests and shutdown paths."""
    with _state.lock:
        if _state.mapping_addr != 0 and _state.libc is not None:
            _state.libc.munmap(_state.mapping_addr, _HEADER_SIZE)
        _state.mapping_addr = 0
        _state.payload_buf = None
        _state.name_buf = None
        _state.attributes = None
        _state.extra_attributes = None
        _state.last_timestamp = 0
        _state.owner_pid = 0
