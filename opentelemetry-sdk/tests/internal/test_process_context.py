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

# pylint: disable=protected-access

import logging
import os
import re
import struct
import sys
import unittest

import pytest

from opentelemetry.sdk._internal import _process_context

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="OTEP 4719 publisher is Linux-only",
)


class VarintEncoderTests(unittest.TestCase):
    def test_varint_known_values(self):
        cases = [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (16383, b"\xff\x7f"),
            (16384, b"\x80\x80\x01"),
            (2**21 - 1, b"\xff\xff\x7f"),
            (2**21, b"\x80\x80\x80\x01"),
        ]
        for value, expected in cases:
            self.assertEqual(_process_context._varint(value), expected)

    def test_varint_rejects_negative(self):
        with self.assertRaises(ValueError):
            _process_context._varint(-1)


class EncoderRoundtripTests(unittest.TestCase):
    """Encoder produces output that the official protobuf parser accepts."""

    def test_roundtrip_via_proto_bindings(self):
        try:
            from opentelemetry.proto.processcontext.v1development.process_context_pb2 import (
                ProcessContext,
            )
        except ImportError:
            self.skipTest("opentelemetry-proto bindings not available")

        attrs = {
            "service.name": "svc",
            "service.instance.id": "uuid",
            "service.version": "1.2.3",
            "deployment.environment.name": "prod",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": "9.9.9",
            "host.name": "h1",
            "process.pid": 12345,
            "is.beta": True,
            "cpu.fraction": 0.5,
            "tags": ["env:prod", "team:profiling"],
        }
        extra = {"datadog.process_tags": "x" * 50_000}
        payload = _process_context.encode_process_context(attrs, extra)

        msg = ProcessContext()
        msg.ParseFromString(payload)

        decoded = {kv.key: kv.value for kv in msg.resource.attributes}
        self.assertEqual(decoded["service.name"].string_value, "svc")
        self.assertEqual(decoded["process.pid"].int_value, 12345)
        self.assertTrue(decoded["is.beta"].bool_value)
        self.assertEqual(decoded["cpu.fraction"].double_value, 0.5)
        self.assertEqual(
            [v.string_value for v in decoded["tags"].array_value.values],
            ["env:prod", "team:profiling"],
        )
        self.assertEqual(len(msg.attributes), 1)
        self.assertEqual(msg.attributes[0].key, "datadog.process_tags")
        self.assertEqual(
            len(msg.attributes[0].value.string_value), 50_000
        )


class _MapsScannerMixin:
    """Helpers shared by Linux-only tests."""

    def _find_otel_ctx_mapping(self):
        with open(f"/proc/{os.getpid()}/maps", "r", encoding="utf-8") as fh:
            for line in fh:
                if "OTEL_CTX" not in line:
                    continue
                m = re.match(
                    r"^([0-9a-f]+)-([0-9a-f]+)\s+\S+\s+\S+\s+\S+\s+\S+\s*(.*)$",
                    line,
                )
                if not m:
                    continue
                start, end, name = m.group(1), m.group(2), m.group(3)
                if (
                    name.startswith("[anon_shmem:OTEL_CTX]")
                    or name.startswith("[anon:OTEL_CTX]")
                    or name.startswith("/memfd:OTEL_CTX")
                ):
                    return int(start, 16), int(end, 16), name
        return None


@linux_only
class PublisherIntegrationTests(unittest.TestCase, _MapsScannerMixin):
    def setUp(self):
        _process_context.drop()

    def tearDown(self):
        _process_context.drop()

    def test_publish_creates_mapping_and_payload_is_valid(self):
        try:
            from opentelemetry.proto.processcontext.v1development.process_context_pb2 import (
                ProcessContext,
            )
        except ImportError:
            self.skipTest("opentelemetry-proto bindings not available")

        attrs = {"service.name": "svc-test", "service.version": "9.9.9"}
        self.assertTrue(_process_context.publish(attrs))

        mapping = self._find_otel_ctx_mapping()
        self.assertIsNotNone(
            mapping,
            "no OTEL_CTX mapping found in /proc/self/maps",
        )
        start, end, _name = mapping
        self.assertGreaterEqual(end - start, _process_context._HEADER_SIZE)

        # Read header from the live mapping via the address /proc/self/maps gives us.
        import ctypes

        header_bytes = bytes(
            (ctypes.c_char * _process_context._HEADER_SIZE).from_address(start)
        )
        sig, version, payload_size, ts, payload_ptr = struct.unpack(
            "<8sIIQQ", header_bytes
        )
        self.assertEqual(sig, b"OTEL_CTX")
        self.assertEqual(version, _process_context._HEADER_VERSION)
        self.assertGreater(ts, 0)
        self.assertGreater(payload_size, 0)
        self.assertNotEqual(payload_ptr, 0)

        payload = bytes(
            (ctypes.c_char * payload_size).from_address(payload_ptr)
        )
        msg = ProcessContext()
        msg.ParseFromString(payload)
        decoded = {kv.key: kv.value.string_value for kv in msg.resource.attributes}
        self.assertEqual(decoded["service.name"], "svc-test")
        self.assertEqual(decoded["service.version"], "9.9.9")

    def test_update_protocol_advances_timestamp(self):
        attrs1 = {"service.name": "svc-1"}
        attrs2 = {"service.name": "svc-2"}

        self.assertTrue(_process_context.publish(attrs1))
        ts1 = _process_context._state.last_timestamp
        self.assertTrue(_process_context.publish(attrs2))
        ts2 = _process_context._state.last_timestamp
        self.assertGreater(ts2, ts1)

    def test_first_wins_logs_warning_on_mismatch(self):
        attrs1 = {"service.name": "svc-1"}
        attrs2 = {"service.name": "svc-2"}
        self.assertTrue(_process_context.try_publish_first_wins(attrs1))
        with self.assertLogs(_process_context.logger, level=logging.WARNING):
            self.assertFalse(
                _process_context.try_publish_first_wins(attrs2)
            )
        # Still publishing the first set.
        self.assertEqual(
            _process_context._state.attributes, attrs1
        )

    def test_first_wins_idempotent_for_matching_attributes(self):
        attrs = {"service.name": "svc"}
        self.assertTrue(_process_context.try_publish_first_wins(attrs))
        self.assertFalse(_process_context.try_publish_first_wins(attrs))
        self.assertEqual(_process_context._state.attributes, attrs)


class EnvVarGateTests(unittest.TestCase):
    def test_env_var_disabled_is_noop(self):
        original = os.environ.pop(
            "OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER", None
        )
        try:
            _process_context.drop()

            class _Resource:
                attributes = {"service.name": "svc"}

            _process_context.maybe_publish_from_resource(_Resource())
            self.assertEqual(_process_context._state.mapping_addr, 0)
        finally:
            if original is not None:
                os.environ[
                    "OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER"
                ] = original

    @linux_only
    def test_env_var_enabled_publishes(self):
        os.environ["OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER"] = "true"
        try:
            _process_context.drop()

            class _Resource:
                attributes = {"service.name": "svc"}

            _process_context.maybe_publish_from_resource(_Resource())
            self.assertNotEqual(
                _process_context._state.mapping_addr, 0
            )
        finally:
            os.environ.pop(
                "OTEL_EXPERIMENTAL_PROCESS_CONTEXT_PUBLISHER", None
            )
            _process_context.drop()
