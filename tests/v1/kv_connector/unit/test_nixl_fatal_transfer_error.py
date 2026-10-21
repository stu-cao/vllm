# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A poisoned CUDA context must stop the engine, not just fail requests.

After an NVLink/MNNVL channel reset, NIXL importers can be revoked while the
engine's CUDA context survives just enough to report the engine as Ready:
every KV pull then fails with ``NIXL_ERR_BACKEND`` / ``cuCtxGetId failed``.
The connector must classify that shape and abort, instead of serving a
decode engine that can never load KV again.
"""

import logging

import pytest

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.errors import (
    KVTransferFailureKind,
    KVTransferFatalError,
    classify_transfer_failure,
)

FATAL = KVTransferFailureKind.FATAL_CUDA_CONTEXT
OTHER = KVTransferFailureKind.OTHER
_NIXL_LOGGER = "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker"


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: unspecified launch failure",
        "cudaErrorLaunchFailure",
        "NIXL_ERR_BACKEND: cuCtxGetId failed: unspecified launch failure",
        "nixlBackendError: NIXL_ERR_BACKEND (cuCtxGetId failed)",
    ],
)
def test_hard_launch_failures_are_fatal(message):
    assert classify_transfer_failure(message) is FATAL


@pytest.mark.parametrize(
    "error",
    [
        None,
        "",
        # A backend error alone is a transport problem, not a broken context.
        "NIXL_ERR_BACKEND",
        # The context query alone is not enough to call the context poisoned.
        "cuCtxGetId failed",
        # The heartbeat release path (#60268) reports remote disconnects;
        # losing a peer must never kill the decoder that noticed.
        "nixlRemoteDisconnectError: remote agent disconnected",
        "Connection reset by peer",
        "KV load timeout",
    ],
)
def test_recoverable_failures_are_not_fatal(error):
    assert classify_transfer_failure(error) is OTHER


def test_exception_objects_are_classified_by_message():
    error = RuntimeError("KV pull failed: CUDA error: unspecified launch failure")
    assert classify_transfer_failure(error) is FATAL


def _worker():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    worker = object.__new__(NixlBaseConnectorWorker)
    worker._fatal_context_error = None
    return worker


def test_poisoned_context_latches_and_raises(caplog):
    worker = _worker()

    with caplog.at_level(logging.ERROR, logger=_NIXL_LOGGER):
        worker._record_fatal_context_failure(
            RuntimeError("CUDA error: unspecified launch failure"), "req-1"
        )

    assert worker._fatal_context_error is not None
    assert len([r for r in caplog.records if "poisoned CUDA context" in r.getMessage()])
    with pytest.raises(KVTransferFatalError, match="poisoned CUDA context"):
        worker._raise_if_cuda_context_poisoned()


def test_poisoned_context_logs_once(caplog):
    worker = _worker()
    with caplog.at_level(logging.ERROR, logger=_NIXL_LOGGER):
        worker._record_fatal_context_failure("unspecified launch failure", "req-1")
        first = [r for r in caplog.records if "poisoned CUDA context" in r.getMessage()]
        caplog.clear()
        worker._record_fatal_context_failure("unspecified launch failure", "req-2")

    assert len(first) == 1
    assert not [r for r in caplog.records if "poisoned CUDA context" in r.getMessage()]


def test_transient_failure_does_not_latch():
    worker = _worker()
    worker._record_fatal_context_failure(RuntimeError("NIXL_ERR_BACKEND"), "req-1")

    assert worker._fatal_context_error is None
    worker._raise_if_cuda_context_poisoned()


def test_fail_fast_can_be_disabled(monkeypatch):
    worker = _worker()
    worker._record_fatal_context_failure("unspecified launch failure", "req-1")

    monkeypatch.setattr(
        envs, "VLLM_KV_CONNECTOR_FAIL_FAST_ON_CUDA_CONTEXT_ERROR", False
    )
    worker._raise_if_cuda_context_poisoned()


def test_remote_peer_loss_does_not_abort_decoder():
    """Regression guard for the #60268 unreachable-peer release path."""
    worker = _worker()
    worker._record_fatal_context_failure(
        RuntimeError("nixlRemoteDisconnectError: heartbeat failed"), "req-1"
    )

    assert worker._fatal_context_error is None
    worker._raise_if_cuda_context_poisoned()
