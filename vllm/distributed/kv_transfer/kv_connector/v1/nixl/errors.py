# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classify KV-transfer failures that make an engine unusable.

Most KV-transfer failures are per-request: the connector logs them, fails the
request (or recomputes), and the engine keeps serving. One shape is not: a
CUDA launch failure poisons the process's CUDA context, so every later NIXL
transfer fails while the engine is still alive, still Ready, and still
receiving traffic.

That is what happened to two decode engines after an NVLink/MNNVL channel
reset revoked their peer's importers: every KV pull failed with
``NIXL_ERR_BACKEND`` / ``cuCtxGetId failed: unspecified launch failure``, the
engines stayed Ready for 12 minutes and failed every request routed to them.

This module holds the (pure) decision of whether a failure is that hard
shape. Keeping it free of NIXL and torch imports makes it cheap to call from
the transfer path and cheap to test.

Two distinctions matter:

* A remote peer that stops responding is a different condition, handled by
  the heartbeat/release path (e.g. #60268): the decoder's own context is
  healthy and it can serve other peers. Only a local context failure is fatal
  here.
* The signal is independent of ``kv_load_failure_policy``. Recomputing on a
  failed KV load recovers a request whose transfer failed, but no policy can
  recover a context that can never load again, so this must not be keyed off
  a count of ordinary load failures.
"""

from __future__ import annotations

import enum
from typing import Final

# A CUDA launch failure invalidates the whole context. UCX/NIXL report the
# same condition as either the CUDA name or the runtime's message.
_CUDA_CONTEXT_ERRORS: Final = (
    "cudaErrorLaunchFailure",
    "unspecified launch failure",
)

# A bare backend error can be a transient network problem. Combined with a
# failed context query it is the poisoned-context signature.
_NIXL_BACKEND_ERROR: Final = "NIXL_ERR_BACKEND"
_CUDA_CONTEXT_QUERY_ERROR: Final = "cuCtxGetId failed"


class KVTransferFailureKind(enum.Enum):
    """How bad a KV-transfer failure is for the process that observed it."""

    FATAL_CUDA_CONTEXT = "fatal_cuda_context"
    """The CUDA context is poisoned; the engine must not serve again."""

    OTHER = "other"
    """Anything else; handled per request by the existing retry policy."""


class KVTransferFatalError(RuntimeError):
    """Raised when a KV-transfer failure leaves the engine unusable.

    Raised from the worker's transfer-results path so it aborts the
    EngineCore: a dead engine fails its readiness probe and is restarted,
    whereas a poisoned engine keeps answering with errors.
    """


def classify_transfer_failure(
    error: BaseException | str | None,
) -> KVTransferFailureKind:
    """Classify a KV-transfer failure from its error text.

    Args:
        error: The exception raised by the NIXL wrapper, its message, or
            None when the failure was reported without one.

    Returns:
        ``FATAL_CUDA_CONTEXT`` when the failure means this engine's CUDA
        context can no longer be used, ``OTHER`` otherwise.

    """
    if error is None:
        return KVTransferFailureKind.OTHER
    message = error if isinstance(error, str) else str(error)
    if not message:
        return KVTransferFailureKind.OTHER

    lowered = message.lower()
    for token in _CUDA_CONTEXT_ERRORS:
        if token.lower() in lowered:
            return KVTransferFailureKind.FATAL_CUDA_CONTEXT
    if _NIXL_BACKEND_ERROR in message and _CUDA_CONTEXT_QUERY_ERROR in message:
        return KVTransferFailureKind.FATAL_CUDA_CONTEXT
    return KVTransferFailureKind.OTHER
