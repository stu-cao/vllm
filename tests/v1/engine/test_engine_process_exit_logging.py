# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine-core supervisor must record how a process died.

An EngineCore that exits without a trace used to leave nothing behind but
``exitcode != 0``, which cannot distinguish "returned status 1" from "was
killed with SIGKILL". These tests pin the decoded form and the log level, so
an unexpected death is always reported before the manager shuts the rest of
the engines down.
"""

import logging
import multiprocessing
import os
import signal
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine import utils as engine_utils

_LOGGER = "vllm.v1.engine.utils"


def _manager_with(processes) -> engine_utils.CoreEngineProcManager:
    """Build a manager without spawning engines or registering a finalizer."""
    manager = object.__new__(engine_utils.CoreEngineProcManager)
    manager.processes = processes
    manager.manager_stopped = threading.Event()
    manager.failed_proc_name = None
    manager._finalizer = MagicMock()
    manager._finalizer.detach.return_value = None
    return manager


def _fake_proc() -> SimpleNamespace:
    return SimpleNamespace(name="EngineCore_DP1", pid=4242)


# ===================================================================
# describe_exitcode
# ===================================================================


def test_describe_exitcode_clean_exit():
    assert engine_utils.describe_exitcode(0) == "0 (clean exit)"


def test_describe_exitcode_self_reported_status():
    assert engine_utils.describe_exitcode(1) == "1 (self-reported exit status 1)"


@pytest.mark.parametrize(
    "signame", ["SIGKILL", "SIGTERM", "SIGSEGV", "SIGABRT", "SIGBUS"]
)
def test_describe_exitcode_names_common_signals(signame):
    signum = getattr(signal, signame)
    assert engine_utils.describe_exitcode(-signum) == (
        f"-{signum} (killed by {signame}, signal {signum})"
    )


def test_describe_exitcode_unknown_signal():
    assert engine_utils.describe_exitcode(-200) == (
        "-200 (killed by unnamed signal, signal 200)"
    )


def test_describe_exitcode_unknown_state():
    assert engine_utils.describe_exitcode(None) == (
        "exit code unknown (process state not collected)"
    )


# ===================================================================
# _log_process_exit
# ===================================================================


def test_unexpected_death_logs_error(caplog):
    manager = _manager_with([])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        manager._log_process_exit(_fake_proc(), -signal.SIGKILL)

    records = [r for r in caplog.records if "EngineCore_DP1" in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    message = record.getMessage()
    assert f"-{signal.SIGKILL}" in message
    assert "SIGKILL" in message
    assert "pid 4242" in message
    assert "manager_stopped=False" in message


def test_expected_shutdown_logs_warning(caplog):
    manager = _manager_with([])
    manager.manager_stopped.set()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        manager._log_process_exit(_fake_proc(), -signal.SIGTERM)

    records = [r for r in caplog.records if "EngineCore_DP1" in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "SIGTERM" in message
    assert "manager_stopped=True" in message


def test_clean_exit_is_not_logged(caplog):
    manager = _manager_with([])

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        manager._log_process_exit(_fake_proc(), 0)

    assert not [r for r in caplog.records if "EngineCore_DP1" in r.getMessage()]


# ===================================================================
# monitor_engine_liveness
# ===================================================================


def _die_by_sigkill():
    os.kill(os.getpid(), signal.SIGKILL)


def _exit_cleanly():
    return


def _run_and_join(ctx, target, name):
    proc = ctx.Process(target=target, name=name)
    proc.start()
    proc.join(timeout=60)
    assert proc.exitcode is not None, "helper process did not exit in time"
    return proc


def test_monitor_reports_signal_death(caplog):
    ctx = multiprocessing.get_context("spawn")
    proc = _run_and_join(ctx, _die_by_sigkill, "EngineCore_DP0")
    manager = _manager_with([proc])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        manager.monitor_engine_liveness()

    assert manager.failed_proc_name == "EngineCore_DP0"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any(
        "EngineCore_DP0" in r.getMessage() and "SIGKILL" in r.getMessage()
        for r in errors
    ), [r.getMessage() for r in errors]


def test_monitor_does_not_flag_clean_exit(caplog):
    ctx = multiprocessing.get_context("spawn")
    proc = _run_and_join(ctx, _exit_cleanly, "EngineCore_DP0")
    manager = _manager_with([proc])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        manager.monitor_engine_liveness()

    assert manager.failed_proc_name is None
    assert not [r for r in caplog.records if "EngineCore_DP0" in r.getMessage()]
