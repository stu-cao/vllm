# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import signature
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest

from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
    flashinfer_autotune,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend)
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ],
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 32, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(skip_attn):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)

    assert get_buckets.call_args_list == [call(8192), call(128)]
    assert autotune.call_args_list == [
        call(tuning_buckets=max_buckets),
        call(tuning_buckets=deferred_buckets),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


@pytest.mark.parametrize(
    ("reuse_cache", "cache_exists", "expect_tuning"),
    [(True, True, False), (True, False, True), (False, True, True)],
)
def test_flashinfer_autotune_reuse_cache_skips_tuning_pass(
    tmp_path, monkeypatch, reuse_cache, cache_exists, expect_tuning
):
    flashinfer_autotuner = pytest.importorskip("flashinfer.autotuner")
    cache_path = tmp_path / "autotune.json"
    if cache_exists:
        cache_path.write_bytes(b"{}")
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_REUSE_CACHE", str(int(reuse_cache)))
    world = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        cpu_group=None,
        broadcast_object=lambda obj, src: obj,
        barrier=Mock(),
    )
    tuner = Mock()
    warmup = "vllm.model_executor.warmup.kernel_warmup."

    with (
        patch("vllm.distributed.parallel_state.get_world_group", return_value=world),
        patch.object(flashinfer_autotuner.AutoTuner, "get", return_value=tuner),
        patch.object(flashinfer_autotuner, "set_autotune_process_group"),
        patch(warmup + "resolve_flashinfer_autotune_file", return_value=cache_path),
        patch(warmup + "write_flashinfer_autotune_cache"),
        patch(warmup + "_flashinfer_autotune_skip_ops", return_value=None),
        patch(warmup + "_run_flashinfer_autotune_dummy_runs") as dummy_runs,
        patch(warmup + "_run_flashinfer_bf16_autotune_dummy_run"),
        patch(warmup + "replayssm_autotune_warmup"),
        patch(warmup + "_autotune_kimi_k3_kda_qkvg"),
        patch("vllm.utils.flashinfer.autotune"),
    ):
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(
                attention_config=SimpleNamespace(hisparse_config=None)
            ),
            get_model=Mock(),
        )
        flashinfer_autotune(runner)

    assert tuner.load_configs.called == cache_exists
    assert dummy_runs.called == expect_tuning
    assert tuner.save_configs.called == expect_tuning
