"""Capability-table load-bearing conformance.

The registered ``BackendCapabilities`` table must be the single authority for
backend feature support: every flag must gate production behavior, and the
gates must be biconditional (flag ``False`` rejects; flag ``True`` admits)
without editing any per-backend message policy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends import (
    BackendCapabilities,
    BackendUnsupportedError,
    get_backend_spec,
    register_backend_spec,
    registered_backend_specs,
)
from torchlens.backends._options import (
    JAX_EXTRA_KWARG_POLICY,
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_EXTRA_KWARG_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_EXTRA_KWARG_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_EXTRA_KWARG_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_EXTRA_KWARG_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    TRACE_OPTION_CAPABILITY_GATES,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
)
from torchlens.data_classes._backend_capability_guards import raise_if_no_backward_capture

pytestmark = [pytest.mark.backend_parity, pytest.mark.smoke]

_TORCHLENS_ROOT = Path(tl.__file__).resolve().parent

_PREVIEW_NAMES = ("mlx", "jax", "tinygrad", "paddle", "tf")

_EXTRA_POLICIES = {
    "mlx": MLX_EXTRA_KWARG_POLICY,
    "jax": JAX_EXTRA_KWARG_POLICY,
    "tinygrad": TINYGRAD_EXTRA_KWARG_POLICY,
    "paddle": PADDLE_EXTRA_KWARG_POLICY,
    "tf": TF_EXTRA_KWARG_POLICY,
}

_OPTION_POLICIES = {
    "mlx": MLX_PREVIEW_TRACE_OPTION_POLICY,
    "jax": JAX_PREVIEW_TRACE_OPTION_POLICY,
    "tinygrad": TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    "paddle": PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    "tf": TF_PREVIEW_TRACE_OPTION_POLICY,
}

_OPTION_POLICY_DEFAULTS: dict[str, Any] = {
    "layers_to_save": "all",
    "input_kwargs": {},
    "output_device": "same",
    "save_raw_activations": True,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
}


def test_every_capability_flag_has_a_production_consumer() -> None:
    """No decorative flags: every field is read by production code or a gate map."""

    gated_fields = set(TRACE_OPTION_CAPABILITY_GATES.values())
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(_TORCHLENS_ROOT.rglob("*.py"))
    )
    for field in dataclasses.fields(BackendCapabilities):
        directly_read = f"capabilities.{field.name}" in source
        assert directly_read or field.name in gated_fields, (
            f"BackendCapabilities.{field.name} gates nothing in production code; "
            "a flag no code reads can silently lie."
        )


@pytest.mark.parametrize("name", _PREVIEW_NAMES)
def test_extra_kwarg_gates_are_biconditional(name: str) -> None:
    """intervene/storage/streaming reject on flag False and admit on flag True."""

    capabilities = get_backend_spec(name).capabilities
    policy = _EXTRA_POLICIES[name]
    sentinel = object()
    for option, flag in (("intervene", "interventions"), ("storage", "streaming"),
                         ("streaming", "streaming")):
        assert not getattr(capabilities, flag)
        with pytest.raises(BackendUnsupportedError):
            reject_extra_trace_kwargs({option: sentinel}, policy, capabilities=capabilities)
        opened = dataclasses.replace(capabilities, **{flag: True})
        reject_extra_trace_kwargs({option: sentinel}, policy, capabilities=opened)


@pytest.mark.parametrize("name", _PREVIEW_NAMES)
def test_option_policy_gates_are_biconditional(name: str) -> None:
    """save_grads/backward_ready/save_rng_states follow the capability table."""

    capabilities = get_backend_spec(name).capabilities
    policy = _OPTION_POLICIES[name]
    cases = [("save_grads", "backward_capture"), ("backward_ready", "backward_capture")]
    if "save_rng_states" in (policy.rejected_truthy_messages or {}):
        cases.append(("save_rng_states", "rng_replay"))
    for option, flag in cases:
        options = dict(_OPTION_POLICY_DEFAULTS)
        options[option] = True
        assert not getattr(capabilities, flag)
        with pytest.raises(BackendUnsupportedError):
            reject_unsupported_trace_options(options, policy, capabilities=capabilities)
        opened = dataclasses.replace(capabilities, **{flag: True})
        reject_unsupported_trace_options(options, policy, capabilities=opened)


def test_record_gate_reads_the_capability_table() -> None:
    """tl.record's torch-only gate is owned by capabilities.fastlog."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    for name in _PREVIEW_NAMES:
        assert not get_backend_spec(name).capabilities.fastlog
        with pytest.raises(BackendUnsupportedError, match="torch-only"):
            tl.record(model, inputs, backend=name)

    original = get_backend_spec("tinygrad")
    opened = dataclasses.replace(
        original,
        capabilities=dataclasses.replace(original.capabilities, fastlog=True),
    )
    register_backend_spec(opened, replace=True)
    try:
        try:
            tl.record(model, inputs, backend="tinygrad", save=tl.func("linear"))
        except BackendUnsupportedError as exc:
            assert "torch-only" not in str(exc), (
                "fastlog gate ignored capabilities.fastlog=True; the table is not "
                "the gate authority."
            )
        except Exception:
            # Any non-gate failure means the call got PAST the fastlog gate,
            # which is exactly the arming proof this test needs.
            pass
    finally:
        register_backend_spec(original, replace=True)


def test_runnable_producer_gate_reads_save_levels() -> None:
    """The runnable producer refusal keys on 'runnable' in save_levels."""

    from torchlens._io.runnable import build_sparse_run_descriptor
    from torchlens.runnable import RunnableErrorCode

    assert "runnable" in get_backend_spec("torch").capabilities.save_levels
    for name in _PREVIEW_NAMES:
        assert "runnable" not in get_backend_spec(name).capabilities.save_levels

    model = nn.Linear(2, 2)
    trace = tl.trace(model, torch.randn(1, 2))
    try:
        descriptor = build_sparse_run_descriptor(trace)
        torch_codes = {d.code for d in descriptor.preflight.diagnostics}
        assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY not in torch_codes

        trace.backend = "paddle"
        descriptor = build_sparse_run_descriptor(trace)
        preview_codes = {d.code for d in descriptor.preflight.diagnostics}
        assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY in preview_codes
    finally:
        trace.backend = "torch"
        tl.release_model(model)


class _StubTrace:
    def __init__(self, backend: str) -> None:
        self.backend = backend


@pytest.mark.parametrize("name", ("jax", "mlx", "tinygrad", "paddle"))
def test_backward_accessor_guard_redirects_to_derived_grads(name: str) -> None:
    """Derived-grads backends refuse with the derived-gradient redirect."""

    with pytest.raises(ValueError, match="derived_grads"):
        raise_if_no_backward_capture(_StubTrace(name), plural_subject="backward_passes")


def test_backward_accessor_guard_tf_has_no_derived_redirect() -> None:
    """tf declares no derived-gradient surface, so the redirect must not appear."""

    with pytest.raises(ValueError, match="declares no derived-gradient surface"):
        raise_if_no_backward_capture(_StubTrace("tf"), plural_subject="backward_passes")


def test_backward_accessor_guard_passes_torch_and_unknown() -> None:
    """torch and unregistered backends never trip the guard."""

    raise_if_no_backward_capture(_StubTrace("torch"), plural_subject="backward_passes")
    raise_if_no_backward_capture(_StubTrace("not-a-backend"), plural_subject="backward_passes")


def test_backward_accessors_raise_for_all_non_backward_backends() -> None:
    """paddle/tf join jax/mlx/tinygrad: no silent-empty backward accessors."""

    model = nn.Linear(2, 2)
    trace = tl.trace(model, torch.randn(1, 2))
    try:
        for name in _PREVIEW_NAMES:
            trace.backend = name
            with pytest.raises(ValueError):
                _ = trace.backward_passes
            with pytest.raises(ValueError):
                _ = trace.saved_grad_ops
            with pytest.raises(ValueError):
                _ = trace.layer_list[0].grads
        trace.backend = "torch"
        _ = trace.backward_passes
        _ = trace.saved_grad_ops
        _ = trace.layer_list[0].grads
    finally:
        trace.backend = "torch"
        tl.release_model(model)


def test_registered_capability_tables_are_truthful_at_registration() -> None:
    """Preview specs declare no capability their gates reject (spot invariants)."""

    for spec in registered_backend_specs():
        capabilities = spec.capabilities
        if str(spec.name) == "torch":
            assert capabilities.backward_capture
            assert capabilities.fastlog
            assert capabilities.interventions
            assert "runnable" in capabilities.save_levels
            continue
        assert not capabilities.backward_capture
        assert not capabilities.fastlog
        assert not capabilities.interventions
        assert not capabilities.streaming
        assert not capabilities.rng_replay
        assert "runnable" not in capabilities.save_levels
