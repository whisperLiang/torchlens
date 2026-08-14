"""fast=True per-iteration faithfulness ceilings (grind B3-L1, Tier-1 #1).

The verify-once gate proves only the FIRST input settled ``verified``; every
later fast iteration runs on a different input and must re-derive the
per-input dynamic ceilings through ``_path_faithfulness`` exactly like the
ordinary provider. These tests pin the two published wrong-value repros
(host-scalar escape, layout twin) to the honest ``unverifiable`` verdict and
prove fast mode can never IMPROVE the verdict over ``fast=False`` on the same
artifact and input.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

pytestmark = pytest.mark.smoke


class HostScalarEscapeModel(nn.Module):
    """Bakes a tensor->host escaped scalar into a downstream literal."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Multiply by a host-escaped derived integer constant."""

        scale = int(value.sum().item()) % 5 + 1
        return value * scale


class LayoutPredicateModel(nn.Module):
    """Branches on a layout predicate derived from the model input."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Take a contiguity-dependent arm on an input-rooted activation."""

        doubled = value * 2
        if doubled.is_contiguous():
            return doubled + 100
        return doubled - 100


def _runnable_artifact(model: nn.Module, inputs: torch.Tensor, path: Path) -> Path:
    """Capture and save one runnable artifact for the fast-path repros."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(
            model,
            inputs,
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    captured.save(path, level="runnable")
    return path


def test_fast_run_host_scalar_escape_changed_input_never_verified(tmp_path: Path) -> None:
    """A changed-input fast iteration with a baked host scalar settles unverifiable."""

    original = torch.ones(4)
    changed = torch.ones(4) * 2
    path = _runnable_artifact(HostScalarEscapeModel(), original, tmp_path / "escape-fast.tlspec")

    ordinary = tl.load(path).run(inputs=changed)
    assert ordinary.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert ordinary.report.poisoned

    loaded = tl.load(path)
    first = loaded.run(inputs=original, fast=True)
    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED

    fast = loaded.run(inputs=changed, fast=True)
    assert fast.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert fast.report.poisoned
    # The wrong replayed value must never be blessed: the report matches the
    # ordinary provider's verdict on the identical artifact and input.
    assert fast.report.path_faithfulness is ordinary.report.path_faithfulness


def test_fast_run_original_input_escape_stays_verified(tmp_path: Path) -> None:
    """Repeating the ORIGINAL input keeps the escape digest fresh: still verified."""

    original = torch.ones(4)
    path = _runnable_artifact(
        HostScalarEscapeModel(), original, tmp_path / "escape-fast-orig.tlspec"
    )

    loaded = tl.load(path)
    first = loaded.run(inputs=original, fast=True)
    second = loaded.run(inputs=original.clone(), fast=True)

    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(second.output, HostScalarEscapeModel()(original))


def test_fast_run_layout_twin_never_verified(tmp_path: Path) -> None:
    """A same-shape stride-twin input settles unverifiable on the fast path too."""

    original = torch.ones(3, 3)
    twin = torch.ones(3, 3).t()
    assert twin.shape == original.shape
    assert twin.stride() != original.stride()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(
            LayoutPredicateModel(),
            original,
            capture=CaptureOptions(
                intervention_ready=True,
                capture_container_structure=True,
                cache=False,
            ),
        )
    path = tmp_path / "layout-fast.tlspec"
    captured.save(path, level="runnable")

    ordinary = tl.load(path).run(inputs=twin)
    assert ordinary.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE

    loaded = tl.load(path)
    first = loaded.run(inputs=original, fast=True)
    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED

    fast = loaded.run(inputs=twin, fast=True)
    assert fast.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert fast.report.poisoned


def test_fast_sparse_run_consumes_alias_unresolved_flag() -> None:
    """The alias-topology ``unresolved`` ceiling is threaded, never discarded.

    Guards the B3-R09-2 discard site structurally: ``_bind_inputs`` returns the
    flag and ``run`` must pass it to ``_path_faithfulness``. A source scan is
    the cheapest tripwire against the discarded-binding regression.
    """

    import inspect

    from torchlens import _fast_run

    source = inspect.getsource(_fast_run._FastSparseSession)
    assert "alias_unresolved" in source
    assert "input_alias_unresolved=input_alias_unresolved" in source
    assert "_, _unresolved" not in source
    run_source = inspect.getsource(_fast_run._FastSparseSession.run)
    assert "provisional_path_faithfulness=PathFaithfulness.VERIFIED" not in run_source


class BufferGainModel(nn.Module):
    """Two linears plus a directly consumed buffer (a non-collectable save)."""

    def __init__(self) -> None:
        """Register the layers and the directly read gain buffer."""

        super().__init__()
        self.first = nn.Linear(3, 3)
        self.second = nn.Linear(3, 3)
        self.register_buffer("gain", torch.tensor([2.0, 1.0, 0.5]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the stacked linear output by the buffer."""

        return self.second(self.first(value)) * self.gain


def test_fast_live_module_refusal_fires_before_activation_wipe() -> None:
    """A typed module-plan refusal must not destroy the user's saved payloads.

    The buffer read is saved but not fast-collectable, so it is exactly the
    payload the unsupported-activation wipe targets; the module-address
    refusal must fire BEFORE that wipe runs (grind b3-l1, F-R07).
    """

    model = BufferGainModel().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    saved_before = {op.label for op in captured.layer_list if op.has_saved_activation}
    assert "buffer_1:1" in saved_before
    del model.second

    with pytest.raises(Exception) as excinfo:
        captured.run(inputs=torch.ones(2, 3), fast=True)
    assert excinfo.value.fields["detection_stage"] == "fast_live_module_plan"

    saved_after = {op.label for op in captured.layer_list if op.has_saved_activation}
    assert saved_after == saved_before
    assert captured.layer_dict_all_keys["buffer_1:1"].out is not None


def test_fast_live_divergence_poisons_half_refreshed_trace() -> None:
    """A diverged fast-live run marks the mixed-activation user Trace poisoned."""

    class BranchingFunctionModel(nn.Module):
        """Choose between two same-shape activation functions from tensor data."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply the branch selected by the runtime sum."""

            if bool((value.sum() > 0).item()):
                return torch.relu(value)
            return torch.sigmoid(value)

    model = BranchingFunctionModel().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(model, torch.ones(2), save=tl.func("relu"))

    from torchlens.errors import PathDivergenceError

    with pytest.raises(PathDivergenceError):
        captured.run(inputs=-torch.ones(2), fast=True)

    # Boundary/site payloads were overwritten in place up to the divergence
    # point, so the user-owned Trace must carry the monotonic poison mark and
    # refuse downstream faithful consumers.
    assert captured._runnable.path_faithfulness is PathFaithfulness.DIVERGED


def test_fast_live_inherited_divergence_never_unregisters_user_trace() -> None:
    """An inherited divergence raise must not evict the user's live Trace."""

    class BranchingFunctionModel(nn.Module):
        """Choose between two same-shape activation functions from tensor data."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply the branch selected by the runtime sum."""

            if bool((value.sum() > 0).item()):
                return torch.relu(value)
            return torch.sigmoid(value)

    from torchlens import _state
    from torchlens.errors import PathDivergenceError

    model = BranchingFunctionModel().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        captured = tl.trace(model, torch.ones(2), save=tl.func("relu"))

    with pytest.raises(PathDivergenceError):
        captured.run(inputs=-torch.ones(2), fast=True)
    # The poisoned mark is monotonic; a later good-input run raises the
    # inherited divergence but must leave the USER-owned Trace registered
    # (every other provider passes a throwaway fork to the discard arm).
    with pytest.raises(PathDivergenceError):
        captured.run(inputs=torch.ones(2), fast=True)
    assert any(log is captured for log in _state.list_logs())


def test_remove_fast_live_hooks_survives_raising_remove() -> None:
    """One raising handle.remove() must not strand the remaining hooks."""

    from torchlens._fast_run import _remove_fast_live_hooks

    removed: list[int] = []

    class Handle:
        """Hook-handle stand-in with an optional failing removal."""

        def __init__(self, key: int, fail: bool = False) -> None:
            """Record the handle identity and failure mode."""

            self.key = key
            self.fail = fail

        def remove(self) -> None:
            """Remove the handle or raise like a torn hook registry."""

            if self.fail:
                raise RuntimeError("torn hook registry")
            removed.append(self.key)

    handles: list = [Handle(1), Handle(2, fail=True), Handle(3)]
    with pytest.raises(RuntimeError, match="torn hook registry"):
        _remove_fast_live_hooks(handles)
    assert removed == [1, 3]
    assert handles == []


def test_close_fast_run_session_retryable_after_raising_close() -> None:
    """A raising close leaves the session attached and retryable, never stranded."""

    from torchlens._fast_run import close_fast_run_session

    class FlakySession:
        """Session stand-in whose first close raises."""

        def __init__(self) -> None:
            """Arm one failing close."""

            self.calls = 0

        def close(self) -> None:
            """Raise once, then succeed."""

            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first close fails")

    class Holder:
        """Bare object with a Trace-like __dict__."""

    trace = Holder()
    session = FlakySession()
    trace.__dict__["_fast_run_session"] = session

    with pytest.raises(RuntimeError, match="first close fails"):
        close_fast_run_session(trace)
    assert trace.__dict__["_fast_run_session"] is session

    close_fast_run_session(trace)
    assert "_fast_run_session" not in trace.__dict__
    assert session.calls == 2


class _PlainLinear(nn.Module):
    """Two-op model for the admission fault-injection probes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(value))


def test_fast_live_admission_guard_failure_refuses_not_fail_open(monkeypatch) -> None:
    """A broken input-contract guard REFUSES the fast path (R22-2 layer 1).

    Opus's fault injection: with the classifier machinery raising internally,
    the typed refusal used to silently vanish and the forward ran unguarded --
    guard failure was indistinguishable from inputs-match.
    """

    import torchlens._runnable_execution as execution
    from torchlens.errors import RunCapabilityUnavailableError

    model = _PlainLinear().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    captured.run(inputs=torch.ones(2, 3), fast=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("injected classifier failure")

    monkeypatch.setattr(execution, "_live_runtime_input_leaves", _boom)
    with pytest.raises(RunCapabilityUnavailableError):
        captured.run(inputs=torch.ones(2, 3), fast=True)


def test_fast_live_input_refresh_arity_guarded_and_poisons(monkeypatch) -> None:
    """The input-payload refresh zip guards arity like the output branch (R22-2 layer 2).

    A truncating zip used to keep STALE capture-time activations on the
    surplus input ops with no disclosure.
    """

    import torchlens._fast_run as fast_run
    from torchlens.errors import PathDivergenceError

    model = _PlainLinear().eval()
    captured = tl.trace(model, torch.ones(2, 3), save=lambda op: True)
    captured.run(inputs=torch.ones(2, 3), fast=True)

    # Patch the REFRESH-side binding only (_fast_run's module global); the
    # admission classifier keeps the real execution-module binding, so the
    # run is admitted and the refresh sees a truncated leaf list.
    real = fast_run._live_runtime_input_leaves

    def _truncating(*args, **kwargs):
        leaves = real(*args, **kwargs)
        if leaves is not None:
            return list(leaves)[:-1]
        return leaves

    monkeypatch.setattr(fast_run, "_live_runtime_input_leaves", _truncating)
    with pytest.raises(PathDivergenceError) as excinfo:
        captured.run(inputs=torch.ones(2, 3), fast=True)
    assert excinfo.value.fields["code"] == "input_tree_mismatch"
