"""Trust-lane inventory and exception restoration for mutable module globals."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _tl as torch_tl, completeness_witness, rescue
from torchlens.capture import projections, trace as capture_trace

_SCOPED_CAPTURE_STATE = frozenset(
    {
        ("torchlens/_state.py", "_active_hook_plan"),
        ("torchlens/_state.py", "_active_intervention_spec"),
        ("torchlens/_state.py", "_active_owner_thread_id"),
        ("torchlens/_state.py", "_active_trace"),
        ("torchlens/_state.py", "_capture_replay_templates"),
        ("torchlens/_state.py", "_dynamo_warning_emitted"),
        ("torchlens/_state.py", "_func_call_id_counter"),
        ("torchlens/_state.py", "_functorch_warning_emitted"),
        ("torchlens/_state.py", "_logging_enabled"),
        ("torchlens/_state.py", "_relationship_input_id"),
        ("torchlens/_state.py", "_relationship_input_shape_hash"),
        ("torchlens/_state.py", "_relationship_model_class"),
        ("torchlens/_state.py", "_relationship_model_id"),
        ("torchlens/_state.py", "_relationship_weight_fingerprint"),
        ("torchlens/backends/torch/_tl.py", "_ACTIVE_LABEL_SESSION"),
        ("torchlens/backends/torch/completeness_witness.py", "_ACTIVE_WITNESS_STATE"),
        ("torchlens/backends/torch/rescue.py", "_rescue_active"),
        ("torchlens/capture/projections.py", "_active_recording_state"),
        ("torchlens/capture/trace.py", "_ACTIVE_CAPTURE_BACKEND"),
    }
)

_INSTALL_STATE_AND_CACHES = frozenset(
    {
        ("torchlens/backends/torch/_tl.py", "_RETIRED_LABEL_SESSION"),
        ("torchlens/backends/torch/backward.py", "_AUTOGRAD_WRAPPERS_INSTALLED"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_BACKWARD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_GRAD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_ENTER"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_INIT"),
        ("torchlens/backends/torch/backward.py", "_SAVED_TENSORS_HOOKS_INIT_PATCHED"),
        ("torchlens/backends/torch/belt.py", "_member_map"),
        ("torchlens/backends/torch/belt.py", "_report"),
        ("torchlens/backends/torch/buffer_writes.py", "_WITNESS_MARKER_STATE"),
        ("torchlens/backends/torch/escape_detection.py", "_TABLES"),
        ("torchlens/backends/torch/wrappers.py", "_DeviceContext"),
        ("torchlens/backends/torch/wrappers.py", "_torchvision_ops_ensured"),
        ("torchlens/capture/arg_positions.py", "_schema_corrections_applied"),
    }
)

_WARN_ONCE_STATE = frozenset(
    {("torchlens/validation/_stock_layer_grads.py", "_PASS_INDEX_PARSE_WARNED")}
)


def _lane_python_paths(repo: Path) -> list[Path]:
    """Return Python files governed by the trust lane.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    list[Path]
        Sorted in-lane Python source paths.
    """

    paths = [repo / "torchlens/_state.py"]
    for relative in ("torchlens/capture", "torchlens/validation", "torchlens/backends/torch"):
        paths.extend((repo / relative).rglob("*.py"))
    return sorted(paths)


def _declared_globals(repo: Path) -> set[tuple[str, str]]:
    """Return exact ``(path, name)`` pairs declared with ``global``.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    set[tuple[str, str]]
        Unique global declarations in trust-lane source files.
    """

    declarations: set[tuple[str, str]] = set()
    for path in _lane_python_paths(repo):
        tree = ast.parse(path.read_text())
        relative = path.relative_to(repo).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                declarations.update((relative, name) for name in node.names)
    return declarations


def _capture_scope_snapshot() -> dict[str, Any]:
    """Return high-risk per-capture global state for restoration checks.

    Returns
    -------
    dict[str, Any]
        Values that must be identical before and after a failed capture.
    """

    return {
        "logging_enabled": _state._logging_enabled,
        "active_trace": _state._active_trace,
        "active_owner_thread_id": _state._active_owner_thread_id,
        "nonowner_belt_armed": _state._nonowner_belt_armed,
        "active_fast_run_collector": _state._active_fast_run_collector,
        "active_hook_plan": _state._active_hook_plan,
        "active_intervention_spec": _state._active_intervention_spec,
        "capture_replay_templates": _state._capture_replay_templates,
        "relationship_model_id": _state._relationship_model_id,
        "relationship_model_class": _state._relationship_model_class,
        "relationship_weight_fingerprint": _state._relationship_weight_fingerprint,
        "relationship_input_id": _state._relationship_input_id,
        "relationship_input_shape_hash": _state._relationship_input_shape_hash,
        "runnable_ledger_armed": _state._runnable_ledger_armed,
        "active_label_session": torch_tl._ACTIVE_LABEL_SESSION,
        "active_witness_state": completeness_witness._ACTIVE_WITNESS_STATE,
        "rescue_active": rescue._rescue_active,
        "active_recording_state": projections._active_recording_state,
        "active_capture_backend": capture_trace._ACTIVE_CAPTURE_BACKEND,
    }


class _InjectedBaseFailure(BaseException):
    """BaseException subclass used to exercise the interruption cleanup arm."""


class _RaiseMidCapture(nn.Module):
    """Run one logged op before raising a selected exception class."""

    def __init__(self, error_type: type[BaseException]) -> None:
        """Store the exception type raised during ``forward``.

        Parameters
        ----------
        error_type:
            Exception class to raise after one tensor operation.
        """

        super().__init__()
        self.error_type = error_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one operation and then raise the injected exception.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This path never returns.

        Raises
        ------
        BaseException
            Always raises ``self.error_type`` after the logged operation.
        """

        _ = torch.relu(x)
        raise self.error_type("injected mid-capture failure")


class _BlockingCapture(nn.Module):
    """Hold a public capture open while a concurrent capture is attempted."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        """Store synchronization events for the capture overlap.

        Parameters
        ----------
        entered:
            Event set after the first logged operation runs.
        release:
            Event that permits the model to finish.
        """

        super().__init__()
        self.entered = entered
        self.release = release

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log one operation and wait until the contender is refused.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logged activation after the release event is set.
        """

        out = torch.relu(x)
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("concurrent capture test did not release the owner")
        return out


def test_global_state_inventory_is_classified_and_shrink_only() -> None:
    """Every trust-lane ``global`` declaration has one lifecycle class."""

    repo = Path(__file__).resolve().parents[1]
    categories = (_SCOPED_CAPTURE_STATE, _INSTALL_STATE_AND_CACHES, _WARN_ONCE_STATE)
    classified = set().union(*categories)

    assert sum(len(category) for category in categories) == len(classified), (
        "global lifecycle classes overlap"
    )
    assert _declared_globals(repo) == classified


@pytest.mark.parametrize("error_type", [RuntimeError, _InjectedBaseFailure])
def test_mid_capture_failure_restores_process_state(
    error_type: type[BaseException],
) -> None:
    """Ordinary and interruption failures leave no stale capture owner.

    Parameters
    ----------
    error_type:
        Failure class injected after one recorded operation.
    """

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()

    with pytest.raises(error_type, match="injected mid-capture failure"):
        tl.trace(_RaiseMidCapture(error_type), torch.ones(2))

    assert _capture_scope_snapshot() == before
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_concurrent_public_capture_refuses_without_corruption() -> None:
    """Overlapping public captures fail loudly and leave the owner intact."""

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Run the capture that owns process-global logging state."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.trace(nn.ReLU(), torch.ones(2))
    finally:
        release.set()
        owner.join(timeout=5.0)

    assert not owner.is_alive(), "owner capture did not finish after release"
    assert owner_errors == []
    assert _capture_scope_snapshot() == before
