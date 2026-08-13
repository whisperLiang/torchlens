"""Trust-lane inventory and exception restoration for mutable module globals."""

from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path
from typing import Any, cast

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
        # The ``global`` statement is LEXICALLY in the split fragment, while the name it
        # binds lives in ``completeness_witness``: the fragment's functions are rebound to
        # that module's globals dict (``_rebind_function(..., globals())``). This inventory
        # keys on the lexical site, which is what the AST detector can see.
        ("torchlens/backends/torch/_completeness_finalize.py", "_ACTIVE_WITNESS_STATE"),
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


class _PauseFromForeignThread(nn.Module):
    """Log ops before and after a NON-OWNER thread enters ``pause_logging()``.

    The foreign thread models the reachable-without-concurrent-capture case: a
    thread merely ANALYZING an older Trace (``tl.save``, validation, an ``.out``
    transform) enters the same process-global pause the owner's forward relies on.

    The foreign pause is HELD OPEN across the owner's second op group, which is
    the realistic shape: an analysis thread's ``pause_logging()`` body spans a
    whole save / validation pass, not a single statement.
    """

    def __init__(self, ops_per_side: int) -> None:
        """Store how many logged ops run on each side of the foreign pause.

        Parameters
        ----------
        ops_per_side:
            Number of logged operations before, during, and after the pause.
        """

        super().__init__()
        self.ops_per_side = ops_per_side
        self.foreign_paused = threading.Event()
        self.foreign_release = threading.Event()
        self.foreign_error: list[BaseException] = []

    def _foreign_pause(self) -> None:
        """Hold ``pause_logging()`` open from a non-owner thread."""

        try:
            with _state.pause_logging():
                self.foreign_paused.set()
                if not self.foreign_release.wait(timeout=10.0):
                    raise TimeoutError("owner never released the foreign pause")
        except BaseException as error:  # pragma: no cover - reported by the test
            self.foreign_error.append(error)
        finally:
            self.foreign_paused.set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ops before, during, and after a held foreign pause.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activation after all three op groups ran.
        """

        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        foreign = threading.Thread(target=self._foreign_pause)
        foreign.start()
        assert self.foreign_paused.wait(timeout=10.0), "foreign pause never entered"
        # These ops run while a NON-OWNER thread holds the pause open.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        self.foreign_release.set()
        foreign.join(timeout=10.0)
        assert not foreign.is_alive(), "foreign pause thread did not exit"
        # And these run after the foreign thread restored what it saved.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        return x


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


def test_foreign_thread_pause_does_not_blind_the_owner_capture() -> None:
    """A non-owner ``pause_logging()`` never drops the owner's later ops.

    ``_PauseLogging.__enter__`` used to clear the process-global toggle
    unconditionally, so ANY thread pausing (even one only analyzing an old
    Trace) silently truncated a live capture from that instant on. The owner
    check now lives in the context manager itself, covering every call site.
    """

    ops_per_side = 3
    model = _PauseFromForeignThread(ops_per_side)
    trace = tl.trace(model, torch.ones(2))

    assert model.foreign_error == [], f"foreign pause raised {model.foreign_error!r}"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 3 * ops_per_side, (
        "ops logged while a non-owner thread held pause_logging() are missing: "
        f"the foreign pause blinded the capture (saw {len(relu_ops)} of "
        f"{3 * ops_per_side} relu ops)"
    )
    assert _state._logging_enabled is False
    assert _state._active_owner_thread_id is None


def test_capture_admission_runs_under_the_admission_lock() -> None:
    """Admission blocks while another thread holds the admission lock.

    ``active_logging``'s refusal check and its publication of ``_active_trace`` /
    ``_active_owner_thread_id`` / ``_logging_enabled`` are separate bytecodes.
    Unlocked, two threads entering together can both pass the check, and the
    loser then overwrites the winner's owner id — after which every op the winner
    logs is dropped by the wrapper's owner-thread fast path and its Trace is
    silently short, with no error anywhere. The check and the publication
    therefore have to happen under one lock; this asserts that directly, because
    the racing window itself is only a few bytecodes wide and a probabilistic
    probe cannot gate it reliably.
    """

    before = _capture_scope_snapshot()
    entered = threading.Event()
    blocked_for_lock = threading.Event()
    admitted = threading.Event()
    failures: list[BaseException] = []

    def admit() -> None:
        """Enter and immediately leave one capture session."""

        try:
            with _state.active_logging(cast("Any", object())):
                admitted.set()
        except BaseException as error:  # pragma: no cover - reported by the test
            failures.append(error)
            admitted.set()

    with _state._capture_admission_lock:
        entered.set()
        worker = threading.Thread(target=admit)
        worker.start()
        # The worker cannot reach the check, let alone publish, while the lock
        # is held here. If admission ran outside the lock it would sail through.
        blocked_for_lock.wait(timeout=0.5)
        assert not admitted.is_set(), (
            "active_logging admitted a capture while the admission lock was "
            "held: the check-then-publish sequence is not serialized"
        )
        assert _state._active_trace is None
        assert _state._active_owner_thread_id is None

    assert admitted.wait(timeout=10.0), "admission never completed after release"
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "admission worker hung"
    assert failures == [], f"admission failed after the lock was released: {failures!r}"
    assert _capture_scope_snapshot() == before


def test_contended_admission_never_publishes_partial_owner_state() -> None:
    """Under contention, an admitted session owns the globals for its whole body.

    Complements the lock test above: whatever the interleaving, a thread that is
    admitted must see its OWN owner id and trace for the entire session, and
    every other thread must get the documented refusal rather than a corrupted
    half-published state. Concurrent capture stays unsupported by design; this
    pins the admission mechanism's behavior under contention.
    """

    before = _capture_scope_snapshot()
    contenders = 4
    rounds = 25
    admitted = 0
    refused = 0
    stolen: list[tuple[int, int | None]] = []
    unexpected: list[BaseException] = []
    lock = threading.Lock()
    # Bytecode-level interleaving is what the admission lock excludes; make the
    # scheduler switch as often as possible so contention is real here.
    prior_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(rounds):
            barrier = threading.Barrier(contenders)

            def contend() -> None:
                """Enter ``active_logging`` at the same instant as the others."""

                nonlocal admitted, refused
                token = object()
                barrier.wait(timeout=10.0)
                try:
                    with _state.active_logging(cast("Any", token)):
                        mine = threading.get_ident()
                        for _ in range(200):
                            owner = _state._active_owner_thread_id
                            if owner != mine or _state._active_trace is not token:
                                with lock:
                                    stolen.append((mine, owner))
                                break
                except _state.ReentrantTraceError:
                    with lock:
                        refused += 1
                except BaseException as error:  # pragma: no cover - test signal
                    with lock:
                        unexpected.append(error)
                else:
                    with lock:
                        admitted += 1

            threads = [threading.Thread(target=contend) for _ in range(contenders)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20.0)
            assert not any(thread.is_alive() for thread in threads), "contender hung"
    finally:
        sys.setswitchinterval(prior_switch_interval)

    assert unexpected == [], f"unexpected admission failure: {unexpected!r}"
    assert stolen == [], (
        "an admitted capture's owner globals were overwritten by a racing "
        f"contender (mine, observed_owner) pairs: {stolen!r}"
    )
    assert admitted + refused == contenders * rounds
    assert admitted >= 1, "no capture was admitted at all"
    assert _capture_scope_snapshot() == before
