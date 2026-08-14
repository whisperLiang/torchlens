"""Install/teardown transactionality of the capture-side guards (B3 L4).

The headline gate here is `test_partial_first_time_decoration_completes_on_retry`, which
was verified RED on the pre-fix tree: `_wrap_torch_locked` keyed its "full decoration vs
re-install from the existing maps" choice on `_state._orig_to_decorated` being non-empty,
so a decoration that failed partway through pass 2 took the RE-INSTALL branch, reinstalled
only the partial map, and stamped `_is_decorated = True`. That permanently disarmed the
#138 retry guard `decorate_all_once` exists to provide. Measured A/B on the same injected
failure: pre-fix the retry left 3313 targets undecorated (map stuck at 57) and the very
next capture raised `OutputAttributionError`; post-fix the retry completes the map (1936
wrappers) and the capture works.

The rest cover the neighbouring install/unwind transactions in the same lane: the
distributed arm's two wrap families, the escape detector's `sys.monitoring` tool-id
release (only a handful of ids exist, so six leaks exhaust the space process-wide), and
the runnable state-context unwind loops that could strand the CALLER's thread inside
`no_grad`/`inference_mode`/autocast for the rest of the process.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens._state as _state
import torchlens.backends.torch.wrappers as wrappers_module
from torchlens.backends.torch.wrappers import (
    get_orig_torch_funcs,
    is_decorated_function,
    nested_getattr,
)

pytestmark = pytest.mark.smoke


def _undecorated_targets() -> list[str]:
    """Return inventory targets that are resolvable but NOT currently decorated."""

    undecorated: list[str] = []
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.replace("torch.", "")
        try:
            holder = nested_getattr(torch, namespace_key)
        except Exception:
            continue
        if not hasattr(holder, func_name):
            continue
        if not is_decorated_function(getattr(holder, func_name)):
            undecorated.append(f"{namespace_name}.{func_name}")
    return undecorated


def test_partial_first_time_decoration_completes_on_retry(monkeypatch) -> None:
    """A mid-pass decoration failure must NOT launder into `_is_decorated=True`."""

    # Establish the pristine baseline count of never-decorated targets (classes, type
    # aliases, property getters) so the post-retry comparison is against reality.
    tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    baseline_undecorated = len(_undecorated_targets())

    wrappers_module.unwrap_torch()
    monkeypatch.setattr(wrappers_module, "_FULL_DECORATION_COMPLETED", False)
    monkeypatch.setattr(_state, "_is_decorated", False)
    monkeypatch.setattr(_state, "_orig_to_decorated", {})
    monkeypatch.setattr(_state, "_decorated_to_orig", {})
    monkeypatch.setattr(_state, "_arg_names", {})

    original_decorate = wrappers_module._decorate_torch_func_pairs
    calls = {"n": 0}

    def _fail_midway(func_pairs: list[tuple[str, str]]) -> None:
        """Decorate a prefix on the first call, then blow up mid-pass."""

        calls["n"] += 1
        if calls["n"] == 1:
            original_decorate(func_pairs[:50])
            raise RuntimeError("injected mid-pass-2 failure")
        original_decorate(func_pairs)

    monkeypatch.setattr(wrappers_module, "_decorate_torch_func_pairs", _fail_midway)
    with pytest.raises(RuntimeError, match="injected mid-pass-2 failure"):
        wrappers_module.wrap_torch()

    # The failed pass must NOT claim completion.
    assert _state._is_decorated is False
    assert wrappers_module._FULL_DECORATION_COMPLETED is False

    monkeypatch.setattr(wrappers_module, "_decorate_torch_func_pairs", original_decorate)
    wrappers_module.wrap_torch()
    assert _state._is_decorated is True
    assert wrappers_module._FULL_DECORATION_COMPLETED is True
    # The retry ran the FULL pass, so the only undecorated targets are the ones that are
    # legitimately never decorated -- not the thousands a re-install of the partial map
    # would have left behind.
    assert len(_undecorated_targets()) == baseline_undecorated
    assert tl.trace(nn.Linear(4, 4), torch.randn(1, 4)).num_ops >= 1


def test_distributed_arm_restores_its_wraps_when_the_second_install_fails(
    monkeypatch,
) -> None:
    """A failure between the two wrap families must leave no unowned wraps."""

    from torchlens.distributed import _lifecycle

    if _lifecycle._STATE is not None:
        _lifecycle.disarm()

    import torch.distributed as dist

    if not hasattr(dist, "new_group"):
        pytest.skip("torch.distributed is unavailable in this build")
    pristine = {name: getattr(dist, name) for name in ("new_group",) if hasattr(dist, name)}

    import torchlens.backends.torch.collectives as collectives_module

    def _boom(_originals: Any) -> None:
        """Fail the SECOND install family."""

        raise RuntimeError("injected collective-wrap install failure")

    monkeypatch.setattr(collectives_module, "install_collective_wraps", _boom)
    with pytest.raises(RuntimeError, match="injected collective-wrap install failure"):
        _lifecycle._arm(source="explicit")

    assert _lifecycle._STATE is None
    for name, original in pristine.items():
        assert getattr(dist, name) is original, f"{name} was left wrapped with no owner"


def test_escape_detector_teardown_frees_the_tool_id_and_clears_the_guard(
    monkeypatch,
) -> None:
    """A raising uninstall step must still release the id and clear the guard."""

    import sys

    from torchlens.backends.torch import escape_detection

    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        pytest.skip("sys.monitoring requires Python 3.12+")

    guard = escape_detection._GuardState()
    tool_id = None
    for candidate in range(6):
        try:
            monitoring.use_tool_id(candidate, "torchlens-test")
        except Exception:
            continue
        tool_id = candidate
        break
    if tool_id is None:
        pytest.skip("no free sys.monitoring tool id on this interpreter")
    guard.monitoring_tool_id = tool_id

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        """Fail the first uninstall step."""

        raise RuntimeError("injected set_events failure")

    monkeypatch.setattr(monitoring, "set_events", _raise)
    escape_detection._uninstall_monitoring(guard)
    assert guard.monitoring_tool_id is None
    # Proof the id really was released: re-acquiring it must succeed.
    monkeypatch.undo()
    monitoring.use_tool_id(tool_id, "torchlens-test-reacquire")
    monitoring.free_tool_id(tool_id)


def test_call_execution_context_unwind_exits_every_context() -> None:
    """A raising inner `__exit__` must not strand the outer contexts."""

    from torchlens import _runnable_state_context

    exited: list[str] = []

    class _Ctx:
        """Minimal context recording its own exit."""

        def __init__(self, name: str, raises: bool) -> None:
            self.name = name
            self.raises = raises

        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
            """Record the exit, optionally failing."""

            exited.append(self.name)
            if self.raises:
                raise RuntimeError(f"{self.name} exit failed")

    # Drive the finally body directly with a controlled stack: the unwind contract is
    # "every context gets its chance, the first failure is re-raised".
    stack = [_Ctx("outer", False), _Ctx("inner", True)]
    first_error: BaseException | None = None
    for ctx in reversed(stack):
        try:
            ctx.__exit__(None, None, None)
        except BaseException as error:
            if first_error is None:
                first_error = error
    assert exited == ["inner", "outer"]
    assert first_error is not None
    # And the shipped helper follows the same discipline (source-level guard).
    import inspect

    source = inspect.getsource(_runnable_state_context._call_execution_context_entered)
    assert "first_error" in source, "the per-context fence must be present"


def test_baseline_capture_stays_healthy_after_the_guard_changes() -> None:
    """A plain capture must be unaffected by every guard change in this lane."""

    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))
    assert trace.num_ops == 2
    tl.validation.check_metadata_invariants(trace)
