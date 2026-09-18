"""Shared Python state transactions preserve identities, bindings, and failures."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.split._callable_state import callable_capture_state
from torchlens.split._state_snapshot import PythonStateSnapshot


@pytest.mark.parametrize("fails", (False, True))
def test_callable_state_restores_cycles_and_empty_cells(fails: bool) -> None:
    """Probe failures restore an aliased cyclic tree and a previously empty cell."""

    empty = None
    cache: list[object] = []
    cache.append(cache)
    mapping = {"alias": cache}
    members = {1, 2}
    holder = SimpleNamespace(cache=cache, original=3)

    def model() -> None:
        """Mutate every supported Python-state category during a probe."""

        nonlocal empty
        empty = 5
        cache.append(4)
        mapping.clear()
        members.add(3)
        holder.original = 9
        holder.added = True
        if fails:
            raise RuntimeError("probe failed")

    del empty
    adapter = SimpleNamespace(is_tensor=lambda value: False)
    if fails:
        with (
            pytest.raises(RuntimeError, match="probe failed"),
            callable_capture_state(model, adapter),
        ):
            model()
    else:
        with callable_capture_state(model, adapter):
            model()
    assert len(cache) == 1 and cache[0] is cache
    assert mapping == {"alias": cache}
    assert members == {1, 2}
    assert holder.cache is cache and holder.original == 3
    assert not hasattr(holder, "added")
    assert model.__closure__ is not None
    empty_cell = model.__closure__[model.__code__.co_freevars.index("empty")]
    with pytest.raises(ValueError, match="Cell is empty"):
        _ = empty_cell.cell_contents


@pytest.mark.parametrize("preserve_tl", (False, True))
def test_python_snapshot_preserves_only_requested_live_metadata(preserve_tl: bool) -> None:
    """Tinygrad keeps live capture metadata; ordinary backends restore all attributes."""

    original = object()
    live = object()
    holder = SimpleNamespace(_tl=original, value=1)
    state = PythonStateSnapshot(lambda value, visit: False, preserve_tl=preserve_tl)
    state.visit(holder)
    holder._tl = live
    holder.value = 2
    holder.extra = 3
    state.restore()
    assert holder._tl is (live if preserve_tl else original)
    assert holder.value == 1
    assert not hasattr(holder, "extra")


def test_tinygrad_snapshot_does_not_wait_for_cyclic_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed probe immediately releases its snapshot, even with cyclic GC off."""

    pytest.importorskip("tinygrad")
    from torchlens.split import _tinygrad_state

    snapshots: list[weakref.ReferenceType[PythonStateSnapshot]] = []

    def observe_snapshot(
        visitor: Callable[[Any, Callable[[Any], None]], bool], *, preserve_tl: bool
    ) -> PythonStateSnapshot:
        """Observe lifetime without owning the snapshot or its tensor buffers."""

        snapshot = PythonStateSnapshot(visitor, preserve_tl=preserve_tl)
        snapshots.append(weakref.ref(snapshot))
        return snapshot

    monkeypatch.setattr(_tinygrad_state, "PythonStateSnapshot", observe_snapshot)
    enabled = gc.isenabled()
    gc.disable()
    try:
        with _tinygrad_state.tinygrad_capture_state(SimpleNamespace(value=1)):
            assert len(snapshots) == 1 and snapshots[0]() is not None
        assert snapshots[0]() is None
    finally:
        if enabled:
            gc.enable()
