"""Regression coverage for batch-4 Lane P cache, I/O, and perf fixes."""

from __future__ import annotations

import gc
import warnings
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import user_funcs
from torchlens._capture_state_helpers import (
    _facet_recipe_cache_key,
    _snapshot_plain_attr_value,
)
from torchlens._io import TorchLensIOError
from torchlens._io.rehydrate import _rehydrate_object
from torchlens._io.scrub import _scrub_value, _ScrubOptions
from torchlens._io.state_keys import (
    _STATIC_ATTR_MEMO,
    invalidate_static_class_attr_cache,
    static_class_attr,
)
from torchlens._trace_selector_helpers import _predicate_cache_key
from torchlens.utils.display import cleanup_trace_visualizer_dir, ensure_trace_visualizer_dir
from torchlens.visualization.auto_collapse import analyze_collapse


class _NonPersistentBufferModel(nn.Module):
    """Tiny model whose output depends on non-persistent state."""

    def __init__(self) -> None:
        """Initialize one non-persistent buffer."""

        super().__init__()
        self.register_buffer("offset", torch.ones(1), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add the current non-persistent buffer value."""

        return value + self.offset


def _deep_list(depth: int) -> list[Any]:
    """Build a one-child list chain of ``depth`` levels."""

    root: list[Any] = []
    cursor = root
    for _ in range(depth):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    return root


def test_capture_cache_separates_training_and_nonpersistent_state(tmp_path: Path) -> None:
    """Cache keys include module mode flags and non-persistent buffers."""

    x = torch.ones(1)
    model = _NonPersistentBufferModel().eval()
    first = tl.trace(model, x, cache=True, cache_dir=tmp_path)
    assert first.capture_cache_hit is False
    assert tl.trace(model, x, cache=True, cache_dir=tmp_path).capture_cache_hit is True

    model.train()
    training = tl.trace(model, x, cache=True, cache_dir=tmp_path)
    assert training.capture_cache_hit is False

    model.offset.fill_(3)
    changed_buffer = tl.trace(model, x, cache=True, cache_dir=tmp_path)
    assert changed_buffer.capture_cache_hit is False
    assert torch.equal(changed_buffer[changed_buffer.output_layers[0]].out, torch.tensor([4.0]))


def test_callable_cache_keys_include_code_and_are_stable() -> None:
    """Redefined nested functions separate by code while equivalent ones match."""

    def make_predicate(index: int) -> Any:
        """Build one closure-backed predicate."""

        return lambda ctx: ctx.raw_index == index

    first = make_predicate(1)
    same = make_predicate(1)
    changed = lambda ctx: ctx.raw_index != 1  # noqa: E731 - cache regression targets lambdas

    assert _predicate_cache_key(first) == _predicate_cache_key(same)
    assert _predicate_cache_key(first) != _predicate_cache_key(changed)
    assert _facet_recipe_cache_key([first]) != _facet_recipe_cache_key([changed])


def test_static_attr_cache_detects_replacement_and_does_not_pin_classes() -> None:
    """Definition fingerprints see same-size replacement and weak keys permit GC."""

    class Ephemeral:
        marker = 1

    assert static_class_attr(Ephemeral, "marker") == 1
    Ephemeral.marker = property(lambda self: 2)
    invalidate_static_class_attr_cache()
    assert isinstance(static_class_attr(Ephemeral, "marker"), property)

    class_ref = weakref.ref(Ephemeral)
    del Ephemeral
    gc.collect()
    assert class_ref() is None
    assert all(key is not class_ref() for key in _STATIC_ATTR_MEMO)


def test_capture_cache_lru_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture cache enforces its entry cap and exposes bounded clearing."""

    monkeypatch.setattr(user_funcs, "_CAPTURE_CACHE_MAX_ENTRIES", 2)
    x = torch.ones(1, 1)
    for value in (1.0, 2.0, 3.0):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(value)
        tl.trace(model, x, cache=True, cache_dir=tmp_path)

    cache_root = tmp_path / "capture"
    assert len(list(cache_root.glob("*.pkl"))) == 2
    assert user_funcs.clear_capture_cache(tmp_path) == 2
    assert list(cache_root.glob("*.pkl")) == []
    assert (cache_root / ".capture_cache_secret").is_file()


def test_scrub_preserves_shared_and_cyclic_mutable_containers() -> None:
    """Portable scrub memoizes ordinary containers before descending."""

    shared = ["value"]
    root: list[Any] = [shared, shared]
    cycle: list[Any] = []
    cycle.append(cycle)
    root.append(cycle)
    scrubbed = _scrub_value(
        root,
        _ScrubOptions(False, False, False, False),
        {},
        [],
        [0],
    )

    assert scrubbed[0] is scrubbed[1]
    assert scrubbed[2][0] is scrubbed[2]


def test_portable_walks_refuse_excessive_depth_typed() -> None:
    """Save and load walkers turn hostile depth into TorchLensIOError."""

    deep = _deep_list(205)
    with pytest.raises(TorchLensIOError, match="maximum depth"):
        _scrub_value(
            deep,
            _ScrubOptions(False, False, False, False),
            {},
            [],
            [0],
        )

    with pytest.raises(TorchLensIOError, match="maximum depth"):
        _rehydrate_object(
            deep,
            {},
            Path("."),
            Path("."),
            False,
            "cpu",
            True,
            None,
            False,
            [],
            {},
        )


def test_rehydrate_preserves_shared_and_cyclic_containers() -> None:
    """Portable rehydration memoizes rebuilt immutable and mutable containers."""

    shared = ("value",)
    cycle: list[Any] = []
    cycle.append(cycle)
    root: list[Any] = [shared, shared, cycle]
    rehydrated = _rehydrate_object(
        root,
        {},
        Path("."),
        Path("."),
        False,
        "cpu",
        True,
        None,
        False,
        [],
        {},
    )

    assert rehydrated[0] is rehydrated[1]
    assert rehydrated[2][0] is rehydrated[2]


def test_plain_attr_snapshot_refuses_cycles_and_excessive_depth() -> None:
    """Validation fallback snapshots bound recursive user-owned state."""

    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(RuntimeError, match="cyclic plain attribute"):
        _snapshot_plain_attr_value(cycle, "model.state")
    with pytest.raises(RuntimeError, match="container levels"):
        _snapshot_plain_attr_value(_deep_list(70), "model.state")


def test_visualizer_scratch_is_removed_by_trace_cleanup() -> None:
    """Explicit trace cleanup removes all trace-owned visualizer artifacts."""

    trace = tl.trace(nn.Identity(), torch.ones(1))
    output_dir = ensure_trace_visualizer_dir(trace)
    (output_dir / "probe.png").write_bytes(b"probe")
    trace.cleanup()

    assert not output_dir.exists()
    assert getattr(trace, "_visualizer_dir", None) is None


def test_log_model_metadata_does_not_self_deprecate() -> None:
    """Canonical metadata helper uses grouped options internally."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.io.log_model_metadata(nn.Identity(), torch.ones(1))
    assert not [warning for warning in caught if warning.category is DeprecationWarning]


def test_visualizer_cleanup_helper_is_idempotent() -> None:
    """Scratch cleanup remains safe after the directory is already gone."""

    class Owner:
        """Weak-referenceable scratch owner."""

    owner = Owner()
    output_dir = ensure_trace_visualizer_dir(owner)
    cleanup_trace_visualizer_dir(owner)
    cleanup_trace_visualizer_dir(owner)
    assert not output_dir.exists()


def test_collapse_analysis_cache_invalidates_after_equal_size_graph_edit() -> None:
    """Visualization caches fingerprint graph content rather than trace identity alone."""

    trace = tl.trace(nn.Sequential(nn.ReLU()), torch.ones(1))
    first = analyze_collapse(trace)
    relu = next(op for op in trace.ops if op.func_name == "relu")
    relu.func_name = "relu_cache_probe"

    second = analyze_collapse(trace)

    assert second is not first
    assert any(
        "relu_cache_probe" in signal.own_func_names for signal in second.signals.values()
    )
