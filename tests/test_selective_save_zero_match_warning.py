"""A save predicate matching ZERO sites must warn, never pass silently.

TF is fail-closed on selector reachability, but the shared non-torch
selective-save resolver accepted a predicate matching nothing with status
COMPLETE and no diagnostic: a typo'd label or function name silently produced
a trace with zero saved activations.  The resolver now warns at predicate-
resolution completion.  (The torch capture-path and paddle intervene-side
equivalents live outside this resolver and are tracked separately.)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import torchlens as tl
from torchlens.backends._selective_save import apply_static_label_save_policy

pytestmark = pytest.mark.smoke


def _stub_trace() -> Any:
    """Return a minimal trace-like object the resolver can filter."""

    ops = [
        SimpleNamespace(
            label="relu_1_1",
            _label_raw="relu_1_1",
            layer_label="relu_1_1",
            layer_label_short="relu_1",
            func_name="relu",
            out=object(),
            has_saved_activation=True,
            is_orphan=False,
            activation_memory=16,
        ),
        SimpleNamespace(
            label="add_1_2",
            _label_raw="add_1_2",
            layer_label="add_1_2",
            layer_label_short="add_1",
            func_name="add",
            out=object(),
            has_saved_activation=True,
            is_orphan=False,
            activation_memory=16,
        ),
    ]
    return SimpleNamespace(layer_list=ops, module_calls=())


def test_zero_match_save_predicate_warns() -> None:
    """A predicate matching no op must surface a warning, not silence."""

    trace = _stub_trace()
    with pytest.warns(UserWarning, match="matched zero"):
        apply_static_label_save_policy(trace, tl.label("no_such_label"), backend_name="mlx")
    assert trace.num_saved_ops == 0


def test_matching_save_predicate_does_not_warn() -> None:
    """A predicate matching at least one op stays silent."""

    import warnings

    trace = _stub_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_static_label_save_policy(trace, tl.label("relu_1_1"), backend_name="mlx")
    assert trace.num_saved_ops == 1


def test_none_predicate_does_not_warn() -> None:
    """The full-save default (no predicate) is untouched."""

    import warnings

    trace = _stub_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_static_label_save_policy(trace, None, backend_name="mlx")
    assert all(op.has_saved_activation for op in trace.layer_list)
