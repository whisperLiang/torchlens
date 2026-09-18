"""Native replay requires its declared session capture sidecars."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.split import graph


@pytest.mark.parametrize(
    ("helper", "field", "empty"),
    [
        (graph._attach_paddle_capture_templates, "_paddle_op_captures", ()),
        (graph._attach_jax_captures, "_jax_capture_index_to_raw_op_label", {}),
        (graph._attach_tf_captures, "_tf_op_captures", ()),
    ],
)
def test_missing_capture_sidecar_never_becomes_empty_replay(
    helper: Callable[..., Any], field: str, empty: Any
) -> None:
    """An absent declared field fails; an explicitly empty capture stays valid."""

    with pytest.raises(AttributeError, match=field):
        helper(SimpleNamespace(), [])
    assert helper(SimpleNamespace(**{field: empty}), []) == []
