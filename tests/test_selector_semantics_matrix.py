"""Characterization matrix for the public selector/predicate surface.

This is the behavioral oracle for the ONE-predicate-interpreter consolidation:
every public selector spelling is evaluated through each user-reachable
lifecycle (capture-time ``save=``, post-hoc ``find_sites``, live hook matching,
and spec round-trip) against fixed models, and the resulting match sets are
snapshotted in ``tests/golden/selector_semantics_matrix.json``.

The golden encodes TODAY'S behavior, including known divergences between the
lifecycles (case sensitivity, label universes, error shapes). A diff against
this file is therefore a *behavior change*: each one must be either an
explicitly intended, enumerated change or a bug. Regenerate deliberately with::

    TL_SELECTOR_MATRIX_REGEN=1 pytest tests/test_selector_semantics_matrix.py

Cell values are either a sorted list of matched labels or ``"ERROR:<Class>"``.
"""

from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.selectors import BaseSelector, grad_fn_label

_GOLDEN_PATH = Path(__file__).parent / "golden" / "selector_semantics_matrix.json"
_REGEN = bool(os.environ.get("TL_SELECTOR_MATRIX_REGEN"))

pytestmark = pytest.mark.smoke


class TinyConvNet(nn.Module):
    """Nested-module CNN with relu/conv/add/flatten/linear ops."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Conv2d(2, 2, 3))
        self.head = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = x + 1
        x = x.flatten(1)
        return self.head(x)


class SplitNet(nn.Module):
    """Multi-output chunk op feeding mul/add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return a * 2 + b


class LoopNet(nn.Module):
    """One linear block called twice (recurrent module passes)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.block = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.block(x))
        x = torch.relu(self.block(x))
        return x


def _model_and_input(model_key: str) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    if model_key == "conv":
        return TinyConvNet(), torch.randn(1, 1, 8, 8)
    if model_key == "split":
        return SplitNet(), torch.randn(1, 4, 4, 4)
    if model_key == "loop":
        return LoopNet(), torch.randn(2, 4)
    raise KeyError(model_key)


@lru_cache(maxsize=None)
def _full_trace(model_key: str) -> Any:
    model, x = _model_and_input(model_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x)


@lru_cache(maxsize=None)
def _backward_trace(model_key: str) -> Any:
    model, x = _model_and_input(model_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(
            model,
            x.requires_grad_(True),
            capture=tl.options.CaptureOptions(save_grads="all", backward_ready=True),
        )
        log.log_backward(log[log.output_layers[0]].out.sum(), retain_graph=True)
    return log


def _error_cell(exc: BaseException) -> str:
    return f"ERROR:{type(exc).__name__}"


def _probe_capture(model_key: str, make_selector: Callable[[], Any]) -> Any:
    """Labels saved by ``tl.trace(model, x, save=selector)``."""

    model, x = _model_and_input(model_key)
    try:
        selector = make_selector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log = tl.trace(
                model,
                x,
                save=selector,
                lookback=4,
                lookback_payload_policy="detached_raw",
            )
    except Exception as exc:  # noqa: BLE001 - characterization records error shape
        return _error_cell(exc)
    return sorted(
        str(op.layer_label)
        for op in log.layer_list
        if getattr(op, "has_saved_activation", False)
    )


def _probe_sites(model_key: str, make_selector: Callable[[], Any], *, backward: bool) -> Any:
    """Labels returned by ``find_sites`` against a cached full trace."""

    try:
        selector = make_selector()
        log = _backward_trace(model_key) if backward else _full_trace(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = log.find_sites(selector, max_fanout=10**6)
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(str(label) for label in table.labels())


def _probe_live(model_key: str, make_selector: Callable[[], Any]) -> Any:
    """Raw labels of sites where a live hook attached to ``selector`` fires."""

    model, x = _model_and_input(model_key)
    fired: list[str] = []

    def _probe_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        fired.append(str(hook.layer_log.get("layer_label")))
        return out

    try:
        selector = make_selector()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.trace(model, x, hooks=[(selector, _probe_hook)])
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    return sorted(fired)


def _probe_spec(make_selector: Callable[[], Any]) -> Any:
    """Round-trip a selector through both spec deserializers."""

    from torchlens.intervention.selectors import _classify_selector_direction
    from torchlens.ir.selector_eval import selector_from_spec

    try:
        selector = make_selector()
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)
    if not isinstance(selector, BaseSelector):
        return f"NOT_A_SELECTOR:{type(selector).__name__}"
    cell: dict[str, Any] = {"kind": str(selector.selector_kind)}
    try:
        cell["direction"] = _classify_selector_direction(selector)
    except Exception as exc:  # noqa: BLE001
        cell["direction"] = _error_cell(exc)
    try:
        spec = selector.to_target_spec()
    except Exception as exc:  # noqa: BLE001
        cell["to_spec"] = _error_cell(exc)
        return cell
    cell["spec_kind"] = str(spec.selector_kind)
    try:
        rebuilt = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="site"
        )
        cell["resolver_repr"] = repr(rebuilt)
        cell["resolver_direction"] = _classify_selector_direction(rebuilt)
    except Exception as exc:  # noqa: BLE001
        cell["resolver_repr"] = _error_cell(exc)
    try:
        rebuilt_live = selector_from_spec(
            spec.selector_kind, spec.selector_value, spec.metadata, lifecycle="live"
        )
        cell["hooks_repr"] = repr(rebuilt_live)
        cell["hooks_direction"] = _classify_selector_direction(rebuilt_live)
    except Exception as exc:  # noqa: BLE001
        cell["hooks_repr"] = _error_cell(exc)
    return cell


def _where_relu() -> Any:
    return tl.where(lambda p: getattr(p, "layer_type", None) == "relu", name_hint="type_relu")


#: (cell_name, model_key, selector factory). Every entry gets capture/sites/live
#: cells on its model plus one spec round-trip cell.
FORWARD_CASES: tuple[tuple[str, str, Callable[[], Any]], ...] = (
    ("label_final", "conv", lambda: tl.label("relu_1_2")),
    ("label_raw", "conv", lambda: tl.label("relu_1_3_raw")),
    ("contains_lower", "conv", lambda: tl.contains("relu")),
    ("contains_upper", "conv", lambda: tl.contains("RELU")),
    ("contains_mixed", "conv", lambda: tl.contains("Conv2d")),
    ("regex_lower", "conv", lambda: tl.regex(r"relu_\d")),
    ("regex_upper", "conv", lambda: tl.regex(r"RELU")),
    ("regex_anchored", "conv", lambda: tl.regex(r"^conv2d_2")),
    ("func_relu", "conv", lambda: tl.func("relu")),
    ("func_add_type", "conv", lambda: tl.func("add")),
    ("func_add_dunder", "conv", lambda: tl.func("__add__")),
    ("func_transform_any", "conv", lambda: tl.func_transform()),
    ("module_container", "conv", lambda: tl.module("features")),
    ("module_address", "conv", lambda: tl.module("features.0")),
    ("in_module_container", "conv", lambda: tl.in_module("features")),
    ("in_module_pass", "conv", lambda: tl.in_module("features:1")),
    ("output_index", "conv", lambda: tl.output(0)),
    ("output_at_0", "conv", lambda: tl.output_at((0,))),
    ("input_at_0", "conv", lambda: tl.input_at(0)),
    ("where_type_relu", "conv", _where_relu),
    ("and_func_inmodule", "conv", lambda: tl.func("relu") & tl.in_module("features")),
    ("or_funcs", "conv", lambda: tl.func("relu") | tl.func("conv2d")),
    ("not_func", "conv", lambda: ~tl.func("relu")),
    (
        "and_nested_three",
        "conv",
        lambda: (tl.func("relu") & tl.in_module("features")) & tl.contains("relu"),
    ),
    (
        "followed_by_combo",
        "conv",
        lambda: tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    ),
    ("followed_by_bare", "conv", lambda: tl.followed_by(tl.func("relu"))),
    ("preceded_by_conv", "conv", lambda: tl.preceded_by(tl.func("conv2d"))),
    ("facet_named", "conv", lambda: tl.facet("q")),
    ("head_indexed", "conv", lambda: tl.head(0)),
    ("grad_fn_on_forward", "conv", lambda: tl.grad_fn("ReluBackward0")),
    ("split_output_0", "split", lambda: tl.output(0)),
    ("split_output_1", "split", lambda: tl.output(1)),
    ("split_func_chunk_out1", "split", lambda: tl.func("chunk", output=1)),
    ("split_contains_mul", "split", lambda: tl.contains("mul")),
    ("loop_in_module_block", "loop", lambda: tl.in_module("block")),
    ("loop_in_module_pass2", "loop", lambda: tl.in_module("block:2")),
    ("loop_module_block", "loop", lambda: tl.module("block")),
    ("loop_label_recurrent", "loop", lambda: tl.label("linear_1_1")),
)

#: Backward selectors probed with find_sites against the armed conv trace.
BACKWARD_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("grad_fn_class", lambda: tl.grad_fn("ReluBackward0")),
    ("grad_fn_type", lambda: tl.grad_fn("relu")),
    ("grad_fn_label_pattern", lambda: tl.grad_fn(label="relu_back")),
    ("grad_fn_not_custom", lambda: tl.grad_fn(is_custom=False)),
    ("grad_fn_label_exact", lambda: grad_fn_label("relu_back_1_9")),
    ("without_op", lambda: tl.without_op()),
    ("grad_input", lambda: tl.grad_input()),
    ("grad_output", lambda: tl.grad_output()),
    ("backward_pass_1", lambda: tl.in_backward_pass(1)),
    ("backward_pass_2", lambda: tl.in_backward_pass(2)),
    (
        "bwd_and_direction_agnostic",
        lambda: tl.grad_fn("ReluBackward0") & tl.contains("relu"),
    ),
    ("bwd_plain_label", lambda: tl.label("relu_back_1_9")),
    ("bwd_not_accumulate", lambda: ~tl.grad_fn("AccumulateGrad")),
)

#: Selector compositions expected to be decided at construction time.
CONSTRUCT_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("cross_direction_and", lambda: tl.func("relu") & tl.grad_fn("ReluBackward0")),
    ("or_with_followed_by", lambda: tl.func("relu") | tl.followed_by(tl.func("relu"))),
    ("not_followed_by", lambda: ~tl.followed_by(tl.func("relu"))),
    ("double_followed_by", lambda: tl.followed_by(tl.func("a")) & tl.followed_by(tl.func("b"))),
    ("func_non_string", lambda: tl.func(torch.relu)),  # type: ignore[arg-type]
)


def _construct_cell(make_selector: Callable[[], Any]) -> Any:
    try:
        return repr(make_selector())
    except Exception as exc:  # noqa: BLE001
        return _error_cell(exc)


def _compute_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for name, model_key, factory in FORWARD_CASES:
        matrix[f"capture/{model_key}/{name}"] = _probe_capture(model_key, factory)
        matrix[f"sites/{model_key}/{name}"] = _probe_sites(model_key, factory, backward=False)
        matrix[f"live/{model_key}/{name}"] = _probe_live(model_key, factory)
        matrix[f"spec/{name}"] = _probe_spec(factory)
    for name, factory in BACKWARD_CASES:
        matrix[f"sites_bwd/conv/{name}"] = _probe_sites("conv", factory, backward=True)
        matrix[f"spec_bwd/{name}"] = _probe_spec(factory)
    for name, factory in CONSTRUCT_CASES:
        matrix[f"construct/{name}"] = _construct_cell(factory)
    return matrix


@lru_cache(maxsize=1)
def _matrix() -> dict[str, Any]:
    return _compute_matrix()


@lru_cache(maxsize=1)
def _golden() -> dict[str, Any]:
    if _REGEN:
        matrix = _matrix()
        _GOLDEN_PATH.write_text(json.dumps(matrix, indent=1, sort_keys=True) + "\n")
        return matrix
    if not _GOLDEN_PATH.exists():
        pytest.fail(
            f"Missing golden {_GOLDEN_PATH}; regenerate with TL_SELECTOR_MATRIX_REGEN=1."
        )
    return json.loads(_GOLDEN_PATH.read_text())


_CELL_KEYS: tuple[str, ...] = tuple(
    [f"{lifecycle}/{model_key}/{name}" for name, model_key, _ in FORWARD_CASES
     for lifecycle in ("capture", "sites", "live")]
    + [f"spec/{name}" for name, _, _ in FORWARD_CASES]
    + [f"sites_bwd/conv/{name}" for name, _ in BACKWARD_CASES]
    + [f"spec_bwd/{name}" for name, _ in BACKWARD_CASES]
    + [f"construct/{name}" for name, _ in CONSTRUCT_CASES]
)


def test_matrix_covers_golden_exactly() -> None:
    """The computed cell-key set and the golden's key set must be identical."""

    assert set(_matrix()) == set(_golden())


@pytest.mark.parametrize("cell_key", _CELL_KEYS)
def test_selector_semantics_cell(cell_key: str) -> None:
    """One selector x lifecycle cell matches its committed characterization."""

    golden = _golden()
    matrix = _matrix()
    assert cell_key in golden, f"cell {cell_key} missing from golden; regenerate deliberately"
    assert matrix[cell_key] == golden[cell_key], (
        f"Behavior change in {cell_key}: golden={golden[cell_key]!r} "
        f"current={matrix[cell_key]!r}. If intended, enumerate it in the "
        "consolidation report and regenerate with TL_SELECTOR_MATRIX_REGEN=1."
    )
