"""Public named-sibling receptive-field extension tests."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import ReceptiveFieldDirection, ReceptiveFieldView, _rules

_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    global _PACK
    original = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _PACK is None:
        module = importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
        _PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_PACK)
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original)
        _rules._RF_RULES_EPOCH = original_epoch


def _trace() -> tuple[object, object]:
    """Capture a one-call convolution trace and its convolution operation."""

    trace = tl.trace(nn.Sequential(nn.Conv2d(1, 1, 3, padding=1)), torch.ones(1, 1, 5, 5))
    op = next(item for item in trace.layer_list if item.func_name == "conv2d")
    return trace, op


def test_named_projective_siblings_and_direction_keyword() -> None:
    """Keep projective ownership source-anchored across all entity siblings."""

    trace, op = _trace()
    source = next(item for item in trace.layer_list if item.is_input)
    view = source.projective_field

    assert isinstance(view, ReceptiveFieldView)
    assert view._direction is ReceptiveFieldDirection.PROJECTIVE
    assert (
        op.receptive_field.at((2, 2), direction="receptive").direction
        is ReceptiveFieldDirection.RECEPTIVE
    )
    assert view.at((2, 2)).direction is ReceptiveFieldDirection.PROJECTIVE
    assert view.at((2, 2), target=trace.output_ops[0]).unit_shape == tuple(source.shape)

    layer = trace[source.layer_label]
    assert layer.projective_field is view
    call = next(iter(trace.module_calls.values()))
    if len(call.output_ops) == 1:
        assert isinstance(call.projective_field, ReceptiveFieldView)
    module = next(iter(trace.modules.values()))
    if module.num_calls == 1 and len(module.calls) == 1:
        assert isinstance(module.projective_field, ReceptiveFieldView)


def test_projective_table_keeps_receptive_default_schema() -> None:
    """Append projective columns without changing the receptive table schema."""

    trace, _ = _trace()
    receptive = trace.receptive_fields().to_pandas()
    projective = trace.projective_fields().to_pandas()

    assert list(receptive.columns) == list(projective.columns[: len(receptive.columns)])
    assert "projective_target" in projective.columns
    assert projective.attrs["direction"] is ReceptiveFieldDirection.PROJECTIVE


def test_configuration_errors_are_typed() -> None:
    """SF-07: user-facing misconfiguration raises the typed RF stratum.

    ``ReceptiveFieldConfigurationError`` subclasses both ``ReceptiveFieldError``
    and ``ValueError``, so callers branching on the historical raw errors keep
    working while typed handling becomes possible.
    """

    from torchlens.receptive_field import (
        ReceptiveFieldConfigurationError,
        ReceptiveFieldError,
        verify,
    )
    from torchlens.receptive_field._rules import register_rf_rule

    assert issubclass(ReceptiveFieldConfigurationError, ReceptiveFieldError)
    assert issubclass(ReceptiveFieldConfigurationError, ValueError)

    trace, _ = _trace()

    with pytest.raises(ReceptiveFieldConfigurationError, match="must be non-negative"):
        verify(trace, empirical_adjoint_atol=-1.0)
    with pytest.raises(ReceptiveFieldConfigurationError, match="level must be"):
        trace.receptive_fields(level="bogus")
    with pytest.raises(ReceptiveFieldConfigurationError, match="at least one function name"):
        register_rf_rule()
    from torchlens.receptive_field._engine_forward import solve_projective

    with pytest.raises(ReceptiveFieldConfigurationError, match="at least one target operation"):
        solve_projective(trace, ())
