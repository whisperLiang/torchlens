"""Regression tests for releasing persistent model preparation."""

from __future__ import annotations

import io
import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import get_module_meta
from torchlens.backends.torch.model_prep import is_forward_call_decorated


class _ReleaseModel(nn.Module):
    """Small nested model whose child receives a persistent forward wrapper."""

    def __init__(self) -> None:
        """Initialize a deterministic linear child."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the child and a functional activation."""
        return torch.relu(self.linear(inputs))


def _new_model() -> _ReleaseModel:
    """Return a deterministically initialized test model."""
    torch.manual_seed(17)
    return _ReleaseModel().eval()


@pytest.mark.smoke
def test_whole_model_pickle_and_torch_save_require_release() -> None:
    """Persistent child wrappers fail whole-model serialization until release."""
    model = _new_model()
    inputs = torch.randn(2, 3)
    tl.trace(model, inputs)

    expected = (
        "Can't pickle <function Linear.forward at ",
        "it's not the same object as torch.nn.modules.linear.Linear.forward",
    )
    for serializer in (
        lambda: pickle.dumps(model),
        lambda: torch.save(model, io.BytesIO()),
    ):
        with pytest.raises(pickle.PicklingError) as exc_info:
            serializer()
        message = str(exc_info.value)
        assert message.startswith(expected[0])
        assert message.endswith(expected[1])

    tl.release_model(model)

    assert pickle.loads(pickle.dumps(model))(inputs).shape == (2, 2)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    assert buffer.tell() > 0


@pytest.mark.smoke
def test_release_is_idempotent_and_never_traced_model_is_a_noop() -> None:
    """Repeated release and release before tracing leave forward behavior intact."""
    model = _new_model()
    inputs = torch.randn(2, 3)
    expected = model(inputs).detach().clone()

    tl.release_model(model)
    assert torch.equal(model(inputs), expected)

    tl.trace(model, inputs)
    assert is_forward_call_decorated(model.linear.forward)
    tl.release_model(model)
    tl.release_model(model)

    assert not is_forward_call_decorated(model.linear.forward)
    assert get_module_meta(model) is None
    assert get_module_meta(model.linear) is None
    assert torch.equal(model(inputs), expected)
    assert all(not name.startswith("tl_") for module in model.modules() for name in vars(module))


@pytest.mark.smoke
def test_retrace_after_release_matches_fresh_model() -> None:
    """A released model is fully re-prepared and captures like a fresh twin."""
    model = _new_model()
    fresh_model = _new_model()
    inputs = torch.randn(2, 3)

    tl.trace(model, inputs)
    tl.release_model(model)
    retraced = tl.trace(model, inputs)
    fresh = tl.trace(fresh_model, inputs)

    assert len(retraced.layer_list) == len(fresh.layer_list)
    assert [op.layer_label for op in retraced.layer_list] == [
        op.layer_label for op in fresh.layer_list
    ]


@pytest.mark.smoke
def test_releasing_one_model_preserves_an_independent_prepared_model() -> None:
    """Release does not disturb persistent preparation for another model tree."""
    first = _new_model()
    second = _new_model()
    inputs = torch.randn(2, 3)
    tl.trace(first, inputs)
    initial_second = tl.trace(second, inputs)

    tl.release_model(first)
    repeated_second = tl.trace(second, inputs)

    assert is_forward_call_decorated(second.linear.forward)
    assert [op.layer_label for op in repeated_second.layer_list] == [
        op.layer_label for op in initial_second.layer_list
    ]


class _HeldActivationModel(nn.Module):
    """Model whose plain attributes hold torch function references."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.act = torch.nn.functional.relu  # whichever epoch is live NOW
        self.extra_acts = [torch.sigmoid, torch.nn.functional.gelu]
        self.act_table = {"tanh": torch.tanh}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.lin(x))
        for act in self.extra_acts:
            y = act(y)
        return self.act_table["tanh"](y)


@pytest.mark.smoke
def test_release_normalizes_prewrap_function_attrs_for_torch_save() -> None:
    """grind-r4 b8 R56 direction 1: a model built BEFORE wrapping holds
    pristine originals; pickled WHILE wrapped, every held ref fails pickle's
    by-reference identity check. release_model must normalize them."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    try:
        model = _HeldActivationModel()  # holds pristine originals
    finally:
        wrap_torch()

    tl.trace(model, torch.randn(1, 4))  # wrappers now live at the public names
    tl.release_model(model)
    torch.save(model, io.BytesIO())
    pickle.dumps(model)
    # The normalized refs still compute the same functions.
    out = model(torch.randn(1, 4))
    assert out.shape == (1, 4)


@pytest.mark.smoke
def test_release_normalizes_wrapper_attrs_after_unwrap() -> None:
    """R56 direction 2: a model built WHILE wrapped holds epoch wrappers;
    after unwrap_torch() those refs fail pickle. release_model normalizes
    them back to the pristine originals."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    wrap_torch()
    model = _HeldActivationModel()  # holds wrap-epoch wrappers
    tl.trace(model, torch.randn(1, 4))
    unwrap_torch()
    try:
        tl.release_model(model)
        torch.save(model, io.BytesIO())
        pickle.dumps(model)
        assert model.act is torch.nn.functional.relu, (
            "the held wrapper was not normalized to the pristine original"
        )
    finally:
        wrap_torch()


@pytest.mark.smoke
def test_release_leaves_foreign_and_user_callables_alone() -> None:
    """The normalization is ledger-fenced: user callables never swap."""

    def user_act(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    model = _HeldActivationModel()
    model.custom = user_act
    tl.trace(model, torch.randn(1, 4))
    tl.release_model(model)
    assert model.custom is user_act
