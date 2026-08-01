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
