"""Factory device arguments follow explicit segment placement."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from v2_helpers import split_request

import torchlens as tl
from torchlens.split import PlacementPlan
from torchlens.split.adapters.torch import (
    _GeneratedSegmentBase,
    _rewrite_placement_device_args,
)
from torchlens.split.placement import DevicePlacement


def _call_node(namespace: str, qualname: str) -> Any:
    """Build only the call metadata consumed by device argument rewriting."""

    return SimpleNamespace(
        args_template=SimpleNamespace(
            func_id=SimpleNamespace(namespace=namespace, qualname=qualname)
        )
    )


@pytest.mark.parametrize("captured_device", ["cpu", torch.device("cpu"), 0, None])
def test_only_device_keyword_is_relocated(captured_device: Any) -> None:
    """Shape, fill and other string/integer literals are not device destinations."""

    args = ((4,), 4)
    kwargs = {"device": captured_device, "label": "cpu", "count": 0}
    actual_args, actual_kwargs = _rewrite_placement_device_args(
        _call_node("torch", "full"), args, kwargs, DevicePlacement("cuda:1")
    )
    assert actual_args == args
    assert actual_kwargs == {"device": torch.device("cuda:1"), "label": "cpu", "count": 0}
    assert kwargs["device"] == captured_device


def test_positional_tensor_to_uses_its_device_overload() -> None:
    """Tensor.to relocation distinguishes device arguments from dtype overloads."""

    node = _call_node("torch.Tensor", "to")
    value = torch.ones(4)
    for destination in ("cpu", torch.device("cpu"), 0):
        args, kwargs = _rewrite_placement_device_args(
            node, (value, destination, torch.float64), {}, DevicePlacement("cuda:0")
        )
        assert args[0] is value
        assert args[1:] == (torch.device("cuda:0"), torch.float64)
        assert kwargs == {}
    args = (value, torch.float64)
    assert _rewrite_placement_device_args(node, args, {}, DevicePlacement("cuda:0"))[0] is args


def test_unplaced_and_non_torch_calls_preserve_literal_arguments() -> None:
    """No device rewrite occurs without placement or outside known Torch calls."""

    args = (4, "cpu")
    kwargs = {"device": "cpu"}
    for node, placement in (
        (_call_node("torch", "ones"), DevicePlacement()),
        (_call_node("custom", "ones"), DevicePlacement("cuda:0")),
    ):
        actual_args, actual_kwargs = _rewrite_placement_device_args(node, args, kwargs, placement)
        assert actual_args is args
        assert actual_kwargs is kwargs


def test_implicit_factory_placement_is_scoped() -> None:
    """Factory allocation follows the segment and restores the ambient default device."""

    segment = object.__new__(_GeneratedSegmentBase)
    segment._state = SimpleNamespace(placement=DevicePlacement("cpu"))
    node = SimpleNamespace(target=torch.ones, op=SimpleNamespace())
    with torch.device("meta"):
        actual = segment._execute_func(node, (4,), {})
        assert actual.device.type == "cpu"
        assert torch.ones(4).device.type == "meta"


@pytest.mark.parametrize("kind", ["ones", "zeros", "arange"])
@pytest.mark.parametrize("explicit_device", [False, True])
def test_cpu_captured_factory_runs_on_cuda_suffix(kind: str, explicit_device: bool) -> None:
    """A CPU-captured factory must not feed a CPU tensor to a CUDA suffix op."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for heterogeneous factory placement.")

    class Model(torch.nn.Module):
        """Create a suffix constant after a trainable prefix."""

        def __init__(self) -> None:
            """Keep all original model state on CPU."""

            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use either a captured or an implicit factory device."""

            x = torch.relu(self.fc(x))
            kwargs = {"dtype": x.dtype}
            if explicit_device:
                kwargs["device"] = x.device
            constant = getattr(torch, kind)(4, **kwargs)
            return x + constant

    model = Model().eval()
    x = torch.randn(3, 4)
    expected = model(x)
    runtime = tl.split.prepare(
        model,
        x,
        split_request("after:relu", placement=PlacementPlan.across("cpu", "cuda:0")),
    )
    assert runtime.batch_validation["status"] == "passed"
    output = runtime.replay(x)
    assert output.device == torch.device("cuda:0")
    torch.testing.assert_close(output.cpu(), expected)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
