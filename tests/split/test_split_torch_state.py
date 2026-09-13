"""Torch split state binding regressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchlens.split.adapters.torch import TorchSplitAdapter, _GeneratedSegmentBase
from torchlens.split.placement import DevicePlacement
from torchlens.split.state import SegmentState


def test_missing_param_module_does_not_drop_live_parameter_handle() -> None:
    """A stale module-log address skips buffer lookup but keeps its parameter."""

    parameter = torch.nn.Parameter(torch.ones(2))

    class MissingModuleParam:
        handle = parameter

        @property
        def module(self) -> object:
            """Raise the same lookup failure as a stale Trace.modules entry."""

            raise KeyError("core.refpoint_embed")

    segment = object.__new__(_GeneratedSegmentBase)
    segment.use_live_param_sources = True
    segment._state = SegmentState(
        adapter=TorchSplitAdapter(),
        placement=DevicePlacement(),
    )
    node: Any = SimpleNamespace(param_refs=(MissingModuleParam(),), label="refpoint")

    handles = segment._param_handles_for_node(node)

    assert handles == [parameter]


def test_capture_state_restores_registered_state_after_reload_failure() -> None:
    """A failed state reload must not skip the remaining model cleanup."""

    from torchlens.split.pipeline import _torch_capture_state

    class MutatingModule(torch.nn.Module):
        """Add registered state and mutate plain state during an internal run."""

        def __init__(self) -> None:
            super().__init__()
            self.marker = "before"

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Create an unexpected buffer and change mode-visible state."""

            if not hasattr(self, "temporary_buffer"):
                self.register_buffer("temporary_buffer", torch.ones(1))
            self.marker = "during"
            return value

        def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
            """Simulate a restoration failure after the forward has mutated state."""

            del state_dict, args, kwargs
            raise RuntimeError("reload failed")

    model = MutatingModule().eval()
    with pytest.raises(RuntimeError, match="reload failed"):
        with _torch_capture_state(model):
            model(torch.ones(1))
            model.train()

    assert not hasattr(model, "temporary_buffer")
    assert model.marker == "before"
    assert model.training is False
