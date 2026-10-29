"""Split capture restores lazy hook registries with their native identity and type."""

from collections import OrderedDict

import torch
from torch import nn

from torchlens.split import prepare_graph


def test_lazy_hook_installation_does_not_corrupt_batch_probe_or_user_handles() -> None:
    """A native hook can be installed again after canonical capture and the probe."""

    class LazyHooks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.installed = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.installed:
                self.linear.register_forward_hook(lambda module, args, output: None)
                self.installed = True
            return self.linear(x)

    model = LazyHooks()
    user_handle = model.linear.register_forward_hook(lambda module, args, output: None)
    registry = model.linear._forward_hooks
    graph = prepare_graph(model, torch.ones(2, 4))
    assert graph.batch_validation["status"] == "passed"
    assert model.linear._forward_hooks is registry
    assert isinstance(registry, OrderedDict)
    user_handle.remove()
    assert not registry
    new_handle = model.linear.register_forward_hook(lambda module, args, output: None)
    new_handle.remove()
    assert not registry
