"""Multi-epoch training parity across eager, TorchLens trace, and split training."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens.options import CaptureOptions, VisualizationOptions


@dataclass(frozen=True)
class _TrainingCase:
    """Configuration for one deterministic training-trajectory parity case."""

    model: nn.Module
    inputs: torch.Tensor
    targets: torch.Tensor
    split_boundary: str
    suffix_param_prefixes: tuple[str, ...]
    epochs: int
    lr: float
    atol: float = 1e-6
    rtol: float = 1e-5


class _TinyTransformerClassifier(nn.Module):
    """Small TransformerEncoder classifier for attention/LayerNorm training parity."""

    def __init__(self) -> None:
        """Initialize the projection, encoder, and head."""

        super().__init__()
        self.proj = nn.Linear(8, 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a Transformer sequence classifier."""

        hidden = self.encoder(self.proj(x))
        pooled = hidden.mean(dim=1)
        return self.head(pooled)


class _ParamSourceSuffixClassifier(nn.Module):
    """Classifier whose suffix derives trainable tensors through no-input ops."""

    def __init__(self) -> None:
        """Initialize prefix, parameter-only suffix source, and head."""

        super().__init__()
        self.prefix = nn.Linear(4, 4)
        self.query = nn.Parameter(torch.randn(4, 10))
        self.head = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a classifier with a parameter-derived suffix tensor."""

        hidden = self.prefix(x)
        query, _unused = self.query.chunk(2, dim=1)
        query = query.transpose(0, 1)[:, :4].transpose(0, 1)
        return self.head(hidden @ query)


class _BufferSourceSuffix(nn.Module):
    """Module whose suffix reads a registered buffer through a no-input source."""

    def __init__(self) -> None:
        """Initialize prefix and registered suffix buffer."""

        super().__init__()
        self.prefix = nn.Linear(4, 4)
        self.register_buffer("scale", torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale prefix features by a live buffer view."""

        hidden = self.prefix(x)
        return hidden * self.scale.view(1, -1)


def _make_resnet18_case() -> _TrainingCase:
    """Return a real torchvision ResNet18 deterministic training-parity case."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(123)
    model = torchvision.models.resnet18(weights=None, num_classes=5).eval()
    inputs = torch.randn(2, 3, 32, 32)
    split_boundary = _boundary_after_last_layer_type(model, inputs, "flatten")
    return _TrainingCase(
        model=model,
        inputs=inputs,
        targets=torch.tensor([1, 3]),
        split_boundary=split_boundary,
        suffix_param_prefixes=("fc.",),
        epochs=2,
        lr=0.01,
        atol=3e-6,
    )


def _make_transformer_encoder_case() -> _TrainingCase:
    """Return a standard TransformerEncoder deterministic training-parity case."""

    torch.manual_seed(456)
    model = _TinyTransformerClassifier().eval()
    inputs = torch.randn(2, 5, 8)
    split_boundary = _boundary_after_last_layer_type(model, inputs, "mean")
    return _TrainingCase(
        model=model,
        inputs=inputs,
        targets=torch.tensor([1, 3]),
        split_boundary=split_boundary,
        suffix_param_prefixes=("head.",),
        epochs=3,
        lr=0.03,
    )


def _make_param_source_suffix_case() -> _TrainingCase:
    """Return a case where split suffix must re-read live parameter sources."""

    torch.manual_seed(789)
    model = _ParamSourceSuffixClassifier().eval()
    inputs = torch.randn(2, 4)
    split_boundary = _boundary_after_first_layer_with_param(model, inputs, "prefix.weight")
    return _TrainingCase(
        model=model,
        inputs=inputs,
        targets=torch.tensor([1, 2]),
        split_boundary=split_boundary,
        suffix_param_prefixes=("query", "head."),
        epochs=3,
        lr=0.02,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "case_factory",
    (_make_resnet18_case, _make_transformer_encoder_case, _make_param_source_suffix_case),
    ids=("torchvision-resnet18", "transformer-encoder", "param-source-suffix"),
)
def test_multi_epoch_training_matches_trace_and_split(
    case_factory: Callable[[], _TrainingCase],
) -> None:
    """Original, trace-backed, and split training follow the same trajectory."""

    case = case_factory()
    eager_model = copy.deepcopy(case.model)
    trace_model = copy.deepcopy(case.model)
    split_model = copy.deepcopy(case.model)

    eager_losses = _train_eager(case, eager_model)
    trace_losses = _train_trace(case, trace_model)
    split_losses = _train_split(case, split_model)

    torch.testing.assert_close(
        torch.stack(trace_losses),
        torch.stack(eager_losses),
        atol=case.atol,
        rtol=case.rtol,
    )
    torch.testing.assert_close(
        torch.stack(split_losses),
        torch.stack(eager_losses),
        atol=case.atol,
        rtol=case.rtol,
    )
    _assert_state_dict_close(trace_model, eager_model, atol=case.atol, rtol=case.rtol)
    _assert_state_dict_close(split_model, eager_model, atol=case.atol, rtol=case.rtol)


def test_trainable_suffix_replays_live_param_derived_sources() -> None:
    """Trainable suffix replay recomputes no-input nodes derived from live params."""

    torch.manual_seed(1234)
    model = _ParamSourceSuffixClassifier().eval()
    x = torch.randn(2, 4)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:prefix", dynamic_batch=(1, 4), trainable=True),
    )
    legacy_runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(
            boundary="after:prefix",
            dynamic_batch=(1, 4),
            trainable=True,
            use_live_param_sources=False,
        ),
    )
    policies = {node.torchlens_label: node.replay_source_policy for node in runtime.trace_graph.ordered_nodes()}
    assert any(policy == "live_param" for policy in policies.values())
    assert any(policy == "live_param_derived" for policy in policies.values())

    original = model(x)
    with torch.no_grad():
        model.query.add_(0.25)
    updated = model(x)

    live_replay = runtime.replay(x)
    legacy_replay = legacy_runtime.replay(x)

    torch.testing.assert_close(live_replay, updated, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(legacy_replay, original, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(legacy_replay, updated, atol=1e-6, rtol=1e-5)


def test_trainable_suffix_replays_live_buffer_sources() -> None:
    """Trainable suffix replay reads current registered buffers."""

    torch.manual_seed(4321)
    model = _BufferSourceSuffix().eval()
    x = torch.randn(2, 4)
    runtime = tl.prepare_split(
        model,
        x,
        tl.SplitSpec(boundary="after:prefix", dynamic_batch=(1, 4), trainable=True),
    )
    policies = {node.torchlens_label: node.replay_source_policy for node in runtime.trace_graph.ordered_nodes()}
    assert any(label.startswith("buffer_") and policy == "live_param" for label, policy in policies.items())

    with torch.no_grad():
        model.scale.add_(1.0)

    torch.testing.assert_close(runtime.replay(x), model(x), atol=1e-6, rtol=1e-5)


def _boundary_after_last_layer_type(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_type: str,
) -> str:
    """Return a split boundary after the last TorchLens layer of ``layer_type``."""

    trace_model = copy.deepcopy(model)
    model_log = tl.log_forward_pass(
        trace_model,
        inputs,
        capture=CaptureOptions(
            layers_to_save="all",
            keep_unsaved_layers=True,
            detach_saved_tensors=False,
            save_function_args=True,
            intervention_ready=True,
        ),
        visualization=VisualizationOptions(view="none"),
    )
    try:
        labels = [
            label
            for label in model_log.layer_labels
            if getattr(model_log[label], "layer_type", None) == layer_type
        ]
    finally:
        model_log.cleanup()
    assert labels, f"Expected at least one {layer_type!r} layer in the traced model."
    return f"after:{labels[-1]}"


def _boundary_after_first_layer_with_param(
    model: nn.Module,
    inputs: torch.Tensor,
    param_name: str,
) -> str:
    """Return a split boundary after the first TorchLens layer using ``param_name``."""

    trace_model = copy.deepcopy(model)
    model_log = tl.log_forward_pass(
        trace_model,
        inputs,
        capture=CaptureOptions(
            layers_to_save="all",
            keep_unsaved_layers=True,
            detach_saved_tensors=False,
            save_function_args=True,
            intervention_ready=True,
        ),
        visualization=VisualizationOptions(view="none"),
    )
    try:
        labels = [
            label
            for label in model_log.layer_labels
            if param_name in (getattr(model_log[label], "params", None) or {})
        ]
    finally:
        model_log.cleanup()
    assert labels, f"Expected at least one layer using parameter {param_name!r}."
    return f"after:{labels[0]}"


def _train_eager(case: _TrainingCase, model: nn.Module) -> list[torch.Tensor]:
    """Run ordinary eager training for the configured number of epochs."""

    optimizer = torch.optim.SGD(model.parameters(), lr=case.lr)
    losses: list[torch.Tensor] = []
    for _epoch in range(case.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(case.inputs), case.targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return losses


def _train_trace(case: _TrainingCase, model: nn.Module) -> list[torch.Tensor]:
    """Run training from the differentiable output saved by TorchLens trace."""

    optimizer = torch.optim.SGD(model.parameters(), lr=case.lr)
    losses: list[torch.Tensor] = []
    for _epoch in range(case.epochs):
        optimizer.zero_grad(set_to_none=True)
        model_log = tl.log_forward_pass(
            model,
            case.inputs,
            capture=CaptureOptions(train_mode=True),
            visualization=VisualizationOptions(view="none"),
        )
        output = model_log[model_log.output_layers[0]].activation
        loss = F.cross_entropy(output, case.targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        model_log.cleanup()
    return losses


def _train_split(case: _TrainingCase, model: nn.Module) -> list[torch.Tensor]:
    """Run split training with disjoint prefix and suffix optimizers."""

    prefix_params, suffix_params = _split_parameters(model, case.suffix_param_prefixes)
    prefix_optimizer = torch.optim.SGD(prefix_params, lr=case.lr)
    suffix_optimizer = torch.optim.SGD(suffix_params, lr=case.lr)
    runtime = tl.prepare_split(
        model,
        case.inputs,
        tl.SplitSpec(
            boundary=case.split_boundary,
            dynamic_batch=(1, int(case.inputs.shape[0])),
            trainable=True,
        ),
    )

    losses: list[torch.Tensor] = []
    for _epoch in range(case.epochs):
        prefix_optimizer.zero_grad(set_to_none=True)
        suffix_optimizer.zero_grad(set_to_none=True)
        boundary = runtime.run_training_prefix(case.inputs)
        loss, boundary_grads = runtime.train_suffix(
            boundary,
            case.targets,
            F.cross_entropy,
            suffix_optimizer,
        )
        runtime.backward_prefix(boundary, boundary_grads, prefix_optimizer)
        losses.append(loss.detach())
    return losses


def _split_parameters(
    model: nn.Module,
    suffix_param_prefixes: tuple[str, ...],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition model parameters into prefix and suffix groups by name prefix."""

    prefix_params: list[nn.Parameter] = []
    suffix_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in suffix_param_prefixes):
            suffix_params.append(param)
        else:
            prefix_params.append(param)
    assert prefix_params, "Expected at least one split-training prefix parameter."
    assert suffix_params, "Expected at least one split-training suffix parameter."
    return prefix_params, suffix_params


def _assert_state_dict_close(
    actual: nn.Module,
    expected: nn.Module,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Assert that parameters and buffers match after training."""

    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, expected_value in expected_state.items():
        torch.testing.assert_close(
            actual_state[key],
            expected_value,
            atol=atol,
            rtol=rtol,
            msg=f"Mismatch in trained state tensor {key!r}.",
        )
