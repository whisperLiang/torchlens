"""Flatten axes are semantic indices, never dynamic-batch shape literals."""

from __future__ import annotations

import pytest
import torch
from v2_helpers import split_request

import torchlens as tl

pytestmark = pytest.mark.smoke


class _ChannelsFirstLayerNorm(torch.nn.LayerNorm):
    """Normalize NCHW channels through ConvNeXt's NHWC permutation pattern."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Preserve the four-dimensional layout around channel normalization.

        Parameters
        ----------
        value:
            NCHW activation to normalize over its channel dimension.

        Returns
        -------
        torch.Tensor
            The normalized activation in its original NCHW layout.
        """

        value = value.permute(0, 2, 3, 1)
        value = torch.nn.functional.layer_norm(
            value, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return value.permute(0, 3, 1, 2)


class _PooledClassifier(torch.nn.Module):
    """Minimal ConvNeXt-style pool, normalization, flatten, and linear head."""

    def __init__(self, spelling: str, start_dim: int, end_dim: int) -> None:
        """Select the captured flatten spelling and fixed axis interval.

        Parameters
        ----------
        spelling:
            Module, tensor method, functional, or keyword-based flatten call.
        start_dim, end_dim:
            Inclusive flatten axis interval, independent of batch size.
        """

        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.norm = _ChannelsFirstLayerNorm(4)
        self.flatten = torch.nn.Flatten(start_dim, end_dim)
        self.head = torch.nn.Linear(4, 3)
        self.spelling = spelling
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map spatial activations to one logit vector per batch member.

        Parameters
        ----------
        value:
            A tensor with four channels and arbitrary spatial dimensions.

        Returns
        -------
        torch.Tensor
            Three logits per example.
        """

        value = self.norm(self.pool(value))
        if self.spelling == "module":
            value = self.flatten(value)
        elif self.spelling == "tensor":
            value = value.flatten(self.start_dim, self.end_dim)
        elif self.spelling == "functional":
            value = torch.flatten(value, self.start_dim, self.end_dim)
        else:
            value = torch.flatten(value, start_dim=self.start_dim, end_dim=self.end_dim)
        return self.head(value)


@pytest.mark.parametrize("boundary", ["before:flatten", "after:flatten"])
@pytest.mark.parametrize(
    ("spelling", "start_dim", "end_dim"),
    [
        ("module", 1, -1),
        ("tensor", 1, 3),
        ("functional", 1, -1),
        ("keywords", 1, -1),
        ("tensor", -3, -1),
    ],
)
def test_pooled_classifier_flatten_axes_survive_batch_changes(
    boundary: str, spelling: str, start_dim: int, end_dim: int
) -> None:
    """B=1 axis literals stay fixed in both replay segments at B=2 and B=3.

    Parameters
    ----------
    boundary:
        Whether flatten executes in the prefix or suffix segment.
    spelling:
        Public Torch flatten spelling exercised by the model.
    start_dim, end_dim:
        Positive or negative axis indices, not target shape dimensions.
    """

    model = _PooledClassifier(spelling, start_dim, end_dim).eval()
    runtime = tl.split.prepare(model, torch.randn(2, 4, 5, 5), split_request(boundary))
    assert runtime.traced_batch_size == 1
    assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
    assert runtime.batch_validation["probe_batch_size"] == 2
    assert runtime.batch_validation["universal_proof"] is False

    shape_program = runtime.trace_graph.shape_program
    assert shape_program is not None
    flatten_node = next(
        node for node in runtime.trace_graph.compute_nodes if node.op_type == "flatten"
    )
    assert not shape_program.recipes.get(flatten_node.canonical_id)
    for batch in (1, 2, 3):
        values = torch.randn(batch, 4, 5, 5)
        assert shape_program.value_shape(
            flatten_node.canonical_id, shape_program.binding_from_batch(batch)
        ) == (batch, 4)
        with torch.no_grad():
            actual = runtime.replay(values)
            expected = model(values)
        assert actual.shape == (batch, 3)
        torch.testing.assert_close(actual, expected)
