"""Regression tests for nested trace refusal."""

import pytest
import torch
from torch import nn

import torchlens as tl


class _NestedTraceModel(nn.Module):
    """Module that attempts to start a nested TorchLens trace."""

    def __init__(self) -> None:
        """Initialize the inner module used by the nested trace."""

        super().__init__()
        self.inner = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attempt a nested trace from inside the active forward pass."""

        tl.trace(self.inner, x)
        return x


def test_nested_trace_raises_typed_error_with_active_model_name() -> None:
    """Nested tracing should raise the public typed re-entrancy error."""

    with pytest.raises(tl.ReentrantTraceError, match="NestedTraceModel") as exc_info:
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))

    assert exc_info.value.fields["code"] == "reentrant_trace"
    assert exc_info.value.fields["remedy"]
    assert exc_info.value.fields["active_model"] is not None
    assert "Remedy:" in str(exc_info.value)


def test_reentrant_trace_error_is_exported() -> None:
    """The re-entrancy exception should be part of the top-level public surface."""

    assert tl.ReentrantTraceError.__name__ == "ReentrantTraceError"
    assert "ReentrantTraceError" in tl.__all__


def test_reentrant_trace_error_joins_the_taxonomy_and_keeps_runtime_error() -> None:
    """The refusal is a taxonomy member without breaking historical handlers.

    It must be catchable as ``tl.errors.CaptureError`` (typed taxonomy) AND as
    ``RuntimeError`` (its historical builtin lineage), and it must resolve
    through the registered ``torchlens.errors`` surface.
    """

    from torchlens import errors

    assert issubclass(tl.ReentrantTraceError, errors.CaptureError)
    assert issubclass(tl.ReentrantTraceError, RuntimeError)
    assert errors.ReentrantTraceError is tl.ReentrantTraceError

    with pytest.raises(RuntimeError):
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))
    with pytest.raises(errors.CaptureError):
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))
