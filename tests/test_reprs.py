"""Tests for Phase 3 repr and tensor display ergonomics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class _InfModel(nn.Module):
    """Tiny model that produces a non-finite out."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an infinite tensor."""

        return x / 0


def _log_for_input(x: torch.Tensor) -> tl.Trace:
    """Capture an identity model for one input.

    Parameters
    ----------
    x:
        Input tensor.

    Returns
    -------
    tl.Trace
        Captured model log.
    """

    return tl.trace(nn.Identity(), x)


def test_print_trace_is_informative(capsys: pytest.CaptureFixture[str]) -> None:
    """Printing a Trace gives a concise model summary."""

    log = _log_for_input(torch.randn(1, 3))
    print(log)
    captured = capsys.readouterr()
    assert "Log of" in captured.out
    assert "Tensor info" in captured.out


def test_trace_repr_html_is_informative() -> None:
    """Trace HTML repr returns an informative string or text fallback."""

    log = _log_for_input(torch.randn(1, 3))
    html = log._repr_html_()
    assert isinstance(html, str)
    assert "Trace" in html or "TorchLens" in html
    assert "Layers" in html or "layers=" in html


@pytest.mark.parametrize("method", ["auto", "heatmap", "channels", "rgb", "hist"])
def test_layer_log_show_custom_methods_return_output(method: str) -> None:
    """Layer.show accepts every Phase 3 display method."""

    x = torch.randn(1, 3, 4, 4) if method == "rgb" else torch.randn(3, 4, 4)
    log = _log_for_input(x)
    output = log.layers[0].show(method=method)
    assert output is not None


def test_op_log_show_returns_output() -> None:
    """Op.show delegates to the tensor display helper."""

    log = _log_for_input(torch.randn(8))
    output = log.layer_list[0].show(method="hist")
    assert output is not None


def test_first_nonfinite_reports_context() -> None:
    """Trace.first_nonfinite reports the first saved NaN or Inf site."""

    log = tl.trace(_InfModel(), torch.ones(1, 2))
    answer = log.first_nonfinite()
    assert "First non-finite" in answer
    assert "shape=" in answer
    assert "dtype=" in answer
    assert "parents=" in answer
    assert "source=" in answer


class _TwoStage(nn.Module):
    """Two-op model so relation interpolation has real parents/children."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear followed by relu."""

        return torch.relu(self.lin(x))


def test_layer_repr_after_trace_collection_degrades_not_raises() -> None:
    """repr/str/format on a Layer whose Trace was collected must not raise.

    R52-A shape 1 (round-4 b7-opus): the natural one-liner
    ``tl.trace(model, x)[label]`` leaves the Layer holding a dead weakref;
    ``__repr__`` is data-model API and must degrade, never raise.
    """

    import gc

    layer = tl.trace(_TwoStage(), torch.randn(2, 4))["relu_1_2"]
    gc.collect()

    text = repr(layer)
    assert "relu_1_2" in text
    assert "detached" in text
    assert str(layer) == text
    assert f"context: {layer}"  # f-string interpolation must not raise
    # Relation accessors still refuse TYPED - repr degrading must not
    # loosen the accessor contract.
    from torchlens._errors import RecordBindingError

    with pytest.raises(RecordBindingError) as exc_info:
        _ = layer.parents
    assert exc_info.value.fields["code"] == "trace_reference_collected"


def test_layer_standalone_pickle_repr_and_accessors_typed() -> None:
    """A standalone-pickled Layer must repr fine and refuse accessors TYPED.

    R52-A shape 2 (round-4 b7-opus): ``__getstate__`` strips the trace
    weakref, so ``source_trace`` used to return a bare ``None`` behind a
    ``-> Trace`` signature and ``repr()`` crashed with an UNTYPED
    ``TypeError: 'NoneType' object is not subscriptable``.
    """

    import pickle

    from torchlens._errors import RecordBindingError

    trace = tl.trace(_TwoStage(), torch.randn(2, 4))
    restored = pickle.loads(pickle.dumps(trace["relu_1_2"]))

    text = repr(restored)
    assert "relu_1_2" in text
    assert "detached" in text
    assert str(restored) == text

    with pytest.raises(RecordBindingError) as exc_info:
        _ = restored.source_trace
    assert exc_info.value.fields["code"] == "record_not_bound"
    assert exc_info.value.fields["remedy"]

    with pytest.raises(RecordBindingError) as parents_exc:
        _ = restored.parents
    assert parents_exc.value.fields["code"] == "record_not_bound"


def test_layer_repr_on_live_trace_unchanged() -> None:
    """A Layer bound to a live Trace keeps the full informative repr."""

    trace = tl.trace(_TwoStage(), torch.randn(2, 4))
    layer = trace["relu_1_2"]
    text = repr(layer)
    assert "Layer relu_1_2" in text
    assert "parents" in text
    assert "detached" not in text
