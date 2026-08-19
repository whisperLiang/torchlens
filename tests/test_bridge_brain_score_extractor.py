"""Brain-Score ``ActivationsExtractorHelper`` adapter tests (bridge P11).

The ``get_activations`` contract is exercised offline (no Brain-Score
install needed); the extractor wiring is verified against a stub mirroring
the real ``ActivationsExtractorHelper`` constructor read from the
brainscore-vision 2.3.22 wheel, plus an importorskip-gated construction test
against the real package where one is installed (it is not installable on
Python < 3.11, so that cell skips here — the adapter is disclosed UNVERIFIED
against a running Brain-Score).
"""

from __future__ import annotations

import sys
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge.brain_score import activations_extractor, get_activations_fn


class _TinyNet(nn.Module):
    """Small named-submodule model mirroring Brain-Score's dotted layer names."""

    def __init__(self) -> None:
        """Initialize a two-stage feature/head stack."""

        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run features then head.

        Parameters
        ----------
        inputs:
            Batched stimulus rows.

        Returns
        -------
        torch.Tensor
            Model logits.
        """

        return self.head(self.features(inputs))


def _image_list(n: int = 5) -> list[np.ndarray]:
    """Return ``n`` preprocessed single-stimulus arrays, Brain-Score style.

    Parameters
    ----------
    n:
        Number of stimuli.

    Returns
    -------
    list[numpy.ndarray]
        Per-stimulus float32 arrays.
    """

    return [np.arange(3, dtype=np.float32) + index for index in range(n)]


def test_get_activations_serves_dotted_paths_and_logits_in_order() -> None:
    """The callable returns an ordered dict of numpy arrays matching extract."""

    model = _TinyNet().eval()
    get_activations = get_activations_fn(model)
    result = get_activations(_image_list(), ["features.1", "logits"])

    assert isinstance(result, OrderedDict)
    assert list(result) == ["features.1", "logits"]
    assert all(isinstance(value, np.ndarray) for value in result.values())
    assert result["features.1"].shape == (5, 4)
    assert result["logits"].shape == (5, 2)

    batch = torch.stack([torch.from_numpy(image) for image in _image_list()])
    direct = tl.extract(model, batch, {"features.1": "features.1", "logits": "output_1"})
    np.testing.assert_array_equal(result["features.1"], direct["features.1"].detach().numpy())
    np.testing.assert_array_equal(result["logits"], direct["logits"].detach().numpy())


def test_get_activations_layer_map_and_model_dtype_matching() -> None:
    """Custom lookups resolve through layer_map; inputs match the model dtype."""

    model = _TinyNet().double().eval()
    get_activations = get_activations_fn(model, layer_map={"pool-ish": "relu"})
    result = get_activations(_image_list(), ["pool-ish"])
    assert list(result) == ["pool-ish"]
    assert result["pool-ish"].dtype == np.float64
    assert result["pool-ish"].shape == (5, 4)


def test_get_activations_accepts_prestacked_batches() -> None:
    """A pre-stacked array or tensor batch works like a per-image list."""

    model = _TinyNet().eval()
    get_activations = get_activations_fn(model)
    stacked = np.stack(_image_list())
    from_array = get_activations(stacked, ["logits"])
    from_tensor = get_activations(torch.from_numpy(stacked), ["logits"])
    np.testing.assert_array_equal(from_array["logits"], from_tensor["logits"])


def test_activations_extractor_refuses_without_brainscore() -> None:
    """Without brainscore-vision the adapter raises a naming ImportError."""

    if "brainscore_vision" in sys.modules or _brainscore_importable():
        pytest.skip("brainscore-vision installed; refusal path not reachable")
    with pytest.raises(ImportError, match="brainscore-vision"):
        activations_extractor(_TinyNet().eval(), preprocessing=None)


def _brainscore_importable() -> bool:
    """Report whether the real brainscore-vision package imports.

    Returns
    -------
    bool
        True when ``brainscore_vision`` is importable in this environment.
    """

    try:
        import brainscore_vision  # noqa: F401
    except ImportError:
        return False
    return True


def test_activations_extractor_wires_the_documented_constructor(monkeypatch) -> None:
    """The adapter passes the exact kwargs the real helper constructor takes.

    The stub mirrors ``ActivationsExtractorHelper.__init__(get_activations,
    preprocessing, identifier=False, batch_size=...)`` as read from the
    brainscore-vision 2.3.22 wheel.
    """

    recorded: dict[str, object] = {}

    class _StubHelper:
        """Constructor-recording stand-in for ActivationsExtractorHelper."""

        def __init__(self, get_activations, preprocessing, identifier=False, batch_size=64):
            """Record the constructor arguments.

            Parameters
            ----------
            get_activations:
                Per-batch activations callable.
            preprocessing:
                Stimulus preprocessing callable.
            identifier:
                Activations identifier.
            batch_size:
                Stimuli per batch.
            """

            recorded["get_activations"] = get_activations
            recorded["preprocessing"] = preprocessing
            recorded["identifier"] = identifier
            recorded["batch_size"] = batch_size

    package = types.ModuleType("brainscore_vision")
    helpers = types.ModuleType("brainscore_vision.model_helpers")
    activations = types.ModuleType("brainscore_vision.model_helpers.activations")
    core = types.ModuleType("brainscore_vision.model_helpers.activations.core")
    core.ActivationsExtractorHelper = _StubHelper
    monkeypatch.setitem(sys.modules, "brainscore_vision", package)
    monkeypatch.setitem(sys.modules, "brainscore_vision.model_helpers", helpers)
    monkeypatch.setitem(sys.modules, "brainscore_vision.model_helpers.activations", activations)
    monkeypatch.setitem(sys.modules, "brainscore_vision.model_helpers.activations.core", core)

    def _preprocess(paths: object) -> object:
        """Pass stimuli through unchanged.

        Parameters
        ----------
        paths:
            Stimulus paths or arrays.

        Returns
        -------
        object
            The unchanged stimuli.
        """

        return paths

    helper = activations_extractor(_TinyNet().eval(), preprocessing=_preprocess, batch_size=8)
    assert isinstance(helper, _StubHelper)
    assert recorded["identifier"] == "_TinyNet"
    assert recorded["preprocessing"] is _preprocess
    assert recorded["batch_size"] == 8

    served = recorded["get_activations"](_image_list(), ["logits"])
    assert list(served) == ["logits"]
    assert served["logits"].shape == (5, 2)


def test_activations_extractor_constructs_against_real_brainscore() -> None:
    """With the real package installed, construction returns the real helper."""

    pytest.importorskip("brainscore_vision")
    from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper

    helper = activations_extractor(_TinyNet().eval(), preprocessing=None, identifier="tiny")
    assert isinstance(helper, ActivationsExtractorHelper)
    assert helper.identifier == "tiny"
