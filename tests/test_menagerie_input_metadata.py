"""CPU-only catalog metadata must not execute accelerator-bound input recipes."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from menagerie import catalog, classics


CUDA_INPUT_NAMES = ("DeepSFM_PSNet", "InstaGraM", "EPCOT-pretraining")


def _forbidden_factory() -> None:
    """Reject accidental execution of a runtime-only input factory."""

    raise AssertionError("Catalog enumeration executed the example-input factory")


@pytest.mark.parametrize("name", CUDA_INPUT_NAMES)
def test_classics_declared_input_metadata_avoids_factory_execution(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared shapes stay available with CUDA hidden and factories forbidden."""

    entry = dict(classics.CLASSICS[name], example_input=_forbidden_factory)
    monkeypatch.setattr(classics, "CLASSICS", {name: entry})

    rows = catalog._classics_source_rows()

    assert len(rows) == 1
    assert rows[0]["name"] == name
    assert rows[0]["input_shape"] == entry["input_metadata"]["input_shape"]
    assert rows[0]["input_dtype"] == entry["input_metadata"]["input_dtype"]


@pytest.mark.parametrize("name", CUDA_INPUT_NAMES)
def test_classics_declared_input_metadata_matches_original_recipe(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test-only transfer stub corroborates metadata without altering recipes."""

    def cpu_transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Keep the actual recipe's tensor operations on the test CPU."""

        assert not args and not kwargs
        return tensor

    monkeypatch.setattr(torch.Tensor, "cuda", cpu_transfer)
    entry = classics.CLASSICS[name]
    with torch.random.fork_rng(devices=[]):
        actual = catalog._shape_dtype_for_input(entry["example_input"]())

    assert catalog._classics_input_metadata(name, entry) == actual


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"input_shape": "(1,)"},
        {"input_shape": "(1,)", "input_dtype": "float32", "extra": "unknown"},
        {"input_shape": [1], "input_dtype": "float32"},
        {"input_shape": "(1,)", "input_dtype": " "},
        ("(1,)", "float32"),
    ],
)
def test_classics_invalid_input_metadata_refuses_without_executing_factory(metadata: Any) -> None:
    """Invalid declarations must not silently execute runtime code instead."""

    entry = {"input_metadata": metadata, "example_input": _forbidden_factory}
    with pytest.raises(ValueError, match="Invalid MENAGERIE_INPUT_METADATA for 'test-model'"):
        catalog._classics_input_metadata("test-model", entry)


def test_classics_undeclared_input_metadata_retains_runtime_inspection() -> None:
    """Existing CPU recipes retain their exact shape/dtype inspection semantics."""

    def example_input() -> tuple[torch.Tensor, torch.Tensor]:
        """Return a mixed-dtype, multi-input fixture."""

        return torch.zeros(1, 2), torch.zeros(3, dtype=torch.int64)

    assert catalog._classics_input_metadata("test-model", {"example_input": example_input}) == (
        "[(1, 2), (3,)]",
        "float32;int64",
    )
