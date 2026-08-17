"""L6 stage-4b attribution-kit behavior and detachability gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens.attribution import occlusion

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ATTRIBUTION_PREFIX = "torchlens.attribution"
_DOCUMENTED_UNSTABLE_TOKENS = {
    "alpha",
    "attribution_sum",
    "baseline",
    "bilinear_display_only",
    "blur",
    "blur_kernel_size",
    "cmap",
    "completeness_residual",
    "grad_cam",
    "image",
    "integrated_gradients",
    "layer",
    "mean",
    "n_steps",
    "native_map_resolution",
    "occluded_score",
    "occlusion",
    "original_score",
    "overlay",
    "relu",
    "rendered_map_resolution",
    "score",
    "selection",
    "selection_digest",
    "target",
    "target_delta",
    "upsampling",
    "zeros",
}


class _ImageScale(nn.Module):
    """Tiny spatial model whose occlusion deltas are easy to inspect."""

    def forward(self, value: Tensor) -> Tensor:
        """Double every input element."""

        return value * 2.0


@pytest.mark.smoke
def test_documented_unstable_attribution_surface_matches_glossary_index() -> None:
    """Every stage-4b spelling carries the exact no-shim glossary marker."""

    glossary = (_REPO_ROOT / "docs/reference/glossary.md").read_text(encoding="utf-8")
    indexed = glossary.split("<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:START -->", 1)[1].split(
        "<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:END -->", 1
    )[0]
    assert set(re.findall(r"`([^`]+)`", indexed)) == _DOCUMENTED_UNSTABLE_TOKENS
    surface_rows = [line for line in indexed.splitlines() if line.startswith("|")][2:]
    assert surface_rows
    assert all("unstable -- no deprecation shim owed" in row for row in surface_rows)


def _output_sum(trace: Any) -> Tensor:
    """Return the single captured output tensor's sum."""

    return trace.output_ops[0].out.sum()


@pytest.mark.smoke
def test_occlusion_routes_named_baselines_through_selection_do() -> None:
    """Zeros, mean, and blur are explicit and produce disclosed scored deltas."""

    inputs = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(),
        inputs,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        selection = tl.units(trace.input_ops[0].label, [(0, 0, 1, 1)])
        results = {
            baseline: occlusion(trace, selection, score=_output_sum, baseline=baseline)
            for baseline in ("zeros", "mean", "blur")
        }
        assert {result.extra["baseline"] for result in results.values()} == {
            "zeros",
            "mean",
            "blur",
        }
        assert results["zeros"].values.item() == pytest.approx(10.0)
        assert results["mean"].values.item() == pytest.approx(0.0)
        assert results["blur"].values.item() == pytest.approx(0.0)
        assert results["blur"].extra["blur_kernel_size"] == 3
        assert results["zeros"].extra["blur_kernel_size"] is None
        assert all(result.extra["selection_digest"] for result in results.values())
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_occlusion_rejects_an_implicit_or_invalid_baseline() -> None:
    """The baseline vocabulary is closed instead of silently choosing a fill."""

    inputs = torch.ones(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(),
        inputs,
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        selection = trace.input_ops[0]
        with pytest.raises(ValueError, match="baseline must be"):
            occlusion(trace, selection, score=_output_sum, baseline="implicit")
    finally:
        trace.cleanup()


def _imported_modules(path: Path) -> set[str]:
    """Return absolute import targets in one Python source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.heavy
def test_attribution_kit_lane_is_detachable_from_stage4a_and_capture_core() -> None:
    """No non-attribution package module or stage-4a file imports the shed lane."""

    package_offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(_imported_modules(path))
        for path in (_REPO_ROOT / "torchlens").rglob("*.py")
        if "attribution" not in path.relative_to(_REPO_ROOT / "torchlens").parts
        and any(
            name == _ATTRIBUTION_PREFIX or name.startswith(f"{_ATTRIBUTION_PREFIX}.")
            for name in _imported_modules(path)
        )
    }
    stage4a_files = (
        "torchlens/selection.py",
        "torchlens/_selection_align.py",
        "tests/test_selection_algebra.py",
        "tests/test_selection_do.py",
        "tests/test_selection_align.py",
        "tests/test_selection_gallery.py",
        "tests/test_dna_canary.py",
    )
    stage4a_offenders = {
        relative: sorted(_imported_modules(_REPO_ROOT / relative))
        for relative in stage4a_files
        if any(
            name == _ATTRIBUTION_PREFIX or name.startswith(f"{_ATTRIBUTION_PREFIX}.")
            for name in _imported_modules(_REPO_ROOT / relative)
        )
    }
    assert not package_offenders
    assert not stage4a_offenders


@pytest.mark.smoke
def test_attribution_detachability_scanner_is_red_capable(tmp_path: Path) -> None:
    """The detachability scanner detects a planted direct import."""

    coupled = tmp_path / "coupled.py"
    coupled.write_text("from torchlens.attribution import occlusion\n", encoding="utf-8")
    assert _ATTRIBUTION_PREFIX in _imported_modules(coupled)
