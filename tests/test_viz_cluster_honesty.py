"""Fixwave-2 FW2-POLISH pins for R19-3/4/5/7: module-cluster honesty.

The dashed cluster style is a semantic claim ("no input ancestor"). The b6
probes found it lying in two ways: the input-connectivity propagation broke
after the outermost module (a tautological loop guard), and clusters whose
ops were condensed away were re-emitted as empty dashed boxes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _Inner(nn.Module):
    """Module whose ONLY edges cross its boundary (no interior op-op edge)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class _Outer(nn.Module):
    """Nested wrapper reproducing the b6 R19-4 dashed-cluster probe."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.wrap = nn.Sequential(_Inner())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrap(self.lin(x))


def _dot_source(tmp_path: Path, trace: tl.Trace, **draw_kwargs) -> str:
    outpath = tmp_path / "cluster_honesty"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(outpath),
        **draw_kwargs,
    )
    return (tmp_path / "cluster_honesty.dot").read_text()


def _cluster_styles(dot_source: str) -> dict[str, str]:
    """Return ``cluster name -> style attr`` for module clusters."""

    styles: dict[str, str] = {}
    current: str | None = None
    for line in dot_source.splitlines():
        stripped = line.strip()
        match = re.match(r'subgraph "?cluster_([^" {]+)"?\s*\{', stripped)
        if match:
            current = match.group(1)
            continue
        style_match = re.match(r'style="?([^"\]]+)"?$', stripped)
        if style_match and current is not None:
            styles.setdefault(current, style_match.group(1))
    return styles


def test_nested_module_without_interior_edge_renders_solid(tmp_path: Path) -> None:
    """A fully input-connected nested module is never styled dashed (R19-4).

    Every op in ``wrap``/``wrap.0`` has ``has_input_ancestor=True``; its only
    edges cross the module boundary. Pre-fix, the connectivity propagation
    broke after the first containing module, so the nested cluster claimed
    "no input ancestor" in the DEFAULT unrolled render while the ROLLED
    render of the same trace showed it solid.
    """

    trace = tl.trace(_Outer(), torch.randn(1, 4))
    for view in ("unrolled", "rolled"):
        source = _dot_source(tmp_path / view, trace, vis_mode=view)
        styles = _cluster_styles(source)
        dashed = {name for name, style in styles.items() if "dashed" in style}
        assert not dashed, (
            f"{view}: input-connected module clusters styled dashed "
            f"('no input ancestor' is a false claim here): {sorted(dashed)}"
        )


def test_unrolled_and_rolled_cluster_styles_agree(tmp_path: Path) -> None:
    """The two views never make contradictory connectivity claims (R19-4)."""

    trace = tl.trace(_Outer(), torch.randn(1, 4))
    unrolled = _cluster_styles(_dot_source(tmp_path / "u", trace, vis_mode="unrolled"))
    rolled = _cluster_styles(_dot_source(tmp_path / "r", trace, vis_mode="rolled"))

    def _solidity(styles: dict[str, str]) -> dict[str, bool]:
        result: dict[str, bool] = {}
        for name, style in styles.items():
            base = re.sub(r"_pass\d+$", "", name)
            result[base] = result.get(base, False) or "dashed" not in style
        return result

    unrolled_solidity = _solidity(unrolled)
    rolled_solidity = _solidity(rolled)
    for base in set(unrolled_solidity) & set(rolled_solidity):
        assert unrolled_solidity[base] == rolled_solidity[base], (
            f"cluster {base!r}: unrolled and rolled renders disagree about "
            "input connectivity"
        )


def test_boundary_crossing_segment_discloses_spanned_modules() -> None:
    """A segment placed above its ops' module homes names them (R19-5).

    The b6/opus probe: a ``collapse="max"`` segment merged three ``@blocks``
    ops with one ``@head`` op and rendered at TOP LEVEL, outside every module
    cluster, with nothing disclosing the containment it erased.
    """

    from types import SimpleNamespace

    from torchlens.visualization.collapse_optimizer import _make_op_segment_descriptor

    trace = tl.trace(_Outer(), torch.randn(1, 4))
    context = SimpleNamespace(vis_mode="unrolled")
    labels = ("linear_1_1", "relu_1_2")
    descriptor = _make_op_segment_descriptor(trace, context, labels, labels)
    assert descriptor.owner is None, "probe expects a top-level (LCA-less) segment"
    assert "spans" in descriptor.label and "@wrap" in descriptor.label, (
        f"top-level segment label discloses no module containment: {descriptor.label!r}"
    )


def test_within_module_segment_label_stays_unchanged() -> None:
    """A segment fully inside one module keeps its plain range label."""

    from types import SimpleNamespace

    from torchlens.visualization.collapse_optimizer import _make_op_segment_descriptor

    deep = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU()))
    trace = tl.trace(deep, torch.randn(1, 4))
    context = SimpleNamespace(vis_mode="unrolled")
    labels = tuple(
        label for label in ("linear_1_1", "relu_1_2", "linear_2_3", "relu_2_4")
    )
    descriptor = _make_op_segment_descriptor(trace, context, labels, labels)
    assert "spans" not in descriptor.label, descriptor.label


def test_condensed_away_clusters_are_not_emitted_empty(tmp_path: Path) -> None:
    """collapse='max' emits no empty dashed module boxes (R19-3).

    The b6/opus probe showed module clusters rendered as labeled dashed
    boxes with ZERO nodes while their ops sat in a segment outside — an
    empty box that both misplaces the ops and falsely claims disconnection.
    A cluster with no nodes, no edges, and no child clusters is noise.
    """

    # Eight blocks reproduce the b6 shape: collapse="max" condenses every
    # block's ops into top-level boundary-crossing segments, which pre-fix
    # left all eight block clusters behind as empty husks (four styled
    # dashed, four solid — style noise on top of the misplacement).
    blocks = nn.Sequential(*[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(8)])
    trace = tl.trace(blocks, torch.randn(1, 4))
    source = _dot_source(tmp_path, trace, vis_mode="unrolled", collapse="max")
    dashed = {name for name, style in _cluster_styles(source).items() if "dashed" in style}
    assert not dashed, (
        f"fully input-connected model rendered dashed clusters at collapse='max': {sorted(dashed)}"
    )
    body_by_cluster: dict[str, list[str]] = {}
    stack: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        match = re.match(r'subgraph "?cluster_([^" {]+)"?\s*\{', stripped)
        if match:
            stack.append(match.group(1))
            body_by_cluster.setdefault(match.group(1), [])
            continue
        if stripped.startswith("}"):
            if stack:
                stack.pop()
            continue
        if stack:
            body_by_cluster[stack[-1]].append(stripped)
    for name, body in body_by_cluster.items():
        has_content = any(
            row and not re.match(r"[a-zA-Z_]+=", row) for row in body
        )  # node/edge statements vs pure attribute rows
        assert has_content, (
            f"cluster {name!r} was emitted with attributes only (no nodes, no "
            "edges) — an empty box making a connectivity claim about nothing"
        )
