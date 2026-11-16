"""Rank-layout DOT exports preserve the rendered artifact after source cleanup."""

from pathlib import Path

import graphviz
import torch
from torch import nn

import torchlens as tl


def test_rank_dot_export_keeps_rendered_file(tmp_path: Path) -> None:
    """The DOT output is a parseable artifact, distinct from temporary input."""
    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.ones(1, 4))
    try:
        trace.draw(
            layout="rank",
            file_format="dot",
            path=str(tmp_path / "rank"),
            open_viewer=False,
        )
        artifact = tmp_path / "rank.dot"
        assert artifact.is_file()
        parsed = graphviz.Source(artifact.read_text(encoding="utf-8")).pipe(format="plain")
        assert parsed.startswith(b"graph ")
        assert b"node " in parsed
    finally:
        trace.cleanup()
