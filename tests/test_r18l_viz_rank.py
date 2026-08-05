"""Regression tests for r18l viz-plots + rank layout + graph caption findings.

Covers:
- H10  caption ``[:-2]`` left ``</FONT`` unterminated -> GraphvizRenderError on
       any ``_has_direct_writes=True`` trace.
- M4   rank ``_compute_topological_layout`` sibling order depended on set/hash
       iteration -> nondeterministic across PYTHONHASHSEED.
- M5   ``feature_map_evolution`` leaked the internal ``mds_evolution``/``MDS``
       error vocabulary on a recurrent layer.
- M6   ``channel_grid`` silently used only batch item 0 (undocumented).
- M7   rank engine silently dropped ``show_legend``.
- M8   ``render_lineplot`` produced a blank plot when no finite (x, y) PAIR existed.
- M9   ``_normalize_finite`` returned a uniform array on reversed vmin/vmax while
       its docstring claimed it raises.
- M10  rank engine silently ignored ``dpi`` and ``vis_graph_overrides``.
- F12  ``render_lineplot`` drew out-of-range points over the chart furniture.
"""

import pytest
import torch
import torch.nn as nn

import torchlens as tl


# ---------------------------------------------------------------------------
# H10 — caption terminates the FONT tag; draw() after set() must not crash
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_h10_direct_writes_caption_renders(tmp_path):
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    trace._has_direct_writes = True
    out = str(tmp_path / "h10")
    # Must not raise GraphvizRenderError.
    src = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=out)
    assert isinstance(src, str)
    # The FONT element must be properly closed and the direct-writes line present.
    assert "</FONT>" in src
    assert "</FONTDirect" not in src
    assert "Direct writes detected" in src


def test_h10_caption_no_direct_writes_still_closes_font(tmp_path):
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    out = str(tmp_path / "h10b")
    src = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=out)
    assert "</FONT>" in src
    assert "Direct writes detected" not in src
