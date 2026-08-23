"""Experimental Dagua rendering bridge exports.

Import this module before selecting ``vis_renderer="dagua"`` from the core
visualization dispatcher.
"""

from ...visualization.node_spec import NodeSpec, render_lines_to_html
from ._bridge import (
    build_render_audit,
    render_trace_with_dagua,
    trace_to_dagua_graph,
)

__torchlens_dagua_opted_in__ = True

__all__ = [
    "NodeSpec",
    "build_render_audit",
    "trace_to_dagua_graph",
    "render_lines_to_html",
    "render_trace_with_dagua",
]
