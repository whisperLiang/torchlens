"""Preview and dry-run RecordContext fidelity tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext, RecordContextFieldError
from torchlens.visualization.fastlog_preview import _build_preview_nodes, _make_node_spec_fn
from torchlens.visualization.node_spec import NodeSpec


class StaticGraph(nn.Module):
    """Static graph model for preview fidelity."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the static graph."""

        return self.layers(x)


def _stable_context_fields(ctx: RecordContext) -> dict[str, object]:
    """Return predicate-visible fields that preview can synthesize from a Trace."""

    return {
        "kind": ctx.kind,
        "label": ctx.label,
        "raw_label": ctx.raw_label,
        "layer_type": ctx.layer_type,
        "type_index": ctx.type_index,
        "raw_index": ctx.raw_index,
        "func_name": ctx.func_name,
        "address": ctx.address,
        "module_type": ctx.module_type,
        "module_pass_index": ctx.module_pass_index,
        "module_stack": tuple(
            (frame.address, frame.module_type, frame.pass_index)
            for frame in ctx.module_stack
        ),
        "parent_labels": ctx.parent_labels,
        "input_output_address": ctx.input_output_address,
        "shape": ctx.shape,
        "dtype": ctx.dtype,
        "output_index": ctx.output_index,
        "is_bottom_level_func": ctx.is_bottom_level_func,
    }


def test_preview_and_dry_run_contexts_match_stable_fields() -> None:
    """Preview-synthesized and real dry-run contexts match stable predicate fields."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    full_trace = tl.trace(model, x)
    dry_trace = tl.fastlog.dry_run(
        model,
        x,
        keep_op=lambda ctx: True,
        include_source_events=True,
    )
    preview_nodes = _build_preview_nodes(full_trace, lambda ctx: True)
    preview_contexts = [node.ctx for node in dict.fromkeys(preview_nodes.values())]
    real_contexts = [
        ctx
        for ctx in dry_trace.contexts
        if ctx.kind in {"input", "op"} and ctx.layer_type != "output"
    ]

    assert [_stable_context_fields(ctx) for ctx in preview_contexts] == [
        _stable_context_fields(ctx) for ctx in real_contexts
    ]


def test_missing_record_context_field_errors_in_preview_and_dry_run() -> None:
    """A nonexistent predicate field fails with RecordContextFieldError in both paths."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)

    def bad_predicate(ctx: RecordContext) -> bool:
        """Access a field outside the RecordContext schema."""

        return bool(ctx.recurrent_ops)

    dot = trace.preview_fastlog(predicate=bad_predicate)

    assert "exception" in dot
    with pytest.raises(RecordContextFieldError):
        tl.fastlog.dry_run(model, x, keep_op=bad_predicate)


def test_preview_nodes_include_short_label_keys() -> None:
    """Preview node lookup includes pass-free short labels."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)

    preview_nodes = _build_preview_nodes(trace, None)

    assert "relu_1" in preview_nodes


def test_preview_node_spec_consumes_short_label_keys() -> None:
    """Preview rendering reads the same short-label keys it writes."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)
    preview_nodes = _build_preview_nodes(trace, lambda ctx: True)
    node_spec_fn = _make_node_spec_fn(
        preview_nodes,
        color_kept="#00FF00",
        color_rejected="#FF0000",
        color_unreachable="#0000FF",
        color_predicate_error="#FFFF00",
        show_predicate_inputs=False,
        show_module_events=False,
    )

    styled = node_spec_fn(
        type(
            "ShortLabelOnlyLayer",
            (),
            {"layer_label": "missing_label", "layer_label_short": "relu_1"},
        )(),
        NodeSpec(lines=["relu"]),
    )

    assert styled.fillcolor == "#00FF00"
    assert "fastlog: kept" in styled.lines
