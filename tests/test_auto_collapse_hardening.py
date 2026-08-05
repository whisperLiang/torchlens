"""Hardening regressions for the intelligent auto-collapse subsystem (round 21).

Covers the round-20 adversarial-audit defect classes:

- multi-pass submodule reuse must not crash the smart-collapse surface;
- segment descriptor identity must be injective (plan/render parity);
- schedule ``collapsed_addresses`` must not report an empty set while hiding nodes;
- range/count labels must never lie about hidden mass;
- weighted optimizer results must not be cache-order dependent;
- ``CollapseSchedule.at(0.0)`` must agree with ``select_collapse_level(0.0)``;
- plans, schedules, and orders must be deterministic across hash seeds.
"""

import re
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.visualization.auto_collapse import _make_run_fold, analyze_collapse
from torchlens.visualization.collapse_optimizer import (
    OptimizerWeights,
    _RESULT_CACHE,
    _child_segment_covered_ops,
    _child_segment_label,
    _make_child_segment_descriptor,
    select_collapse_level,
    select_collapse_plan,
)
from torchlens.visualization.collapse_plan import (
    ChildSegment,
    ModuleBox,
    OpSegment,
    RawOp,
    RenderContext,
    RepeatFold,
)
from torchlens.visualization._render_edges import _run_fold_ellipsis_label


class ResidualBlock(nn.Module):
    """Small residual block used to build multi-pass reuse models."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class DoubleCall(nn.Module):
    """Calls every block twice per forward, creating multi-pass submodules."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            x = block(x)
        return x


class SegmentLeaf(nn.Module):
    """Three-op leaf block used by segment-collision models."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(self.lin(x)))


class CollidingSegments(nn.Module):
    """Nested ``a.b0..a.b9`` and top-level ``a_b0..a_b9`` chains.

    Both child-segment endpoint pairs mangle to the same legacy Graphviz name,
    which historically made the descriptor dict non-injective.
    """

    class Inner(nn.Module):
        def __init__(self, width: int = 8) -> None:
            super().__init__()
            for index in range(10):
                setattr(self, f"b{index}", SegmentLeaf(width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for index in range(10):
                x = getattr(self, f"b{index}")(x)
            return x

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.a = self.Inner(width)
        for index in range(10):
            setattr(self, f"a_b{index}", SegmentLeaf(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(10):
            x = getattr(self, f"a_b{index}")(x)
        return self.a(x)


class LoopedFunctionalAroundBlock(nn.Module):
    """Loop of shared functional ops around a boxable block.

    Every iteration re-executes the same functional layer labels, so the max
    plan contains several op segments whose BASE label runs are identical and
    only the pass identity distinguishes them.
    """

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        layers = []
        for _ in range(5):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.blk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(torch.sigmoid(torch.relu(x)))
            x = self.blk(x)
        return x


class UnevenRecurrent(nn.Module):
    """Two sibling blocks called three and five times per forward."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.first = SegmentLeaf(width)
        self.second = SegmentLeaf(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.first(x)
        for _ in range(5):
            x = self.second(x)
        return x


def _svg_node_group_count(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return len(re.findall(r'class="node', handle.read()))


def _plan_segment_node_count(plan) -> int:
    return sum(isinstance(node, (ChildSegment, OpSegment)) for node in plan.nodes)


@pytest.mark.parametrize("depth", [3, 4, 6])
def test_collapse_surface_survives_multipass_submodule(depth, tmp_path):
    """Every public smart-collapse surface must survive multi-pass reuse."""

    model = DoubleCall(depth).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    for mode in ("auto", "max", 0.5):
        plan = trace.collapse_plan(mode=mode)
        assert plan.total > 0
    schedule = trace.collapse_schedule()
    assert schedule.steps[0].visible_count >= schedule.steps[-1].visible_count
    order = trace.module_collapse_order
    assert isinstance(order, list) and order
    for mode in ("auto", "max"):
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(tmp_path / f"dc{depth}_{mode}"),
        )


def test_segment_descriptor_cardinality_matches_plan():
    """Colliding segment endpoints must yield one descriptor per plan segment."""

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    result = select_collapse_plan(trace, context, mode="max")
    plan_segments = _plan_segment_node_count(result.plan)
    descriptors = result.segments or {}
    assert plan_segments == 2
    assert len(descriptors) == plan_segments
    names = sorted(descriptors)
    assert len(set(names)) == plan_segments
    member_tuples = {descriptor.members for descriptor in descriptors.values()}
    assert len(member_tuples) == plan_segments


def test_segment_collision_render_parity(tmp_path):
    """The rendered SVG node count must equal the max plan's node count."""

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    plan = trace.collapse_plan(mode="max")
    out = tmp_path / "collision"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    assert _svg_node_group_count(str(out) + ".svg") == plan.total


@pytest.mark.parametrize(
    "builder",
    [lambda: DoubleCall(2), lambda: LoopedFunctionalAroundBlock()],
    ids=["double_call", "looped_functional"],
)
def test_multipass_op_segments_render_distinct(builder, tmp_path):
    """Per-pass op segments must not silently merge into one rendered node.

    ``looped_functional`` produces several op segments whose BASE label runs
    are byte-identical (only the pass differs), so any pass-free segment
    identity collides and silently drops rendered structure.
    """

    model = builder().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    plan = trace.collapse_plan(mode="max")
    result = select_collapse_plan(trace, RenderContext(), mode="max")
    descriptors = result.segments or {}
    assert len(descriptors) == _plan_segment_node_count(result.plan)
    assert len(set(descriptors)) == len(descriptors)
    out = tmp_path / "multipass_max"
    trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        collapse="max",
        vis_outpath=str(out),
    )
    assert _svg_node_group_count(str(out) + ".svg") == plan.total


def _plan_hidden_and_visible_ops(trace, result):
    """Split the trace's concrete ops into visible and hidden-with-witness sets."""

    concrete = {str(op.label) for op in trace.ops}
    hidden: set[str] = set()
    visible: set[str] = set()
    analysis = analyze_collapse(trace)
    occurrence: dict[str, int] = {}
    for node in result.plan.nodes:
        if isinstance(node, RawOp) and isinstance(node.op, str):
            base = node.op
            if base in concrete:
                visible.add(base)
                continue
            occurrence[base] = occurrence.get(base, 0) + 1
            qualified = f"{base}:{occurrence[base]}"
            visible.add(qualified if qualified in concrete else base)
        elif isinstance(node, OpSegment):
            for base in node.ops:
                occurrence[base] = occurrence.get(base, 0) + 1
    for descriptor in (result.segments or {}).values():
        hidden.update(str(op) for op in descriptor.ops)
    for node in result.plan.nodes:
        if isinstance(node, ChildSegment):
            for address in node.members:
                signal = analysis.signals.get(address)
                if signal is not None:
                    hidden.update(str(label) for label in signal.subtree_ops)
        elif isinstance(node, (ModuleBox, RepeatFold)):
            addresses = (
                node.members if isinstance(node, RepeatFold) else (node.call.rsplit(":", 1)[0],)
            )
            for address in addresses:
                signal = analysis.signals.get(address)
                if signal is not None:
                    hidden.update(str(label) for label in signal.subtree_ops)
    return visible, hidden


@pytest.mark.parametrize("mode", ["auto", "max"])
def test_no_silent_node_drop(mode):
    """Every concrete op is either visible or witnessed by a hiding unit."""

    battery = [
        DoubleCall(3).eval(),
        CollidingSegments().eval(),
        nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval(),
    ]
    for model in battery:
        trace = tl.trace(model, torch.randn(2, 8))
        result = select_collapse_plan(trace, RenderContext(), mode=mode)
        if result.declined:
            continue
        visible, hidden = _plan_hidden_and_visible_ops(trace, result)
        concrete = {str(op.label) for op in trace.ops}
        unaccounted = {
            label
            for label in concrete
            if label not in hidden
            and label not in visible
            and label.rsplit(":", 1)[0] not in visible
        }
        assert not unaccounted, (
            f"{type(model).__name__} mode={mode}: ops silently dropped: {sorted(unaccounted)[:8]}"
        )


def test_schedule_collapsed_addresses_not_empty_when_hiding():
    """Schedule steps that hide nodes must report a nonempty collapsed set."""

    model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    schedule = trace.collapse_schedule()
    full = schedule.steps[0].visible_count
    for step in schedule.steps:
        if step.visible_count < full:
            assert step.collapsed_addresses, (
                f"step t={step.t} hides {full - step.visible_count} nodes but "
                "reports an empty collapsed set"
            )


def test_schedule_addresses_monotone_superset():
    """Collapsed sets must stay nested as t increases (with the honest final set)."""

    for model in (
        nn.Sequential(*[nn.Linear(8, 8) for _ in range(20)]).eval(),
        nn.Sequential(*[ResidualBlock() for _ in range(6)]).eval(),
    ):
        trace = tl.trace(model, torch.randn(2, 8))
        schedule = trace.collapse_schedule()
        previous: frozenset = frozenset()
        for step in schedule.steps:
            assert step.collapsed_addresses >= previous
            previous = step.collapsed_addresses


def test_child_segment_range_label_honest():
    """Name-noncontiguous members must not be labeled as a numeric interval."""

    contiguous = _child_segment_label(("blocks.0", "blocks.1", "blocks.2"), 9, 0, 0)
    assert "blocks.0-2" in contiguous
    gapped = _child_segment_label(("blocks.0", "blocks.2", "blocks.4"), 9, 0, 0)
    assert "blocks.0-4" not in gapped
    assert "3 blocks" in gapped
    backwards = _child_segment_label(("stage.5", "stage.3", "stage.1"), 9, 0, 0)
    assert "stage.5-1" not in backwards
    assert "3 blocks" in backwards
    long_gapped = _child_segment_label(
        ("blocks.0", "blocks.2", "blocks.4", "blocks.6", "blocks.8"), 15, 0, 0
    )
    assert "blocks.0-8" not in long_gapped
    assert "5 blocks" in long_gapped


def test_child_segment_descriptor_counts_concrete_ops():
    """Segment descriptors must count every hidden concrete op, not base labels."""

    model = UnevenRecurrent().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    analysis = analyze_collapse(trace)
    expected = len(analysis.signals["first"].subtree_ops) + len(
        analysis.signals["second"].subtree_ops
    )
    covered = _child_segment_covered_ops(analysis, ("first", "second"))
    assert len(covered) == expected
    descriptor = _make_child_segment_descriptor(
        trace, RenderContext(), ("first", "second"), covered
    )
    assert descriptor.num_ops + descriptor.num_buffers == expected
    assert f"{expected} ops" in descriptor.label


def test_fold_ellipsis_discloses_hidden_calls():
    """Repeat-fold ellipses must disclose hidden call mass beyond addresses."""

    model = UnevenRecurrent().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    fold = _make_run_fold(trace, ("first", "second"))
    label = _run_fold_ellipsis_label(fold)
    assert label.startswith(f"... +{fold.multiplicity - 1} more ")
    hidden_calls = sum(
        int(getattr(trace.modules[address], "num_calls", 1) or 1) for address in fold.addresses[1:]
    )
    assert hidden_calls == 5
    assert "5 calls" in label

    single_model = nn.Sequential(*[ResidualBlock() for _ in range(4)]).eval()
    single_trace = tl.trace(single_model, torch.randn(2, 8))
    single_fold = _make_run_fold(single_trace, ("0", "1", "2", "3"))
    single_label = _run_fold_ellipsis_label(single_fold)
    assert single_label == f"... +3 more {single_fold.class_name}"


def test_weighted_result_cache_not_order_dependent():
    """Weighted plan selection must not be poisoned by earlier weighted calls."""

    model = nn.Sequential(*[ResidualBlock() for _ in range(6)]).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    prefer_small = OptimizerWeights(w_k=100.0)
    prefer_large = OptimizerWeights(w_k=-100.0)

    def fingerprint(result):
        return (result.visible_count, tuple(sorted(result.selected)))

    _RESULT_CACHE.pop(trace, None)
    fresh_small = fingerprint(select_collapse_plan(trace, context, prefer_small))
    _RESULT_CACHE.pop(trace, None)
    fresh_large = fingerprint(select_collapse_plan(trace, context, prefer_large))
    assert fresh_small != fresh_large

    _RESULT_CACHE.pop(trace, None)
    first = fingerprint(select_collapse_plan(trace, context, prefer_small))
    second = fingerprint(select_collapse_plan(trace, context, prefer_large))
    assert (first, second) == (fresh_small, fresh_large)

    _RESULT_CACHE.pop(trace, None)
    first = fingerprint(select_collapse_plan(trace, context, prefer_large))
    second = fingerprint(select_collapse_plan(trace, context, prefer_small))
    assert (first, second) == (fresh_large, fresh_small)


def test_at_zero_agrees_with_select_level_zero():
    """``at(0.0)`` and ``select_collapse_level(0.0)`` must both report no collapse."""

    layers = []
    for _ in range(12):
        layers += [nn.Linear(8, 8), nn.ReLU()]
    model = nn.Sequential(*layers).eval()
    trace = tl.trace(model, torch.randn(2, 8))
    schedule = trace.collapse_schedule()
    full = schedule.steps[0].visible_count
    step = schedule.at(0.0)
    assert step.visible_count == full
    assert step.collapsed_addresses == frozenset()
    level = select_collapse_level(trace, RenderContext(), 0.0)
    assert level.visible_count == full
    assert level.selected == frozenset()
    assert level.selected == step.collapsed_addresses


_DETERMINISM_SNIPPET = textwrap.dedent(
    """
    import hashlib, warnings
    warnings.filterwarnings("ignore")
    import torch, torch.nn as nn
    import torchlens as tl

    torch.manual_seed(0)

    class Blk(nn.Module):
        def __init__(s, w=8):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())

        def forward(s, x):
            return s.net(x) + x

    class DC(nn.Module):
        def __init__(s, depth):
            super().__init__()
            s.blocks = nn.ModuleList([Blk() for _ in range(depth)])

        def forward(s, x):
            for b in s.blocks:
                x = b(x)
                x = b(x)
            return x

    m = DC(3).eval()
    log = tl.trace(m, torch.randn(2, 8))
    pieces = [
        repr(log.collapse_plan(mode="max")),
        repr(log.collapse_plan(mode="auto")),
        repr(log.module_collapse_order),
        repr([(s.t, s.visible_count, tuple(sorted(s.collapsed_addresses)))
              for s in log.collapse_schedule().steps]),
    ]
    print(hashlib.sha256("\\n".join(pieces).encode()).hexdigest())
    """
)


@pytest.mark.heavy
def test_collapse_determinism_across_hashseed():
    """Plans, orders, and schedules must be byte-identical across hash seeds."""

    digests = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SNIPPET],
            capture_output=True,
            text=True,
            env={
                "PYTHONHASHSEED": seed,
                "PATH": "/usr/bin:/bin",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": ":".join(sys.path),
            },
            check=True,
        )
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, f"hash-seed dependent collapse output: {digests}"
