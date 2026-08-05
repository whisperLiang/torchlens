"""Hardening regressions for the intelligent auto-collapse subsystem (round 21+).

Covers the round-20 adversarial-audit defect classes:

- multi-pass submodule reuse must not crash the smart-collapse surface;
- segment descriptor identity must be injective (plan/render parity);
- schedule ``collapsed_addresses`` must not report an empty set while hiding nodes;
- range/count labels must never lie about hidden mass;
- weighted optimizer results must not be cache-order dependent;
- ``CollapseSchedule.at(0.0)`` must agree with ``select_collapse_level(0.0)``;
- plans, schedules, and orders must be deterministic across hash seeds.

Round-23 seal finding C1: when auto's band-pressure branch returns an
op-segment-condensed plan, the max ladder re-condenses it, so pre-existing
segment nodes must keep their descriptors (pass-through) and the L3 auto
fallback must carry ``auto.segments`` instead of stripping them; every
max/float/schedule/draw/order surface must return honestly for that class.
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
    _condense_plan_with_child_segments,
    _make_child_segment_descriptor,
    _optimizer_total_units,
    _rendered_module_hidden_counts,
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


class MultiPassBlock(nn.Module):
    """Three-op leaf block (linear, relu, mul) reused by multi-pass fixtures."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(torch.relu(self.lin(x)), 1.0)


class SharedMultiCall(nn.Module):
    """One shared n-child module called ``calls`` times plus a linear tail.

    Reconstructs the round-22 collapse-seal H1 fixture: functional
    relu/sigmoid/tanh before every shared call, a seven-child shared body,
    and a ten-leaf tail. Under max/1.0 the selected plan contains multi-pass
    module boxes and a fold whose representative is called ``calls`` times,
    which the pre-fix optimizer counted once per ADDRESS while the renderer
    emits one box per CALL.
    """

    class Shared(nn.Module):
        def __init__(self, width: int, children: int) -> None:
            super().__init__()
            self.body = nn.Sequential(*[MultiPassBlock(width) for _ in range(children)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)

    def __init__(
        self,
        width: int = 8,
        children: int = 7,
        calls: int = 4,
        tail: int = 10,
    ) -> None:
        super().__init__()
        self.calls = calls
        self.shared = self.Shared(width, children)
        self.tail = nn.Sequential(*[nn.Linear(width, width) for _ in range(tail)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.calls):
            x = torch.relu(x)
            x = torch.sigmoid(x)
            x = torch.tanh(x)
            x = self.shared(x)
        return self.tail(x)


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


def _occurrence_witness_partition(trace, result):
    """Partition concrete PASS-QUALIFIED op occurrences into visible/hidden.

    Unlike :func:`_plan_hidden_and_visible_ops`, which witnesses whole
    addresses, this attributes every concrete pass-qualified occurrence:

    - visible: raw plan nodes, with the ``k``-th plan occurrence of a
      pass-free base label attributed to pass ``k`` (render order);
    - hidden: occurrences whose renderer-effective module stack (innermost
      dropped for atomic exits) contains an exact plan box CALL, any call of
      a fold's hidden members, or a segment member address, plus the exact
      pass-qualified ops of op-segment descriptors.

    An occurrence with no witness at all is a silently dropped node.
    """

    concrete = {str(op.label) for op in trace.ops}
    visible: set[str] = set()
    occurrence: dict[str, int] = {}
    box_calls: set[str] = set()
    fold_hidden_addresses: set[str] = set()
    child_members: set[str] = set()
    op_segment_ops: set[str] = set()
    for node in result.plan.nodes:
        if isinstance(node, RawOp):
            text = str(node.op)
            if text in concrete:
                visible.add(text)
                continue
            occurrence[text] = occurrence.get(text, 0) + 1
            qualified = f"{text}:{occurrence[text]}"
            if qualified in concrete:
                visible.add(qualified)
        elif isinstance(node, ModuleBox):
            box_calls.add(node.call)
        elif isinstance(node, RepeatFold):
            box_calls.add(node.rep.call)
            fold_hidden_addresses.update(node.members[1:])
        elif isinstance(node, ChildSegment):
            child_members.update(node.members)
    for descriptor in (result.segments or {}).values():
        if descriptor.kind == "op":
            op_segment_ops.update(str(op) for op in descriptor.ops)
        else:
            child_members.update(descriptor.members)
    hidden: set[str] = set()
    for op in trace.ops:
        label = str(op.label)
        modules = [str(call) for call in (getattr(op, "modules", ()) or ())]
        # Segment and fold absorption match the op's ORIGINAL module stack
        # (mirroring ``_segment_for_node`` / ``_run_fold_ancestor_for_node``);
        # box hiding matches the renderer-EFFECTIVE stack, where the innermost
        # module of an atomic exit op is dropped and the op stays visible.
        original_bases = {call.rsplit(":", 1)[0] for call in modules}
        effective = modules
        if getattr(op, "is_atomic_module", False) and effective:
            effective = effective[:-1]
        if (
            any(call in box_calls for call in effective)
            or original_bases & fold_hidden_addresses
            or original_bases & child_members
            or label in op_segment_ops
        ):
            hidden.add(label)
    return concrete, visible, hidden


def _multi_call_plan_boxes(trace, plan):
    """Return plan boxes and fold representatives of multi-call addresses."""

    boxes = []
    folds = []
    for node in plan.nodes:
        if isinstance(node, ModuleBox):
            address = node.call.rsplit(":", 1)[0]
        elif isinstance(node, RepeatFold):
            address = node.rep.call.rsplit(":", 1)[0]
        else:
            continue
        if address not in trace.modules:
            continue
        if int(getattr(trace.modules[address], "num_calls", 1) or 1) > 1:
            (folds if isinstance(node, RepeatFold) else boxes).append(node)
    return boxes, folds


@pytest.mark.parametrize(
    "builder, require_fold",
    [
        (lambda: SharedMultiCall(children=7, calls=4, tail=10), False),
        (lambda: SharedMultiCall(children=7, calls=2, tail=4), True),
    ],
    ids=["shared_four_calls", "folded_two_calls"],
)
def test_multipass_pass_occurrence_conservation_and_render_parity(builder, require_fold, tmp_path):
    """Multi-pass plans must count rendered CALLS and witness every occurrence.

    Round-22 seal finding H1: ``_module_box_plan_nodes`` and the folded
    instantiation emitted ``address:1`` once while the renderer collapses
    every call of a multi-pass address, so ``plan.total``, the max count
    gate, and ``collapse_schedule().steps[-1].visible_count`` under-counted
    the rendered graph and pass-2+ occurrences had no plan witness. For every
    public mode this pins:

    - ``plan.total`` equals the rendered SVG node count, and
    - the full concrete pass-qualified occurrence set equals the union of
      visible plan nodes and hidden-with-witness occurrences.
    """

    torch.manual_seed(0)
    model = builder().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()

    max_result = select_collapse_plan(trace, context, mode="max")
    assert not max_result.declined
    multi_boxes, multi_folds = _multi_call_plan_boxes(trace, max_result.plan)
    assert multi_boxes, "fixture must exercise multi-call module boxes at max"
    if require_fold:
        assert multi_folds, "fixture must exercise a multi-call fold representative at max"
        rep_call = multi_folds[0].rep.call
        rep_address = rep_call.rsplit(":", 1)[0]
        later_pass_boxes = [
            node
            for node in max_result.plan.nodes
            if isinstance(node, ModuleBox)
            and node.call.rsplit(":", 1)[0] == rep_address
            and node.call != rep_call
        ]
        assert later_pass_boxes, "multi-call fold representative must render later passes"

    schedule = trace.collapse_schedule()
    assert schedule.steps[-1].visible_count == max_result.plan.total

    for mode in ("auto", "max", 0.5, 1.0):
        if isinstance(mode, float):
            result = select_collapse_level(trace, context, mode)
        else:
            result = select_collapse_plan(trace, context, mode=mode)
        if result.declined:
            continue
        out = tmp_path / f"multipass_{str(mode).replace('.', '_')}"
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(out),
        )
        rendered = _svg_node_group_count(str(out) + ".svg")
        assert result.plan.total == rendered, (
            f"mode={mode}: plan.total={result.plan.total} but the renderer emitted {rendered} nodes"
        )
        assert result.visible_count == result.plan.total
        concrete, visible, hidden = _occurrence_witness_partition(trace, result)
        orphaned = concrete - visible - hidden
        assert not orphaned, (
            f"mode={mode}: {len(orphaned)} of {len(concrete)} concrete "
            f"pass-qualified occurrences have no visible or hidden plan "
            f"witness: {sorted(orphaned)[:8]}"
        )
        assert concrete == visible | hidden


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


class PureFunctionalLoops(nn.Module):
    """Zero-submodule functional loop: tanh; sigmoid; relu(x + 0.01) per pass.

    At ten or more loops the full graph exceeds the readable band with no
    DP-selectable module, so auto's band-pressure branch returns an
    op-segment-condensed plan (round-23 seal C1 model class).
    """

    def __init__(self, loops: int) -> None:
        super().__init__()
        self.loops = loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.loops):
            x = torch.tanh(x)
            x = torch.sigmoid(x)
            x = torch.relu(x + 0.01)
        return x


class AtomicWrapper(nn.Module):
    """Bare-linear wrapper whose single op renders as an atomic raw op."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class SharedChildFunctionalLoops(nn.Module):
    """Real submodule called once per functional loop iteration.

    The child's ops join the surrounding functional run, so auto's op
    segments span multi-pass module ops rather than pure functional labels.
    """

    def __init__(self, loops: int, width: int = 8) -> None:
        super().__init__()
        self.child = AtomicWrapper(width)
        self.loops = loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.loops):
            x = torch.tanh(x)
            x = self.child(x)
            x = torch.sigmoid(x)
            x = torch.relu(x + 0.01)
        return x


def _op_segment_condensed_trace(kind: str, loops: int):
    torch.manual_seed(0)
    model = (
        PureFunctionalLoops(loops) if kind == "functional" else SharedChildFunctionalLoops(loops)
    ).eval()
    return tl.trace(model, torch.randn(1, 8))


_SEGMENTED_AUTO_CASES = [
    ("functional", 9),
    ("functional", 10),
    ("functional", 11),
    ("functional", 12),
    ("functional", 20),
    ("shared_child", 8),
    ("shared_child", 12),
    ("shared_child", 16),
]
_SEGMENTED_AUTO_IDS = [f"{kind}_{loops}" for kind, loops in _SEGMENTED_AUTO_CASES]


@pytest.mark.parametrize("kind, loops", _SEGMENTED_AUTO_CASES, ids=_SEGMENTED_AUTO_IDS)
def test_op_segment_condensed_auto_max_ladder_surfaces(kind, loops, tmp_path):
    """Round-23 C1: every max/float/schedule/draw/order surface must be honest.

    The size range spans the segmentation boundary (nine functional loops
    keep auto un-segmented; ten push it into the band-pressure branch) and
    both the zero-submodule and the real-submodule-in-functional-runs
    variants. The pre-fix max ladder re-condensed auto's already-segmented
    plan, dropped the pre-existing ``OpSegment`` descriptors, and crashed the
    r21 parity tripwire (``AssertionError``) on all six public surfaces.
    """

    trace = _op_segment_condensed_trace(kind, loops)
    context = RenderContext()

    auto_result = select_collapse_plan(trace, context, mode="auto")
    max_result = select_collapse_plan(trace, context, mode="max")
    results = {"auto": auto_result, "max": max_result}
    for t in (0.25, 0.5, 0.75, 1.0):
        results[f"t={t}"] = select_collapse_level(trace, context, t)
    for label, result in results.items():
        assert not result.declined, f"{label}: unexpectedly declined"
        assert result.visible_count == result.plan.total, (
            f"{label}: visible_count {result.visible_count} != plan.total {result.plan.total}"
        )
        descriptors = result.segments or {}
        plan_segments = _plan_segment_node_count(result.plan)
        assert len(descriptors) == plan_segments, (
            f"{label}: descriptor cardinality {len(descriptors)} != "
            f"plan segment nodes {plan_segments}"
        )
        assert len(set(descriptors)) == len(descriptors)
    assert results["t=1.0"].plan == max_result.plan

    schedule = trace.collapse_schedule()
    assert schedule.steps[-1].visible_count == max_result.plan.total
    for step in schedule.steps:
        assert step.visible_count == step.plan.total

    order = trace.collapse_order(mode="max")
    assert isinstance(order, list)

    concrete, visible, hidden = _occurrence_witness_partition(trace, max_result)
    assert concrete == visible | hidden, (
        f"max: {len(concrete - visible - hidden)} occurrences have no plan witness"
    )

    for mode in ("max", 0.5):
        out = tmp_path / f"segauto_{kind}_{loops}_{str(mode).replace('.', '_')}"
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            collapse=mode,
            vis_outpath=str(out),
        )
        plan = trace.collapse_plan(mode)
        rendered = _svg_node_group_count(str(out) + ".svg")
        assert plan.total == rendered, (
            f"draw({mode}): plan.total {plan.total} != rendered SVG nodes {rendered}"
        )


def test_max_ladder_carries_auto_op_segment_descriptors():
    """A pass-through max plan must keep auto's descriptors byte-identical.

    For the op-segment-condensed auto class nothing further condenses, so the
    max ladder falls back to auto; the fallback must carry auto's plan AND
    auto's segment descriptors (the pre-r21 L3 fallback stripped
    ``segments={}``, a silent plan/render parity lie). The op segments must
    span multi-pass op labels, pinning the multi-pass absorption variant.
    """

    trace = _op_segment_condensed_trace("shared_child", 12)
    context = RenderContext()
    auto_result = select_collapse_plan(trace, context, mode="auto")
    max_result = select_collapse_plan(trace, context, mode="max")
    assert _plan_segment_node_count(auto_result.plan) > 0, (
        "fixture must produce an op-segment-condensed auto plan"
    )
    assert max_result.plan == auto_result.plan
    assert dict(max_result.segments or {}) == dict(auto_result.segments or {})
    assert max_result.segments, "max fallback stripped auto's segment descriptors"
    multi_pass = any(
        ":" in op and op.rsplit(":", 1)[1] not in ("", "1")
        for descriptor in max_result.segments.values()
        for op in descriptor.ops
    )
    assert multi_pass, "op segments must span multi-pass op labels"


def test_condense_pass_through_preserves_segment_descriptors():
    """Re-condensing an already-segmented plan must rebuild its descriptors.

    Feeds a max plan that carries ``ChildSegment`` nodes back through
    ``_condense_plan_with_child_segments``: pre-existing segment nodes must
    pass through verbatim with descriptors equal to the originals (the
    pre-fix loop appended them descriptor-less, firing the parity tripwire).
    """

    model = CollidingSegments().eval()
    trace = tl.trace(model, torch.randn(2, 8))
    context = RenderContext()
    max_result = select_collapse_plan(trace, context, mode="max")
    child_segments = sum(isinstance(node, ChildSegment) for node in max_result.plan.nodes)
    assert child_segments > 0, "fixture must produce child segments at max"
    analysis = analyze_collapse(trace)
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    total_units = _optimizer_total_units(trace, context)
    replan, redescriptors = _condense_plan_with_child_segments(
        trace,
        context,
        analysis,
        max_result.plan,
        hidden_counts,
        total_units,
        dominance_limit=0.75,
        k_hi=20,
    )
    assert replan.nodes == max_result.plan.nodes
    assert len(redescriptors) == _plan_segment_node_count(replan)
    assert dict(redescriptors) == dict(max_result.segments or {})
