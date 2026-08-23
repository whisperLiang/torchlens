"""Sibling-ordering resolution and verification for the forward DOT pipeline.

Extracted verbatim from ``_render_dot.py`` (renderer-thinning pass): these
helpers decide whether sibling ordering is in scope for a render, verify the
injected rank chains against a plain-layout baseline, and rewrite the DOT
source with the surviving chains. They consume :class:`RenderIR`-level
ordering constraints and never touch Trace state.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import TYPE_CHECKING, cast

from . import _render_utils
from ._render_common import (
    _SIBLING_ORDER_WARNING_EMITTED,
    SIBLING_ORDER_EPSILON,
    SIBLING_ORDER_NODE_CAP,
    SIBLING_ORDER_STRETCH_CAP,
    CapturedForwardEdge,
    CollapseFn,
    PlainLayout,
    SiblingOrderChain,
    SiblingOrderDecision,
    strict_collapse_checks_enabled,
)
from ._render_flow import (
    _assert_sibling_backstops,
    _filter_sibling_chains_to_rendered_nodes,
    _flow_span,
    _inject_sibling_rank_groups,
)

if TYPE_CHECKING:
    from typing import Any

    from .._literals import VisInterventionModeLiteral
    from ..data_classes.module import Module
    from .render_ir import RenderIROrderingConstraint


def _should_order_siblings(
    *,
    order_siblings: bool,
    engine: str,
    vis_mode: str,
    num_nodes: int,
    module: Module | str | None,
    vis_intervention_mode: VisInterventionModeLiteral,
    collapse_fn: CollapseFn | None,
    vis_call_depth: int,
) -> bool:
    """Return whether sibling ordering is in scope for this render."""

    return (
        order_siblings
        and engine == "dot"
        and vis_mode == "unrolled"
        and num_nodes <= SIBLING_ORDER_NODE_CAP
        and module is None
        and vis_intervention_mode == "node_mark"
        and vis_call_depth >= 1000
    )


def _queue_sibling_rank_group(
    module_edge_dict: dict[str, Any],
    top_level_rank_groups: list[SiblingOrderChain],
    chain: SiblingOrderChain,
) -> None:
    """Queue a sibling rank group in the cluster dictionary."""

    if chain.lca_key == -1:
        top_level_rank_groups.append(chain)
    else:
        module_edge_dict[cast(str, chain.lca_key)]["rank_groups"].append(chain)


def _verify_and_apply_sibling_ordering(
    source: str,
    chains: tuple[SiblingOrderChain | RenderIROrderingConstraint, ...],
    captured_edges: list[CapturedForwardEdge],
    rankdir: str,
) -> tuple[str, SiblingOrderDecision]:
    """Verify sibling rank chains and return final DOT source."""

    baseline_source = _strip_sibling_rank_groups(source)
    baseline = _layout_dot_plain(baseline_source, rankdir, captured_edges)
    chains = _filter_sibling_chains_to_rendered_nodes(
        cast(tuple[SiblingOrderChain, ...], chains), baseline.nodes
    )
    if not chains:
        return baseline_source, _sibling_order_decision((), (), {})
    injected = _layout_dot_plain(source, rankdir, captured_edges)
    _assert_sibling_backstops(baseline, injected, chains, captured_edges)

    ratios = {
        _sibling_chain_key(chain): _sibling_chain_stretch_ratio(
            chain, captured_edges, baseline, injected
        )
        for chain in chains
    }
    survivors = tuple(
        chain for chain in chains if ratios[_sibling_chain_key(chain)] <= SIBLING_ORDER_STRETCH_CAP
    )
    current_source = (
        source if survivors == chains else _inject_sibling_rank_groups(baseline_source, survivors)
    )
    current_layout = (
        injected
        if survivors == chains
        else _layout_dot_plain(current_source, rankdir, captured_edges)
    )

    for _ in range(2):
        bad_chains = tuple(
            chain
            for chain in survivors
            if _sibling_chain_stretch_ratio(chain, captured_edges, baseline, current_layout)
            > SIBLING_ORDER_STRETCH_CAP
        )
        if not bad_chains:
            return current_source, _sibling_order_decision(chains, survivors, ratios)
        survivors = tuple(chain for chain in survivors if chain not in bad_chains)
        current_source = _inject_sibling_rank_groups(baseline_source, survivors)
        current_layout = _layout_dot_plain(current_source, rankdir, captured_edges)
    return current_source, _sibling_order_decision(chains, survivors, ratios)


# r-b7 R42-9: one shared TORCHLENS_COLLAPSE_STRICT parser (_render_common).
_strict_sibling_order_checks_enabled = strict_collapse_checks_enabled


def _warn_sibling_order_fallback_once(exc: BaseException) -> None:
    """Warn once when sibling-order verification is skipped in production.

    Parameters
    ----------
    exc:
        Verification failure that triggered the fallback.
    """

    global _SIBLING_ORDER_WARNING_EMITTED
    if _SIBLING_ORDER_WARNING_EMITTED:
        return
    _SIBLING_ORDER_WARNING_EMITTED = True
    warnings.warn(
        "Sibling-order verification failed; rendering without the optional sibling-order "
        f"post-pass. ({type(exc).__name__}: {exc})",
        RuntimeWarning,
        stacklevel=3,
    )


def _sibling_chain_key(chain: SiblingOrderChain) -> tuple[str, tuple[str, ...]]:
    """Return a stable key for decision reporting."""

    return chain.source_name, chain.targets


def _sibling_order_decision(
    chains: tuple[SiblingOrderChain, ...],
    survivors: tuple[SiblingOrderChain, ...],
    ratios: dict[tuple[str, tuple[str, ...]], float],
) -> SiblingOrderDecision:
    """Build a sibling-order decision record."""

    return SiblingOrderDecision(
        candidate_count=len(chains),
        survivor_count=len(survivors),
        ratios=ratios,
        surviving_keys=tuple(_sibling_chain_key(chain) for chain in survivors),
    )


def _layout_dot_plain(
    source: str,
    rankdir: str,
    captured_edges: list[CapturedForwardEdge],
) -> PlainLayout:
    """Run ``dot -Tplain`` and parse coordinates and real-edge spans."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    with tempfile.NamedTemporaryFile(
        "w", suffix=".dot", delete=False, encoding="utf-8"
    ) as source_file:
        source_file.write(source)
        source_path = source_file.name
    try:
        proc = _render_utils.run_bounded_subprocess(
            ["dot", "-Tplain", source_path],
            text=True,
            timeout=120,
        )
    finally:
        os.remove(source_path)

    nodes: dict[str, tuple[float, float]] = {}
    pending_edges: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4:
            nodes[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            edge_key = (parts[1], parts[2])
            if edge_key in real_edges:
                pending_edges.append(edge_key)

    edge_spans: dict[tuple[str, str], float] = {}
    for edge_key in pending_edges:
        if edge_key[0] in nodes and edge_key[1] in nodes:
            edge_spans[edge_key] = _flow_span(nodes[edge_key[0]], nodes[edge_key[1]], rankdir)
    return PlainLayout(nodes=nodes, edge_spans=edge_spans)


def _sibling_chain_stretch_ratio(
    chain: SiblingOrderChain,
    captured_edges: list[CapturedForwardEdge],
    baseline: PlainLayout,
    candidate: PlainLayout,
) -> float:
    """Return the local incident-edge stretch ratio for ``chain``."""

    local_nodes = {chain.source_name, *chain.targets}
    ratios: list[float] = []
    for edge in captured_edges:
        edge_key = (edge.tail_name, edge.head_name)
        if edge.tail_name not in local_nodes and edge.head_name not in local_nodes:
            continue
        if edge_key not in baseline.edge_spans or edge_key not in candidate.edge_spans:
            continue
        ratios.append(
            candidate.edge_spans[edge_key]
            / max(SIBLING_ORDER_EPSILON, baseline.edge_spans[edge_key])
        )
    return max(ratios, default=1.0)


def _strip_sibling_rank_groups(source: str) -> str:
    """Remove TorchLens sibling-order rank-group blocks from DOT source."""

    lines = source.splitlines()
    stripped: list[str] = []
    skipping = False
    for line in lines:
        if "tl:sibling-order:start" in line:
            skipping = True
            continue
        if "tl:sibling-order:end" in line:
            skipping = False
            continue
        if not skipping:
            stripped.append(line)
    return "\n".join(stripped) + "\n"


__all__ = [
    "_layout_dot_plain",
    "_queue_sibling_rank_group",
    "_should_order_siblings",
    "_sibling_chain_key",
    "_sibling_chain_stretch_ratio",
    "_sibling_order_decision",
    "_strict_sibling_order_checks_enabled",
    "_strip_sibling_rank_groups",
    "_verify_and_apply_sibling_ordering",
    "_warn_sibling_order_fallback_once",
]
