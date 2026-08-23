"""Rendered-edge multiplicity disclosure (r19): dedupe registry helpers.

Split out of ``_render_edges.py`` under the R43 file-size ratchet. When the
visual dedupe merges genuinely distinct dataflow edges, the emitted edge
gains an ``xN`` label instead of silently collapsing them.
"""

from typing import Any

import graphviz

__all__ = ["_bump_deduped_edge_multiplicity", "_register_deduped_edge"]


def _register_deduped_edge(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    visual_dedupe_key: tuple[Any, ...],
    raw_edge_identity: tuple[str, str],
    edge_dict: dict[str, Any],
    body_index: int | None,
) -> None:
    """Record one emitted rendered edge for later multiplicity disclosure.

    Parameters
    ----------
    registry:
        Cross-call registry owned by the render entrypoint, or ``None``.
    visual_dedupe_key:
        The rendered-edge visual identity the edge was emitted under.
    raw_edge_identity:
        ``(source_layer_label, target_layer_label)`` of the underlying
        dataflow edge.
    edge_dict:
        The emitted edge attributes (queued dicts stay mutable in
        ``module_edge_dict``).
    body_index:
        Index of the emitted statement in ``graphviz_graph.body`` for
        directly-emitted edges, or ``None`` for cluster-queued edges.
    """

    if registry is None:
        return
    registry[visual_dedupe_key] = {
        "identities": {raw_edge_identity},
        "edge_dict": edge_dict,
        "body_index": body_index,
        "base_label": edge_dict.get("label"),
    }


def _bump_deduped_edge_multiplicity(
    registry: dict[tuple[Any, ...], dict[str, Any]] | None,
    visual_dedupe_key: tuple[Any, ...],
    raw_edge_identity: tuple[str, str],
    graphviz_graph: graphviz.Digraph,
) -> None:
    """Disclose multiplicity when distinct dataflow edges merge in a render.

    r19 (b6-fable carried LOW, 3rd round): a collapsed module returning TWO
    tensors both consumed by one exterior op rendered as ONE unlabeled edge
    — the visual dedupe silently merged genuinely distinct dataflow edges
    with no multiplicity disclosure. When a deduped edge's underlying
    ``(source, target)`` identity is NEW (not a repeat occurrence of the
    same logical edge), the emitted edge gains an ``xN`` label. Queued
    cluster edges are updated in place; directly-emitted edges are rewritten
    in ``graphviz_graph.body`` at their recorded index, so statement order —
    and therefore DOT byte determinism for undisclosed renders — is
    unchanged.
    """

    if registry is None:
        return
    entry = registry.get(visual_dedupe_key)
    if entry is None or raw_edge_identity in entry["identities"]:
        return
    entry["identities"].add(raw_edge_identity)
    count = len(entry["identities"])
    base_label = entry["base_label"]
    entry["edge_dict"]["label"] = f"{base_label} x{count}" if base_label else f"x{count}"
    body_index = entry["body_index"]
    if body_index is None:
        return
    calls = getattr(graphviz_graph, "calls", None)
    if calls is not None:
        from .render_ir import RenderIRDotStatement

        calls[body_index] = RenderIRDotStatement("edge", (), tuple(entry["edge_dict"].items()))
    else:
        rewrite = graphviz.Digraph()
        rewrite.edge(**entry["edge_dict"])
        graphviz_graph.body[body_index] = rewrite.body[-1]
