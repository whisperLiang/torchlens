"""Last-use schedules for generated-eager Torch split segments."""

from __future__ import annotations

from .graph import SplitTraceGraph, iter_replay_value_refs


def release_schedule(
    graph: SplitTraceGraph,
    node_ids: frozenset[str],
    retained_ids: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    """Plan value-reference releases after each executed call, preserving aliases.

    Parameters
    ----------
    graph:
        Normalized graph whose multi-output calls execute once as a group.
    node_ids:
        Nodes executed by this segment.
    retained_ids:
        Boundary or final-output values needed after the segment finishes.

    Returns
    -------
    dict
        First node of each executed call mapped to values whose last use is
        that call. Only Python references are released: views and autograd
        retain their own storage and saved tensors normally.
    """

    steps: list[str] = []
    last_use: dict[str, int] = {}
    executed: set[str] = set()
    for node in graph.nodes:
        if node.canonical_id not in node_ids or node.is_input or node.is_output:
            continue
        source = node.is_buffer or (node.target is None and not node.parents)
        call = None if source else graph.replay_call_by_output_id.get(node.canonical_id)
        call_id = node.canonical_id if call is None else call.call_id
        if call_id in executed:
            continue
        executed.add(call_id)
        index = len(steps)
        steps.append(node.canonical_id)
        members = (
            (node,)
            if call is None
            else tuple(
                graph.node_by_id[node_id] for node_id in call.output_node_ids if node_id in node_ids
            )
        )
        for member in members:
            last_use.setdefault(member.canonical_id, index)
            # Include both normalized templates and graph parents: metadata
            # edges may conservatively retain more, but must never drop a
            # dependency carried only in a nested argument template.
            parents = (
                *member.parents,
                *(ref.value_id for ref in iter_replay_value_refs(member.args_template)),
                *(ref.value_id for ref in iter_replay_value_refs(member.kwargs_template)),
            )
            for parent in parents:
                parent_id = graph.node_id_by_alias.get(parent, parent)
                last_use[parent_id] = index

    releases: dict[str, list[str]] = {}
    for value_id, index in last_use.items():
        if value_id not in retained_ids:
            releases.setdefault(steps[index], []).append(value_id)
    return {node_id: tuple(values) for node_id, values in releases.items()}
