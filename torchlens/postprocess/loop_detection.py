"""Step 7: Adapt Trace state to the backend-neutral recurrence grouper.

Production loop detection lives in :mod:`loop_grouping_adapter`. This module
retains only the Trace adapter, assignment application, and the lightweight
shared-parameter grouping used when full recurrence detection is disabled.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _group_by_shared_params(self: "Trace") -> None:
    """Group repeated uses of the same parameterized function.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.

    Notes
    -----
    Operations without parameters remain individual single-pass layers. The
    helper sets ``_layer_label_raw``, ``recurrent_ops``, ``pass_index``, and
    ``num_passes`` on every retained op.
    """

    param_barcode_groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for label in self._raw_layer_labels_list:
        node = self[label]
        if getattr(node, "is_orphan", False):
            continue
        if node.uses_params and node._param_barcodes:
            key = (node.func_name, tuple(sorted(node._param_barcodes)))
            param_barcode_groups[key].append(label)

    for members in param_barcode_groups.values():
        if len(members) <= 1:
            continue
        leader = min(members, key=lambda label: self[label].raw_index)
        leader_raw = self[leader]._layer_label_raw
        for label in members:
            self[label]._layer_label_raw = leader_raw

    _rebuild_pass_assignments(self)


def _detect_and_label_loops(self: "Trace") -> None:
    """Delegate recurrence grouping to the backend-neutral implementation.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.
    """

    grouping_graph = _build_recurrence_grouping_graph(self)
    assignments = group_recurrent_nodes(grouping_graph)
    _apply_recurrence_assignments(self, assignments)


def _build_recurrence_grouping_graph(self: "Trace") -> RecurrenceGroupingGraph:
    """Build the backend-neutral recurrence graph from Trace postprocess state.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.

    Returns
    -------
    RecurrenceGroupingGraph
        Neutral graph containing only the fields the shared grouper needs.
    """

    nodes: dict[str, RecurrenceNode] = {}
    eligible_labels: list[str] = []
    raw_labels = tuple(self._raw_layer_labels_list)
    raw_label_set = set(raw_labels)

    for label in raw_labels:
        node = self[label]
        is_pruned = bool(getattr(node, "is_orphan", False))
        retain = not is_pruned
        if retain:
            eligible_labels.append(label)
        nodes[label] = RecurrenceNode(
            label=label,
            raw_order=node.raw_index,
            equivalence_key=node.equivalence_class,
            equivalent_labels=tuple(node.equivalent_ops),
            data_parents=tuple(parent for parent in node.parents if parent in raw_label_set),
            data_children=tuple(child for child in node.children if child in raw_label_set),
            layer_label=node._layer_label_raw,
            recurrent_labels=tuple(node.recurrent_ops),
            uses_params=bool(node.uses_params),
            func_name=node.func_name,
            param_barcodes=tuple(node._param_barcodes),
            retain=retain,
            pruned=is_pruned,
            recurrence_anchored=(
                bool(getattr(node, "modules", None)) or bool(getattr(node, "is_buffer", False))
            ),
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=raw_labels,
        source_labels=tuple(self.input_layers + self.internal_source_ops),
        eligible_labels=tuple(eligible_labels),
    )


def _apply_recurrence_assignments(
    self: "Trace",
    assignments: dict[str, RecurrenceAssignment],
) -> None:
    """Apply neutral recurrence assignments back onto Trace ops.

    Parameters
    ----------
    self:
        Trace currently running Step 7 postprocessing.
    assignments:
        Neutral assignments returned by ``group_recurrent_nodes``.
    """

    for label, assignment in assignments.items():
        node = self[label]
        node._layer_label_raw = assignment.layer_label
        node.recurrent_ops = list(assignment.recurrent_labels)
        node.pass_index = assignment.pass_index
        node.num_passes = assignment.num_passes
        node.equivalence_class = assignment.equivalence_key


def _rebuild_pass_assignments(self: "Trace") -> None:
    """Rebuild recurrence membership and pass numbers from authoritative labels.

    Parameters
    ----------
    self:
        Trace whose ``_layer_label_raw`` values define recurrence groups.
    """

    groups: dict[str, list[str]] = defaultdict(list)
    for entry in self:
        if getattr(entry, "is_orphan", False):
            continue
        groups[entry._layer_label_raw].append(entry._label_raw)

    for members in groups.values():
        members_sorted = sorted(members, key=lambda label: self[label].raw_index)
        for pass_index, member_label in enumerate(members_sorted, start=1):
            member = self[member_label]
            member.recurrent_ops = members_sorted
            member.pass_index = pass_index
            member.num_passes = len(members_sorted)
