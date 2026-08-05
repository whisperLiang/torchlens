"""Backend-neutral recurrence grouping model and service.

This module contains the graph-only inputs needed by Step 7 loop grouping. Backend
finishers are responsible for adapting their postprocess state into this model,
passing only data-flow edges in ``data_parents`` and ``data_children``.
"""

import heapq
import itertools as it
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Tuple


FrontierNodes = OrderedDict[str, dict[str, deque[str]]]

# Minimum number of operations a parameter-free loop body must span before adjacent
# repetitions are grouped as a recurrent loop. Raising the bar from 1 to 2 kills the
# "two adjacent same-op nodes are a 2-pass loop" false positive (e.g. y = tanh(x);
# z = tanh(y)) without touching parameter-based recurrence (RNNs, weight tying), which
# is decided by shared parameters rather than body size.
_MIN_PARAM_FREE_LOOP_BODY_OPS = 2


@dataclass(frozen=True)
class RecurrenceNode:
    """Graph-only node input for recurrence grouping.

    Parameters
    ----------
    label:
        Stable backend-local node label.
    raw_order:
        Capture order used for deterministic traversal and pass ordering.
    equivalence_key:
        Backend-provided structural key used for isomorphic matching.
    equivalent_labels:
        Backend-provided candidate labels considered equivalent enough to seed
        an expansion round. Torch supplies its existing ``equivalent_ops`` set.
    data_parents:
        Parent labels connected by value/data edges only.
    data_children:
        Child labels connected by value/data edges only.
    layer_label:
        Current raw layer label assignment for this node.
    recurrent_labels:
        Current recurrent-op assignment, if any, used by conservative merge guards.
    uses_params:
        Whether this operation uses learned parameters.
    func_name:
        Function name used for same-function plus same-params merging.
    param_barcodes:
        Stable parameter identifiers used for same-params merging.
    retain:
        Whether the node should be retained in grouping.
    pruned:
        Whether the node has been pruned from the user-visible graph.
    """

    label: str
    raw_order: int
    equivalence_key: str
    equivalent_labels: tuple[str, ...]
    data_parents: tuple[str, ...]
    data_children: tuple[str, ...]
    layer_label: str
    recurrent_labels: tuple[str, ...]
    uses_params: bool
    func_name: str
    param_barcodes: tuple[str, ...]
    retain: bool = True
    pruned: bool = False
    output_slot: Optional[int] = None
    """Zero-based output slot for multi-output operations, ``None`` for single-output.

    Co-outputs of ONE call (``h, c = lstm_cell(...)``) occupy distinct output slots.
    Distinct slots of the same call are distinct layers, never sequential passes of
    each other: an N-step ``nn.LSTMCell`` loop is one N-pass h-layer plus one N-pass
    c-layer, mirroring how ``torch.max`` values/indices already split. Same-function,
    same-parameter merging must therefore never fuse nodes across output slots."""
    module_site: Optional[tuple[str, ...]] = None
    """Module ADDRESS stack (addresses only, no pass numbers), or ``None`` if unknown.

    Exact parameter identity must never override module identity: two DISTINCT modules
    that deliberately share a weight tensor (a tied ``encoder``/``decoder`` pair) are
    two semantic sites, not two passes of one recurrent layer. A genuinely reused
    module (ALBERT-style, one ``nn.Module`` called N times) keeps ONE address across
    calls and still groups. ``None`` (backends that do not supply the field) preserves
    the historical parameter-only behavior."""
    arg_signature: Optional[str] = None
    """Structural fingerprint of the call's NON-TENSOR arguments, or ``None`` if unknown.

    Same-parameter identity must not override call semantics: two ``F.conv2d`` calls
    sharing one kernel but differing in ``padding``/``stride``/``dilation``/``groups``
    are different operations, never recurrent passes of one layer. The fingerprint
    deliberately excludes tensor arguments AND their shapes so genuine variable-length
    recurrence (a loop whose activations shrink each step) keeps one signature across
    passes. ``None`` preserves the historical behavior."""
    recurrence_anchored: bool = False
    """Whether this op has a reused persistent identity that anchors genuine recurrence.

    Anchored ops are calls of a named submodule (non-empty ``modules``) or stateful buffer
    nodes (``is_buffer``). Repeating such an identity -- one ``nn.ReLU`` called four times, a
    buffer rewritten each iteration -- is real recurrence even when the body is a single op,
    so the param-free minimum-body-size guard does not apply to them. Bare functional ops
    (``torch.tanh(x)`` in the parent forward) are not anchored and must clear the body-size
    bar to be grouped, which rejects the ``y = tanh(x); z = tanh(y)`` false positive."""


@dataclass(frozen=True)
class RecurrenceGroupingGraph:
    """Backend-neutral graph input for recurrence grouping.

    Parameters
    ----------
    nodes:
        Mapping from node label to recurrence node data.
    raw_labels:
        Labels in raw capture order.
    source_labels:
        Input and internally initialized source labels used to seed traversal.
    eligible_labels:
        Labels considered by grouping. Backends may retain pruned nodes in
        ``nodes`` for diagnostics, but only eligible labels participate.
    """

    nodes: Mapping[str, RecurrenceNode]
    raw_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    eligible_labels: tuple[str, ...]


@dataclass(frozen=True)
class RecurrenceAssignment:
    """Computed recurrence assignment for one eligible node.

    Parameters
    ----------
    layer_label:
        Raw layer label leader assigned to this node.
    recurrent_labels:
        Raw labels in this layer, sorted by raw capture order.
    pass_index:
        One-indexed pass number within ``recurrent_labels``.
    num_passes:
        Number of passes represented by this layer.
    equivalence_key:
        Canonical structural key assigned to the grouped nodes.
    """

    layer_label: str
    recurrent_labels: tuple[str, ...]
    pass_index: int
    num_passes: int
    equivalence_key: str


_ParamCallIdentity = Tuple[
    str,
    Tuple[str, ...],
    Optional[int],
    Optional[Tuple[str, ...]],
    Optional[str],
]


@dataclass
class _MutableRecurrenceNode:
    """Mutable working copy of a neutral recurrence node."""

    label: str
    raw_order: int
    equivalence_key: str
    equivalent_labels: tuple[str, ...]
    data_parents: tuple[str, ...]
    data_children: tuple[str, ...]
    layer_label: str
    recurrent_labels: list[str]
    uses_params: bool
    func_name: str
    param_barcodes: tuple[str, ...]
    output_slot: Optional[int] = None
    recurrence_anchored: bool = False
    module_site: Optional[tuple[str, ...]] = None
    arg_signature: Optional[str] = None


@dataclass
class SubgraphInfo:
    """Track nodes belonging to one isomorphic subgraph.

    Parameters
    ----------
    starting_node:
        Label of the equivalent operation that anchors this subgraph.
    param_nodes:
        Labels in this subgraph that use learned parameters.
    node_set:
        Labels already assigned to this subgraph.
    """

    starting_node: str
    param_nodes: Set[str] = field(default_factory=set)
    node_set: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Register the starting node in this subgraph."""
        self.node_set.add(self.starting_node)


@dataclass
class IsomorphicExpansionState:
    """Mutable state threaded through BFS isomorphic-subgraph expansion."""

    iso_node_groups: OrderedDict[str, list[str]]
    node_to_iso_leader: OrderedDict[str, str]
    subgraph_info: Dict[str, SubgraphInfo]
    node_to_subgraph: Dict[str, SubgraphInfo]
    adjacent_subgraphs: Dict[str, set[str]]
    node_stack: deque[list[str]]


@dataclass
class _GroupingWorkspace:
    """Mutable recurrence grouping workspace built from a neutral graph."""

    nodes: dict[str, _MutableRecurrenceNode]
    raw_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    eligible_labels: set[str]
    _param_contexts: Optional[dict[str, frozenset[_ParamCallIdentity]]] = None

    @classmethod
    def from_graph(
        cls: type["_GroupingWorkspace"],
        graph: RecurrenceGroupingGraph,
    ) -> "_GroupingWorkspace":
        """Build a mutable workspace from a neutral recurrence graph.

        Parameters
        ----------
        graph:
            Backend-neutral recurrence graph.

        Returns
        -------
        _GroupingWorkspace
            Mutable grouping workspace.
        """
        eligible = set(graph.eligible_labels)
        nodes = {
            label: _MutableRecurrenceNode(
                label=node.label,
                raw_order=node.raw_order,
                equivalence_key=node.equivalence_key,
                equivalent_labels=tuple(node.equivalent_labels),
                data_parents=tuple(node.data_parents),
                data_children=tuple(node.data_children),
                layer_label=node.layer_label,
                recurrent_labels=list(node.recurrent_labels),
                uses_params=node.uses_params,
                func_name=node.func_name,
                param_barcodes=tuple(node.param_barcodes),
                output_slot=node.output_slot,
                recurrence_anchored=node.recurrence_anchored,
                module_site=node.module_site,
                arg_signature=node.arg_signature,
            )
            for label, node in graph.nodes.items()
            if label in eligible and node.retain and not node.pruned
        }
        return cls(
            nodes=nodes,
            raw_labels=tuple(label for label in graph.raw_labels if label in nodes),
            source_labels=tuple(label for label in graph.source_labels if label in nodes),
            eligible_labels=set(nodes),
        )

    def equivalent_labels(self, label: str) -> tuple[str, ...]:
        """Return eligible equivalent labels for ``label``.

        Parameters
        ----------
        label:
            Node label to inspect.

        Returns
        -------
        tuple[str, ...]
            Equivalent labels present in the grouping workspace.
        """
        node = self.nodes[label]
        return tuple(equiv for equiv in node.equivalent_labels if equiv in self.nodes)

    def param_contexts(self) -> dict[str, frozenset[_ParamCallIdentity]]:
        """Return each node's nearest-parameter-ancestor call-identity context.

        A node's parametric context is the union, over its data parents, of the
        parent's own SITE-QUALIFIED call identity (:func:`_param_call_identity`:
        function, parameter barcodes, output slot, module site, and non-tensor arg
        signature) when the parent is parameterized, else the parent's context. It
        identifies WHICH parametric loop a parameter-free op sits inside: a
        ``tanh`` fed by parameterized site A carries context ``{A}``, one fed by
        site B carries ``{B}``. Two bare functional ops whose nonempty contexts
        DIFFER AT ALL provably sit after different parent topologies and must not
        be merged as recurrent passes of one layer -- disjointness is not
        required: with a shared module reused in both loops next to
        site-specific halves, the contexts ``{shared, enc}`` and
        ``{shared, dec}`` overlap yet still mark different loops. The sole
        honest unequal case, the not-yet-saturated loop-entry call, is
        re-admitted through :func:`_entry_adoption_pairs`.

        The context element must be the full call identity, never the bare
        parameter barcodes: two DISTINCT weight-tied modules (a tied
        ``encoder``/``decoder`` pair) and one kernel applied under different
        non-tensor args share barcodes yet are different semantic sites, so a bare
        functional interior (``torch.tanh``) between their loops would otherwise
        inherit one indistinguishable context and merge ACROSS the loop boundary --
        yielding a 5-pass layer bridging the 3-pass and 2-pass parameterized
        layers that the disjoint-context veto exists to keep apart. A genuinely
        reused module (ALBERT-style, one ``nn.Module`` called in both loops) keeps
        ONE identity across calls, so its interiors still share a context and still
        merge.

        The computation is a single pass over capture order (parents precede
        children in a captured DAG) and depends only on set-valued inputs, so it is
        invariant to sibling capture order.

        Returns
        -------
        dict[str, frozenset[_ParamCallIdentity]]
            Parameterized-ancestor call-identity context keyed by node label.
            Cached after the first call.
        """
        if self._param_contexts is not None:
            return self._param_contexts
        contexts: dict[str, frozenset[_ParamCallIdentity]] = {}
        for label in self.raw_labels:
            node = self.nodes[label]
            accumulated: set[_ParamCallIdentity] = set()
            for parent in node.data_parents:
                parent_node = self.nodes.get(parent)
                if parent_node is None:
                    continue
                if parent_node.uses_params and parent_node.param_barcodes:
                    accumulated.add(_param_call_identity(parent_node))
                else:
                    accumulated.update(contexts.get(parent, frozenset()))
            contexts[label] = frozenset(accumulated)
        self._param_contexts = contexts
        return contexts

    def assignments(self) -> dict[str, RecurrenceAssignment]:
        """Return computed assignments for every eligible node.

        Returns
        -------
        dict[str, RecurrenceAssignment]
            Assignments keyed by node label.
        """
        return {
            label: RecurrenceAssignment(
                layer_label=node.layer_label,
                recurrent_labels=tuple(node.recurrent_labels),
                pass_index=index + 1,
                num_passes=len(node.recurrent_labels),
                equivalence_key=node.equivalence_key,
            )
            for label, node in self.nodes.items()
            for index, recurrent_label in enumerate(node.recurrent_labels)
            if recurrent_label == label
        }


def group_recurrent_nodes(graph: RecurrenceGroupingGraph) -> dict[str, RecurrenceAssignment]:
    """Group recurrent nodes in a backend-neutral graph.

    Parameters
    ----------
    graph:
        Backend-neutral recurrence graph.

    Returns
    -------
    dict[str, RecurrenceAssignment]
        Final recurrence assignments keyed by eligible node label.
    """
    workspace = _GroupingWorkspace.from_graph(graph)
    _detect_and_label_workspace_loops(workspace)
    return workspace.assignments()


def _detect_and_label_workspace_loops(workspace: _GroupingWorkspace) -> None:
    """Detect loops and assign recurrence groups in a mutable workspace.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    None
        Mutates ``workspace``.
    """
    sort_keys = {label: workspace.nodes[label].raw_order for label in workspace.raw_labels}
    node_heap = [(sort_keys[label], label) for label in workspace.source_labels]
    heapq.heapify(node_heap)
    heap_seen = set(workspace.source_labels)
    equivalence_keys_seen: set[str] = set()

    while node_heap:
        _, node_label = heapq.heappop(node_heap)
        node = workspace.nodes[node_label]
        node_equivalence_key = node.equivalence_key

        if node_equivalence_key in equivalence_keys_seen:
            continue
        equivalence_keys_seen.add(node_equivalence_key)

        equivalent_labels = workspace.equivalent_labels(node_label)
        for equiv_label in equivalent_labels:
            for child in workspace.nodes[equiv_label].data_children:
                if child not in heap_seen and child in workspace.nodes:
                    heap_seen.add(child)
                    heapq.heappush(node_heap, (sort_keys[child], child))

        if len(equivalent_labels) == 1:
            node.recurrent_labels = [node_label]
            continue

        if len(equivalent_labels) == len(node.recurrent_labels):
            continue

        _expand_isomorphic_subgraphs(workspace, node_label)

    _rebuild_pass_assignments(workspace)


def _rebuild_pass_assignments(workspace: _GroupingWorkspace) -> None:
    """Rebuild recurrent labels and pass counts from authoritative layer labels.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    None
        Mutates each node's recurrent label assignment.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for label in workspace.raw_labels:
        node = workspace.nodes[label]
        groups[node.layer_label].append(label)

    for members in groups.values():
        members_sorted = sorted(members, key=lambda label: workspace.nodes[label].raw_order)
        for member_label in members_sorted:
            workspace.nodes[member_label].recurrent_labels = members_sorted


def _expand_isomorphic_subgraphs(workspace: _GroupingWorkspace, node_label: str) -> None:
    """Expand isomorphic subgraphs from one equivalent operation group.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node_label:
        Label whose equivalent group seeds expansion.

    Returns
    -------
    None
        Mutates ``workspace`` assignments.
    """
    node = workspace.nodes[node_label]
    equivalent_operation_starting_labels = sorted(workspace.equivalent_labels(node_label))
    if not equivalent_operation_starting_labels:
        return

    sg_info: dict[str, SubgraphInfo] = {}
    for starting_label in equivalent_operation_starting_labels:
        sg_info[starting_label] = SubgraphInfo(starting_node=starting_label)
        if node.uses_params:
            sg_info[starting_label].param_nodes.add(starting_label)

    state = IsomorphicExpansionState(
        iso_node_groups=OrderedDict(
            {equivalent_operation_starting_labels[0]: equivalent_operation_starting_labels}
        ),
        node_to_iso_leader=OrderedDict(
            {
                label: equivalent_operation_starting_labels[0]
                for label in equivalent_operation_starting_labels
            }
        ),
        subgraph_info=sg_info,
        node_to_subgraph=OrderedDict(
            {label: sg_info[label] for label in equivalent_operation_starting_labels}
        ),
        adjacent_subgraphs={},
        node_stack=deque([equivalent_operation_starting_labels[:]]),
    )

    is_first_node = True
    while state.node_stack:
        isomorphic_nodes = sorted(state.node_stack.popleft())
        if len(isomorphic_nodes) == 1:
            continue
        _advance_bfs_frontier(workspace, isomorphic_nodes, state, is_first_node)
        is_first_node = False

    _refine_iso_groups(workspace, state)
    _finalize_layer_assignments(workspace, state)


def _refine_iso_groups(
    workspace: _GroupingWorkspace,
    state: IsomorphicExpansionState,
) -> None:
    """Split iso groups whose members do not share directional neighbor signatures.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state``.
    """
    for group_leader, members in list(state.iso_node_groups.items()):
        if len(members) <= 1:
            continue

        member_neighbor_isos: dict[str, set[tuple[str, str]]] = {}
        for member_label in members:
            member_node = workspace.nodes[member_label]
            neighbor_groups: set[tuple[str, str]] = set()
            for child in member_node.data_children:
                if child in state.node_to_iso_leader:
                    neighbor_groups.add(("child", state.node_to_iso_leader[child]))
            for parent in member_node.data_parents:
                if parent in state.node_to_iso_leader:
                    neighbor_groups.add(("parent", state.node_to_iso_leader[parent]))
            member_neighbor_isos[member_label] = neighbor_groups

        uf_parent = {member: member for member in members}

        def find(x: str, uf: dict[str, str] = uf_parent) -> str:
            """Return the union-find root for a group member."""
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(x: str, y: str, uf: dict[str, str] = uf_parent) -> None:
            """Merge the union-find sets for two group members."""
            rx, ry = find(x), find(y)
            if rx != ry:
                uf[rx] = ry

        reverse_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        for member_label in members:
            for neighbor_key in member_neighbor_isos[member_label]:
                reverse_index[neighbor_key].append(member_label)
        for members_with_key in reverse_index.values():
            if len(members_with_key) > 1:
                first = members_with_key[0]
                for other in members_with_key[1:]:
                    union(first, other)

        # Directly chained ANCHORED members are consecutive passes of one reused
        # persistent identity (one ``nn.ReLU`` module applied to its own output) and
        # must survive refinement together. Without this, a group of exactly two has
        # asymmetric endpoint signatures (one member only feeds the group, the other
        # is only fed by it), gets split into singletons here, and the singletons
        # produce zero merge candidates downstream -- so the documented
        # ``recurrence_anchored`` bypass never ran for its canonical two-call case.
        # Groups of three or more survive because middle members carry both
        # directional signatures. Bare functional chains (``tanh(tanh(x))``) are not
        # anchored and still split, preserving the straight-chain false-positive
        # guard.
        member_set = set(members)
        for member_label in members:
            member_node = workspace.nodes[member_label]
            if not member_node.recurrence_anchored:
                continue
            for neighbor in member_node.data_children:
                if neighbor in member_set and workspace.nodes[neighbor].recurrence_anchored:
                    union(member_label, neighbor)

        components: dict[str, list[str]] = defaultdict(list)
        for member in members:
            components[find(member)].append(member)

        if len(components) <= 1:
            continue

        del state.iso_node_groups[group_leader]
        for comp_members in components.values():
            sorted_members = sorted(comp_members)
            new_leader = sorted_members[0]
            state.iso_node_groups[new_leader] = sorted_members
            for member in sorted_members:
                state.node_to_iso_leader[member] = new_leader


def _advance_bfs_frontier(
    workspace: _GroupingWorkspace,
    current_iso_nodes: list[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> None:
    """Process one BFS frontier step.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    current_iso_nodes:
        Labels occupying the same structural position across subgraphs.
    state:
        Mutable isomorphic expansion state.
    is_first_node:
        Whether this is the first expansion step from the starting nodes.

    Returns
    -------
    None
        Mutates ``state``.
    """
    frontier_nodes = _collect_frontier_and_detect_adjacency(
        workspace,
        current_iso_nodes,
        state,
        is_first_node,
    )

    while True:
        (
            candidate_node_label,
            candidate_node_neighbor_type,
            candidate_node_subgraph,
        ) = _pop_frontier_node(frontier_nodes)
        if candidate_node_label is None:
            break

        if candidate_node_label in state.node_to_subgraph:
            # The candidate was absorbed into another subgraph after this frontier
            # was collected (a loop-carried value: child of body ``i``, parent of
            # body ``i + 1``). It cannot be matched again, but its presence on this
            # subgraph's frontier is exactly the evidence that consecutive
            # iterations are directly adjacent -- record that instead.
            _record_frontier_adjacency(
                candidate_node_subgraph,  # type: ignore[arg-type]
                candidate_node_label,
                state,
            )
            continue

        new_equivalent_nodes = _find_isomorphic_matches(
            workspace,
            candidate_node_label,
            candidate_node_neighbor_type,  # type: ignore[arg-type]
            candidate_node_subgraph,  # type: ignore[arg-type]
            frontier_nodes,
        )

        _register_isomorphic_group(workspace, new_equivalent_nodes, state)


def _collect_frontier_and_detect_adjacency(
    workspace: _GroupingWorkspace,
    current_iso_nodes: list[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> FrontierNodes:
    """Collect frontier nodes and detect inter-subgraph adjacency.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    current_iso_nodes:
        Labels occupying the same structural position across subgraphs.
    state:
        Mutable isomorphic expansion state.
    is_first_node:
        Whether to expand only children.

    Returns
    -------
    FrontierNodes
        Candidate frontier nodes by subgraph and neighbor direction.
    """
    node_types_to_use = ["children"] if is_first_node else ["children", "parents"]
    frontier_nodes: FrontierNodes = OrderedDict()

    for node_label in current_iso_nodes:
        node = workspace.nodes[node_label]
        node_subgraph = state.node_to_subgraph[node_label]
        node_subgraph_label = node_subgraph.starting_node
        subgraph_successor_nodes: dict[str, deque[str]] = {
            "children": deque(),
            "parents": deque(),
        }
        added_neighbors: set[str] = set()
        for node_type in node_types_to_use:
            neighbor_labels = node.data_children if node_type == "children" else node.data_parents
            for neighbor_label in neighbor_labels:
                if neighbor_label not in workspace.nodes:
                    continue
                if neighbor_label in node_subgraph.node_set:
                    continue
                if neighbor_label in state.node_to_subgraph:
                    _record_subgraph_adjacency(node_label, neighbor_label, state)
                elif neighbor_label not in added_neighbors:
                    subgraph_successor_nodes[node_type].append(neighbor_label)
                    added_neighbors.add(neighbor_label)
        frontier_nodes[node_subgraph_label] = subgraph_successor_nodes

    _canonicalize_frontier(workspace, frontier_nodes)
    return frontier_nodes


def _canonicalize_frontier(
    workspace: _GroupingWorkspace,
    frontier_nodes: FrontierNodes,
) -> None:
    """Make frontier contents a canonical function of the graph, not capture order.

    Grouping must be a well-defined function of the captured DAG: two mathematically
    identical models differing only in independent-sibling statement order (or a
    cross-backend feed ordering children differently) must produce the SAME layer
    partition. Two canonicalizations enforce this:

    * Every frontier deque is sorted by ``raw_order``, so which same-key neighbor an
      isomorphic match absorbs no longer depends on the incidental order of a node's
      ``data_children`` tuple.
    * A label appearing in the SAME direction of MORE THAN ONE subgraph's frontier is
      a single node shared between candidate loop bodies at the same relative
      position. One node cannot be a per-iteration isomorphic copy in two bodies at
      once, so such labels are dropped from that direction instead of being greedily
      absorbed by whichever subgraph popped first (the previous behavior, which let
      sibling capture order fabricate or miss loops and inflate the body-size
      guard). A label shared across OPPOSITE directions is kept: a loop-carried
      value is simultaneously a child of body ``i`` and a parent of body ``i + 1``,
      and dropping it would sever the chain that makes consecutive iterations
      adjacent (the pop path records that adjacency instead).

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    None
        Mutates ``frontier_nodes`` in place.
    """
    for direction in ("children", "parents"):
        label_counts: Counter[str] = Counter()
        for direction_buckets in frontier_nodes.values():
            label_counts.update(direction_buckets[direction])
        shared_labels = {label for label, count in label_counts.items() if count > 1}
        for direction_buckets in frontier_nodes.values():
            bucket = direction_buckets[direction]
            kept = [label for label in bucket if label not in shared_labels]
            kept.sort(key=lambda label: workspace.nodes[label].raw_order)
            direction_buckets[direction] = deque(kept)


def _record_subgraph_adjacency(
    node_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Mark two subgraphs as DIRECTLY adjacent in the expansion state.

    Adjacency is pairwise, never transitive. Consecutive iterations of one loop are
    directly adjacent and still chain into a single layer through union-find in
    :func:`_merge_iso_groups_to_layers`; recording transitively merged adjacency
    SETS (the previous behavior) instead let two subgraphs many hops apart -- e.g.
    the first ``tanh`` of loop A and the last ``tanh`` of a chained but distinct
    loop B -- count as "adjacent" and merge into one incoherent recurrent layer.

    Parameters
    ----------
    node_label:
        Current node label.
    neighbor_label:
        Neighbor label already assigned to another subgraph.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state.adjacent_subgraphs``.
    """
    node_subgraph_label = state.node_to_subgraph[node_label].starting_node
    _record_frontier_adjacency(node_subgraph_label, neighbor_label, state)


def _record_frontier_adjacency(
    subgraph_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Record direct adjacency between a subgraph and a neighbor's subgraph.

    Parameters
    ----------
    subgraph_label:
        Starting-node label of the subgraph whose frontier met the neighbor.
    neighbor_label:
        Neighbor label already assigned to a subgraph.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state.adjacent_subgraphs``.
    """
    node_subgraph = state.subgraph_info[subgraph_label]
    neighbor_subgraph_label = state.node_to_subgraph[neighbor_label].starting_node
    if neighbor_subgraph_label == subgraph_label:
        return

    neighbor_iso_group = state.node_to_iso_leader[neighbor_label]
    nodes_isomorphic_to_neighbor_node = state.iso_node_groups[neighbor_iso_group]
    if len(node_subgraph.node_set.intersection(nodes_isomorphic_to_neighbor_node)) == 0:
        return

    adj = state.adjacent_subgraphs
    adj.setdefault(subgraph_label, set()).add(neighbor_subgraph_label)
    adj.setdefault(neighbor_subgraph_label, set()).add(subgraph_label)


def _pop_frontier_node(
    frontier_nodes: FrontierNodes,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pop the next frontier candidate for isomorphic matching.

    Parameters
    ----------
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Candidate label, neighbor type, and subgraph label, or all ``None``.

    Notes
    -----
    Iteration is direction-major (every subgraph's children before any parents) so
    that a loop-carried value shared between a child frontier and a parent frontier
    is deterministically absorbed through its child position first; its parent-side
    appearance then records iteration adjacency at pop time.
    """
    for neighbor_type, subgraph_label in it.product(["children", "parents"], frontier_nodes):
        subgraph_neighbors = frontier_nodes[subgraph_label][neighbor_type]
        if len(subgraph_neighbors) > 0:
            candidate_node_label = subgraph_neighbors.popleft()
            return candidate_node_label, neighbor_type, subgraph_label
    return None, None, None


def _find_isomorphic_matches(
    workspace: _GroupingWorkspace,
    candidate_node_label: str,
    candidate_node_neighbor_type: str,
    candidate_node_subgraph: str,
    frontier_nodes: FrontierNodes,
) -> list[tuple[str, str]]:
    """Find candidate-equivalent nodes across other subgraph frontiers.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    candidate_node_label:
        Candidate node label from one subgraph.
    candidate_node_neighbor_type:
        Neighbor direction, ``"children"`` or ``"parents"``.
    candidate_node_subgraph:
        Starting-node label for the candidate's subgraph.
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    list[tuple[str, str]]
        Matched ``(node_label, subgraph_label)`` pairs.
    """
    candidate_node = workspace.nodes[candidate_node_label]
    candidate_node_equivalence_key = candidate_node.equivalence_key
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in frontier_nodes:
        if subgraph_label == candidate_node_subgraph:
            continue
        other_subgraph_nodes = frontier_nodes[subgraph_label][candidate_node_neighbor_type]
        for comparison_index, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = workspace.nodes[comparison_node_label]
            if comparison_node.equivalence_key == candidate_node_equivalence_key:
                del other_subgraph_nodes[comparison_index]
                new_equivalent_nodes.append((comparison_node_label, subgraph_label))
                break
    new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda item: item[0])

    seen_labels: set[str] = set()
    dupe_labels: set[str] = set()
    for node in new_equivalent_nodes:
        if node[0] in seen_labels:
            dupe_labels.add(node[0])
        seen_labels.add(node[0])
    if dupe_labels:
        new_equivalent_nodes = [node for node in new_equivalent_nodes if node[0] not in dupe_labels]
    return new_equivalent_nodes


def _register_isomorphic_group(
    workspace: _GroupingWorkspace,
    new_isomorphic_nodes: list[tuple[str, str]],
    state: IsomorphicExpansionState,
) -> None:
    """Register a newly discovered iso group.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    new_isomorphic_nodes:
        ``(node_label, subgraph_label)`` pairs in the new group.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state``.
    """
    if len(new_isomorphic_nodes) == 0:
        return
    iso_group_label = new_isomorphic_nodes[0][0]
    equivalent_node_labels = [tup[0] for tup in new_isomorphic_nodes]
    state.iso_node_groups[iso_group_label] = equivalent_node_labels[:]
    for node_label in equivalent_node_labels:
        state.node_to_iso_leader[node_label] = iso_group_label
    for node_label, node_subgraph in new_isomorphic_nodes:
        node = workspace.nodes[node_label]
        state.subgraph_info[node_subgraph].node_set.add(node_label)
        if node.uses_params:
            state.subgraph_info[node_subgraph].param_nodes.add(node_label)
        state.node_to_subgraph[node_label] = state.subgraph_info[node_subgraph]
    state.node_stack.append(equivalent_node_labels)


def _finalize_layer_assignments(
    workspace: _GroupingWorkspace,
    state: IsomorphicExpansionState,
) -> None:
    """Assign same-layer labels from iso groups, parameters, and adjacency.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``workspace`` assignments.
    """
    merged_layer_groups = _merge_iso_groups_to_layers(
        workspace, state.iso_node_groups, state.node_to_subgraph, state.adjacent_subgraphs
    )

    for layer_label, layer_nodes_set in merged_layer_groups.items():
        layer_nodes = sorted(
            list(layer_nodes_set), key=lambda layer: workspace.nodes[layer].raw_order
        )
        if len(layer_nodes) < max(
            [len(workspace.nodes[layer].recurrent_labels) for layer in layer_nodes]
        ):
            continue
        canonical_equiv_type = workspace.nodes[layer_nodes[0]].equivalence_key
        for pass_index, grouped_node_label in enumerate(layer_nodes):
            node = workspace.nodes[grouped_node_label]
            node.layer_label = layer_label
            node.recurrent_labels = layer_nodes
            node.equivalence_key = canonical_equiv_type


def _param_free_adjacency_merge_allowed(
    workspace: "_GroupingWorkspace",
    node1_label: str,
    node2_label: str,
    subgraph_a: SubgraphInfo,
    subgraph_b: SubgraphInfo,
) -> bool:
    """Return whether an adjacency-only, param-free merge is a real loop.

    Two structurally isomorphic subgraphs that are adjacent but share no parameters are only
    a genuine recurrent loop when EITHER:

    * a seed op is recurrence-anchored -- a reused submodule call (one ``nn.ReLU`` invoked
      four times) or a stateful buffer node (a buffer rewritten each iteration) -- whose
      repeated persistent identity is real recurrence even with a single-op body; or
    * the repeated body spans at least ``_MIN_PARAM_FREE_LOOP_BODY_OPS`` operations.

    This rejects the historical false positive where two bare adjacent same-op calls in the
    parent forward -- ``y = tanh(x); z = tanh(y)`` -- were grouped as a two-pass loop despite
    being a straight-line chain. Parameter-based merges (RNNs, weight tying) never reach this
    guard; they are decided by shared parameters and are left untouched.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace, used to read the seed ops' module binding.
    node1_label:
        Seed op label of the first subgraph.
    node2_label:
        Seed op label of the second subgraph.
    subgraph_a:
        One candidate loop-body subgraph.
    subgraph_b:
        The structurally isomorphic partner subgraph.

    Returns
    -------
    bool
        ``True`` when the two subgraphs may be merged into a recurrent group.
    """
    if (
        workspace.nodes[node1_label].recurrence_anchored
        or workspace.nodes[node2_label].recurrence_anchored
    ):
        return True
    smallest_body = min(len(subgraph_a.node_set), len(subgraph_b.node_set))
    return smallest_body >= _MIN_PARAM_FREE_LOOP_BODY_OPS


def _seed_reaches(
    workspace: _GroupingWorkspace,
    node1_label: str,
    node2_label: str,
    memo: dict[tuple[str, str], bool],
) -> bool:
    """Return whether one seed reaches the other along directed data edges.

    Capture order is a topological order (parents precede children), so
    reachability is only possible from the earlier-captured seed to the later one,
    and the search window is bounded by the later seed's ``raw_order``.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node1_label:
        First seed label.
    node2_label:
        Second seed label.
    memo:
        Per-merge cache of resolved reachability queries.

    Returns
    -------
    bool
        ``True`` when a directed data path connects the two seeds.
    """
    src_label, dst_label = node1_label, node2_label
    if workspace.nodes[src_label].raw_order > workspace.nodes[dst_label].raw_order:
        src_label, dst_label = dst_label, src_label
    key = (src_label, dst_label)
    cached = memo.get(key)
    if cached is not None:
        return cached
    dst_order = workspace.nodes[dst_label].raw_order
    stack = [src_label]
    seen = {src_label}
    found = False
    while stack:
        current = stack.pop()
        if current == dst_label:
            found = True
            break
        for child in workspace.nodes[current].data_children:
            child_node = workspace.nodes.get(child)
            if child_node is None or child in seen or child_node.raw_order > dst_order:
                continue
            seen.add(child)
            stack.append(child)
    memo[key] = found
    return found


def _param_call_identity(node: _MutableRecurrenceNode) -> _ParamCallIdentity:
    """Return the full call identity two parameterized ops must share to be one layer.

    Recurrent passes of ONE layer are repeated executions of the SAME call: same
    function, same parameters, same output slot, same module address, and same
    non-tensor structural arguments. Exact parameter identity alone must not
    override the other axes:

    * ``module_site`` -- two DISTINCT modules deliberately sharing a weight tensor
      (tied ``encoder``/``decoder``) are two semantic sites, never a false 2-pass
      recurrent layer. A genuinely reused module keeps one address and still groups.
    * ``arg_signature`` -- one kernel applied with ``padding=0`` and ``padding=1``
      is two different operations with different output structure, not recurrence.

    ``None`` values (backends that do not supply the newer fields) compare equal to
    each other, preserving the historical parameter-only behavior for those feeds.

    Parameters
    ----------
    node:
        Parameterized workspace node.

    Returns
    -------
    _ParamCallIdentity
        Hashable identity tuple.
    """

    return (
        node.func_name,
        tuple(sorted(node.param_barcodes)),
        node.output_slot,
        node.module_site,
        node.arg_signature,
    )


def _param_free_flow_reachable(
    workspace: _GroupingWorkspace,
    src_label: str,
    max_order: int,
) -> set[str]:
    """Return labels reachable from ``src_label`` along parameter-free data paths.

    Traversal follows ``data_children`` edges but never enters a parameterized
    node -- the same ``uses_params and param_barcodes`` predicate that cuts
    :meth:`_GroupingWorkspace.param_contexts` propagation. Membership in the
    result therefore certifies that the source's parametric context FLOWED to
    the target along the wire: for every returned label ``m``,
    ``context(m) >= context(src_label)`` holds by construction of the context
    recurrence. Containment that instead arises from RE-CALLING a shared
    parameterized identity behind a parametric cut (a second loop re-invoking a
    shared module on the first loop's output) is deliberately NOT certified --
    that is a loop-boundary crossing, not context flow.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    src_label:
        Parameter-free source node label.
    max_order:
        Raw-order upper bound; children past every candidate target are pruned.

    Returns
    -------
    set[str]
        Parameter-free labels reachable from ``src_label`` without crossing a
        parameterized node.
    """
    reachable: set[str] = set()
    seen: set[str] = {src_label}
    stack: list[str] = [src_label]
    while stack:
        current = stack.pop()
        for child in workspace.nodes[current].data_children:
            child_node = workspace.nodes.get(child)
            if child_node is None or child in seen or child_node.raw_order > max_order:
                continue
            seen.add(child)
            if child_node.uses_params and child_node.param_barcodes:
                continue
            reachable.add(child)
            stack.append(child)
    return reachable


def _entry_adoption_pairs(
    workspace: _GroupingWorkspace,
    iso_nodes: list[str],
    param_contexts: Dict[str, frozenset[_ParamCallIdentity]],
) -> set[tuple[str, str]]:
    """Return the loop-entry pairs exempt from the unequal-context veto.

    The veto in :func:`_merge_iso_groups_to_layers` keeps a bare param-free op
    from grouping across calls whose nearest-parameterized-ancestor contexts
    differ AT ALL. One honest case has genuinely unequal contexts: the LOOP
    ENTRY. On pass 1 a recurrent op reads state produced OUTSIDE the loop, so
    its context is missing the in-loop sites that saturate every later pass
    through the recurrent feedback wire (``h = emb(x); for: h = h + attn(h)``
    gives the first ``add`` context ``{emb, attn}`` and every later one
    ``{emb, attn, mlp, ...}``). Splitting pass 1 off would shear the entry call
    from its own loop.

    An exemption is granted only when ALL of the following hold, each one
    adversarially load-bearing:

    * ``context(entry)`` is a STRICT SUBSET of ``context(target)`` -- entry
      saturation only ever grows the context. Disjoint or crosswise-different
      contexts (two loops with different site-specific halves) never qualify.
    * The entry call is the UNIQUE carrier of its context among the iso group's
      veto-governed candidates. A multi-call context class is a recurrent site
      of its own (a 3-pass loop upstream of a 2-pass loop), never a dangling
      entry, so it must stay a separate layer.
    * The entry call reaches the target through a PARAM-FREE data path
      (:func:`_param_free_flow_reachable`), certifying the subset relation came
      from context flow along the feedback wire -- not from a second loop
      re-calling a shared parameterized identity behind a parametric cut.
    * Each entry call adopts AT MOST ONE target: the earliest reachable
      qualifying call. Union-find merges are transitive, so a pairwise
      exemption must not let one entry bridge two mutually-vetoed classes.
      With out-degree <= 1 and strictly increasing raw order, adoption edges
      form an in-forest whose components terminate at exactly one class, so
      two multi-member classes can never merge through adoptions.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    iso_nodes:
        Iso-group member labels sorted by raw capture order.
    param_contexts:
        Nearest-parameterized-ancestor contexts keyed by node label.

    Returns
    -------
    set[tuple[str, str]]
        ``(entry_label, target_label)`` pairs, entry strictly earlier in raw
        capture order than target.
    """
    classes: Dict[frozenset[_ParamCallIdentity], list[str]] = defaultdict(list)
    for label in iso_nodes:
        node = workspace.nodes[label]
        if node.uses_params or node.recurrence_anchored:
            continue
        context = param_contexts.get(label, frozenset())
        if context:
            classes[context].append(label)
    if len(classes) < 2:
        return set()
    pairs: set[tuple[str, str]] = set()
    for context, members in classes.items():
        if len(members) != 1:
            continue
        entry = members[0]
        entry_order = workspace.nodes[entry].raw_order
        candidates = [
            member
            for other_context, other_members in classes.items()
            if context < other_context
            for member in other_members
            if workspace.nodes[member].raw_order > entry_order
        ]
        if not candidates:
            continue
        max_order = max(workspace.nodes[member].raw_order for member in candidates)
        reachable = _param_free_flow_reachable(workspace, entry, max_order)
        reachable_candidates = [member for member in candidates if member in reachable]
        if not reachable_candidates:
            continue
        target = min(reachable_candidates, key=lambda member: workspace.nodes[member].raw_order)
        pairs.add((entry, target))
    return pairs


def _merge_iso_groups_to_layers(
    workspace: _GroupingWorkspace,
    iso_node_groups: Dict[str, list[str]],
    node_to_subgraph: Dict[str, SubgraphInfo],
    adjacent_subgraphs: Dict[str, set[str]],
) -> Dict[str, Set[str]]:
    """Merge iso groups into same-layer groups using union-find.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    iso_node_groups:
        Iso-group leader to member labels.
    node_to_subgraph:
        Node label to subgraph info.
    adjacent_subgraphs:
        Subgraph adjacency union-find map.

    Returns
    -------
    dict[str, set[str]]
        Merged layer groups with at least two members.
    """
    uf_parent: Dict[str, str] = {}

    def find(x: str) -> str:
        """Return the union-find root for a node label."""
        if x not in uf_parent:
            uf_parent[x] = x
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Merge two union-find sets."""
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx > ry:
                rx, ry = ry, rx
            uf_parent[ry] = rx

    all_iso_nodes: set[str] = set()
    for iso_nodes_orig in iso_node_groups.values():
        all_iso_nodes.update(iso_nodes_orig)

    sg_param_types: dict[str, frozenset[str]] = {}
    for iso_nodes_orig in iso_node_groups.values():
        for node_label in iso_nodes_orig:
            sg = node_to_subgraph[node_label]
            sg_label = sg.starting_node
            if sg_label not in sg_param_types:
                sg_param_types[sg_label] = frozenset(
                    workspace.nodes[pnode].equivalence_key for pnode in sg.param_nodes
                )

    param_contexts = workspace.param_contexts()
    reach_memo: dict[tuple[str, str], bool] = {}

    for iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(
            iso_nodes_orig, key=lambda node_label: workspace.nodes[node_label].raw_order
        )
        entry_adoptions = _entry_adoption_pairs(workspace, iso_nodes, param_contexts)
        # Consecutive pairs first: in a genuine loop they carry the unions, so the
        # full pairwise sweep afterwards short-circuits on shared union-find roots
        # instead of re-deriving (and re-checking reachability for) distant pairs.
        pair_iter = it.chain(zip(iso_nodes, iso_nodes[1:]), it.combinations(iso_nodes, 2))
        for node1_label, node2_label in pair_iter:
            if find(node1_label) == find(node2_label):
                continue
            node1_subgraph_label = node_to_subgraph[node1_label].starting_node
            node2_subgraph_label = node_to_subgraph[node2_label].starting_node
            subgraphs_are_adjacent = (
                node1_subgraph_label in adjacent_subgraphs
                and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            node1 = workspace.nodes[node1_label]
            node2 = workspace.nodes[node2_label]
            if (
                node1.uses_params
                and node2.uses_params
                and _param_call_identity(node1) != _param_call_identity(node2)
            ):
                # Two parameterized ops that differ in module address, output slot,
                # or non-tensor call structure are different semantic sites; no
                # adjacency or shared-parameter evidence may unite them as passes
                # of one recurrent layer. Identity equality is transitive, so
                # allowed unions can never chain around this veto.
                continue
            pair_anchored = node1.recurrence_anchored or node2.recurrence_anchored
            if not (node1.uses_params or node2.uses_params or pair_anchored):
                # Two bare functional ops sitting inside DIFFERENT parametric loops
                # are passes of different loops; merging them straddles the loop
                # boundary and yields incoherent pass counts (a 3-pass and a 2-pass
                # layer cannot share a 5-pass neighbor). The veto fires whenever the
                # nonempty nearest-param-ancestor contexts differ AT ALL: disjoint
                # contexts (untied loops), overlapping-but-unequal contexts (a
                # shared module reused in both loops next to site-specific halves,
                # where each bridge op must follow its OWN loop's parent topology),
                # and strict-subset contexts alike. Equality is transitive, so
                # allowed unions can never chain around the veto -- the one honest
                # unequal case, the loop-entry call whose context has not yet
                # saturated through the recurrent feedback, is re-admitted solely
                # through its precomputed adoption pair (see
                # :func:`_entry_adoption_pairs`). Anchored ops are exempt: a reused
                # module identity is real recurrence wherever its calls sit.
                context1 = param_contexts.get(node1_label, frozenset())
                context2 = param_contexts.get(node2_label, frozenset())
                if (
                    context1
                    and context2
                    and context1 != context2
                    and (node1_label, node2_label) not in entry_adoptions
                ):
                    continue
            overlapping_param_types = (
                sg_param_types[node1_subgraph_label] & sg_param_types[node2_subgraph_label]
            )
            if overlapping_param_types:
                # Shared parametric body content marks two iterations of one
                # weight-tied loop -- but only when the iterations actually CHAIN:
                # directly adjacent, or one seed feeds the other through the graph
                # (interleaved bodies, e.g. tied-linear/mul/tied-linear/log). A
                # shared weight captured inside two structurally-disjoint parallel
                # branches is NOT recurrence between the branches' ops; without the
                # connectivity requirement, independent terminal reductions on
                # parallel streams were grouped as spurious recurrent passes.
                if subgraphs_are_adjacent or _seed_reaches(
                    workspace, node1_label, node2_label, reach_memo
                ):
                    union(node1_label, node2_label)
            elif subgraphs_are_adjacent and _param_free_adjacency_merge_allowed(
                workspace,
                node1_label,
                node2_label,
                node_to_subgraph[node1_label],
                node_to_subgraph[node2_label],
            ):
                union(node1_label, node2_label)

    param_barcode_groups: dict[_ParamCallIdentity, list[str]] = defaultdict(list)
    for node_label in all_iso_nodes:
        node = workspace.nodes[node_label]
        if node.uses_params and node.param_barcodes:
            # The FULL call identity is the key, not bare parameter identity. The
            # output slot is part of it: co-outputs of one call (h and c of an
            # LSTMCell) share function name and parameters but are DISTINCT layers,
            # not sequential passes of each other (omitting the slot doubled
            # ``num_passes`` for every multi-output recurrent cell). The module
            # address and non-tensor arg signature are equally part of it: tied
            # distinct modules and same-kernel/different-padding convolutions were
            # unconditionally unioned here into false recurrent layers.
            param_barcode_groups[_param_call_identity(node)].append(node_label)

    for _identity_key, nodes_with_same_params in param_barcode_groups.items():
        if len(nodes_with_same_params) > 1:
            first = nodes_with_same_params[0]
            for other in nodes_with_same_params[1:]:
                union(first, other)

    merged_layer_groups: Dict[str, Set[str]] = defaultdict(set)
    for node_label in all_iso_nodes:
        root = find(node_label)
        merged_layer_groups[root].add(node_label)

    return {leader: nodes for leader, nodes in merged_layer_groups.items() if len(nodes) > 1}


__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
]
