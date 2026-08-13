"""Trace consistency and graph topology invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import (
        _SPECIAL_LIST_FLAG_PAIRS,
        MetadataInvariantError,
    )

__all__ = (
    "_check_trace_self_consistency",
    "_retained_orphan_computational_count",
    "_retained_orphan_op_labels",
    "_retained_orphan_layer_labels",
    "_check_special_layer_lists",
    "_check_capture_edge_survival",
    "_check_graph_topology",
)


def _check_trace_self_consistency(ml: Trace) -> None:
    """Check A: Trace aggregate counts and metadata are internally consistent.

    Validates:
    - layer_labels length matches layer_list length, no duplicates.
    - num_ops == count of computational (non-input, non-output,
      non-buffer) layers.
    - Param counts (total, trainable, frozen) are consistent and sum correctly.
      Uses deduplication by layer_label to match labeling.py logic.
    - At least one output layer exists.
    - Timing values are non-negative and ordered.
    - Tensor counts: total >= saved.
    """
    name = "trace_self_consistency"

    # op_labels vs layer_list length
    if len(ml.op_labels) != len(ml.layer_list):
        raise MetadataInvariantError(
            name,
            f"len(op_labels)={len(ml.op_labels)} != len(layer_list)={len(ml.layer_list)}",
        )

    # No duplicate labels
    if len(ml.op_labels) != len(set(ml.op_labels)):
        dupes = [lbl for lbl in ml.op_labels if ml.op_labels.count(lbl) > 1]
        raise MetadataInvariantError(name, f"Duplicate op_labels: {set(dupes)}")

    # num_ops counts computational layers only (excludes input, output,
    # buffer).  We check per-layer flags instead of comparing against label
    # sets because buffer_layers stores pass-qualified labels while
    # layer_labels strips the pass suffix -- they use different formats.
    expected_ops = sum(
        1 for lpl in ml.layer_list if not (lpl.is_input or lpl.is_output or lpl.is_buffer)
    )
    expected_ops += _retained_orphan_computational_count(ml, name)
    if ml.num_ops != expected_ops:
        raise MetadataInvariantError(
            name,
            f"num_ops={ml.num_ops} != expected computational layers={expected_ops}",
        )

    # Param counts must be deduplicated by layer_label because
    # multi-pass layers share the same params -- counting each pass would
    # double-count.  This matches the summation logic in labeling.py:116-122.
    seen_no_pass: set[str] = set()
    expected_param_sum = 0
    expected_num_params = 0
    for lpl in ml.layer_list:
        if lpl.layer_label not in seen_no_pass:
            expected_param_sum += lpl.num_param_tensors
            expected_num_params += lpl.num_params
            seen_no_pass.add(lpl.layer_label)
    if ml.num_param_tensors != expected_param_sum:
        raise MetadataInvariantError(
            name,
            f"num_param_tensors={ml.num_param_tensors} != "
            f"sum(unique num_param_tensors)={expected_param_sum}",
        )
    if ml.num_params != expected_num_params:
        raise MetadataInvariantError(
            name,
            f"num_params={ml.num_params} != sum(unique num_params)={expected_num_params}",
        )

    if ml.num_params_trainable + ml.num_params_frozen != ml.num_params:
        raise MetadataInvariantError(
            name,
            f"trainable({ml.num_params_trainable}) + frozen({ml.num_params_frozen}) "
            f"!= total({ml.num_params})",
        )

    # At least one output layer
    if len(ml.output_layers) == 0:
        raise MetadataInvariantError(name, "No output layers found")

    # Timing
    if ml.capture_duration < 0:
        raise MetadataInvariantError(name, f"capture_duration={ml.capture_duration} < 0")
    if ml.capture_start_time > ml.capture_end_time:
        raise MetadataInvariantError(
            name,
            f"capture_start_time={ml.capture_start_time} > capture_end_time={ml.capture_end_time}",
        )

    # Tensor counts
    if ml.num_tensors < ml.num_saved_ops:
        raise MetadataInvariantError(
            name,
            f"num_tensors={ml.num_tensors} < num_saved_ops={ml.num_saved_ops}",
        )


def _retained_orphan_computational_count(ml: Trace, name: str) -> int:
    """Return retained orphan ops after proving their narrow island contract.

    Parameters
    ----------
    ml:
        Trace whose aggregate operation count is being checked.
    name:
        Invariant name to report if retained-orphan metadata is inconsistent.

    Returns
    -------
    int
        Number of retained orphan ops that have the same computational role as
        operations counted in ``layer_list``.

    Raises
    ------
    MetadataInvariantError
        If purported retained orphans are not exactly the detached island
        records established during postprocessing.
    """
    orphan_logs = tuple(getattr(ml, "_orphan_logs", ()))
    retained_orphans = tuple(log for log in orphan_logs if getattr(log, "is_orphan", False))
    if not retained_orphans:
        return 0

    if not getattr(ml, "keep_orphans", False):
        raise MetadataInvariantError(name, "retained orphan logs require keep_orphans=True")

    orphan_raw_labels = list(getattr(ml, "_orphan_labels", ()))
    orphan_log_raw_labels = [getattr(log, "_label_raw", None) for log in orphan_logs]
    if orphan_raw_labels != orphan_log_raw_labels:
        raise MetadataInvariantError(
            name,
            "_orphan_labels must exactly match _orphan_logs raw labels for retained islands",
        )
    if len(orphan_raw_labels) != len(set(orphan_raw_labels)):
        raise MetadataInvariantError(name, "_orphan_labels contains duplicate raw labels")
    if len(retained_orphans) != len(orphan_logs):
        raise MetadataInvariantError(
            name,
            "retained orphan islands must mark every _orphan_logs entry is_orphan=True",
        )

    active_labels = {layer.layer_label for layer in ml.layer_list}
    for orphan in retained_orphans:
        island_edges = set(orphan.parents) | set(orphan.children)
        active_edges = island_edges & active_labels
        if active_edges:
            raise MetadataInvariantError(
                name,
                f"Retained orphan {orphan.layer_label} has edges into active graph: {active_edges}",
            )

    return sum(
        1
        for orphan in retained_orphans
        if not (orphan.is_input or orphan.is_output or orphan.is_buffer)
    )


def _retained_orphan_op_labels(ml: Trace) -> set[str]:
    """Return final labels belonging to explicitly retained orphan operations.

    Parameters
    ----------
    ml:
        Trace containing postprocessed orphan metadata.

    Returns
    -------
    set[str]
        Labels for the exact retained-island class, which is outside the active
        ``op_labels`` projection by design.
    """
    return {
        orphan.label
        for orphan in getattr(ml, "_orphan_logs", ())
        if getattr(orphan, "is_orphan", False)
    }


def _retained_orphan_layer_labels(ml: Trace) -> set[str]:
    """Return no-pass labels belonging to explicitly retained orphan operations.

    Parameters
    ----------
    ml:
        Trace containing postprocessed orphan metadata.

    Returns
    -------
    set[str]
        No-pass labels for the same narrow retained-island class as
        :func:`_retained_orphan_op_labels`.
    """
    return {
        orphan.layer_label
        for orphan in getattr(ml, "_orphan_logs", ())
        if getattr(orphan, "is_orphan", False)
    }


def _check_special_layer_lists(ml: Trace) -> None:
    """Check B: special layer lists (input, output, buffer, etc.) match per-layer boolean flags.

    For each (list_attr, flag_attr) pair, verifies bidirectional consistency:
    - Forward: every label in the list has the flag set on its Op.
    - Reverse: every Op with the flag set appears in the list.

    Retained orphan ops are the sole exception to active ``op_labels``
    membership: they are intentionally absent from the active graph but retain
    their internal-source/sink flags in raw metadata.
    """
    name = "special_layer_lists"
    retained_orphans_by_label = {
        orphan.label: orphan
        for orphan in getattr(ml, "_orphan_logs", ())
        if getattr(orphan, "is_orphan", False)
    }
    for list_attr, flag_attr, label_kind in _SPECIAL_LIST_FLAG_PAIRS:
        special_list = getattr(ml, list_attr)
        special_set = set(special_list)
        label_set = set(ml.op_labels if label_kind == "op" else ml.layer_labels)
        label_field = "op_labels" if label_kind == "op" else "layer_labels"

        # All entries must be valid labels
        missing = special_set - label_set
        if label_kind == "op":
            missing -= _retained_orphan_op_labels(ml)
        if missing:
            raise MetadataInvariantError(
                name, f"{list_attr} contains labels not in {label_field}: {missing}"
            )

        # Forward: every label in the list has the flag set
        for label in special_list:
            lpl = (
                retained_orphans_by_label[label]
                if label in retained_orphans_by_label
                else ml[label]
            )
            if not getattr(lpl, flag_attr):
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} is in {list_attr} but {flag_attr}=False",
                )

        # Reverse: every layer/op with the flag is in the list.
        for lpl in ml.layer_list:
            label = lpl.label if label_kind == "op" else lpl.layer_label
            if getattr(lpl, flag_attr) and label not in special_set:
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} has {flag_attr}=True but is not in {list_attr}",
                )


def _check_capture_edge_survival(trace: Trace) -> None:
    """Reconcile the final graph against the sealed capture-time edge truth (r29 F3b).

    The per-op identity witness (``dropped_edge_tensor_args``) is stamped at
    CAPTURE, so an edge dropped anywhere DOWNSTREAM of it -- the 20-step
    postprocess pipeline, a later graph mutation -- was invisible whenever the
    payload was trivial (the value sweep's blind class): removing an edge from
    ``parents`` + ``parent_arg_positions`` symmetrically left every
    self-consistency check green. This invariant closes that stage gap: every
    capture-observed (slot -> producer) pair sealed on the Trace at op-record
    time (``_capture_parent_edge_truth``, keyed by raw label) must survive into
    the final op's ``parents``, unless the PRODUCER itself left the graph (an
    orphan prune / merge is visible as an absent ``raw -> final`` mapping and
    is the accounted rewrite class).

    Fail-open boundaries, deliberate and narrow: a trace with no sealed truth
    (loaded artifacts, non-exhaustive captures) has nothing to reconcile; a
    producer raw label with no final mapping is accounted as pruned; an op
    whose record carries interventions is the user's deliberate rewiring.

    Parameters
    ----------
    trace:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a capture-witnessed parent edge between two SURVIVING ops is absent
        from the final graph.
    """

    truth = trace.__dict__.get("_capture_parent_edge_truth")
    if not truth:
        return
    # Each capture CALL keeps its unique raw label on exactly one surviving
    # record, so the raw index resolves the producer's exact final spelling
    # set (pass-qualified ``label`` for multi-pass ops, plain ``layer_label``
    # otherwise) -- the many-to-one raw->layer_label map cannot.
    ops_by_raw = {
        getattr(op, "_label_raw", None): op
        for op in trace.layer_list
        if getattr(op, "_label_raw", None) is not None
    }
    for op in trace.layer_list:
        raw_label = getattr(op, "_label_raw", None)
        edges = truth.get(raw_label) if raw_label is not None else None
        if not edges:
            continue
        if getattr(op, "interventions", None):
            continue
        positions = getattr(op, "parent_arg_positions", None) or {}
        recorded = set(op.parents or ())
        recorded.update(
            label
            for domain in ("args", "kwargs")
            for label in (positions.get(domain) or {}).values()
        )
        # r33 F-1: reconcile each sealed triplet AT ITS EXACT SLOT, not by
        # label-set membership. Set membership let two corruptions through:
        # a slot PERMUTATION between value-identical surviving producers (the
        # swapped labels are both still "present somewhere"), and an argpos
        # entry DROP with parents intact (the label survives via ``parents``).
        # Positional truth for args/kwargs slots is the sealed slot itself;
        # plain-parent truth (slot ``None``) keeps the membership check.
        for arg_type, slot, parent_raw in edges:
            producer = ops_by_raw.get(parent_raw)
            if producer is None:
                continue  # producer pruned/merged out of the final graph: accounted
            spellings = {
                spelling
                for spelling in (
                    getattr(producer, "layer_label", None),
                    getattr(producer, "label", None),
                )
                if spelling is not None
            }
            if arg_type in ("args", "kwargs"):
                final_slot_label = (positions.get(arg_type) or {}).get(slot)
                if final_slot_label in spellings:
                    continue
                raise MetadataInvariantError(
                    "capture_edge_survival",
                    f"capture-witnessed parent edge {producer.layer_label!r} (raw "
                    f"{parent_raw!r}) of {op.layer_label!r} (raw {raw_label!r}) at "
                    f"{arg_type}[{slot!r}] resolves to {final_slot_label!r} in the "
                    "final graph while its producer survives -- the slot's edge was "
                    "dropped or rewired after the capture witness was stamped",
                )
            if spellings & recorded:
                continue
            raise MetadataInvariantError(
                "capture_edge_survival",
                f"capture-witnessed parent edge {producer.layer_label!r} (raw "
                f"{parent_raw!r}) of {op.layer_label!r} (raw {raw_label!r}) is absent "
                "from the final graph while its producer survives -- an edge was "
                "dropped after the capture witness was stamped",
            )


def _check_graph_topology(ml: Trace) -> None:
    """Check C: parent-child edge bidirectionality and stored-flag consistency.

    Validates:
    - Every parent edge has a corresponding child edge (and vice versa).
    - The stored has_children flag matches the actual child list.
      Note: has_children excludes output layers (added during postprocessing,
      not during capture when the flag was set).
    - Every parent_arg_positions entry references an op that is actually in
      ``parents`` (the arg map may not invent edges the graph does not have).
    - Input layers have no parents.
    - out_versions_by_child keys are a subset of children.

    Round-26 W3-3 note: this check previously also "compared"
    ``has_parents``/``has_siblings``/``has_co_parents`` against
    ``len(parents)``/``len(siblings)``/``len(co_parents)``. Those three are
    read-only ``Op`` PROPERTIES defined as exactly those length tests
    (op.py), so the comparisons were tautologies that could never fail on any
    trace -- security theater, not a tripwire. They were removed rather than
    kept; ``has_children`` is the only stored capture-time flag with
    independent information, and the ``parent_arg_positions`` cross-check
    below is a real two-independent-structures consistency test that replaces
    them.
    """
    name = "graph_topology"
    label_set = set(ml.layer_labels) | set(ml.op_labels)
    output_set = set(ml.output_layers)

    def label_aliases(entry: Any, fallback: str) -> set[str]:
        """Return stored layer/op labels without invoking fragile accessors.

        Parameters
        ----------
        entry:
            Layer-like record whose independently stored labels should be read.
        fallback:
            Lookup label used to resolve ``entry``.

        Returns
        -------
        set[str]
            Layer, lookup, and recurrence-member labels available without
            consulting ``Layer.label``. That accessor intentionally raises when
            pass-count metadata is corrupt, but invariant checks must report the
            owning corruption contract rather than leak that accessor error.
        """

        aliases = {fallback}
        layer_label = getattr(entry, "layer_label", None)
        if isinstance(layer_label, str):
            aliases.add(layer_label)
        recurrent_ops = getattr(entry, "recurrent_ops", ()) or ()
        aliases.update(label for label in recurrent_ops if isinstance(label, str))
        return aliases

    for lpl in ml.layer_list:
        label = lpl.layer_label
        lpl_aliases = label_aliases(lpl, label)

        # Parent-child bidirectionality
        for p in lpl.parents:
            parent = ml[p]
            if p not in label_set and parent.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has parent {p} not in layer_labels"
                )
            if not lpl_aliases.intersection(parent.children):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {p} as parent, but {p} does not list {label} as child",
                )

        for c in lpl.children:
            child = ml[c]
            if c not in label_set and child.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has child {c} not in layer_labels"
                )
            if not lpl_aliases.intersection(child.parents):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {c} as child, but {c} does not list {label} as parent",
                )

        # Stored-flag consistency. has_children is set during capture and does
        # not account for output layers added during postprocessing, so output
        # layers are excluded from the child count for this check. (This is the
        # ONLY stored flag with independent information; the derived
        # has_parents/has_siblings/has_co_parents properties recompute from the
        # very lists a comparison would use, so checking them is vacuous -- see
        # the round-26 W3-3 note in the docstring.)
        non_output_children = [
            c for c in lpl.children if c not in output_set and ml[c].layer_label not in output_set
        ]
        if lpl.has_children != (len(non_output_children) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_children={lpl.has_children} but "
                f"non-output children={non_output_children}",
            )

        # Arg-map/graph cross-consistency: parent_arg_positions and parents are
        # two INDEPENDENTLY stored structures, so this comparison has real
        # teeth (unlike the removed property tautologies). Every op the arg map
        # attributes an argument slot to must actually be a recorded parent;
        # an arg-map entry naming a non-parent is graph corruption. The
        # reverse direction (a data parent missing from the arg map) is
        # replay-validation's job (``_check_layer_arguments_logged_correctly``),
        # which also has value evidence for it.
        parent_alias_set = set(lpl.parents)
        for parent_label in lpl.parents:
            parent_entry = ml[parent_label]
            parent_alias_set.update(label_aliases(parent_entry, parent_label))
        for arg_domain in ("args", "kwargs"):
            for position, attributed_label in lpl.parent_arg_positions.get(arg_domain, {}).items():
                try:
                    attributed_entry = ml[attributed_label]
                except (KeyError, ValueError):
                    # A label that does not resolve at all is the
                    # ``edge_use_parent_arg`` invariant's finding ("references
                    # missing parent"); this check owns only the
                    # resolvable-but-not-a-parent inconsistency.
                    continue
                attributed_aliases = label_aliases(attributed_entry, attributed_label)
                if attributed_aliases & parent_alias_set:
                    continue
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: parent_arg_positions[{arg_domain!r}][{position!r}] "
                    f"names {attributed_label!r}, which is not a recorded parent "
                    f"(parents={lpl.parents})",
                )

        # Input layers have no parents
        if lpl.is_input and len(lpl.parents) > 0:
            raise MetadataInvariantError(
                name,
                f"Input layer {label} has parents={lpl.parents}",
            )

        # out_versions_by_child keys subset of children
        ctv_keys = set(lpl.out_versions_by_child.keys())
        child_set = set(lpl.children)
        extra = ctv_keys - child_set
        if extra:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: out_versions_by_child has keys not in children: {extra}",
            )
