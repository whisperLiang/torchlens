"""Distance, connectivity, lookup, and contract helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..data_classes.module import Module


if TYPE_CHECKING:
    from ..backends import BackendSpec
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantContract,
        MetadataInvariantError,
    )
    from .invariants import (
        METADATA_INVARIANT_CONTRACTS,
        _retained_orphan_layer_labels,
        _retained_orphan_op_labels,
    )

__all__ = (
    "_check_distance_invariants",
    "_op_follows_recorded_backward_trigger",
    "_consumed_unattributed_data_operand",
    "_check_graph_connectivity",
    "_check_module_containment_logic",
    "_check_lookup_key_consistency",
    "_metadata_invariant_contracts_for_trace",
    "_metadata_invariant_contracts_for_backend",
    "_metadata_invariant_applies",
)


def _check_distance_invariants(ml: "Trace") -> None:
    """Check O: distance and reachability invariants.

    Only runs when ``mark_layer_depths`` was enabled during logging.

    Validates:
    - min_distance <= max_distance for both input and output distances.
    - Input layers have distance_from_input == 0.
    - Output layers have distance_from_output == 0.
    - has_input_ancestor <-> input_ancestors is non-empty.
    - has_output_descendant <-> output_descendants is non-empty.
    - input_ancestors subset of input_layers; output_descendants subset of
      output_layers.
    """
    if not ml.mark_layer_depths:
        return

    name = "distance_invariants"
    input_set = set(ml.input_layers)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # min <= max for input distances
        if lpl.min_distance_from_input is not None and lpl.max_distance_from_input is not None:
            if lpl.min_distance_from_input > lpl.max_distance_from_input:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_from_input="
                    f"{lpl.min_distance_from_input} > max={lpl.max_distance_from_input}",
                )

        # min <= max for output distances
        if lpl.min_distance_to_output is not None and lpl.max_distance_to_output is not None:
            if lpl.min_distance_to_output > lpl.max_distance_to_output:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_to_output="
                    f"{lpl.min_distance_to_output} > max={lpl.max_distance_to_output}",
                )

        # Input layers: distance from input == 0
        if label in input_set:
            if lpl.min_distance_from_input != 0 or lpl.max_distance_from_input != 0:
                raise MetadataInvariantError(
                    name,
                    f"Input layer '{label}': distance_from_input should be 0, got "
                    f"min={lpl.min_distance_from_input}, max={lpl.max_distance_from_input}",
                )

        # Output layers: distance from output == 0
        if label in output_set:
            if lpl.min_distance_to_output != 0 or lpl.max_distance_to_output != 0:
                raise MetadataInvariantError(
                    name,
                    f"Output layer '{label}': distance_from_output should be 0, got "
                    f"min={lpl.min_distance_to_output}, max={lpl.max_distance_to_output}",
                )

        # has_input_ancestor ↔ input_ancestors non-empty
        has_ancestors = len(lpl.input_ancestors) > 0
        if lpl.has_input_ancestor != has_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_input_ancestor={lpl.has_input_ancestor} but "
                f"len(input_ancestors)={len(lpl.input_ancestors)}",
            )

        # has_output_descendant ↔ output_descendants non-empty
        has_descendents = len(lpl.output_descendants) > 0
        if lpl.has_output_descendant != has_descendents:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_output_descendant={lpl.has_output_descendant} but "
                f"len(output_descendants)={len(lpl.output_descendants)}",
            )

        # input_ancestors subset of input_layers
        extra_ancestors = lpl.input_ancestors - input_set
        if extra_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': input_ancestors contains labels not in "
                f"input_layers: {extra_ancestors}",
            )

        # output_descendants subset of output_layers
        extra_desc = lpl.output_descendants - output_set
        if extra_desc:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': output_descendants contains labels not in "
                f"output_layers: {extra_desc}",
            )


def _op_follows_recorded_backward_trigger(ml: "Trace", layer: "Op") -> bool:
    """Return whether an op ran after a RECORDED mid-forward backward trigger.

    ``torch.autograd.grad`` / ``loss.backward()`` fired mid-forward produce
    gradient tensors whose construction is legitimately untraceable, and a
    downstream op consuming one is a genuine parentless consumer of an
    unattributed tensor (e.g. MAML-style ``weight_grad.mean()``) -- NOT a
    silent capture drop. The trigger itself is positively recorded in the
    capture's backward events, so the exemption is anchored to trace-level
    evidence, not the op's self-claims: a plain capture with no recorded
    backward trigger can never use it.

    Parameters
    ----------
    ml:
        Trace being validated.
    layer:
        Parentless op under the connectivity check.

    Returns
    -------
    bool
        True when a backward trigger was recorded at a forward-op position
        before this op ran.
    """

    capture_events = getattr(ml, "_capture_events", None) or getattr(ml, "capture_events", None)
    backward_events = getattr(capture_events, "backward_events", None) or ()
    trigger_positions = [
        event.forward_op_count_at_trigger
        for event in backward_events
        if type(event).__name__ == "BackwardPassStart"
        and getattr(event, "forward_op_count_at_trigger", None) is not None
    ]
    if not trigger_positions:
        return False
    step_index = getattr(layer, "step_index", None)
    return step_index is not None and step_index > min(trigger_positions)


def _consumed_unattributed_data_operand(layer: "Op") -> bool:
    """Return whether the capture witness flagged a DATA-operand consumption.

    Parameters
    ----------
    layer:
        Layer pass whose ``unattributed_tensor_args`` witness is inspected.

    Returns
    -------
    bool
        True when the capture-side witness recorded any unattributed tensor
        argument. A runtime tensor at ANY input slot is a data dependency
        (round-31 H2) -- including schema-typed ``int``/``Scalar`` control
        slots -- so every witness position counts; the former ATen-schema
        metadata-slot exclusion is gone.
    """

    return bool(tuple(getattr(layer, "unattributed_tensor_args", ()) or ()))


def _check_graph_connectivity(ml: "Trace") -> None:
    """Check P: graph connectivity invariants.

    Validates:
    - Every non-input, non-buffer, non-internally-initialized, non-output
      layer has at least one parent (no dangling computational nodes).
    - _orphan_labels (removed during postprocessing) do NOT appear in the
      active layer_labels (they were pruned from the graph).
    - No pruned orphan raw label was minted a final label in
      ``_raw_to_final_op_labels`` (proof the prune actually took).
    - No retained orphan island label leaked into the active label sets.
    """
    name = "graph_connectivity"
    label_set = set(ml.layer_labels)
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Non-input, non-buffer, non-internally-initialized layers must have parents
        if (
            label not in input_set
            and label not in buffer_set
            and not lpl.is_internal_source
            and not lpl.is_output
            and len(lpl.parents) == 0
        ):
            raise MetadataInvariantError(
                name,
                f"Layer '{label}' has no parents but is not input, buffer, "
                f"internally initialized, or output",
            )

        # Round-26 W3-4c: the ``is_internal_source`` exemption above is only
        # legitimate for GENUINE graph sources (factories such as
        # ``torch.ones``, vmap-built masks, buffers). Capture stamps
        # ``is_internal_source = (len(parents) == 0)``, so a parentless op
        # that in fact CONSUMED an unaccounted tensor at a data-operand slot
        # (the capture-side witness recorded it in ``unattributed_tensor_args``,
        # e.g. the consumer of a silently-dropped ``torch.ops.aten.*`` call)
        # was blessed as a "source" and dodged the dangling-node check
        # entirely -- letting a silent op drop validate as ``passed``. A
        # demonstrated tensor CONSUMER is not a source; fail it. Genuine
        # sources have no tensor-data args and keep the exemption, and an op
        # consuming a known-provenance outside tensor (module attribute,
        # input, param, buffer) is never flagged by the witness in the first
        # place. Non-data (size/shape/metadata) positions are excluded via the
        # same ATen-schema classifier the capture witness uses.
        if (
            lpl.is_internal_source
            and len(lpl.parents) == 0
            and callable(getattr(lpl, "func", None))
            and label not in input_set
            and label not in buffer_set
            and not lpl.is_output
            and _consumed_unattributed_data_operand(lpl)
            # Mid-forward autograd products (torch.autograd.grad /
            # loss.backward fired inside forward) are legitimately
            # untraceable tensors, and their consumers are genuine parentless
            # sources-of-record. The exemption keys on the trace-level
            # RECORDED backward trigger (never op self-claims), so a plain
            # capture cannot use it.
            and not _op_follows_recorded_backward_trigger(ml, lpl)
        ):
            raise MetadataInvariantError(
                name,
                f"Layer '{label}' is marked is_internal_source but consumed "
                f"unattributed tensor data "
                f"(unattributed_tensor_args="
                f"{tuple(getattr(lpl, 'unattributed_tensor_args', ()) or ())}); "
                f"a tensor consumer with no recorded parents is a dangling "
                f"computational node (silent capture drop), not a graph source",
            )

    raw_orphan_in_list = set(ml._orphan_labels) & label_set
    if raw_orphan_in_list:
        raise MetadataInvariantError(
            name,
            f"_orphan_labels contains labels still in layer_labels: {raw_orphan_in_list}",
        )

    # The survival check above compares RAW labels against FINAL labels, which are
    # disjoint domains on a well-formed trace, so on its own it can only catch a
    # trace whose final labels were corrupted back into raw form. The check with
    # teeth is on the RAW side, where both operands live in the same domain: a
    # pruned orphan must never have been minted a final label. Every surviving op
    # gets exactly one ``_raw_to_final_op_labels`` entry during finalization, so a
    # pruned orphan raw label appearing as a key in that map is proof the prune
    # did not take and the node reached the final graph.
    #
    # A pruned orphan has NO final label to compare against -- its Op never
    # reaches labeling, so ``label``/``_label_raw``/``is_orphan`` are unset -- and
    # final labels are RENUMBERED over the survivors, so a raw label is NOT its
    # final label with the ``_raw`` suffix stripped. Deriving one that way is
    # actively unsafe: the survivors of
    # ``ones(); relu(ones); x + 1; * 2; relu()`` renumber such that the stripped
    # orphan raw label ``relu_1_3`` is byte-identical to a LIVE op's final label,
    # so a stripped-suffix comparison hard-fails a correct capture.
    #
    # ``keep_orphans=True`` is the ONE case where an orphan raw label legitimately
    # owns a final-label entry: the island is retained, labeled, and held outside
    # the active projection. It is excluded by checking what the raw label MAPPED
    # TO, not by disabling the check -- a retained orphan whose raw label was
    # mapped onto a LIVE final label still fires. With ``keep_orphans=False`` the
    # retained set is empty, so the check keeps its full teeth there.
    retained_orphan_labels = _retained_orphan_op_labels(ml) | _retained_orphan_layer_labels(ml)
    raw_to_final_op_labels = getattr(ml, "_raw_to_final_op_labels", {}) or {}
    orphan_raw_labels = {label for label in ml._orphan_labels if isinstance(label, str)}
    resurrected_orphans = {
        raw_label
        for raw_label in orphan_raw_labels & set(raw_to_final_op_labels)
        if raw_to_final_op_labels[raw_label] not in retained_orphan_labels
    }
    if resurrected_orphans:
        raise MetadataInvariantError(
            name,
            f"Pruned orphan raw labels were mapped to final labels: {sorted(resurrected_orphans)}",
        )

    # Retained orphan islands (``keep_orphans=True``) DO carry final labels and
    # are outside the active projection by design; one showing up in the active
    # label sets means a retained island leaked into the live graph.
    leaked_retained = retained_orphan_labels & (label_set | set(ml.op_labels))
    if leaked_retained:
        raise MetadataInvariantError(
            name,
            f"Retained orphan labels survive in active labels: {sorted(leaked_retained)}",
        )


def _check_module_containment_logic(ml: "Trace") -> None:
    """Check Q: module containment logical consistency.

    Validates:
    - Address tree is acyclic (walking address_parent chain reaches None
      without revisiting a node).
    - Root module 'self' has address_depth == 0; others have
      address_depth == addr.count('.') + 1.
    - Per-layer modules (call nesting stack):
      - Last element matches module.
      - Every element is a known module address.
      - No duplicate addresses (can't be inside the same module twice).
    """
    name = "module_containment_logic"
    mod_accessor = ml.modules

    # Build set of known module addresses
    known_addrs = set()
    for mod_log in mod_accessor:
        known_addrs.add(mod_log.address)

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Address tree acyclicity: walk address_parent to root
        visited: set[str] = set()
        current: str | None = addr
        while current is not None:
            if current in visited:
                raise MetadataInvariantError(
                    name,
                    f"Cycle in address_parent chain starting from '{addr}': revisited '{current}'",
                )
            visited.add(current)
            try:
                parent_mod: Module = mod_accessor[current]  # type: ignore[assignment]
            except (KeyError, IndexError):
                break
            current = parent_mod.address_parent

        # Address depth consistency
        if addr == "self":
            if mod_log.address_depth != 0:
                raise MetadataInvariantError(
                    name,
                    f"Root module 'self' has address_depth={mod_log.address_depth}, expected 0",
                )
        else:
            expected_depth = addr.count(".") + 1
            if mod_log.address_depth != expected_depth:
                raise MetadataInvariantError(
                    name,
                    f"Module '{addr}': address_depth={mod_log.address_depth} "
                    f"!= expected {expected_depth} (addr.count('.')+1)",
                )

    # Per-layer: modules path validity
    # Format after postprocessing: list of "addr:pass" strings, ordered from
    # outermost enclosing submodule to innermost. Does NOT include "self".
    for lpl in ml.layer_list:
        nested = lpl.modules
        if not nested:
            continue

        # Leaf consistency: last element matches module
        if lpl.module is not None:
            if nested[-1] != lpl.module:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': last nested module '{nested[-1]}' "
                    f"!= module '{lpl.module}'",
                )

        # Path validity: each element must be a known module, and no
        # duplicate addresses (a module can't appear twice in the same call
        # stack).  We do NOT check address depth ordering — address depth
        # (position in the nn.Module tree) is independent of call nesting
        # depth (position on the forward() call stack).  A module at a
        # shallow address can be called from inside a deeply-addressed
        # module's forward(), e.g., encoder.blocks.1.1.attention calling
        # encoder.attention_structure.sin_dropout.
        seen_addrs = set()
        for entry in nested:
            addr = entry.split(":")[0] if ":" in entry else entry
            if addr not in known_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': nested path contains unknown "
                    f"module address '{addr}'",
                )
            if addr in seen_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': duplicate module address "
                    f"'{addr}' in nested path {nested}",
                )
            seen_addrs.add(addr)


def _check_lookup_key_consistency(ml: "Trace") -> None:
    """Check R: lookup key bidirectional consistency.

    Validates:
    - _lookup_keys_to_layer_num_dict (forward: key->num) and
      _layer_num_to_lookup_keys_dict (reverse: num->[keys]) are consistent:
      forward[key]=num implies key in reverse[num], and vice versa.
    - _raw_to_final_layer_labels and _final_to_raw_layer_labels are inverse
      bijections.
    - All final labels in the raw->final map exist in layer_labels.
    """
    name = "lookup_key_consistency"

    # _lookup_keys_to_layer_num_dict maps key→num (last assigned wins).
    # _layer_num_to_lookup_keys_dict maps num→[keys] (accumulates all assignments).
    # Forward → reverse: for every (key, num) in forward, key must be in reverse[num].
    fwd = ml._lookup_keys_to_layer_num_dict
    rev = ml._layer_num_to_lookup_keys_dict

    for key, num in fwd.items():
        if num not in rev:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"{num} not in _layer_num_to_lookup_keys_dict",
            )
        if key not in rev[num]:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"'{key}' not in _layer_num_to_lookup_keys_dict[{num}]",
            )

    # Reverse → forward: every key in reverse must exist in forward (but may
    # point to a different num if the key was reassigned to a later layer).
    for num, keys in rev.items():
        for key in keys:
            if key not in fwd:
                raise MetadataInvariantError(
                    name,
                    f"_layer_num_to_lookup_keys_dict[{num}] has '{key}' but "
                    f"'{key}' not in _lookup_keys_to_layer_num_dict",
                )

    # _raw_to_final_layer_labels ↔ _final_to_raw_layer_labels
    raw_fwd = ml._raw_to_final_layer_labels
    raw_rev = ml._final_to_raw_layer_labels

    for raw, final in raw_fwd.items():
        if final not in raw_rev:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"'{final}' not in _final_to_raw_layer_labels",
            )
        if raw_rev[final] != raw:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"_final_to_raw_layer_labels['{final}']='{raw_rev[final]}'",
            )

    for final, raw in raw_rev.items():
        if raw not in raw_fwd:
            raise MetadataInvariantError(
                name,
                f"_final_to_raw_layer_labels['{final}']='{raw}' but "
                f"'{raw}' not in _raw_to_final_layer_labels",
            )

    # All final labels are valid lookup labels. Multi-pass raw labels map to
    # pass-qualified Op labels, while single-pass raw labels may map to Layer
    # labels for compatibility lookup.
    label_set = set(ml.layer_labels) | set(ml.op_labels) | _retained_orphan_layer_labels(ml)
    for final in raw_fwd.values():
        if final not in label_set:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels maps to '{final}' which is not a valid label",
            )


def _metadata_invariant_contracts_for_trace(
    trace: "Trace",
) -> tuple[MetadataInvariantContract, ...]:
    """Return metadata invariant contracts applicable to ``trace``.

    Parameters
    ----------
    trace:
        Trace whose backend selects the contract subset.

    Returns
    -------
    tuple[MetadataInvariantContract, ...]
        Ordered invariant contracts matching the trace backend and capability
        requirements.
    """

    from ..backends import get_backend_spec

    spec = get_backend_spec(getattr(trace, "backend", "torch"))
    torch_spec = get_backend_spec("torch")
    backend_family: Literal["torch", "non_torch"] = "torch" if spec is torch_spec else "non_torch"
    return _metadata_invariant_contracts_for_backend(backend_family, spec=spec)


def _metadata_invariant_contracts_for_backend(
    backend_family: Literal["torch", "non_torch"],
    *,
    spec: "BackendSpec | None" = None,
) -> tuple[MetadataInvariantContract, ...]:
    """Return ordered metadata invariant contracts for a backend family.

    Parameters
    ----------
    backend_family:
        ``"torch"`` for torch traces, otherwise ``"non_torch"``.
    spec:
        Optional backend registry spec used for capability-gated contracts.

    Returns
    -------
    tuple[MetadataInvariantContract, ...]
        Ordered invariant contracts whose backend and capability contracts match.
    """

    return tuple(
        contract
        for contract in METADATA_INVARIANT_CONTRACTS
        if _metadata_invariant_applies(contract, backend_family, spec)
    )


def _metadata_invariant_applies(
    contract: MetadataInvariantContract,
    backend_family: Literal["torch", "non_torch"],
    spec: "BackendSpec | None",
) -> bool:
    """Return whether ``contract`` applies to a backend family and spec.

    Parameters
    ----------
    contract:
        Metadata invariant contract to evaluate.
    backend_family:
        Backend family selected from the trace backend.
    spec:
        Backend registry spec, when available.

    Returns
    -------
    bool
        True when backend applicability and optional capability gates match.
    """

    if contract.applies_to not in {"all", backend_family}:
        return False
    if contract.requires_capability is None:
        return True
    if spec is None:
        return False
    return bool(getattr(spec.capabilities, contract.requires_capability, False))
