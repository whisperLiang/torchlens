"""Load-boundary validation for persisted cross-record claim families.

The fields in this module are ordinary Python objects after restricted
unpickling, but their values remain hostile artifact input.  Validation here
therefore uses closed vocabularies and independently persisted structural
anchors before a rehydrated :class:`~torchlens.data_classes.trace.Trace` is
exposed to callers.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

from . import TorchLensIOError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_RELATIONS = frozenset({"exact", "upper_bound", "lower_bound", "unknown"})
_CHECKPOINT_FLAGS = frozenset(
    {
        "classifier_unavailable",
        "patch_unavailable",
        "exotic_subclass",
        "unmatched_backward_warn",
        "reentrant_node_discovered",
        "unwitnessed_checkpoint_enter",
    }
)
_CHECKPOINT_VERDICTS = frozenset(
    {
        "checkpoint_invocations_observed",
        "evidence_incomplete",
        "no_checkpoint_invocation_observed",
    }
)
_KERNEL_ATTRIBUTION_STATUSES = frozenset({"ambiguous", "attributed", "unattributed", "unavailable"})
_KERNEL_LAUNCH_FIELDS = frozenset(
    {
        "attribution_status",
        "device",
        "duration",
        "launch_name",
        "runtime_correlation",
        "stream",
    }
)
_MAX_REPR_TUPLE_LENGTH = 4096


def validate_persisted_forgery_surfaces(trace: Trace) -> None:
    """Validate the persisted claim families before load returns.

    Parameters
    ----------
    trace:
        Trace whose restricted-unpickled state has been restored but has not
        yet been exposed by the portable loader.

    Raises
    ------
    TorchLensIOError
        If a persisted field violates its closed schema or structural anchor.
    """

    _validate_site_keys(trace)
    _validate_distributed_scope(trace)
    _validate_timing_provenance(trace)
    _validate_checkpoint_witness(trace)
    _validate_intervention_audit(trace)
    _validate_kernel_telemetry(trace)
    _validate_structure_only_coherence(trace)


def _refuse(
    message: str,
    *,
    code: str,
    field: str,
    reason: str,
    remedy: str,
) -> NoReturn:
    """Raise one teaching portable-field refusal.

    Parameters
    ----------
    message:
        Human-readable description of the structural violation.
    code:
        Stable refusal code for the owning field family.
    field:
        Fully-qualified persisted field name.
    reason:
        Machine-readable structural failure class.
    remedy:
        Concrete recovery action.

    Raises
    ------
    TorchLensIOError
        Always.
    """

    raise TorchLensIOError(
        f"{field} is invalid in the loaded artifact: {message}. Remedy: {remedy}.",
        code=code,
        field=field,
        reason=reason,
        remedy=remedy,
    )


def _is_int(value: Any) -> TypeGuard[int]:
    """Return whether ``value`` is an integer but not a boolean.

    Parameters
    ----------
    value:
        Candidate integer value.

    Returns
    -------
    bool
        ``True`` only for non-boolean integers.
    """

    return isinstance(value, int) and not isinstance(value, bool)


def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    """Return whether ``value`` is a non-string sequence.

    Parameters
    ----------
    value:
        Candidate sequence value.

    Returns
    -------
    bool
        ``True`` for sequence containers other than strings and bytes.
    """

    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _is_digest(value: Any) -> TypeGuard[str]:
    """Return whether ``value`` is a canonical lowercase SHA-256 digest.

    Parameters
    ----------
    value:
        Candidate digest.

    Returns
    -------
    bool
        ``True`` for exactly 64 lowercase hexadecimal characters.
    """

    return isinstance(value, str) and _HEX_DIGEST.fullmatch(value) is not None


def _trace_ops(trace: Trace) -> tuple[Any, ...]:
    """Return retained ops in their persisted execution order.

    Parameters
    ----------
    trace:
        Trace carrying the retained op list.

    Returns
    -------
    tuple[Any, ...]
        Retained ops in the order used by the site-key minter.
    """

    return tuple(getattr(trace, "layer_list", ()) or ())


def _validate_site_keys(trace: Trace) -> None:
    """Validate ``Op.site_key`` by parsing and byte-exact recomputation.

    Parameters
    ----------
    trace:
        Loaded trace carrying zero or more ``site_key_v1`` values.
    """

    from ..backends.registry import JAX_BACKEND_NAME
    from ..postprocess._site_key import SiteKeyMinter, parse_site_key

    ops = _trace_ops(trace)
    keys = tuple(getattr(op, "site_key", None) for op in ops)
    if not any(key is not None for key in keys):
        return
    if any(key is None for key in keys):
        _refuse(
            "site_key_v1 is present on only part of the retained op family; "
            "the structural-position relation must be total once present",
            code="artifact_site_key_invalid",
            field="Op.site_key",
            reason="partial_family",
            remedy=(
                "re-capture and save with one current TorchLens version; artifacts at the "
                "tlspec v6/v7 boundary may legitimately be wholly keyless, never partly keyed"
            ),
        )

    for op, key in zip(ops, keys, strict=True):
        if not isinstance(key, str):
            _refuse(
                f"op {getattr(op, 'label', '<unknown>')!r} carries non-string "
                f"site_key_v1 value {key!r}",
                code="artifact_site_key_invalid",
                field="Op.site_key",
                reason="type",
                remedy="re-capture and re-save the trace; do not hand-edit site_key_v1",
            )
        try:
            parse_site_key(key)
        except (UnicodeError, ValueError) as exc:
            _refuse(
                f"op {getattr(op, 'label', '<unknown>')!r} carries malformed "
                f"site_key_v1 value {key!r} ({exc})",
                code="artifact_site_key_invalid",
                field="Op.site_key",
                reason="syntax",
                remedy="re-capture and re-save the trace; do not hand-edit site_key_v1",
            )

    minter = SiteKeyMinter()
    backend = str(getattr(trace, "backend", "torch") or "torch")
    for op, key in zip(ops, keys, strict=True):
        if backend == JAX_BACKEND_NAME:
            from ..backends.jax._site_dialect import _jax_site_components

            module_site, call_instance = _jax_site_components(op)
            expected = minter.mint_at(
                module_site,
                call_instance,
                str(getattr(op, "type", "") or ""),
                (
                    getattr(op, "multi_output_index", None)
                    if getattr(op, "in_multi_output", False)
                    else None
                ),
            )
        else:
            expected = minter.mint(
                tuple(getattr(op, "module_call_stack", ()) or ()),
                str(getattr(op, "type", "") or ""),
                (
                    getattr(op, "multi_output_index", None)
                    if getattr(op, "in_multi_output", False)
                    else None
                ),
            )
        if key != expected:
            _refuse(
                f"op {getattr(op, 'label', '<unknown>')!r} stores {key!r}, but its "
                f"persisted module/type/output-slot facts recompute to {expected!r}",
                code="artifact_site_key_invalid",
                field="Op.site_key",
                reason="structural_mismatch",
                remedy=(
                    "re-capture and re-save with one current TorchLens version; artifacts at "
                    "the tlspec v6/v7 boundary may be wholly keyless"
                ),
            )


def _validate_distributed_scope(trace: Trace) -> None:
    """Validate the closed ``Trace.distributed_scope`` vocabulary.

    Parameters
    ----------
    trace:
        Loaded trace carrying the shard-local disclosure marker.
    """

    value = getattr(trace, "distributed_scope", None)
    if value not in (None, "rank_local_shard"):
        _refuse(
            f"expected None or 'rank_local_shard', got {value!r}",
            code="artifact_distributed_scope_invalid",
            field="Trace.distributed_scope",
            reason="vocabulary",
            remedy="re-save the source trace; do not invent distributed-scope markers",
        )


def _validate_timing_provenance(trace: Trace) -> None:
    """Validate the closed backward timing-provenance vocabulary.

    Parameters
    ----------
    trace:
        Loaded trace carrying the timing source discriminator.
    """

    value = getattr(trace, "grad_fn_timing_provenance", None)
    if value not in (None, "unmeasured", "perf_counter"):
        _refuse(
            f"expected None, 'unmeasured', or 'perf_counter', got {value!r}",
            code="artifact_grad_fn_timing_provenance_invalid",
            field="Trace.grad_fn_timing_provenance",
            reason="vocabulary",
            remedy="re-run backward capture and re-save without editing the clock source",
        )


def _checkpoint_invalid(message: str, reason: str) -> NoReturn:
    """Raise a typed checkpoint-witness load refusal.

    Parameters
    ----------
    message:
        Structural violation detail.
    reason:
        Machine-readable failure class.

    Raises
    ------
    TorchLensIOError
        Always.
    """

    _refuse(
        message,
        code="artifact_checkpoint_witness_invalid",
        field="Trace.checkpoint_invocation_witness",
        reason=reason,
        remedy="re-run backward capture and re-save; do not hand-edit checkpoint evidence",
    )


def _validate_checkpoint_witness(trace: Trace) -> None:
    """Validate checkpoint token geometry, vocabularies, and site relations.

    Parameters
    ----------
    trace:
        Loaded trace carrying the checkpoint invocation witness.
    """

    witness = getattr(trace, "checkpoint_invocation_witness", None)
    if witness is None:
        return
    if not isinstance(witness, Mapping):
        _checkpoint_invalid(f"expected a mapping, got {type(witness).__name__}", "type")
    expected_fields = {"token_count", "tokens", "degrade_flags", "verdict"}
    if set(witness) != expected_fields:
        _checkpoint_invalid(
            f"expected fields {sorted(expected_fields)!r}, got {sorted(map(str, witness))!r}",
            "schema",
        )

    token_count = witness["token_count"]
    tokens = witness["tokens"]
    if not _is_int(token_count) or token_count < 0:
        _checkpoint_invalid(f"token_count must be a non-negative int, got {token_count!r}", "count")
    if not isinstance(tokens, Mapping):
        _checkpoint_invalid(f"tokens must be a mapping, got {type(tokens).__name__}", "tokens")
    if set(tokens) != set(range(1, token_count + 1)):
        _checkpoint_invalid(
            "token keys must be the contiguous per-trace ordinals 1..token_count",
            "token_ordinals",
        )
    _validate_checkpoint_verdict(token_count, witness["degrade_flags"], witness["verdict"])

    site_keys = {
        key for op in _trace_ops(trace) if isinstance((key := getattr(op, "site_key", None)), str)
    }
    for token, record in tokens.items():
        _validate_checkpoint_token_record(token, record, site_keys)


def _validate_checkpoint_verdict(token_count: int, flags: Any, verdict: Any) -> None:
    """Validate degrade flags and the evidence-scoped verdict coherence rule.

    Parameters
    ----------
    token_count:
        Validated non-negative token count.
    flags:
        Candidate degrade-flag sequence.
    verdict:
        Candidate evidence-scoped verdict.
    """

    if not _is_sequence(flags) or any(not isinstance(flag, str) for flag in flags):
        _checkpoint_invalid("degrade_flags must be a sequence of strings", "degrade_flags")
    if list(flags) != sorted(set(flags)) or not set(flags) <= _CHECKPOINT_FLAGS:
        _checkpoint_invalid(
            f"degrade_flags must be sorted, unique, and drawn from {sorted(_CHECKPOINT_FLAGS)!r}",
            "degrade_flags",
        )
    if verdict not in _CHECKPOINT_VERDICTS:
        _checkpoint_invalid(
            f"verdict must be one of {sorted(_CHECKPOINT_VERDICTS)!r}, got {verdict!r}",
            "verdict",
        )
    expected_verdict = (
        "evidence_incomplete"
        if flags
        else (
            "checkpoint_invocations_observed"
            if token_count
            else "no_checkpoint_invocation_observed"
        )
    )
    if verdict != expected_verdict:
        _checkpoint_invalid(
            f"verdict {verdict!r} disagrees with token_count={token_count} and "
            f"degrade_flags={list(flags)!r}; expected {expected_verdict!r}",
            "verdict_coherence",
        )


def _validate_checkpoint_token_record(
    token: Any,
    record: Any,
    trace_site_keys: set[str],
) -> None:
    """Validate one checkpoint-token evidence summary.

    Parameters
    ----------
    token:
        Positive per-trace token ordinal.
    record:
        Persisted token evidence mapping.
    trace_site_keys:
        Site keys independently carried by retained ops.
    """

    if not isinstance(record, Mapping):
        _checkpoint_invalid(f"token {token!r} record must be a mapping", "token_record_type")
    expected_fields = {
        "pack_count",
        "site_key_candidates",
        "unpack_evidence_count",
        "window_evidence",
    }
    if set(record) != expected_fields:
        _checkpoint_invalid(
            f"token {token!r} expected fields {sorted(expected_fields)!r}, got "
            f"{sorted(map(str, record))!r}",
            "token_record_schema",
        )
    for field_name in ("pack_count", "unpack_evidence_count"):
        value = record[field_name]
        if not _is_int(value) or value < 0:
            _checkpoint_invalid(
                f"token {token!r} {field_name} must be a non-negative int, got {value!r}",
                "token_record_count",
            )

    candidates = record["site_key_candidates"]
    if not _is_sequence(candidates) or any(not isinstance(key, str) for key in candidates):
        _checkpoint_invalid(
            f"token {token!r} site_key_candidates must be a sequence of strings",
            "site_candidates",
        )
    if list(candidates) != sorted(set(candidates)):
        _checkpoint_invalid(
            f"token {token!r} site_key_candidates must be sorted and unique",
            "site_candidates",
        )
    unknown_sites = set(candidates) - trace_site_keys
    if unknown_sites:
        _checkpoint_invalid(
            f"token {token!r} cites site keys absent from retained ops: {sorted(unknown_sites)!r}",
            "site_relation",
        )

    _validate_checkpoint_windows(token, record["window_evidence"], record["unpack_evidence_count"])


def _validate_checkpoint_windows(token: Any, windows: Any, expected_count: Any) -> None:
    """Validate one token's unpack window-evidence rows.

    Parameters
    ----------
    token:
        Owning token ordinal used in diagnostics.
    windows:
        Candidate window-evidence sequence.
    expected_count:
        Validated ``unpack_evidence_count`` the row count must equal.
    """

    if not _is_sequence(windows):
        _checkpoint_invalid(
            f"token {token!r} window_evidence must be a sequence",
            "window_evidence",
        )
    if len(windows) != expected_count:
        _checkpoint_invalid(
            f"token {token!r} has {len(windows)} window rows but "
            f"unpack_evidence_count={expected_count!r}",
            "window_count",
        )
    for window in windows:
        if not _is_sequence(window) or len(window) != 3:
            _checkpoint_invalid(
                f"token {token!r} window row must be a three-item sequence, got {window!r}",
                "window_schema",
            )
        pass_index, label, call_index = window
        if pass_index is not None and (not _is_int(pass_index) or pass_index < 0):
            _checkpoint_invalid(
                f"token {token!r} window pass index is invalid: {pass_index!r}",
                "window_pass_index",
            )
        if label is not None and not isinstance(label, str):
            _checkpoint_invalid(
                f"token {token!r} window label is invalid: {label!r}",
                "window_label",
            )
        if call_index is not None and (not _is_int(call_index) or call_index < 0):
            _checkpoint_invalid(
                f"token {token!r} window call index is invalid: {call_index!r}",
                "window_call_index",
            )


def _audit_invalid(message: str, reason: str) -> NoReturn:
    """Raise a typed intervention-audit load refusal.

    Parameters
    ----------
    message:
        Structural violation detail.
    reason:
        Machine-readable failure class.

    Raises
    ------
    TorchLensIOError
        Always.
    """

    _refuse(
        message,
        code="artifact_intervention_audit_invalid",
        field="Trace.intervention_audit",
        reason=reason,
        remedy="re-apply the intervention and re-save; do not hand-edit audit or recipe rows",
    )


def _selection_recipes(trace: Trace) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return named helper selection recipes from retained fire records.

    Parameters
    ----------
    trace:
        Loaded trace carrying intervention fire records.

    Returns
    -------
    tuple[tuple[str, Mapping[str, Any]], ...]
        ``(helper_name, recipe)`` pairs in retained-op order.
    """

    recipes: list[tuple[str, Mapping[str, Any]]] = []
    for op in _trace_ops(trace):
        for record in getattr(op, "interventions", ()) or ():
            helper = getattr(record, "helper", None)
            recipe = getattr(helper, "selection_recipe", None) if helper is not None else None
            if recipe is None:
                continue
            if not isinstance(recipe, Mapping):
                _audit_invalid(
                    f"HelperSpec.selection_recipe must be a mapping, got {type(recipe).__name__}",
                    "recipe_type",
                )
            helper_name = getattr(helper, "helper_name", None)
            if not isinstance(helper_name, str):
                _audit_invalid(
                    "HelperSpec.selection_recipe has no string helper_name anchor",
                    "recipe_helper_name",
                )
            recipes.append((helper_name, recipe))
    return tuple(recipes)


def _validate_intervention_audit(trace: Trace) -> None:
    """Validate audit schemas and their helper-recipe digest relation.

    Parameters
    ----------
    trace:
        Loaded trace carrying intervention audit and fire records.
    """

    audit = getattr(trace, "intervention_audit", None)
    if audit is None:
        return
    if not _is_sequence(audit):
        _audit_invalid(f"expected a sequence, got {type(audit).__name__}", "type")
    audit_rows: list[tuple[str, str, frozenset[tuple[str, str]]]] = []
    for index, row in enumerate(audit):
        audit_rows.append(_validate_audit_row(index, row))
    audit_digests = {digest for digest, _edit, _targets in audit_rows}

    recipe_digests: set[str] = set()
    recipes_by_anchor: dict[tuple[str, str, str], set[str]] = {}
    for index, (helper_name, recipe) in enumerate(_selection_recipes(trace)):
        digest, kind, target = _validate_selection_recipe(index, recipe)
        recipe_digests.add(digest)
        recipes_by_anchor.setdefault((helper_name, kind, target), set()).add(digest)
    if not recipe_digests <= audit_digests:
        _audit_invalid(
            "HelperSpec.selection_recipe contains resolve digests absent from "
            f"Trace.intervention_audit: {sorted(recipe_digests - audit_digests)!r}",
            "recipe_relation",
        )
    for digest, edit, targets in audit_rows:
        for kind, target in targets:
            anchored_digests = recipes_by_anchor.get((edit, kind, target))
            if anchored_digests is not None and digest not in anchored_digests:
                _audit_invalid(
                    f"audit digest {digest!r} disagrees with the HelperSpec.selection_recipe "
                    f"rows anchored to edit={edit!r}, {kind} target={target!r}",
                    "digest_relation",
                )


def _validate_audit_row(index: int, row: Any) -> tuple[str, str, frozenset[tuple[str, str]]]:
    """Validate one ACT- or EDGE-kind intervention audit row.

    Parameters
    ----------
    index:
        Row index used in diagnostics.
    row:
        Candidate audit mapping.

    Returns
    -------
    tuple[str, str, frozenset[tuple[str, str]]]
        Resolve digest, edit identity, and typed target anchors.
    """

    if not isinstance(row, Mapping):
        _audit_invalid(f"row {index} must be a mapping, got {type(row).__name__}", "row_type")
    kind = row.get("kind")
    common_fields = {"edit", "kind", "resolve_digest", "selection_repr"}
    if kind == "ACT":
        allowed_fields = common_fields | {"patch_source", "sites"}
        required_fields = common_fields | {"sites"}
        if not required_fields <= set(row) or not set(row) <= allowed_fields:
            _audit_invalid(
                f"ACT row {index} has an invalid field set {sorted(map(str, row))!r}", "schema"
            )
        targets = frozenset(("ACT", site) for site in _validate_audit_sites(index, row["sites"]))
        if "patch_source" in row:
            _validate_patch_source(index, row["patch_source"])
    elif kind == "EDGE":
        expected_fields = common_fields | {"edges"}
        if set(row) != expected_fields:
            _audit_invalid(
                f"EDGE row {index} expected fields {sorted(expected_fields)!r}, got "
                f"{sorted(map(str, row))!r}",
                "schema",
            )
        targets = frozenset(("EDGE", edge) for edge in _validate_audit_edges(index, row["edges"]))
    else:
        _audit_invalid(f"row {index} kind must be 'ACT' or 'EDGE', got {kind!r}", "kind")
    if not isinstance(row["selection_repr"], str) or not isinstance(row["edit"], str):
        _audit_invalid(
            f"row {index} selection_repr and edit must be strings",
            "string_fields",
        )
    digest = row["resolve_digest"]
    if not _is_digest(digest):
        _audit_invalid(f"row {index} resolve_digest is not a canonical SHA-256", "digest")
    return digest, row["edit"], targets


def _validate_audit_sites(index: int, sites: Any) -> tuple[str, ...]:
    """Validate ACT audit site rows.

    Parameters
    ----------
    index:
        Owning audit-row index.
    sites:
        Candidate site-row sequence.

    Returns
    -------
    tuple[str, ...]
        Validated site-key representations.
    """

    if not _is_sequence(sites):
        _audit_invalid(f"ACT row {index} sites must be a sequence", "sites_type")
    expected_fields = {"relation", "selected", "site_key"}
    anchors: list[str] = []
    for site in sites:
        if not isinstance(site, Mapping) or set(site) != expected_fields:
            _audit_invalid(f"ACT row {index} has a malformed site row {site!r}", "site_schema")
        if site["relation"] not in _SELECTION_RELATIONS:
            _audit_invalid(
                f"ACT row {index} site relation is outside the closed vocabulary: "
                f"{site['relation']!r}",
                "site_relation",
            )
        if not _is_int(site["selected"]) or site["selected"] < 0:
            _audit_invalid(
                f"ACT row {index} selected count must be a non-negative int",
                "site_selected",
            )
        _validate_repr_tuple(site["site_key"], f"ACT row {index} site_key")
        anchors.append(site["site_key"])
    return tuple(anchors)


def _validate_patch_source(index: int, source: Any) -> None:
    """Validate the optional ``patch_from`` source-identity disclosure.

    Parameters
    ----------
    index:
        Owning audit-row index.
    source:
        Candidate source-identity mapping.
    """

    expected_fields = {"source_model_class", "source_object_id", "source_trace_label"}
    if (
        not isinstance(source, Mapping)
        or set(source) != expected_fields
        or any(not isinstance(value, str) for value in source.values())
    ):
        _audit_invalid(
            f"ACT row {index} patch_source must contain exactly three string identity fields",
            "patch_source",
        )


def _validate_audit_edges(index: int, edges: Any) -> tuple[str, ...]:
    """Validate EDGE audit relation rows.

    Parameters
    ----------
    index:
        Owning audit-row index.
    edges:
        Candidate edge-row sequence.

    Returns
    -------
    tuple[str, ...]
        Validated edge-address representations.
    """

    if not _is_sequence(edges):
        _audit_invalid(f"EDGE row {index} edges must be a sequence", "edges_type")
    expected_fields = {"child", "edge_address", "parent", "value_digest"}
    anchors: list[str] = []
    for edge in edges:
        if not isinstance(edge, Mapping) or set(edge) != expected_fields:
            _audit_invalid(f"EDGE row {index} has a malformed edge row {edge!r}", "edge_schema")
        if not isinstance(edge["parent"], str) or not isinstance(edge["child"], str):
            _audit_invalid(f"EDGE row {index} parent and child must be strings", "edge_labels")
        _validate_repr_tuple(edge["edge_address"], f"EDGE row {index} edge_address")
        if not _is_digest(edge["value_digest"]):
            _audit_invalid(
                f"EDGE row {index} value_digest is not a canonical SHA-256", "edge_digest"
            )
        anchors.append(edge["edge_address"])
    return tuple(anchors)


def _validate_repr_tuple(value: Any, field: str) -> tuple[Any, ...]:
    """Parse a persisted tuple representation without executing code.

    Parameters
    ----------
    value:
        Candidate ``repr(tuple)`` string.
    field:
        Diagnostic field description.

    Returns
    -------
    tuple[Any, ...]
        Parsed tuple.
    """

    if not isinstance(value, str):
        _audit_invalid(f"{field} must be a tuple representation string", "tuple_repr")
    if len(value) > _MAX_REPR_TUPLE_LENGTH:
        _audit_invalid(
            f"{field} exceeds the {_MAX_REPR_TUPLE_LENGTH}-character parse ceiling",
            "tuple_repr_size",
        )
    try:
        parsed = ast.literal_eval(value)
    except (MemoryError, RecursionError, SyntaxError, ValueError) as exc:
        _audit_invalid(f"{field} is not a literal tuple ({exc})", "tuple_repr")
    if not isinstance(parsed, tuple):
        _audit_invalid(f"{field} must decode to a tuple, got {type(parsed).__name__}", "tuple_repr")
    return parsed


def _validate_selection_recipe(index: int, recipe: Mapping[str, Any]) -> tuple[str, str, str]:
    """Validate one persisted helper selection recipe.

    Parameters
    ----------
    index:
        Recipe index used in diagnostics.
    recipe:
        Candidate recipe mapping.

    Returns
    -------
    tuple[str, str, str]
        Resolve digest, target kind, and target representation.
    """

    digest = recipe.get("resolve_digest")
    if not _is_digest(digest):
        _audit_invalid(f"selection recipe {index} has an invalid resolve_digest", "recipe_digest")
    if set(recipe) == {"edge_address", "resolve_digest"}:
        _validate_repr_tuple(recipe["edge_address"], f"selection recipe {index} edge_address")
        return digest, "EDGE", str(recipe["edge_address"])
    expected_fields = {"relation", "resolve_digest", "selected", "site_key", "source"}
    if set(recipe) != expected_fields:
        _audit_invalid(
            f"selection recipe {index} expected ACT or EDGE schema, got "
            f"{sorted(map(str, recipe))!r}",
            "recipe_schema",
        )
    if recipe["relation"] not in _SELECTION_RELATIONS:
        _audit_invalid(
            f"selection recipe {index} relation is outside the closed vocabulary",
            "recipe_relation_vocabulary",
        )
    if not _is_int(recipe["selected"]) or recipe["selected"] < 0:
        _audit_invalid(
            f"selection recipe {index} selected count must be a non-negative int",
            "recipe_selected",
        )
    if not isinstance(recipe["source"], str):
        _audit_invalid(f"selection recipe {index} source must be a string", "recipe_source")
    _validate_repr_tuple(recipe["site_key"], f"selection recipe {index} site_key")
    return digest, "ACT", str(recipe["site_key"])


#: Op payload fields whose presence contradicts the structure-only marker
#: (a structure-only capture never retains values). Version metadata and
#: RNG-state families are deliberately not in this set.
_OP_VALUE_PAYLOAD_FIELDS = (
    "out",
    "saved_args",
    "saved_kwargs",
    "transformed_out",
    "grad",
    "transformed_grad",
)


def _structure_only_invalid(message: str, reason: str) -> NoReturn:
    """Raise a typed structure-only marker-coherence load refusal.

    Parameters
    ----------
    message:
        Structural violation detail.
    reason:
        Machine-readable failure class.

    Raises
    ------
    TorchLensIOError
        Always.
    """

    _refuse(
        message,
        code="artifact_structure_only_incoherent",
        field="Trace.structure_only",
        reason=reason,
        remedy=(
            "re-capture and re-save with one current TorchLens version; do "
            "not hand-edit the structure-only marker, payloads, or the "
            "verification verdict"
        ),
    )


def _validate_structure_only_coherence(trace: Trace) -> None:
    """Validate the structure-only marker's coherence rules (M-C2/M-C3).

    M-C2: a marked trace carrying ANY retained value payload is incoherent
    (structure-only captures retain no values). M-C3: a marked trace claiming
    ``capture_verified=True`` is incoherent (verification requires values);
    this runs BEFORE the ``__setstate__`` no-claim degradation, so the forged
    positive claim refuses loudly rather than degrading silently.

    SCOPE STATEMENT (M-C1, form (a)): a stripped-marker structure-only
    artifact carries no load-visible anchor on the current persistence
    surface — its input digests are absent (no payloads to hash), its
    param/buffer geometry records are device-neutral, and payload-free
    ordinary captures are legal — so marker ABSENCE is undetectable without
    refusing legal artifacts (the same t3b conclusion the L7a memo states
    for form (b)). Coherence therefore validates the marker's PRESENCE,
    never its absence.

    Parameters
    ----------
    trace:
        Loaded trace carrying the optional structure-only marker.
    """

    marker = trace.__dict__.get("structure_only")
    if marker in (None, False):
        return
    if marker is not True:
        _structure_only_invalid(
            f"expected a bool marker, got {marker!r}",
            "type",
        )
    if trace.__dict__.get("capture_verified") is True:
        _structure_only_invalid(
            "capture_verified=True on a structure-only capture (verification requires values)",
            "verification_claim",
        )
    for op in _trace_ops(trace):
        for field_name in _OP_VALUE_PAYLOAD_FIELDS:
            if getattr(op, field_name, None) is not None:
                _structure_only_invalid(
                    f"op {getattr(op, 'label', '<unknown>')!r} retains a value "
                    f"payload in {field_name!r} under the structure-only marker",
                    "value_payload_present",
                )


def _kernel_invalid(message: str, reason: str) -> NoReturn:
    """Raise a typed kernel-telemetry load refusal.

    Parameters
    ----------
    message:
        Structural violation detail.
    reason:
        Machine-readable failure class.

    Raises
    ------
    TorchLensIOError
        Always.
    """

    _refuse(
        message,
        code="artifact_kernel_telemetry_invalid",
        field="Trace.annotations._kernel_telemetry",
        reason=reason,
        remedy="re-profile and re-save the trace; do not hand-edit launch or relation rows",
    )


def _validate_kernel_telemetry(trace: Trace) -> None:
    """Validate telemetry rows and their primitive-sequence foreign keys.

    Parameters
    ----------
    trace:
        Loaded trace carrying optional kernel telemetry.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, Mapping) or "_kernel_telemetry" not in annotations:
        return
    payload = annotations["_kernel_telemetry"]
    expected_fields = {"_available", "_launches", "_relations"}
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        _kernel_invalid(
            f"payload must be a mapping with fields {sorted(expected_fields)!r}",
            "payload_schema",
        )
    available = payload["_available"]
    launches = payload["_launches"]
    relations = payload["_relations"]
    if not isinstance(available, bool):
        _kernel_invalid(f"_available must be bool, got {available!r}", "available")
    if not _is_sequence(launches):
        _kernel_invalid("_launches must be a sequence", "launches_type")
    for index, launch in enumerate(launches):
        _validate_kernel_launch(index, launch)
    if not _is_sequence(relations):
        _kernel_invalid(
            "_relations must be a sequence of (sequence, launch_index) pairs", "relations_type"
        )
    _validate_kernel_relations(trace, relations, launches)
    if not available and (len(launches) != 1 or not _is_unavailable_launch(launches[0])):
        _kernel_invalid(
            "_available=False must carry exactly one fact-free 'unavailable' launch row",
            "unavailable_coherence",
        )


def _validate_kernel_relations(
    trace: Trace,
    relations: Sequence[Any],
    launches: Sequence[Any],
) -> None:
    """Validate telemetry relation rows against their two foreign-key spaces.

    Parameters
    ----------
    trace:
        Loaded trace carrying the primitive profile the rows cite.
    relations:
        Candidate ``(sequence, launch_index)`` relation rows.
    launches:
        Validated launch rows the indices must land in.
    """

    profile = trace.__dict__.get("_primitive_op_profile")
    primitive_rows = tuple(getattr(profile, "primitive_ops", ()) or ())
    primitive_sequences = {
        sequence for row in primitive_rows if _is_int(sequence := getattr(row, "sequence", None))
    }
    seen_relations: set[tuple[int, int]] = set()
    for relation in relations:
        if not _is_sequence(relation) or len(relation) != 2:
            _kernel_invalid(
                f"relation row must contain two items, got {relation!r}", "relation_schema"
            )
        sequence, launch_index = relation
        if not _is_int(sequence) or sequence not in primitive_sequences:
            _kernel_invalid(
                f"relation sequence {sequence!r} does not name a primitive profile row",
                "primitive_foreign_key",
            )
        if not _is_int(launch_index) or not 0 <= launch_index < len(launches):
            _kernel_invalid(
                f"relation launch index {launch_index!r} is outside 0..{len(launches) - 1}",
                "launch_foreign_key",
            )
        normalized = (sequence, launch_index)
        if normalized in seen_relations:
            _kernel_invalid(f"duplicate relation row {normalized!r}", "duplicate_relation")
        seen_relations.add(normalized)


def _validate_kernel_launch(index: int, launch: Any) -> None:
    """Validate one kernel-launch row against its closed field vocabularies.

    Parameters
    ----------
    index:
        Launch-row index used in diagnostics.
    launch:
        Candidate launch mapping or typed launch object.
    """

    if isinstance(launch, Mapping):
        values = launch
    else:
        values = {field: getattr(launch, field, None) for field in _KERNEL_LAUNCH_FIELDS}
        if any(not hasattr(launch, field) for field in _KERNEL_LAUNCH_FIELDS):
            _kernel_invalid(f"launch {index} has an invalid representation", "launch_type")
    if set(values) != _KERNEL_LAUNCH_FIELDS:
        _kernel_invalid(
            f"launch {index} has an invalid field set {sorted(map(str, values))!r}", "launch_schema"
        )
    status = values["attribution_status"]
    if status not in _KERNEL_ATTRIBUTION_STATUSES:
        _kernel_invalid(
            f"launch {index} attribution_status must be one of "
            f"{sorted(_KERNEL_ATTRIBUTION_STATUSES)!r}, got {status!r}",
            "attribution_status",
        )
    for field_name in ("launch_name", "device"):
        value = values[field_name]
        if value is not None and not isinstance(value, str):
            _kernel_invalid(f"launch {index} {field_name} must be str or None", "launch_field_type")
    for field_name in ("stream", "runtime_correlation"):
        value = values[field_name]
        if value is not None and not (isinstance(value, str) or _is_int(value)):
            _kernel_invalid(
                f"launch {index} {field_name} must be str, int, or None",
                "launch_field_type",
            )
    duration = values["duration"]
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(float(duration))
        or duration < 0
    ):
        _kernel_invalid(
            f"launch {index} duration must be a finite non-negative number or None",
            "duration",
        )


def _is_unavailable_launch(launch: Any) -> bool:
    """Return whether a launch is the canonical fact-free disclosure row.

    Parameters
    ----------
    launch:
        Validated launch mapping or typed launch object.

    Returns
    -------
    bool
        ``True`` for the single canonical unavailable row.
    """

    getter = launch.get if isinstance(launch, Mapping) else lambda field: getattr(launch, field)
    return getter("attribution_status") == "unavailable" and all(
        getter(field) is None for field in _KERNEL_LAUNCH_FIELDS - {"attribution_status"}
    )
