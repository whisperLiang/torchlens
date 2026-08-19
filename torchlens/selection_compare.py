"""Comparative selection producers: differential and cross-pass (L6 producer wave).

Selects by HOW VALUES DIFFER, across runs or across passes. Two producer
families, both returning :class:`~torchlens.selection.Selection` queries that
compose with the full ``| & - ~`` algebra and resolve explicitly against one
trace:

- DIFFERENTIAL producers compare the resolution trace (the SUBJECT — the
  trace passed to ``resolve()``, where the masks land) against ONE explicit
  REFERENCE trace: ``changed`` (elementwise bounds on the delta — "units the
  intervention actually moved", "what changed when I swapped the input") and
  ``top_changed`` (global deterministic ranking of the delta — "units that
  changed most"). The delta is DIRECTIONAL: ``subject - reference``,
  elementwise in float64. Comparison is deliberately pairwise against one
  reference (multi-sample dispersion is the different spelling
  ``low_variance(samples=...)``).

- CROSS-PASS producers read ONE capture's multi-pass layers, pass-qualified
  throughout: ``stable_across_passes`` (per-element range across the pass
  window within ``tol``) and ``pass_variance`` (bounds on the per-element
  variance across the window). Evidence windows are explicit (``passes=``);
  a layer contributing fewer than TWO passes refuses typed — a cross-pass
  claim about a single-pass layer is vacuous, and vacuous truths are
  refusals here, never silent empty masks.

STRUCTURE MATCHING (differential): the population comes from the SUBJECT;
the reference must hold a retained, shape-identical activation at the same
pass-qualified ``(layer_label, pass_index)`` address — missing sites, unsaved
payloads, and shape drift refuse typed with the reference named, exactly the
multi-sample evidence contract. Structures are NEVER silently intersected.
When both sides carry L1 structural site keys, a key disagreement also
refuses (label coincidence across different architectures is caught, not
compared); either side keyless proceeds on address+shape.

PROVENANCE HONESTY (pinned by tests): every producer here names a statistic
of complete retained evidence — the pairwise delta between THESE two
captures, or the dispersion across THIS capture's passes — so entries declare
``relation="exact"`` (the ``low_variance`` precedent). None of these names
makes an open-world dispositional claim ("input-sensitive" would be one; it
is deliberately not spelled here). Population restriction composes through
the normative JOIN table, never by overwriting a relation.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (megasprint provisional-name protocol), with the interface
flagged for the UI-sprint review. Producers are ACT-kind only: PARAM
populations refuse ``selection_kind_incompatible`` — ``Param`` records hold
only a LIVE parameter reference, never capture-time payloads, so a
"weights that shifted between checkpoints" claim cannot be made honestly
from Trace records (compare runnable-save ``state_dict_v1`` blobs instead;
a capture-time weight differential is a named possibility, not a promise).

Elements whose compared quantity is NaN never satisfy any criterion; rank
producers exclude them from the candidate population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .selection import (
    ResolvedSelection,
    Selection,
    SelectionProvenance,
    SiteEntry,
    _join_relation,
    _mask_from_dense,
    _unresolvable,
    register_term_resolver,
)
from .selection_values import (
    _entry_with_mask,
    _lift_within,
    _op_for_entry,
    _read_saved_value,
    _resolve_population,
    _validate_real_number,
)

__all__ = [
    "changed",
    "pass_variance",
    "stable_across_passes",
    "top_changed",
]


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _reference_display(reference: Any) -> str:
    """Return a compact reference identity for reprs/sources (never the trace)."""

    label = getattr(reference, "trace_label", None)
    return str(label) if label else "<reference trace>"


def _validate_reference(reference: Any, producer: str) -> Any:
    """Validate the reference operand: exactly ONE Trace, refused with teaching."""

    from .data_classes.trace import Trace

    if isinstance(reference, Trace):
        return reference
    if isinstance(reference, (list, tuple, set)) or type(reference).__name__ == "Bundle":
        raise ValueError(
            f"{producer} compares against ONE reference trace (pairwise delta), "
            "not an evidence set. For dispersion across many samples use "
            "low_variance(samples=...)."
        )
    raise ValueError(f"{producer} `reference` must be a Trace (got {type(reference).__name__}).")


def _validate_by(by: str, producer: str) -> str:
    """Validate the delta-comparison mode (closed vocabulary)."""

    if by not in ("abs", "signed"):
        raise ValueError(f"{producer} `by` must be 'abs' or 'signed'; got {by!r}.")
    return by


def _validate_passes(passes: Any, producer: str) -> tuple[int, ...] | None:
    """Validate an explicit pass window: >= 2 distinct 1-based pass indices."""

    if passes is None:
        return None
    collected: list[int] = []
    for index in passes:
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError(
                f"{producer} `passes` must contain 1-based pass indices (ints >= 1); got {index!r}."
            )
        collected.append(index)
    window = tuple(sorted(set(collected)))
    if len(window) < 2:
        raise ValueError(
            f"{producer} makes a CROSS-PASS claim and needs a window of at least 2 "
            f"passes (got {len(window)}). For single-pass value claims use the "
            "value producers (threshold / sign / top_k)."
        )
    return window


def _delta_for_entry(node: _CompareTerm, trace: Any, entry: SiteEntry) -> torch.Tensor:
    """Compute one entry's subject-minus-reference delta with typed refusals.

    Reads both sides through the shared evidence reader (missing site /
    unsaved payload / shape drift refuse typed with the reference named) and
    corroborates L1 structural site keys when both sides carry them.
    """

    reference_op = _op_for_entry(node.reference, entry, sample_name="reference")
    subject_key = entry.structural_site_key
    reference_key = getattr(reference_op, "site_key", None)
    if subject_key is not None and reference_key is not None and subject_key != reference_key:
        raise _unresolvable(
            "site_not_in_trace",
            f"the reference trace's op at address {entry.site_key!r} is a "
            f"structurally DIFFERENT site (structural site key {reference_key!r} "
            f"vs subject {subject_key!r}): the traces do not share this "
            "position, and a delta between structurally different sites would "
            "be a false comparison. Restrict `within=` to genuinely shared "
            "structure.",
            site=repr(entry.site_key),
            sample="reference",
        )
    subject_value = _read_saved_value(trace, entry)
    reference_value = _read_saved_value(node.reference, entry, sample_name="reference")
    if subject_value.is_complex() or reference_value.is_complex():
        return subject_value.to(torch.complex128) - reference_value.to(torch.complex128)
    # float64 exactly represents every float32/16/bfloat16/int32 value; the
    # int64 tail beyond 2**53 is a documented precision residual.
    return subject_value.to(torch.float64) - reference_value.to(torch.float64)


def _compared_delta(delta: torch.Tensor, by: str, producer: str, site: Any) -> torch.Tensor:
    """Return the compared quantity (|delta| or signed delta; complex refuses ordered)."""

    if by == "abs":
        return delta.abs()
    if delta.is_complex():
        raise _unresolvable(
            "value_criterion_invalid",
            f"{producer} with by='signed' orders raw deltas, and complex deltas "
            f"have no total order (site {site!r}). Use by='abs' (delta magnitude).",
            site=site,
        )
    return delta


# ---------------------------------------------------------------------------
# Differential producers (subject vs one explicit reference trace).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CompareTerm:
    """AST leaf for the differential producers (changed / top_changed)."""

    criterion: str
    reference: Any
    within: Any
    by: str = "abs"
    above: float | None = None
    below: float | None = None
    k: int | None = None
    fraction: float | None = None
    largest: bool = True

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure (never the trace)."""

        population = "saved_sites" if self.within is None else repr(self.within)
        vs = _reference_display(self.reference)
        if self.criterion == "changed":
            head = f"changed(vs={vs!r}, above={self.above}, below={self.below}, by={self.by!r}"
        elif self.k is not None:
            head = f"top_changed(vs={vs!r}, k={self.k}, by={self.by!r}, largest={self.largest}"
        else:
            head = (
                f"top_changed(vs={vs!r}, fraction={self.fraction}, by={self.by!r}, "
                f"largest={self.largest}"
            )
        return f"{head}, within={population})"


def _refuse_self_comparison(node: _CompareTerm, trace: Any) -> None:
    """Refuse resolving a differential producer against its own reference."""

    if node.reference is trace:
        raise _unresolvable(
            "value_criterion_invalid",
            f"{node.criterion} resolved against its own reference trace: the "
            "delta of a capture with itself is identically zero, so the "
            "criterion is vacuous by construction. Resolve on the SUBJECT run "
            "and pass the OTHER run as the reference (a fork is a different "
            "trace object).",
        )


def _resolve_changed(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve the elementwise delta-bound criterion exactly."""

    _refuse_self_comparison(node, trace)
    population = _resolve_population(node.within, trace)
    source = (
        f"changed(above={node.above}, below={node.below}, by={node.by!r}, "
        f"vs={_reference_display(node.reference)!r})"
    )
    entries: list[SiteEntry] = []
    for entry in population:
        delta = _delta_for_entry(node, trace, entry)
        compared = _compared_delta(delta, node.by, "changed", entry.site_key)
        dense = torch.ones(entry.shape, dtype=torch.bool)
        if node.above is not None:
            dense &= compared > node.above
        if node.below is not None:
            dense &= compared < node.below
        dense &= entry._mask._dense_ro()
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_top_changed(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve the global delta-rank criterion exactly.

    Ranking is GLOBAL across the population with a deterministic tie-break:
    stable sort over the concatenation of sites in canonical order, so ties
    resolve by (site order, flat index). NaN deltas never enter the candidate
    population.
    """

    _refuse_self_comparison(node, trace)
    population = _resolve_population(node.within, trace)
    per_entry: list[tuple[SiteEntry, torch.Tensor, torch.Tensor]] = []
    for entry in population:
        delta = _delta_for_entry(node, trace, entry)
        keys = _compared_delta(delta, node.by, "top_changed", entry.site_key)
        keys = keys.to(torch.float64)
        valid = entry._mask._dense_ro() & ~torch.isnan(keys)
        per_entry.append((entry, keys.reshape(-1), valid.reshape(-1)))
    total_valid = int(sum(valid.sum().item() for _, _, valid in per_entry))
    vs = _reference_display(node.reference)
    if node.fraction is not None:
        k = math.ceil(node.fraction * total_valid)
        source = (
            f"top_changed(fraction={node.fraction}, by={node.by!r}, "
            f"largest={node.largest}, vs={vs!r})"
        )
    else:
        if node.k is None:
            raise RuntimeError("top_changed node lost its k")
        k = node.k
        source = f"top_changed(k={k}, by={node.by!r}, largest={node.largest}, vs={vs!r})"
        if k > total_valid:
            raise _unresolvable(
                "population_too_small",
                f"top_changed needs {k} elements but the population has only "
                f"{total_valid} rankable (non-NaN, in-population) elements.",
                requested=k,
                available=total_valid,
            )
    sentinel = float("-inf") if node.largest else float("inf")
    flat_keys = (
        torch.cat(
            [
                torch.where(valid, keys, torch.tensor(sentinel, dtype=torch.float64))
                for _, keys, valid in per_entry
            ]
        )
        if per_entry
        else torch.zeros(0, dtype=torch.float64)
    )
    order = torch.argsort(flat_keys, descending=node.largest, stable=True)[:k]
    selected_flat = torch.zeros(flat_keys.shape[0], dtype=torch.bool)
    selected_flat[order] = True
    entries: list[SiteEntry] = []
    offset = 0
    for entry, keys, _ in per_entry:
        span = keys.shape[0]
        dense = selected_flat[offset : offset + span].reshape(entry.shape)
        offset += span
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_compare_term(node: _CompareTerm, trace: Any) -> ResolvedSelection:
    """Resolve one differential term against the subject trace's payloads."""

    if node.criterion == "changed":
        return _resolve_changed(node, trace)
    return _resolve_top_changed(node, trace)


def changed(
    reference: Any,
    within: Any = None,
    *,
    above: float | None = None,
    below: float | None = None,
    by: str = "abs",
) -> Selection:
    """Select elements whose value differs from a reference run's ("what moved?").

    The delta is DIRECTIONAL — ``subject - reference``, where the SUBJECT is
    the trace the selection resolves against and ``reference`` is one other
    Trace (a fork after ``do()``, a capture on another input, ...). ``by``
    picks the compared quantity: ``'abs'`` (default) compares ``|delta|``,
    ``'signed'`` the signed delta (so ``above=`` selects INCREASED elements,
    ``below=`` with a negative bound the decreased). ``above=`` / ``below=``
    are strict bounds (both = open band); with NEITHER given the default is
    ``above=0.0`` — bare ``changed(ref)`` selects every element that moved at
    all, the intervention-effect mask. Both runs must retain the compared
    payloads at the same pass-qualified site with the same shape: missing
    sites, unsaved payloads, shape drift, and structural-site-key
    disagreement refuse typed (structures never silently intersect).
    Resolving against the reference itself refuses (vacuous by construction).
    NaN deltas never satisfy a bound; complex deltas refuse ``by='signed'``.
    ``provenance.relation`` is ``exact``: the claim is the pairwise delta
    between THESE two captures. PARAM populations refuse
    ``selection_kind_incompatible`` (capture-time weights are not retained on
    Trace records). DOCUMENTED-UNSTABLE spelling.
    """

    reference = _validate_reference(reference, "changed")
    _validate_by(by, "changed")
    if above is None and below is None:
        above = 0.0
    if above is not None:
        above = _validate_real_number(above, "above", "changed")
    if below is not None:
        below = _validate_real_number(below, "below", "changed")
    return Selection(
        _CompareTerm(
            criterion="changed",
            reference=reference,
            within=_lift_within(within, "changed"),
            by=by,
            above=above,
            below=below,
        ),
        kind="ACT",
    )


def top_changed(
    reference: Any,
    within: Any = None,
    k: int | None = None,
    *,
    fraction: float | None = None,
    by: str = "abs",
    largest: bool = True,
) -> Selection:
    """Select the elements that changed most (or least) versus a reference run.

    Globally ranks ``subject - reference`` deltas across the whole population
    (``within=None`` means every retained tensor site) and selects exactly
    ``k`` elements — or ``ceil(fraction * population)`` with ``fraction=``;
    exactly ONE of the two must be given — without replacement, with a
    deterministic tie-break (stable sort; canonical site order, then flat
    index). ``by='abs'`` (default) ranks ``|delta|``; ``by='signed'`` ranks
    the signed delta (``largest=True`` = most increased). ``largest=False``
    selects the LEAST-moved elements (a real control: what the intervention
    did NOT touch). NaN deltas never enter the candidate population; a
    population with fewer than ``k`` rankable elements refuses
    ``population_too_small``. Structure matching, self-comparison, and
    provenance follow :func:`changed` exactly. DOCUMENTED-UNSTABLE spelling.
    """

    reference = _validate_reference(reference, "top_changed")
    _validate_by(by, "top_changed")
    if not isinstance(largest, bool):
        raise ValueError(f"top_changed `largest` must be a bool; got {largest!r}.")
    if (k is None) == (fraction is None):
        raise ValueError(
            "top_changed requires exactly one of `k=` (element count) or "
            "`fraction=` (share of the rankable population)."
        )
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
        raise ValueError(f"top_changed `k` must be a non-negative int; got {k!r}.")
    if fraction is not None:
        fraction = _validate_real_number(fraction, "fraction", "top_changed")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"top_changed `fraction` must be in [0, 1]; got {fraction!r}.")
    return Selection(
        _CompareTerm(
            criterion="top_changed",
            reference=reference,
            within=_lift_within(within, "top_changed"),
            by=by,
            k=k,
            fraction=fraction,
            largest=largest,
        ),
        kind="ACT",
    )


# ---------------------------------------------------------------------------
# Cross-pass producers (one capture; evidence = a multi-pass layer's passes).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PassTerm:
    """AST leaf for the cross-pass producers (stable_across_passes / pass_variance)."""

    stat: str
    within: Any
    tol: float = 0.0
    above: float | None = None
    below: float | None = None
    passes: tuple[int, ...] | None = None

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure."""

        population = "saved_sites" if self.within is None else repr(self.within)
        window = "all" if self.passes is None else repr(list(self.passes))
        if self.stat == "stable":
            head = f"stable_across_passes(tol={self.tol}, passes={window}"
        else:
            head = f"pass_variance(above={self.above}, below={self.below}, passes={window}"
        return f"{head}, within={population})"


def _pass_groups(
    node: _PassTerm, population: ResolvedSelection, producer: str
) -> list[tuple[str, list[SiteEntry]]]:
    """Group population entries per layer and select each layer's pass window.

    Enforces the cross-pass honesty floor (>= 2 window passes per layer, else
    ``population_too_small`` with the teaching message), explicit-window
    presence (a requested pass missing from the population refuses
    ``site_not_in_trace``), and a constant index space across the window
    (shape drift refuses ``mask_shape_mismatch`` — restrict ``passes=`` to a
    constant-shape window).
    """

    by_layer: dict[str, dict[int, SiteEntry]] = {}
    layer_order: list[str] = []
    for entry in population:
        layer_label, pass_index = entry.site_key
        if layer_label not in by_layer:
            by_layer[layer_label] = {}
            layer_order.append(layer_label)
        by_layer[layer_label][int(pass_index)] = entry
    groups: list[tuple[str, list[SiteEntry]]] = []
    for layer_label in layer_order:
        passes_present = by_layer[layer_label]
        if node.passes is None:
            window = sorted(passes_present)
        else:
            window = list(node.passes)
            for pass_index in window:
                if pass_index not in passes_present:
                    raise _unresolvable(
                        "site_not_in_trace",
                        f"{producer} window names pass {pass_index} of layer "
                        f"{layer_label!r}, which is not in the population "
                        "(absent from the trace, or excluded by `within=`).",
                        site=f"{layer_label}:{pass_index}",
                    )
        if len(window) < 2:
            raise _unresolvable(
                "population_too_small",
                f"{producer} makes a CROSS-PASS claim, and layer {layer_label!r} "
                f"contributes only {len(window)} pass to the window — the claim "
                "would be vacuously true. Restrict `within=` to the recurrent "
                "(multi-pass) layers, or use the single-capture value producers "
                "(threshold / sign / top_k) for one-pass claims.",
                site=layer_label,
                requested=2,
                available=len(window),
            )
        window_entries = [passes_present[pass_index] for pass_index in window]
        first_shape = window_entries[0].shape
        for entry in window_entries[1:]:
            if entry.shape != first_shape:
                raise _unresolvable(
                    "mask_shape_mismatch",
                    f"layer {layer_label!r} changes shape across the pass window "
                    f"({first_shape!r} vs {entry.shape!r} at pass "
                    f"{entry.site_key[1]}): elementwise cross-pass statistics "
                    "need one index space. Restrict `passes=` to a "
                    "constant-shape window.",
                    site=layer_label,
                )
        groups.append((layer_label, window_entries))
    return groups


def _resolve_pass_term(node: _PassTerm, trace: Any) -> ResolvedSelection:
    """Resolve one cross-pass statistic against a capture's per-pass payloads.

    The element population per layer is the INTERSECTION of the window
    entries' masks (a cross-pass claim needs the element in evidence at EVERY
    window pass); the resulting mask lands on every window pass-site, since
    the claim is about the unit across the whole window.
    """

    producer = "stable_across_passes" if node.stat == "stable" else "pass_variance"
    population = _resolve_population(node.within, trace)
    entries: list[SiteEntry] = []
    for layer_label, window_entries in _pass_groups(node, population, producer):
        values = []
        for entry in window_entries:
            value = _read_saved_value(trace, entry)
            if value.is_complex():
                raise _unresolvable(
                    "value_criterion_invalid",
                    f"{producer} orders/spreads raw values across passes, and "
                    f"complex payloads have no total order (layer "
                    f"{layer_label!r}).",
                    site=layer_label,
                )
            values.append(value.to(torch.float64))
        stacked = torch.stack(values)
        shared = window_entries[0]._mask._dense_ro().clone()
        relation = window_entries[0].provenance.relation
        for entry in window_entries[1:]:
            shared &= entry._mask._dense_ro()
            relation = _join_relation(relation, entry.provenance.relation)
        n_passes = len(window_entries)
        if node.stat == "stable":
            spread = stacked.max(dim=0).values - stacked.min(dim=0).values
            dense = spread <= node.tol
            source = f"stable_across_passes(n_passes={n_passes}, tol={node.tol})"
        else:
            variance = torch.var(stacked, dim=0)
            dense = torch.ones(window_entries[0].shape, dtype=torch.bool)
            if node.above is not None:
                dense &= variance > node.above
            if node.below is not None:
                dense &= variance < node.below
            source = f"pass_variance(n_passes={n_passes}, above={node.above}, below={node.below})"
        dense &= shared
        for entry in window_entries:
            entries.append(
                SiteEntry(
                    kind=entry.kind,
                    site_key=entry.site_key,
                    provenance=SelectionProvenance(
                        relation=_join_relation(relation, "exact"), source=source
                    ),
                    _mask=_mask_from_dense(entry.shape, dense),
                    structural_site_key=entry.structural_site_key,
                )
            )
    return ResolvedSelection(trace, "ACT", entries)


def stable_across_passes(
    within: Any = None,
    *,
    tol: float = 0.0,
    passes: Any = None,
) -> Selection:
    """Select units stable across a recurrent layer's passes ("all timesteps").

    Per element of each multi-pass layer in the population, computes the
    RANGE (max - min, in float64) of the retained activation across the pass
    window and selects elements whose range is ``<= tol``. ``passes=None``
    (default) uses every population pass of each layer; an explicit iterable
    of 1-based pass indices restricts the window (every named pass must be in
    the population). Addressing is pass-qualified throughout; a bare layer
    label in ``within=`` is the all-passes Layer spelling, never one silent
    pass. A layer contributing fewer than two window passes refuses
    ``population_too_small`` (a single-pass "stability" claim is vacuous);
    cross-pass shape drift refuses ``mask_shape_mismatch``. The mask lands on
    EVERY window pass-site (the claim is about the unit across the window,
    so ``do()`` on it edits every window pass); the element population is the
    intersection of the window entries' masks. An element with a NaN at any
    window pass has a NaN range and is never selected. Complex payloads
    refuse (no total order). ``provenance.relation`` is ``exact`` — the name
    scopes the claim to THIS capture's passes, complete evidence (the
    ``low_variance`` precedent); the unstable selection is the touched-family
    complement ``~stable_across_passes(...)``. DOCUMENTED-UNSTABLE spelling.
    """

    tol = _validate_real_number(tol, "tol", "stable_across_passes")
    if tol < 0:
        raise ValueError(f"stable_across_passes `tol` must be non-negative; got {tol!r}.")
    return Selection(
        _PassTerm(
            stat="stable",
            within=_lift_within(within, "stable_across_passes"),
            tol=tol,
            passes=_validate_passes(passes, "stable_across_passes"),
        ),
        kind="ACT",
    )


def pass_variance(
    within: Any = None,
    *,
    above: float | None = None,
    below: float | None = None,
    passes: Any = None,
) -> Selection:
    """Select units by their variance across a recurrent layer's passes.

    Per element, computes the unbiased (n-1) variance in float64 of the
    retained activation across the pass window and applies strict bounds:
    ``below=`` selects low-variance (steady) units, ``above=`` high-variance
    (swinging) units, both together the open band; at least one bound is
    required. "Variance explodes late in the sequence" is
    ``pass_variance(above=t, passes=[8, 9, 10])``, optionally composed with
    ``pass_variance(below=t, passes=[1, 2, 3])``. Window semantics, the
    two-pass honesty floor, shape-drift and complex refusals, NaN exclusion,
    mask placement (every window pass-site), and the ``exact`` provenance
    relation all follow :func:`stable_across_passes`. DOCUMENTED-UNSTABLE
    spelling.
    """

    if above is None and below is None:
        raise ValueError(
            "pass_variance requires at least one bound: `above=` (high-variance) "
            "and/or `below=` (low-variance)."
        )
    if above is not None:
        above = _validate_real_number(above, "above", "pass_variance")
    if below is not None:
        below = _validate_real_number(below, "below", "pass_variance")
    return Selection(
        _PassTerm(
            stat="variance",
            within=_lift_within(within, "pass_variance"),
            above=above,
            below=below,
            passes=_validate_passes(passes, "pass_variance"),
        ),
        kind="ACT",
    )


register_term_resolver(_CompareTerm, _resolve_compare_term)
register_term_resolver(_PassTerm, _resolve_pass_term)
