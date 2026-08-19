"""Value-based and statistical selection producers (L6 producer wave).

Selects by WHAT VALUES ARE, not just where. Two producer families, both
returning :class:`~torchlens.selection.Selection` queries that compose with
the full ``| & - ~`` algebra and resolve explicitly against one trace:

- VALUE producers read the resolution trace's OWN retained activations:
  ``top_k`` / ``top_fraction`` (rank criteria), ``threshold`` (magnitude
  bands), ``sign`` (positive / negative / zero / nonzero — the sparsity
  masks). Their masks are computed from saved payloads at resolve time, so
  every claim is about THIS capture: ``provenance.relation == "exact"``.

- STATISTICAL producers are inherently MULTI-SAMPLE: "dead" means "never
  fired across N inputs", which no single trace can express. They take an
  explicit ``samples=`` evidence set (an iterable of Traces — a ``Bundle``
  iterates its members — with at least TWO samples; the single-trace
  "zero in this capture" form is deliberately a DIFFERENT spelling,
  ``sign(site, "zero")``, so one name never silently means two things).
  The resolution trace contributes the site GEOMETRY and the population;
  the samples contribute the evidence. Include the resolution trace in
  ``samples`` if its values should count as evidence.

PROVENANCE HONESTY (pinned by tests): ``dead`` and ``saturated`` make
DISPOSITIONAL claims — a unit observed silent/pinned on N samples may still
fire elsewhere, so the observed mask is a SUPERSET of the true support and
declares ``relation="upper_bound"``. ``low_variance`` names the sample
statistic itself — an exact claim about the provided evidence set — and
declares ``relation="exact"`` with the sample count disclosed in
``provenance.source``. Population restriction composes through the normative
JOIN table, never by overwriting a relation.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (megasprint provisional-name protocol). Producers are ACT-kind
only in v1: a PARAM/EDGE ``within`` refuses ``selection_kind_incompatible``
(``Param`` records carry no tensor payload to read; a PARAM value producer
is a named possibility, not a promise).

Elements whose value is NaN never satisfy any value criterion (comparison
semantics); rank producers exclude them from the candidate population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .errors._base import TorchLensError
from .selection import (
    ResolvedSelection,
    Selection,
    SelectionError,
    SelectionProvenance,
    SiteEntry,
    _act_entry,
    _find_act_ops,
    _forward_ops,
    _join_relation,
    _lift,
    _mask_from_dense,
    _mask_whole,
    _unresolvable,
    _WholeSiteTerm,
    register_term_resolver,
)

__all__ = [
    "dead",
    "low_variance",
    "saturated",
    "sign",
    "threshold",
    "top_fraction",
    "top_k",
]

_RANK_CRITERIA = ("top_k", "top_fraction")
_SIGN_CLASSES = ("positive", "negative", "zero", "nonzero")


# ---------------------------------------------------------------------------
# Shared operand handling.
# ---------------------------------------------------------------------------


def _lift_within(within: Any, producer: str) -> Any:
    """Normalize a ``within`` operand to a Selection-shaped population.

    Accepts ``None`` (default population: every retained tensor site of the
    resolution trace), a site label string (whole-site population; a bare
    layer label on a multi-pass layer is the all-passes Layer spelling), or
    anything selection-shaped. Non-ACT kinds refuse typed.
    """

    if within is None:
        return None
    if isinstance(within, str):
        if not within:
            raise ValueError(f"{producer} `within` site label must be a non-empty string.")
        return Selection(_WholeSiteTerm(site_label=within, pass_index=None), kind="ACT")
    lifted = _lift(within)
    if lifted is None:
        raise ValueError(
            f"{producer} `within` must be None, a site label string, or a "
            "selection-shaped producer implementing __selection__; got "
            f"{type(within).__name__}."
        )
    if lifted.kind != "ACT":
        raise SelectionError(
            f"{producer} reads retained ACTIVATION payloads, so its population "
            f"must be an ACT selection; got a {lifted.kind} selection. "
            "(A PARAM value producer is a named possibility, not a promise: "
            "Param records carry no tensor payload to read.)",
            code="selection_kind_incompatible",
            left_kind="ACT",
            right_kind=lifted.kind,
            operator="within",
        )
    return lifted


def _resolve_population(within: Any, trace: Any) -> ResolvedSelection:
    """Resolve the population operand against the resolution trace."""

    if within is None:
        return _default_population(trace)
    if isinstance(within, ResolvedSelection):
        if within._trace is not trace:
            raise SelectionError(
                "the `within` population is bound to a different trace; "
                "re-resolve it against this trace first.",
                code="selection_trace_mismatch",
            )
        return within
    return within.resolve(trace)


def _default_population(trace: Any) -> ResolvedSelection:
    """Build the default population: every retained single-tensor ACT site.

    The population is DEFINED as the sites whose payloads this capture
    retained — value claims can only be made about values that exist. Sites
    with unsaved or non-tensor outputs are not in the population by
    definition (they are not silently "checked and passed").
    """

    entries = []
    for op in _forward_ops(trace):
        if not getattr(op, "has_saved_activation", False):
            continue
        shape = getattr(op, "shape", None)
        if not isinstance(shape, tuple) or not all(isinstance(extent, int) for extent in shape):
            continue
        entries.append(_act_entry(op, _mask_whole(shape), "exact", "saved_sites"))
    return ResolvedSelection(trace, "ACT", entries)


def _op_for_entry(trace: Any, entry: SiteEntry, *, sample_name: str | None = None) -> Any:
    """Return the op backing one population entry on one trace."""

    layer_label, pass_index = entry.site_key
    ops = [
        op
        for op in _find_act_ops(trace, layer_label)
        if (getattr(op, "pass_index", 1) or 1) == pass_index
    ]
    if not ops:
        where = "this trace" if sample_name is None else sample_name
        raise _unresolvable(
            "site_not_in_trace",
            f"site {entry.site_key!r} is not present on {where}.",
            site=layer_label,
            **({} if sample_name is None else {"sample": sample_name}),
        )
    return ops[0]


def _read_saved_value(
    trace: Any, entry: SiteEntry, *, sample_name: str | None = None
) -> torch.Tensor:
    """Read one entry's retained activation, refusing typed when absent."""

    op = _op_for_entry(trace, entry, sample_name=sample_name)
    where = "this capture" if sample_name is None else sample_name
    if not getattr(op, "has_saved_activation", False):
        raise _unresolvable(
            "value_not_saved",
            f"value criteria need the saved activation at {op.label!r}, which "
            f"{where} did not retain. Re-capture with a `save=` predicate "
            "covering this site.",
            site=op.label,
            **({} if sample_name is None else {"sample": sample_name}),
        )
    try:
        value = op.out
    except TorchLensError as exc:
        raise _unresolvable(
            "value_not_saved",
            f"value criteria need the saved activation at {op.label!r}, which "
            f"{where} cannot serve: {exc}",
            site=op.label,
            **({} if sample_name is None else {"sample": sample_name}),
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise _unresolvable(
            "non_tensor_site",
            f"site {op.label!r} has a non-tensor output; value criteria address "
            "single-tensor outputs only.",
            site=op.label,
        )
    if tuple(value.shape) != entry.shape:
        raise _unresolvable(
            "mask_shape_mismatch",
            f"saved activation at {op.label!r} on {where} has shape "
            f"{tuple(value.shape)!r}, which does not match the population's "
            f"index space {entry.shape!r}.",
            site=op.label,
            **({} if sample_name is None else {"sample": sample_name}),
        )
    return value.detach().cpu()


def _refuse_complex(value: torch.Tensor, producer: str, site: Any) -> None:
    """Refuse ordered comparisons on complex payloads (no total order)."""

    if value.is_complex():
        raise _unresolvable(
            "value_criterion_invalid",
            f"{producer} orders/compares raw values, and complex tensors have "
            f"no total order (site {site!r}). Use by='abs' (magnitude) where "
            "the producer offers it.",
            site=site,
        )


def _entry_with_mask(
    entry: SiteEntry, dense: torch.Tensor, relation: str, source: str
) -> SiteEntry:
    """Rebuild one population entry with a criterion mask and provenance."""

    return SiteEntry(
        kind=entry.kind,
        site_key=entry.site_key,
        provenance=SelectionProvenance(
            relation=_join_relation(entry.provenance.relation, relation),
            source=source,
        ),
        _mask=_mask_from_dense(entry.shape, dense),
        structural_site_key=entry.structural_site_key,
    )


def _validate_real_number(value: Any, name: str, producer: str) -> float:
    """Validate one real, finite-or-infinite numeric criterion parameter."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{producer} `{name}` must be a real number; got {value!r}.")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{producer} `{name}` must not be NaN.")
    return float(value)


# ---------------------------------------------------------------------------
# Value producers (single-capture; read the resolution trace's payloads).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValueTerm:
    """AST leaf for the value-criterion producers (top_k / threshold / sign)."""

    criterion: str
    within: Any
    k: int | None = None
    fraction: float | None = None
    by: str = "value"
    largest: bool = True
    above: float | None = None
    below: float | None = None
    which: str | None = None
    tol: float = 0.0

    def __repr__(self) -> str:
        """Return the compact constructor-shaped disclosure."""

        population = "saved_sites" if self.within is None else repr(self.within)
        if self.criterion == "top_k":
            head = f"top_k(k={self.k}, by={self.by!r}, largest={self.largest}"
        elif self.criterion == "top_fraction":
            head = f"top_fraction(fraction={self.fraction}, by={self.by!r}, largest={self.largest}"
        elif self.criterion == "threshold":
            head = f"threshold(above={self.above}, below={self.below}, by={self.by!r}"
        else:
            head = f"sign({self.which!r}, tol={self.tol}"
        return f"{head}, within={population})"


def _rank_keys(value: torch.Tensor, by: str, producer: str, site: Any) -> torch.Tensor:
    """Return float64 ranking keys for one site's values (NaN preserved)."""

    if by == "abs":
        keys = value.abs()
    else:
        _refuse_complex(value, producer, site)
        keys = value
    if keys.dtype == torch.bool:
        keys = keys.to(torch.uint8)
    # float64 exactly represents every float32/16/bfloat16/int32 value; the
    # int64 tail beyond 2**53 is a documented precision residual.
    return keys.to(torch.float64)


def _resolve_rank_criterion(node: _ValueTerm, trace: Any) -> ResolvedSelection:
    """Resolve a global rank criterion (top_k / top_fraction) exactly.

    Ranking is GLOBAL across the population with a deterministic tie-break:
    stable sort over the concatenation of sites in canonical order, so ties
    resolve by (site order, flat index). NaN elements never enter the
    candidate population.
    """

    population = _resolve_population(node.within, trace)
    per_entry: list[tuple[SiteEntry, torch.Tensor, torch.Tensor]] = []
    for entry in population:
        value = _read_saved_value(trace, entry)
        keys = _rank_keys(value, node.by, node.criterion, entry.site_key)
        valid = entry._mask._dense_ro() & ~torch.isnan(keys)
        per_entry.append((entry, keys.reshape(-1), valid.reshape(-1)))
    total_valid = int(sum(valid.sum().item() for _, _, valid in per_entry))
    if node.criterion == "top_fraction":
        assert node.fraction is not None
        k = math.ceil(node.fraction * total_valid)
        source = f"top_fraction(fraction={node.fraction}, by={node.by!r}, largest={node.largest})"
    else:
        assert node.k is not None
        k = node.k
        source = f"top_k(k={k}, by={node.by!r}, largest={node.largest})"
        if k > total_valid:
            raise _unresolvable(
                "population_too_small",
                f"top_k needs {k} elements but the population has only "
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


def _resolve_elementwise_criterion(node: _ValueTerm, trace: Any) -> ResolvedSelection:
    """Resolve an elementwise value criterion (threshold / sign) exactly."""

    population = _resolve_population(node.within, trace)
    entries: list[SiteEntry] = []
    for entry in population:
        value = _read_saved_value(trace, entry)
        if node.criterion == "threshold":
            if node.by == "abs":
                compared = value.abs()
            else:
                _refuse_complex(value, "threshold", entry.site_key)
                compared = value
            dense = torch.ones(entry.shape, dtype=torch.bool)
            if node.above is not None:
                dense &= compared > node.above
            if node.below is not None:
                dense &= compared < node.below
            source = f"threshold(above={node.above}, below={node.below}, by={node.by!r})"
        else:
            which = node.which
            if which in ("zero", "nonzero"):
                magnitude = value.abs()
                dense = magnitude <= node.tol if which == "zero" else magnitude > node.tol
            else:
                _refuse_complex(value, "sign", entry.site_key)
                dense = value > node.tol if which == "positive" else value < -node.tol
            source = f"sign({which!r}, tol={node.tol})"
        dense &= entry._mask._dense_ro()
        entries.append(_entry_with_mask(entry, dense, "exact", source))
    return ResolvedSelection(trace, "ACT", entries)


def _resolve_value_term(node: _ValueTerm, trace: Any) -> ResolvedSelection:
    """Resolve one value-criterion term against the trace's saved payloads."""

    if node.criterion in _RANK_CRITERIA:
        return _resolve_rank_criterion(node, trace)
    return _resolve_elementwise_criterion(node, trace)


def _validate_rank_common(by: str, largest: Any, producer: str) -> None:
    """Validate the shared rank-producer parameters."""

    if by not in ("value", "abs"):
        raise ValueError(f"{producer} `by` must be 'value' or 'abs'; got {by!r}.")
    if not isinstance(largest, bool):
        raise ValueError(f"{producer} `largest` must be a bool; got {largest!r}.")


def top_k(
    within: Any = None,
    k: int = 1,
    *,
    by: str = "value",
    largest: bool = True,
) -> Selection:
    """Select the k globally highest-valued elements of a population.

    Ranks the resolution trace's retained activations across the whole
    population (``within=None`` means every retained tensor site) and selects
    exactly ``k`` elements, without replacement, with a deterministic
    tie-break (stable sort; ties resolve by canonical site order, then flat
    index). ``by='abs'`` ranks magnitudes; ``largest=False`` selects the
    bottom-k. NaN elements never enter the candidate population. A
    population with fewer than ``k`` rankable elements refuses
    ``selection_unresolvable`` / ``population_too_small`` at resolve time.
    ``provenance.relation`` is ``exact``: the claim is about THIS capture's
    values. DOCUMENTED-UNSTABLE spelling.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"top_k `k` must be a non-negative int; got {k!r}.")
    _validate_rank_common(by, largest, "top_k")
    return Selection(
        _ValueTerm(
            criterion="top_k", within=_lift_within(within, "top_k"), k=k, by=by, largest=largest
        ),
        kind="ACT",
    )


def top_fraction(
    within: Any = None,
    fraction: float = 0.01,
    *,
    by: str = "value",
    largest: bool = True,
) -> Selection:
    """Select the top ``fraction`` of a population by value.

    Exactly :func:`top_k` with ``k = ceil(fraction * population)`` computed
    at resolve time against the rankable (non-NaN) population size, so
    "top 1%" of a non-empty population is never zero elements. ``fraction``
    must lie in ``[0, 1]``. DOCUMENTED-UNSTABLE spelling.
    """

    fraction = _validate_real_number(fraction, "fraction", "top_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"top_fraction `fraction` must be in [0, 1]; got {fraction!r}.")
    _validate_rank_common(by, largest, "top_fraction")
    return Selection(
        _ValueTerm(
            criterion="top_fraction",
            within=_lift_within(within, "top_fraction"),
            fraction=fraction,
            by=by,
            largest=largest,
        ),
        kind="ACT",
    )


def threshold(
    within: Any = None,
    *,
    above: float | None = None,
    below: float | None = None,
    by: str = "value",
) -> Selection:
    """Select elements by strict value comparison ("everything above 0.5").

    ``above=`` keeps elements with value strictly greater; ``below=``
    strictly less; both together select the open band. At least one bound is
    required. ``by='abs'`` compares magnitudes. NaN elements never satisfy a
    comparison. ``provenance.relation`` is ``exact``. DOCUMENTED-UNSTABLE
    spelling.
    """

    if above is None and below is None:
        raise ValueError("threshold requires at least one bound: `above=` and/or `below=`.")
    if above is not None:
        above = _validate_real_number(above, "above", "threshold")
    if below is not None:
        below = _validate_real_number(below, "below", "threshold")
    if by not in ("value", "abs"):
        raise ValueError(f"threshold `by` must be 'value' or 'abs'; got {by!r}.")
    return Selection(
        _ValueTerm(
            criterion="threshold",
            within=_lift_within(within, "threshold"),
            above=above,
            below=below,
            by=by,
        ),
        kind="ACT",
    )


def sign(
    within: Any = None,
    which: str = "positive",
    *,
    tol: float = 0.0,
) -> Selection:
    """Select elements by sign class: the negatives, the zeros, the sparsity mask.

    ``which`` is one of ``'positive'`` (value > tol), ``'negative'``
    (value < -tol), ``'zero'`` (|value| <= tol — the sparsity mask), or
    ``'nonzero'`` (|value| > tol). ``'zero'``/``'nonzero'`` accept complex
    payloads (magnitude); ordered classes refuse them typed. This is also
    the SINGLE-CAPTURE "didn't fire on this input" spelling — the
    multi-sample "never fires" claim is :func:`dead`, deliberately a
    different name. ``provenance.relation`` is ``exact``.
    DOCUMENTED-UNSTABLE spelling.
    """

    if which not in _SIGN_CLASSES:
        raise ValueError(f"sign `which` must be one of {_SIGN_CLASSES}; got {which!r}.")
    tol = _validate_real_number(tol, "tol", "sign")
    if tol < 0:
        raise ValueError(f"sign `tol` must be non-negative; got {tol!r}.")
    return Selection(
        _ValueTerm(criterion="sign", within=_lift_within(within, "sign"), which=which, tol=tol),
        kind="ACT",
    )


# ---------------------------------------------------------------------------
# Statistical producers (multi-sample; explicit evidence set).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StatTerm:
    """AST leaf for the multi-sample statistical producers."""

    stat: str
    samples: tuple[Any, ...]
    within: Any
    tol: float = 0.0
    low: float | None = None
    high: float | None = None
    threshold: float | None = None

    def __repr__(self) -> str:
        """Return the compact disclosure (never dumps sample traces)."""

        population = "saved_sites" if self.within is None else repr(self.within)
        if self.stat == "dead":
            head = f"dead(n_samples={len(self.samples)}, tol={self.tol}"
        elif self.stat == "saturated":
            head = (
                f"saturated(n_samples={len(self.samples)}, low={self.low}, "
                f"high={self.high}, tol={self.tol}"
            )
        else:
            head = f"low_variance(n_samples={len(self.samples)}, threshold={self.threshold}"
        return f"{head}, within={population})"


def _sample_values(node: _StatTerm, entry: SiteEntry) -> list[torch.Tensor]:
    """Read one entry's activation from every sample trace (typed refusals)."""

    values = []
    for index, sample in enumerate(node.samples):
        values.append(_read_saved_value(sample, entry, sample_name=f"samples[{index}]"))
    return values


def _resolve_stat_term(node: _StatTerm, trace: Any) -> ResolvedSelection:
    """Resolve one multi-sample statistic against the population geometry.

    The resolution trace supplies geometry and population; each sample must
    hold a retained, shape-identical activation at every population site
    (missing sites, unsaved payloads, and shape drift refuse typed rather
    than silently shrinking the evidence set).
    """

    population = _resolve_population(node.within, trace)
    entries: list[SiteEntry] = []
    for entry in population:
        values = _sample_values(node, entry)
        if node.stat == "dead":
            dense = torch.ones(entry.shape, dtype=torch.bool)
            for value in values:
                dense &= value.abs() <= node.tol
            relation = "upper_bound"
            source = f"dead(n_samples={len(values)}, tol={node.tol})"
        elif node.stat == "saturated":
            dense = torch.zeros(entry.shape, dtype=torch.bool)
            for bound in (node.low, node.high):
                if bound is None:
                    continue
                pinned = torch.ones(entry.shape, dtype=torch.bool)
                for value in values:
                    pinned &= (value - bound).abs() <= node.tol
                dense |= pinned
            relation = "upper_bound"
            source = (
                f"saturated(n_samples={len(values)}, low={node.low}, "
                f"high={node.high}, tol={node.tol})"
            )
        else:
            stacked = torch.stack([value.to(torch.float64) for value in values])
            dense = torch.var(stacked, dim=0) < node.threshold
            relation = "exact"
            source = f"low_variance(n_samples={len(values)}, threshold={node.threshold})"
        dense &= entry._mask._dense_ro()
        entries.append(_entry_with_mask(entry, dense, relation, source))
    return ResolvedSelection(trace, "ACT", entries)


def _validate_samples(samples: Any, producer: str, single_trace_hint: str) -> tuple[Any, ...]:
    """Validate the evidence set: an iterable of >= 2 Traces (Bundle iterates)."""

    from .data_classes.trace import Trace

    if isinstance(samples, (str, bytes, torch.Tensor, Trace)):
        raise ValueError(
            f"{producer} `samples` must be an iterable of Traces (a Bundle "
            "iterates its members), not a single object."
        )
    collected = tuple(samples)
    for index, sample in enumerate(collected):
        if not isinstance(sample, Trace):
            raise ValueError(
                f"{producer} `samples[{index}]` is not a Trace (got {type(sample).__name__})."
            )
    if len(collected) < 2:
        raise ValueError(
            f"{producer} is a MULTI-SAMPLE claim and needs at least 2 sample "
            f"traces (got {len(collected)}). {single_trace_hint}"
        )
    return collected


def dead(
    samples: Any,
    *,
    within: Any = None,
    tol: float = 0.0,
) -> Selection:
    """Select units that never fired across an explicit multi-sample evidence set.

    An element is selected when its magnitude is ``<= tol`` in EVERY sample
    trace. "Dead" is a dispositional claim — silent-on-N-samples is a
    SUPERSET of truly-dead — so entries declare
    ``provenance.relation="upper_bound"`` with the sample count disclosed in
    ``provenance.source``. Requires at least TWO samples; the single-capture
    "zero on this input" form is deliberately the different spelling
    ``sign(site, 'zero')``. The resolution trace supplies geometry and
    population only; include it in ``samples`` to count it as evidence.
    DOCUMENTED-UNSTABLE spelling.
    """

    tol = _validate_real_number(tol, "tol", "dead")
    if tol < 0:
        raise ValueError(f"dead `tol` must be non-negative; got {tol!r}.")
    collected = _validate_samples(
        samples,
        "dead",
        "For the single-capture zero-activation mask, spell it sign(site, 'zero').",
    )
    return Selection(
        _StatTerm(stat="dead", samples=collected, within=_lift_within(within, "dead"), tol=tol),
        kind="ACT",
    )


def saturated(
    samples: Any,
    *,
    low: float | None = None,
    high: float | None = None,
    tol: float = 0.0,
    within: Any = None,
) -> Selection:
    """Select units always pinned at a declared bound across the evidence set.

    An element is selected when it sits within ``tol`` of the SAME declared
    bound (``low=`` and/or ``high=``; at least one required — saturation is
    relative to the nonlinearity's range, e.g. ``low=0.0, high=1.0`` for a
    sigmoid) in EVERY sample trace. Like :func:`dead`, this is a
    dispositional claim from finite evidence: ``provenance.relation`` is
    ``upper_bound`` and requires at least two samples. DOCUMENTED-UNSTABLE
    spelling.
    """

    if low is None and high is None:
        raise ValueError(
            "saturated requires at least one declared bound (`low=` and/or "
            "`high=`): saturation is relative to the nonlinearity's range."
        )
    if low is not None:
        low = _validate_real_number(low, "low", "saturated")
    if high is not None:
        high = _validate_real_number(high, "high", "saturated")
    tol = _validate_real_number(tol, "tol", "saturated")
    if tol < 0:
        raise ValueError(f"saturated `tol` must be non-negative; got {tol!r}.")
    collected = _validate_samples(
        samples,
        "saturated",
        "For a single capture, spell the pinned-band mask with threshold(...).",
    )
    return Selection(
        _StatTerm(
            stat="saturated",
            samples=collected,
            within=_lift_within(within, "saturated"),
            low=low,
            high=high,
            tol=tol,
        ),
        kind="ACT",
    )


def low_variance(
    samples: Any,
    *,
    threshold: float,
    within: Any = None,
) -> Selection:
    """Select units whose variance across the evidence set is below a threshold.

    Elementwise unbiased (n-1) variance across the sample traces, computed
    in float64, strictly less than ``threshold``. Unlike :func:`dead`, the
    name claims the SAMPLE STATISTIC itself, so ``provenance.relation`` is
    ``exact`` (about the provided evidence set, whose size the source
    disclosure carries); it is not a bound on the distributional variance.
    The high-variance selection is its touched-family complement:
    ``~low_variance(...)``. Requires at least two samples.
    DOCUMENTED-UNSTABLE spelling.
    """

    threshold = _validate_real_number(threshold, "threshold", "low_variance")
    collected = _validate_samples(
        samples,
        "low_variance",
        "Variance of one sample is identically zero and claims nothing.",
    )
    return Selection(
        _StatTerm(
            stat="low_variance",
            samples=collected,
            within=_lift_within(within, "low_variance"),
            threshold=threshold,
        ),
        kind="ACT",
    )


register_term_resolver(_ValueTerm, _resolve_value_term)
register_term_resolver(_StatTerm, _resolve_stat_term)
