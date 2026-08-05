"""Interpolation receptive-field rules with branch-correct source scales."""

from __future__ import annotations

from fractions import Fraction
from math import floor

from .._query import _IndexSet, map_interpolation_index_set
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import number_tuple


_ANTIALIAS_INTERP_SIZE = {"bilinear": 2, "bicubic": 4}
_CUBIC_A = Fraction(-3, 4)
# Rational quantities this close to a filter zero or an integer window bound can
# land on either side of it under ATen's float64 evaluation; such taps and
# bounds are kept (containment) and the axis result is downgraded to inexact.
_FLOAT_AMBIGUITY_MARGIN = Fraction(1, 2**40)


def _antialias_filter_verdict(mode: str, magnitude: Fraction) -> str:
    """Classify one antialiased filter argument as zero, nonzero, or ambiguous.

    Parameters
    ----------
    mode:
        Certified interpolation mode (``"bilinear"`` or ``"bicubic"``).
    magnitude:
        Exact non-negative filter argument ``|t|``.

    Returns
    -------
    str
        ``"zero"`` when the tap weight is provably zero under float64,
        ``"nonzero"`` when provably nonzero, ``"ambiguous"`` at boundaries
        where float rounding can produce either outcome.
    """

    if mode == "bicubic":
        if (
            abs(magnitude - 1) < _FLOAT_AMBIGUITY_MARGIN
            or abs(magnitude - 2) < _FLOAT_AMBIGUITY_MARGIN
        ):
            return "ambiguous"
        # torch's cubic convolution (A=-0.75) is zero exactly at |t|=1 and
        # for |t|>=2, but nonzero on (1, 2).
        if magnitude == 1 or magnitude >= 2:
            return "zero"
        return "nonzero"
    if abs(magnitude - 1) < _FLOAT_AMBIGUITY_MARGIN:
        return "ambiguous"
    return "zero" if magnitude >= 1 else "nonzero"


def _floor_with_margin(value: Fraction, *, prefer_low: bool) -> tuple[int, bool]:
    """Floor a rational bound, widening when float trunc could disagree."""

    low = int(floor(value - _FLOAT_AMBIGUITY_MARGIN))
    high = int(floor(value + _FLOAT_AMBIGUITY_MARGIN))
    if low == high:
        return low, True
    return (low if prefer_low else high), False


def _antialias_output_taps(
    output_index: int, input_extent: int, scale: Fraction, mode: str
) -> tuple[list[int], bool]:
    """Return one output's antialiased source taps with a provable-exactness flag.

    Mirrors ATen's separable antialiased upsampling window (align_corners
    False): ``center = scale * (i + 1/2)``, support ``interp/2 * max(scale, 1)``,
    taps ``[trunc(center - support + 1/2), trunc(center + support + 1/2))``
    clipped to the input, with zero-weight taps removed. Filter arguments that
    sit within float-rounding distance of a zero keep their tap and downgrade
    exactness so the result always contains the true support.
    """

    interp = _ANTIALIAS_INTERP_SIZE[mode]
    support = Fraction(interp, 2) * scale if scale >= 1 else Fraction(interp, 2)
    center = scale * (Fraction(output_index) + Fraction(1, 2))
    lower, lower_ok = _floor_with_margin(center - support + Fraction(1, 2), prefer_low=True)
    upper, upper_ok = _floor_with_margin(center + support + Fraction(1, 2), prefer_low=False)
    exact = lower_ok and upper_ok
    start = max(lower, 0)
    stop = min(upper, input_extent)
    inverse_scale = Fraction(1) / scale if scale >= 1 else Fraction(1)
    taps: list[int] = []
    for tap in range(start, stop):
        magnitude = abs((Fraction(tap) - center + Fraction(1, 2)) * inverse_scale)
        verdict = _antialias_filter_verdict(mode, magnitude)
        if verdict == "zero":
            continue
        if verdict == "ambiguous":
            exact = False
        taps.append(tap)
    return taps, exact


def _antialias_result(
    context: ReceptiveFieldRuleContext,
    mode: str,
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    *,
    align_corners: bool | None,
    scale_factor: tuple[int | float | Fraction, ...] | None,
    recompute_scale_factor: bool | None,
) -> _RuleResult:
    """Emit certified antialiased-interpolation geometry or fail closed."""

    if mode not in _ANTIALIAS_INTERP_SIZE:
        return context.unknown("antialias interpolation is certified only for bilinear/bicubic")
    if align_corners is True:
        return context.unknown(
            "antialias interpolation with align_corners=True is not geometrically certified"
        )
    scales: list[Fraction] = []
    for axis, (input_size, output_size) in enumerate(zip(inputs, outputs, strict=True)):
        if scale_factor is not None and recompute_scale_factor is not True:
            scales.append(Fraction(1) / Fraction(scale_factor[axis]))
        else:
            scales.append(Fraction(input_size, output_size))
    if any(scale <= 0 for scale in scales):
        return context.unknown("antialias interpolation scale is malformed")
    edges = []
    for scale in scales:
        interp = _ANTIALIAS_INTERP_SIZE[mode]
        support = Fraction(interp, 2) * scale if scale >= 1 else Fraction(interp, 2)
        base = scale / 2
        edges.append(
            (
                (scale, base - support - Fraction(1, 2)),
                (scale, base + support - Fraction(1, 2)),
            )
        )

    def map_index_set(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Map one output progression through exact antialiased tap windows."""

        taps: set[int] = set()
        exact = True
        for output_index in output_set.values():
            output_taps, output_exact = _antialias_output_taps(
                int(output_index), inputs[axis], scales[axis], mode
            )
            taps.update(output_taps)
            exact = exact and output_exact
        return (
            _IndexSet.from_values(sorted(taps), exact=exact and output_set.exact),
            exact,
        )

    return context.window_edges(
        tuple(edges),
        exact=False,
        note="antialiased interpolation uses the exact separable tap window",
        map_index_set=map_index_set,
    )


def _interpolation_edges(
    mode: str,
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    *,
    align_corners: bool | None,
    scale_factor: tuple[int | float | Fraction, ...] | None,
    recompute_scale_factor: bool | None,
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]:
    """Return a containing affine envelope for the captured interpolation branch."""

    edges = []
    for axis, (input_size, output_size) in enumerate(zip(inputs, outputs, strict=True)):
        supplied_scale = None if scale_factor is None else Fraction(scale_factor[axis])
        if align_corners and mode in {"linear", "bilinear", "trilinear", "bicubic"}:
            slope = Fraction(0) if output_size == 1 else Fraction(input_size - 1, output_size - 1)
            offset = Fraction(0)
        else:
            slope = (
                Fraction(input_size, output_size)
                if supplied_scale is None or recompute_scale_factor is True
                else Fraction(1, 1) / supplied_scale
            )
            if mode == "nearest":
                offset = Fraction(0)
            elif mode == "nearest-exact":
                offset = slope / 2
            else:
                offset = slope / 2 - Fraction(1, 2)
        radius = 2 if mode == "bicubic" else 1
        edges.append(((slope, offset - radius), (slope, offset + radius)))
    return tuple(edges)


@register_rf_rule(
    "interpolate",
    "upsample",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_bilinear2d",
    "upsample_trilinear3d",
)
def interpolate(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit a phase envelope and exact supported-mode per-unit tap callback."""

    if not context.in_shapes:
        return context.unknown("interpolation is missing its input shape")
    mode = context.cfg("mode", context.arg("mode", None))
    if not isinstance(mode, str):
        return context.unknown("interpolation mode was not captured")
    antialias = context.cfg("antialias", context.arg("antialias", False))
    size = context.cfg("size", context.arg("size", None))
    scale = context.cfg("scale_factor", context.arg("scale_factor", None))
    align_corners = context.arg("align_corners", None)
    recompute = context.arg("recompute_scale_factor", None)
    if size is None and scale is None:
        return context.unknown("interpolation size/scale branch was not captured")
    rank = (
        len(size)
        if isinstance(size, (tuple, list))
        else (len(scale) if isinstance(scale, (tuple, list)) else len(context.out_shape) - 2)
    )
    if rank <= 0:
        return context.unknown("interpolation spatial rank is undeterminable")
    inputs = tuple(int(value) for value in context.in_shapes[0][-rank:])
    outputs = tuple(int(value) for value in context.out_shape[-rank:])
    if mode not in {"linear", "bilinear", "trilinear", "bicubic", "nearest", "nearest-exact"}:
        return context.unknown("interpolation mode is not supported")
    scale_factor = None if scale is None else number_tuple(scale, rank)
    if scale is not None and scale_factor is None:
        return context.unknown("interpolation scale_factor was malformed")
    if antialias is True:
        return _antialias_result(
            context,
            mode,
            inputs,
            outputs,
            align_corners=align_corners if isinstance(align_corners, bool) else None,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute if isinstance(recompute, bool) else None,
        )
    edges = _interpolation_edges(
        mode,
        inputs,
        outputs,
        align_corners=align_corners if isinstance(align_corners, bool) else None,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute if isinstance(recompute, bool) else None,
    )

    def map_index_set(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Map one output progression through branch-correct interpolation taps."""

        return map_interpolation_index_set(
            axis,
            output_set,
            mode=mode,
            input_extent=inputs,
            output_extent=outputs,
            align_corners=align_corners if isinstance(align_corners, bool) else None,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute if isinstance(recompute, bool) else None,
        )

    return context.window_edges(edges, exact=False, map_index_set=map_index_set)
