"""Selection-addressed occlusion attribution over captured traces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from torchlens.attribution._core import AttributionError, AttributionResult
from torchlens.intervention import mean_ablate, zero_ablate
from torchlens.selection import ResolvedSelection, Selection

_OcclusionBaseline: TypeAlias = Literal["zeros", "mean", "blur"]
_OcclusionScore: TypeAlias = Callable[[Any], Tensor | float]


def _score_trace(trace: Any, score: _OcclusionScore) -> Tensor:
    """Evaluate and validate one scalar trace score.

    Parameters
    ----------
    trace
        Original or occluded Trace passed to the user scorer.
    score
        Callable returning one real scalar.

    Returns
    -------
    Tensor
        Detached scalar score.

    Raises
    ------
    AttributionError
        If the scorer does not return one finite real scalar.
    """

    value = score(trace)
    if isinstance(value, bool) or not isinstance(value, (Tensor, int, float)):
        raise AttributionError("occlusion score must return one real scalar")
    tensor = value if isinstance(value, Tensor) else torch.tensor(float(value))
    if tensor.numel() != 1 or tensor.is_complex():
        raise AttributionError("occlusion score must return one real scalar")
    scalar = tensor.detach().reshape(())
    if not bool(torch.isfinite(scalar)):
        raise AttributionError("occlusion score must return a finite scalar")
    return scalar


def _blur_edit(kernel_size: int) -> Callable[..., Tensor]:
    """Build a spatial mean-blur edit for the selection scatter engine.

    Parameters
    ----------
    kernel_size
        Odd spatial averaging kernel width.

    Returns
    -------
    Callable[..., Tensor]
        Hook that blurs an entire ``N,C,H,W`` activation. The selection engine
        scatters only selected elements from that replacement.
    """

    def blur(out: Tensor, *, hook: Any) -> Tensor:
        """Return a same-shaped spatially blurred activation."""

        del hook
        if out.ndim != 4 or not (out.is_floating_point() or out.is_complex()):
            raise AttributionError(
                "baseline='blur' requires a floating or complex N,C,H,W activation"
            )
        padding = kernel_size // 2
        if out.is_complex():
            real = F.avg_pool2d(out.real, kernel_size, stride=1, padding=padding)
            imag = F.avg_pool2d(out.imag, kernel_size, stride=1, padding=padding)
            return torch.complex(real, imag)
        return F.avg_pool2d(out, kernel_size, stride=1, padding=padding)

    return blur


def _occlusion_edit(baseline: _OcclusionBaseline, blur_kernel_size: int) -> Any:
    """Return the shipped or local edit implementing an explicit baseline.

    Parameters
    ----------
    baseline
        Named replacement policy.
    blur_kernel_size
        Odd width used only for ``baseline="blur"``.

    Returns
    -------
    Any
        Edit accepted by ``Trace.do``.
    """

    if baseline == "zeros":
        return zero_ablate()
    if baseline == "mean":
        return mean_ablate()
    if baseline == "blur":
        if isinstance(blur_kernel_size, bool) or blur_kernel_size < 3 or blur_kernel_size % 2 == 0:
            raise AttributionError("blur_kernel_size must be an odd integer of at least 3")
        return _blur_edit(blur_kernel_size)
    raise AttributionError("baseline must be 'zeros', 'mean', or 'blur'")


def _resolve_selection(trace: Any, selection: Any) -> ResolvedSelection:
    """Resolve a Selection or region producer against ``trace``.

    Parameters
    ----------
    trace
        Trace owning the values to occlude.
    selection
        Selection query, resolved selection, or ``__selection__`` producer.

    Returns
    -------
    ResolvedSelection
        Session-bound concrete selection.

    Raises
    ------
    AttributionError
        If ``selection`` does not implement the Selection protocol.
    """

    lifted = selection
    if not isinstance(lifted, (Selection, ResolvedSelection)):
        converter = getattr(lifted, "__selection__", None)
        if not callable(converter):
            raise AttributionError("occlusion selection must implement __selection__()")
        lifted = converter()
    if isinstance(lifted, ResolvedSelection):
        return lifted if lifted._trace is trace else lifted.align_to(trace)
    if isinstance(lifted, Selection):
        return lifted.resolve(trace)
    raise AttributionError("occlusion selection did not produce a Selection")


def occlusion(
    trace: Any,
    selection: Any,
    *,
    score: _OcclusionScore,
    baseline: _OcclusionBaseline = "zeros",
    blur_kernel_size: int = 3,
) -> AttributionResult:
    """Score the effect of explicitly occluding one resolved activation region.

    The result is ``score(original) - score(occluded)``. ``baseline`` is
    mandatory vocabulary even though it has a documented ``"zeros"`` default:
    zeros, the selected site's global mean, and a spatial blur answer different
    counterfactual questions. Occlusion runs through ``Trace.fork().do(...)`` so
    it inherits Selection masking and the shipped edit-then-scatter contract.

    Parameters
    ----------
    trace
        Live captured Trace with replayable saved values.
    selection
        Selection query, resolved selection, or region producer to occlude.
    score
        Callable receiving a Trace and returning one finite real scalar. It is
        invoked once on the original and once on the occluded fork.
    baseline
        Replacement policy: ``"zeros"`` (default), ``"mean"``, or ``"blur"``.
    blur_kernel_size
        Odd spatial averaging width for ``baseline="blur"``.

    Returns
    -------
    AttributionResult
        Scalar scored delta in ``values`` plus both endpoint scores and baseline
        disclosure in ``extra``.
    """

    resolved = _resolve_selection(trace, selection)
    if resolved._kind != "ACT":
        raise AttributionError("occlusion currently supports ACT selections only")
    edit = _occlusion_edit(baseline, blur_kernel_size)
    original_score = _score_trace(trace, score)
    fork = trace.fork(name="attribution_occlusion")
    try:
        fork_selection = resolved.align_to(fork)
        fork.do(fork_selection, edit)
        occluded_score = _score_trace(fork, score)
    finally:
        fork.cleanup()
    delta = original_score - occluded_score
    return AttributionResult(
        method="occlusion",
        values=delta,
        target_repr="trace score",
        extra={
            "baseline": baseline,
            "blur_kernel_size": blur_kernel_size if baseline == "blur" else None,
            "original_score": original_score,
            "occluded_score": occluded_score,
            "selection_digest": resolved.resolve_digest,
        },
    )


__all__ = ["occlusion"]
