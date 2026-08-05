"""Brute-force oracle hardening for receptive/projective-field geometry (r22).

Every geometry claim here is pinned against an independent ground truth that
never consults the TorchLens geometry engine: potential support is measured by
perturbing one element at a time and observing which outputs change (reverse
for receptive fields). Exact boxes must equal the true hull; upper bounds must
contain it. The suites were mutation-proven against the r21 audit defects
(max-pool dilation dropped, positional antialias missed, strided-slice
projective lattice loss, the line-637 bare assert, empty-box ``slices()``).
"""

from __future__ import annotations


import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.receptive_field import ReceptiveFieldValidationStatus


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Independent brute-force oracles (no TorchLens geometry involved)
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).detach().clone()


def true_receptive_support(
    model: nn.Module,
    x: torch.Tensor,
    out_index: tuple[int, ...],
    deltas: tuple[float, ...] = (1000.0, -1000.0, 0.5),
) -> list[tuple[int, ...]]:
    """Input elements whose perturbation changes ``out[out_index]``."""

    base = _forward(model, x)
    shape = tuple(x.shape)
    hits: list[tuple[int, ...]] = []
    for flat in range(x.numel()):
        index = []
        remaining = flat
        for extent in reversed(shape):
            index.append(remaining % extent)
            remaining //= extent
        index_tuple = tuple(reversed(index))
        for delta in deltas:
            perturbed = x.detach().clone()
            perturbed[index_tuple] += delta
            out = _forward(model, perturbed)
            if not torch.allclose(out[out_index], base[out_index]):
                hits.append(index_tuple)
                break
    return hits


def true_projective_support(
    model: nn.Module,
    x: torch.Tensor,
    source_index: tuple[int, ...],
    deltas: tuple[float, ...] = (1000.0, -1000.0, 0.5),
) -> list[tuple[int, ...]]:
    """Output elements whose value changes when ``x[source_index]`` moves."""

    base = _forward(model, x)
    hits: set[tuple[int, ...]] = set()
    for delta in deltas:
        perturbed = x.detach().clone()
        perturbed[source_index] += delta
        out = _forward(model, perturbed)
        for row in (out != base).nonzero(as_tuple=False).tolist():
            hits.add(tuple(int(value) for value in row))
    return sorted(hits)


def hull(indices: list[tuple[int, ...]], axis: int) -> tuple[int, int] | None:
    """Half-open hull of one axis over a support set, or ``None`` when empty."""

    if not indices:
        return None
    values = [index[axis] for index in indices]
    return (min(values), max(values) + 1)


def box_bounds(box: object, spatial_rank: int) -> list[tuple[int | None, int | None]]:
    """Clipped bounds of the trailing ``spatial_rank`` axes of a box."""

    return [(axis.clipped_start, axis.clipped_stop) for axis in box.axes[-spatial_rank:]]


def capture(model: nn.Module, x: torch.Tensor) -> object:
    """Capture with the full gradient-verification triple armed."""

    return tl.trace(
        model,
        x.detach().clone().requires_grad_(True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def sole_input(trace: object) -> object:
    return next(op for op in trace.layer_list if op.is_input)


def op_named(trace: object, fragment: str) -> object:
    matches = [op for op in trace.layer_list if fragment in op.func_name]
    assert matches, f"no op matching {fragment!r}"
    return matches[-1]


def assert_box_against_truth(
    box: object,
    truth: list[tuple[int, ...]],
    spatial_axes: tuple[int, ...],
    *,
    context: str,
) -> None:
    """Exact boxes equal the true hull; upper bounds contain it; empty is empty."""

    bounds = [(axis.clipped_start, axis.clipped_stop) for axis in box.axes]
    if not truth:
        if box.exact:
            assert box.empty, f"{context}: exact box must be empty (true support empty)"
        return
    assert not box.empty, f"{context}: box empty but true support {truth}"
    for axis in spatial_axes:
        true_hull = hull(truth, axis)
        assert true_hull is not None
        start, stop = bounds[axis]
        assert start is not None and stop is not None, f"{context}: axis {axis} unbounded"
        if box.exact:
            assert (start, stop) == true_hull, (
                f"{context}: exact axis {axis} reported {(start, stop)} != true {true_hull}"
            )
        else:
            assert start <= true_hull[0] and stop >= true_hull[1], (
                f"{context}: axis {axis} bound {(start, stop)} does not contain {true_hull}"
            )


# ---------------------------------------------------------------------------
# G1 — max-pool dilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_maxpool1d_dilation_rf_pf_exact(kernel: int, stride: int, dilation: int) -> None:
    """Exact RF/PF hulls across the max-pool dilation matrix vs perturbation truth."""

    extent = 12
    model = nn.MaxPool1d(kernel_size=kernel, stride=stride, dilation=dilation)
    x = torch.linspace(0.0, 1.0, extent, dtype=torch.float64).reshape(1, 1, extent)
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    source = sole_input(trace)
    n_out = int(pool.shape[-1])
    for out_pos in range(n_out):
        truth = true_receptive_support(model, x, (0, 0, out_pos))
        box = pool.receptive_field.at((out_pos,))
        assert box.exact, f"dilated max-pool RF must stay exact (out {out_pos})"
        assert_box_against_truth(
            box, truth, (2,), context=f"RF k{kernel}s{stride}d{dilation} out{out_pos}"
        )
    for src in range(extent):
        truth = true_projective_support(model, x, (0, 0, src))
        box = source.projective_field.at((src,))
        assert_box_against_truth(
            box, truth, (2,), context=f"PF k{kernel}s{stride}d{dilation} src{src}"
        )


def test_maxpool_dilation_check_and_verify_pass() -> None:
    """The r21 dilated-pool repro now passes containment and full verify()."""

    model = nn.MaxPool1d(kernel_size=3, stride=1, dilation=2)
    x = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8) * 1.0
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    box = pool.receptive_field.at((1,))
    axis = box.axes[-1]
    assert (axis.clipped_start, axis.clipped_stop) == (1, 6)
    verification = tl.receptive_field.verify(trace, units="center")
    assert verification.passed
    assert all(
        result.status is ReceptiveFieldValidationStatus.PASS for result in verification.containment
    )


def test_pool_ceil_mode_positional_and_kwarg_downgrade_exactness() -> None:
    """ceil_mode reaches the rule in both spellings and stays an honest envelope."""

    for model in (
        nn.MaxPool2d(3, stride=2, ceil_mode=True),  # forwarded as a keyword
        nn.AvgPool2d(3, stride=2, ceil_mode=True),  # forwarded positionally
    ):
        trace = capture(model, torch.randn(1, 1, 10, 10, dtype=torch.float64))
        pool = op_named(trace, "pool")
        last = (int(pool.shape[-2]) - 1, int(pool.shape[-1]) - 1)
        box = pool.receptive_field.at(last)
        assert not box.exact, f"{type(model).__name__} ceil_mode window must not claim exact"


def test_plain_maxpool_stays_exact() -> None:
    """Dilation/ceil handling must not disturb the default pooling geometry."""

    model = nn.MaxPool2d(3, stride=2)
    x = torch.randn(1, 1, 10, 10, dtype=torch.float64)
    trace = capture(model, x)
    pool = op_named(trace, "max_pool")
    truth = true_receptive_support(model, x, (0, 0, 1, 1))
    box = pool.receptive_field.at((1, 1))
    assert box.exact
    assert_box_against_truth(box, truth, (2, 3), context="plain maxpool RF")
