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
import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# G2 — antialiased interpolation
# ---------------------------------------------------------------------------


class _Interp(nn.Module):
    """Interpolate wrapper covering keyword and positional argument spellings."""

    def __init__(self, *, positional: bool = False, **kwargs: object) -> None:
        super().__init__()
        self.positional = positional
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional:
            return F.interpolate(
                x,
                self.kwargs.get("size"),
                self.kwargs.get("scale_factor"),
                self.kwargs.get("mode", "nearest"),
                self.kwargs.get("align_corners"),
                self.kwargs.get("recompute_scale_factor"),
                self.kwargs.get("antialias", False),
            )
        return F.interpolate(x, **self.kwargs)


@pytest.mark.parametrize(
    ("mode", "in_size", "kwargs"),
    [
        ("bilinear", 4, {"scale_factor": (0.5, 0.5)}),
        ("bilinear", 6, {"size": (2, 2)}),
        ("bilinear", 13, {"size": (7, 7)}),  # odd/odd scale: float-boundary taps
        ("bilinear", 8, {"size": (3, 3)}),
        ("bicubic", 9, {"size": (3, 3)}),  # interior filter zeros (holes)
        ("bicubic", 4, {"scale_factor": (0.5, 0.5)}),
        ("bilinear", 4, {"size": (6, 6)}),  # antialiased upsampling
    ],
)
def test_antialias_interpolate_rf_pf_against_truth(mode: str, in_size: int, kwargs: dict) -> None:
    """Exact AA boxes equal, and inexact ones contain, the perturbation truth."""

    model = _Interp(mode=mode, align_corners=False, antialias=True, **kwargs)
    x = torch.randn(1, 1, in_size, in_size, dtype=torch.float64)
    trace = capture(model, x)
    interp = op_named(trace, "interpolate")
    source = sole_input(trace)
    out_extent = int(interp.shape[-1])
    for out_pos in ((0, 0), (out_extent - 1, out_extent - 1), (0, out_extent // 2)):
        truth = true_receptive_support(model, x, (0, 0, *out_pos), deltas=(0.5, -0.5))
        box = interp.receptive_field.at(out_pos)
        assert_box_against_truth(
            box, truth, (2, 3), context=f"AA RF {mode} in{in_size} out{out_pos}"
        )
    for src in ((0, 1), (in_size - 1, in_size - 1), (in_size // 2, 0)):
        truth = true_projective_support(model, x, (0, 0, *src), deltas=(0.5, -0.5))
        box = source.projective_field.at(src)
        assert_box_against_truth(box, truth, (2, 3), context=f"AA PF {mode} in{in_size} src{src}")


def test_antialias_positional_and_keyword_spellings_agree() -> None:
    """The r21 positional-antialias repro: both spellings give the same exact box."""

    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    boxes = []
    for positional in (False, True):
        model = _Interp(
            positional=positional,
            scale_factor=(0.5, 0.5),
            mode="bilinear",
            align_corners=False if positional else None,
            antialias=True,
        )
        trace = capture(model, x)
        interp = op_named(trace, "interpolate")
        box = interp.receptive_field.at((0, 0))
        boxes.append([(axis.clipped_start, axis.clipped_stop) for axis in box.axes[-2:]])
        assert box.exact
        verification = tl.receptive_field.verify(trace, units="center")
        assert verification.passed
    assert boxes[0] == boxes[1] == [(0, 3), (0, 3)]


def test_antialias_align_corners_true_fails_closed() -> None:
    """Uncertified AA configurations must refuse rather than claim geometry."""

    model = _Interp(size=(3, 3), mode="bicubic", align_corners=True, antialias=True)
    trace = capture(model, torch.randn(1, 1, 7, 7, dtype=torch.float64))
    interp = op_named(trace, "interpolate")
    with pytest.raises(Exception, match="(?i)geometry|gradient"):
        interp.receptive_field.at((0, 0))


class _GetItem(nn.Module):
    """Basic-indexing wrapper for slice-geometry probes."""

    def __init__(self, key: tuple) -> None:
        super().__init__()
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.key]


@pytest.mark.parametrize("start", [0, 1, 2])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_strided_slice_projective_lattice(start: int, step: int) -> None:
    """PF of every source under a strided slice equals perturbation truth.

    Off-lattice sources must be exactly EMPTY; on-lattice sources must map to
    their single surviving output. Guards the forward window-edge transpose's
    integer membership proof.
    """

    if (start, step) == (0, 1):
        pytest.skip("identity slice is a passthrough with no windowed axes")
    extent = 9
    key = (slice(None), slice(None), slice(start, None, step))
    model = _GetItem(key)
    x = torch.arange(extent, dtype=torch.float64).reshape(1, 1, extent) * 1.0
    trace = capture(model, x)
    source = sole_input(trace)
    for src in range(extent):
        truth = true_projective_support(model, x, (0, 0, src))
        box = source.projective_field.at((src,))
        assert box.exact, f"slice start={start} step={step} src={src} must stay exact"
        assert_box_against_truth(
            box, truth, (2,), context=f"slice PF start={start} step={step} src={src}"
        )
    getitem = op_named(trace, "getitem")
    for out_pos in range(int(getitem.shape[-1])):
        truth = true_receptive_support(model, x, (0, 0, out_pos))
        box = getitem.receptive_field.at((out_pos,))
        assert_box_against_truth(
            box, truth, (2,), context=f"slice RF start={start} step={step} out={out_pos}"
        )


def test_strided_slice_spurious_nonempty_pin() -> None:
    """The r21 repro: source 1 of ``x[:, :, ::2]`` must report an EMPTY box."""

    model = _GetItem((slice(None), slice(None), slice(None, None, 2)))
    x = torch.arange(5, dtype=torch.float64).reshape(1, 1, 5) * 1.0
    trace = capture(model, x)
    source = sole_input(trace)
    off_lattice = source.projective_field.at((1,))
    assert off_lattice.empty, "off-lattice source must have an empty projective field"
    assert off_lattice.exact
    on_lattice = source.projective_field.at((2,))
    axis = on_lattice.axes[-1]
    assert (axis.clipped_start, axis.clipped_stop) == (1, 2)


# ---------------------------------------------------------------------------
# Metamorphic laws: adjoint duality, composition, translation equivariance
# ---------------------------------------------------------------------------


def _windowed_hull(box: object) -> tuple[tuple[int, int] | None, ...]:
    """Clipped hull over the trailing windowed axes, ``None`` for empty."""

    if box.empty:
        return (None,)
    return tuple(
        (axis.clipped_start, axis.clipped_stop) for axis in box.axes if axis.kind == "windowed"
    )


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: _GetItem((slice(None), slice(None), slice(1, None, 2))),
        lambda: nn.Conv1d(1, 1, 3, stride=2, bias=False),
        lambda: nn.AvgPool1d(2, stride=2),
        lambda: _Interp(scale_factor=(0.5,), mode="linear", align_corners=False),
    ],
)
def test_adjoint_duality_law(model_factory) -> None:
    """For dense exact windows: ``u in PF(p)`` iff ``p in RF(u)`` (no autograd)."""

    extent = 8
    model = model_factory()
    x = torch.randn(1, 1, extent)
    trace = capture(model, x)
    source = sole_input(trace)
    target = [op for op in trace.layer_list if not op.is_input and not op.is_output][-1]
    out_extent = int(target.shape[-1])
    for src in range(extent):
        pf_box = source.projective_field.at((src,))
        for out_pos in range(out_extent):
            rf_box = target.receptive_field.at((out_pos,))
            if not (pf_box.exact and rf_box.exact):
                continue
            pf_axis = pf_box.axes[-1]
            rf_axis = rf_box.axes[-1]
            in_pf = (
                not pf_box.empty
                and pf_axis.clipped_start is not None
                and pf_axis.clipped_start <= out_pos < pf_axis.clipped_stop
            )
            in_rf = (
                not rf_box.empty
                and rf_axis.clipped_start is not None
                and rf_axis.clipped_start <= src < rf_axis.clipped_stop
            )
            assert in_pf == in_rf, (
                f"{type(model).__name__}: adjoint duality broken at src={src} "
                f"out={out_pos}: u-in-PF={in_pf} p-in-RF={in_rf}"
            )


def test_layer_to_layer_composition_law() -> None:
    """box_input(u) equals the bbox of box_input over box_mid(u) members."""

    model = nn.Sequential(
        nn.Conv1d(1, 1, 3, stride=2, bias=False),
        nn.Conv1d(1, 1, 3, bias=False),
    )
    x = torch.randn(1, 1, 17)
    trace = capture(model, x)
    convs = [op for op in trace.layer_list if "conv" in op.func_name]
    mid, final = convs[0], convs[1]
    source = sole_input(trace)
    for out_pos in range(int(final.shape[-1])):
        full_box = final.receptive_field.at((out_pos,), input=source)
        mid_box = final.receptive_field.at((out_pos,), source=mid)
        mid_axis = mid_box.axes[-1]
        assert mid_axis.clipped_start is not None
        starts, stops = [], []
        for mid_pos in range(mid_axis.clipped_start, mid_axis.clipped_stop):
            inner = mid.receptive_field.at((mid_pos,), input=source)
            inner_axis = inner.axes[-1]
            starts.append(inner_axis.clipped_start)
            stops.append(inner_axis.clipped_stop)
        composed = (min(starts), max(stops))
        full_axis = full_box.axes[-1]
        assert (full_axis.clipped_start, full_axis.clipped_stop) == composed, (
            f"composition broken at out={out_pos}: "
            f"full={(full_axis.clipped_start, full_axis.clipped_stop)} composed={composed}"
        )


def test_translation_equivariance() -> None:
    """Interior units of a pure conv stack shift RFs by exactly jump*d."""

    model = nn.Sequential(
        nn.Conv1d(1, 1, 3, stride=2, bias=False),
        nn.Conv1d(1, 1, 3, dilation=2, bias=False),
    )
    x = torch.randn(1, 1, 33)
    trace = capture(model, x)
    final = [op for op in trace.layer_list if "conv" in op.func_name][-1]
    view = final.receptive_field
    jump = view.jump[-1]
    base = view.at((5,), clip=False)
    base_axis = base.axes[-1]
    for shift in (1, 2, 3):
        shifted = view.at((5 + shift,), clip=False)
        axis = shifted.axes[-1]
        assert axis.index_start - base_axis.index_start == jump * shift
        assert axis.index_stop - base_axis.index_stop == jump * shift


# ---------------------------------------------------------------------------
# Crash-class pins — rank-changing partial-full transposes (line-637 assert)
# ---------------------------------------------------------------------------


class _HyperLinear(nn.Module):
    """F.linear with a computed weight: the audited line-637 crash model."""

    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.randn(4, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.tanh(self.raw))


class _ComputedParent(nn.Module):
    """Partial-axes full rules (softmax/cumsum) fed by a rank-mismatched parent."""

    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.op == "softmax":
            return F.softmax(x + y.mean(), dim=-1)
        return torch.cumsum(x * y.mean(), dim=-1)


def test_hypernetwork_projective_solve_degrades_typed() -> None:
    """The r21 4-op hypernetwork must pass invariants with an UNKNOWN branch."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    trace = tl.trace(_HyperLinear(), torch.randn(2, 5, 3))
    assert check_geometric_metadata_invariants(trace) is True
    tanh = op_named(trace, "tanh")
    descriptors = tanh.projective_field.per_input
    assert descriptors, "weight branch must still produce a projective descriptor"
    for descriptor in descriptors.values():
        assert descriptor.status.value == "unknown"
        assert descriptor.status.value != "exact"


def test_hypernetwork_input_weight_receptive_degrades_typed() -> None:
    """Input-fed computed weights hit the same class in the receptive engine."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    class HyperInput(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.linear(x, torch.tanh(y))

    trace = tl.trace(HyperInput(), (torch.randn(2, 5, 3), torch.randn(4, 3)))
    assert check_geometric_metadata_invariants(trace) is True
    linear = op_named(trace, "linear")
    statuses = {
        role: descriptor.status.value
        for role, descriptor in linear.receptive_field.per_input.items()
    }
    assert statuses.get("input.y") == "unknown"
    assert len(trace.projective_fields().to_pandas()) > 0


@pytest.mark.parametrize("op", ["softmax", "cumsum"])
def test_partial_full_rule_computed_parent_never_crashes(op: str) -> None:
    """softmax/cumsum with rank-mismatched parents solve without assertions."""

    from torchlens.receptive_field._validation import check_geometric_metadata_invariants

    trace = tl.trace(_ComputedParent(op), (torch.randn(2, 5, 3), torch.randn(4, 3)))
    assert check_geometric_metadata_invariants(trace) is True


# ---------------------------------------------------------------------------
# Empty-box slices()
# ---------------------------------------------------------------------------


def test_empty_box_slices_select_nothing() -> None:
    """An empty RF box's slices() must select zero elements, never the whole input."""

    model = nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1, output_padding=2, bias=False)
    x = torch.randn(1, 1, 6, 6)
    trace = capture(model, x)
    conv = op_named(trace, "conv_transpose")
    last = int(conv.shape[-1]) - 1
    truth = true_receptive_support(model, x, (0, 0, last, last), deltas=(0.5, -0.5))
    assert truth == [], "the output_padding artifact unit must have no true support"
    box = conv.receptive_field.at((last, last))
    assert box.empty
    selected = x[box.slices()]
    assert selected.numel() == 0, f"empty box selected {tuple(selected.shape)}"
    # Pointwise batch/channel axes keep their same-index full-slice semantics.
    assert selected.shape[:2] == (1, 1)


def test_nonempty_box_slices_match_support_hull() -> None:
    """Non-empty boxes still slice exactly their clipped spatial hull."""

    model = nn.ConvTranspose2d(1, 1, 3, stride=3, padding=1, output_padding=2, bias=False)
    x = torch.randn(1, 1, 6, 6)
    trace = capture(model, x)
    conv = op_named(trace, "conv_transpose")
    box = conv.receptive_field.at((4, 4))
    assert not box.empty
    truth = true_receptive_support(model, x, (0, 0, 4, 4), deltas=(0.5, -0.5))
    selected = x[box.slices()]
    rows = hull(truth, 2)
    cols = hull(truth, 3)
    assert rows is not None and cols is not None
    assert selected.shape[-2:] == (rows[1] - rows[0], cols[1] - cols[0])


def test_non_antialiased_interpolate_regression() -> None:
    """The AA branch must not disturb ordinary interpolation geometry."""

    for mode, align in (("bilinear", False), ("bilinear", True), ("nearest", None)):
        kwargs = {"size": (3, 3), "mode": mode}
        if align is not None:
            kwargs["align_corners"] = align
        model = _Interp(**kwargs)
        x = torch.randn(1, 1, 7, 7, dtype=torch.float64)
        trace = capture(model, x)
        interp = op_named(trace, "interpolate")
        truth = true_receptive_support(model, x, (0, 0, 1, 1), deltas=(0.5, -0.5))
        box = interp.receptive_field.at((1, 1))
        assert_box_against_truth(box, truth, (2, 3), context=f"non-AA {mode} ac={align}")
