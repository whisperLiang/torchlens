"""Hardening tests for param-free interiors at loop boundaries (round-22).

Locks the round-22 fix for the loop-core seal's MED residual: a bare param-free
functional interior (``torch.tanh``) between two chained loops must never merge
ACROSS the loop boundary just because the flanking PARAMETERIZED sites share
parameter barcodes. The disjoint-context veto in
``loop_grouping_adapter._merge_iso_groups_to_layers`` keys interiors by their
nearest parameterized ancestor's context, and that context must carry the same
SITE-QUALIFIED call identity (function, barcodes, output slot, module site,
non-tensor arg signature) that r21-loop-2 added to the parameterized union key.

Before the fix, ``param_contexts()`` accumulated raw parameter barcodes, so in
exactly the two flagship r21-loop-2 model classes -- tied DISTINCT modules, and
one kernel applied under different non-tensor args -- the interiors on both
sides of the boundary carried identical contexts, the veto never fired, and the
result was a 5-pass ``tanh`` layer bridging a 3-pass and a 2-pass parameterized
neighbor: the exact neighbor incoherence the boundary guard's own contract
forbids ("a 3-pass and a 2-pass layer cannot share a 5-pass neighbor").

Controls lock the boundary of the fix: untied chained loops keep splitting,
distinct anchored ``nn.Tanh`` modules keep splitting, and genuine same-module
reuse across chained loops (ALBERT policy) keeps merging 5/5 -- the fix must
not over-split legitimate recurrence.
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlens import trace as trace_fn
from torchlens.validation.invariants import check_metadata_invariants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_passes(traced) -> "OrderedDict[str, int]":
    """Return ordered mapping of layer_label -> number of grouped passes."""
    layers: "OrderedDict[str, list]" = OrderedDict()
    for op in traced:
        layers.setdefault(op.layer_label, []).append(op)
    return OrderedDict((label, len(ops)) for label, ops in layers.items())


def _pass_counts(traced, stem: str) -> list[int]:
    """Return the per-layer pass counts of every layer whose label starts with ``stem``.

    Ordered by first appearance in the trace, so a chained 3-iteration loop
    followed by a 2-iteration loop reads ``[3, 2]``.
    """
    return [count for label, count in _layer_passes(traced).items() if label.startswith(f"{stem}_")]


def _assert_coherent_two_loop_partition(traced, param_stem: str, interior_stem: str) -> None:
    """Assert the chained 3+2 loop pair splits into pass-coherent neighborhoods.

    The parameterized flanks must be [3, 2], and the param-free interior must
    mirror its own loop -- two layers of [3, 2] passes, NOT one 5-pass layer
    bridging the boundary.
    """
    assert _pass_counts(traced, param_stem) == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, interior_stem) == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Fixtures: chained 3-iteration + 2-iteration loops with param-free interiors
# ---------------------------------------------------------------------------


class _TiedTwoLoops(nn.Module):
    """Two DISTINCT weight-tied modules driving chained loops with tanh interiors."""

    def __init__(self, tied: bool = True) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        if tied:
            self.dec.weight = self.enc.weight
            self.dec.bias = self.enc.bias

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(self.enc(x))
        for _ in range(2):
            x = torch.tanh(self.dec(x))
        return x


class _SharedKernelTwoLoops(nn.Module):
    """ONE kernel applied under different non-tensor args in two chained loops.

    Both calls are shape-preserving on the input (3x3 kernel: padding=1, and
    padding=2 with dilation=2), so the tanh interiors of both loops share one
    structural equivalence key -- the merge is blocked only by the loop
    boundary, exactly the axis under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(F.conv2d(x, self.kernel, padding=1))
        for _ in range(2):
            x = torch.tanh(F.conv2d(x, self.kernel, padding=2, dilation=2))
        return x


class _SameModuleTwoLoops(nn.Module):
    """Genuine reuse control: ONE module drives BOTH chained loops (ALBERT)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        for _ in range(2):
            x = torch.tanh(self.cell(x))
        return x


class _DistinctTanhModuleLoops(nn.Module):
    """Anchored-interior control: tied linears with DISTINCT nn.Tanh interiors."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        self.dec.weight = self.enc.weight
        self.dec.bias = self.enc.bias
        self.tanh_a = nn.Tanh()
        self.tanh_b = nn.Tanh()

    def forward(self, x):
        for _ in range(3):
            x = self.tanh_a(self.enc(x))
        for _ in range(2):
            x = self.tanh_b(self.dec(x))
        return x


# ---------------------------------------------------------------------------
# Defect cases: interiors must not bridge the loop boundary
# ---------------------------------------------------------------------------


def test_tied_distinct_modules_interiors_respect_loop_boundary() -> None:
    """Tied enc/dec chained 3+2 loops: tanh interiors split 3/2, never 5.

    r21-loop-2 correctly splits the weight-tied parameterized flanks into a
    3-pass and a 2-pass layer, but the barcode-keyed ``param_contexts()`` left
    both loops' interiors with one indistinguishable context, so the bare
    ``torch.tanh`` merged across the boundary into a single 5-pass layer -- a
    5-pass neighbor sandwiched between a 3-pass and a 2-pass layer, which the
    boundary guard's own contract documents as forbidden.
    """
    torch.manual_seed(0)
    traced = trace_fn(_TiedTwoLoops(tied=True), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_shared_kernel_divergent_args_interiors_respect_loop_boundary() -> None:
    """Same kernel under different args in chained loops: interiors split 3/2.

    The second flagship r21-loop-2 class: one ``F.conv2d`` kernel applied with
    ``padding=1`` in loop 1 and ``padding=2, dilation=2`` in loop 2 shares
    parameter barcodes across the boundary, and the shape-preserving choice of
    args gives both loops' tanh interiors one equivalence key -- so only the
    site-qualified context keeps them apart.
    """
    torch.manual_seed(0)
    traced = trace_fn(_SharedKernelTwoLoops(), torch.randn(1, 1, 8, 8))
    _assert_coherent_two_loop_partition(traced, "conv2d", "tanh")


# ---------------------------------------------------------------------------
# Controls: the fix must not over-split legitimate recurrence
# ---------------------------------------------------------------------------


def test_untied_chained_loops_control() -> None:
    """Untied enc/dec chained loops already split 3/2 and must stay split."""
    torch.manual_seed(0)
    traced = trace_fn(_TiedTwoLoops(tied=False), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_distinct_tanh_modules_control() -> None:
    """Distinct anchored nn.Tanh interiors split 3/2 through module identity."""
    torch.manual_seed(0)
    traced = trace_fn(_DistinctTanhModuleLoops(), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_same_module_reuse_still_merges_across_chained_loops() -> None:
    """ALBERT policy: ONE module reused in both loops keeps merging 5/5.

    A genuinely reused module keeps ONE site-qualified identity across both
    loops, so its interiors share a context and the disjoint-context veto must
    NOT fire: the linear groups as one 5-pass layer and the tanh interior as
    one coherent 5-pass neighbor. This is the boundary of the fix -- contexts
    must become disjoint only for genuinely-distinct semantic sites.
    """
    torch.manual_seed(0)
    traced = trace_fn(_SameModuleTwoLoops(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [5], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Within-body boundary: two tied sites inside ONE loop body
# ---------------------------------------------------------------------------


def test_tied_sites_within_one_loop_body_keep_per_site_interiors() -> None:
    """Tied enc/dec called inside ONE 3-iteration body: two 3-pass tanh layers.

    Each iteration runs ``tanh(enc(x))`` then ``tanh(dec(x))`` with tied
    weights. The two tanh SITES sit after genuinely-distinct parameterized
    sites, so they must stay two coherent 3-pass layers (matching their own
    flanks), not fuse into one 6-pass layer through the shared barcodes.
    """

    class TiedSitesOneBody(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)
            self.dec.weight = self.enc.weight
            self.dec.bias = self.enc.bias

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.enc(x))
                x = torch.tanh(self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(TiedSitesOneBody(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
