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


# ---------------------------------------------------------------------------
# Round 23: OVERLAPPING-BUT-UNEQUAL contexts (a shared module reused in both
# loops next to site-specific halves). r22 vetoed only DISJOINT contexts, so
# the shared element defeated the veto and the param-free bridges (add, tanh)
# over-merged into boundary-straddling 5-pass layers. The veto now fires on
# ANY context inequality; the sole honest unequal case -- the not-yet-saturated
# loop-entry call -- is re-admitted through a guarded adoption pair.
# ---------------------------------------------------------------------------


class _PartialOverlapLoops(nn.Module):
    """Shared linear reused in BOTH loops plus per-loop site-specific linears.

    ``tanh(shared(x) + enc(x))`` for three iterations, then
    ``tanh(shared(x) + dec(x))`` for two. ``shared`` is genuine ALBERT reuse
    (one 5-pass layer); ``enc``/``dec`` are distinct 3-/2-pass sites; the
    bridging ``add``/``tanh`` contexts are {shared, enc} vs {shared, dec} --
    OVERLAPPING on ``shared`` yet unequal, the exact quadrant r22 left open.
    """

    def __init__(self, chained: bool = True) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        self.chained = chained

    def forward(self, x):
        if self.chained:
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
        else:
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
                x = torch.tanh(self.shared(x) + self.dec(x))
        return x


def test_overlapping_unequal_contexts_respect_loop_boundary() -> None:
    """(b) Chained 3+2 loops with a shared reused flank: bridges split 3/2.

    The shared linear stays ONE 5-pass layer (locked ALBERT genuine-reuse
    policy) and enc/dec stay 3/2, so the param-free ``add`` and ``tanh``
    bridges must follow their differing parent topology (``shared+enc`` vs
    ``shared+dec``) and split 3/2 -- never a 5-pass layer neighboring a 3-pass
    and a 2-pass layer.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PartialOverlapLoops(chained=True), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5, 3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_overlapping_unequal_sites_within_one_loop_body() -> None:
    """(e) Both overlapping-unequal sites inside ONE 3-iteration body: 3/3.

    Two ``tanh(shared(x) + site(x))`` statements per iteration must yield two
    coherent 3-pass add layers and two 3-pass tanh layers, never 6-pass fusions,
    while ``shared`` stays one 6-pass reuse layer.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PartialOverlapLoops(chained=False), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [6, 3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_subset_context_across_structurally_unequal_bodies() -> None:
    """(b) Strict-subset contexts across visibly different bodies split 3/2.

    ``tanh(shared(x))`` for three iterations then ``tanh(shared(x) + dec(x))``
    for two: the first loop's tanh context {shared} is a strict SUBSET of the
    second's {shared, dec}, and the grouped calls do not even share parent
    topology (param parent vs add parent). The 3-member context class is a
    recurrent site of its own, so entry adoption must not apply.
    """

    class SubsetOverlap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(SubsetOverlap(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_multi_output_tied_cells_with_shared_reuse_bridge() -> None:
    """(d) LSTMCell slots stay honest AND the combining bridges split 3/2.

    A reused shared LSTMCell fires in both chained loops (two 5-pass slot
    layers), fully weight-tied DISTINCT enc/dec cells fire 3/2 (honest
    [5, 5, 3, 3, 2, 2] slots), and the ``tanh(h_shared + h_site)`` bridges must
    follow the site half: add/tanh 3/2, never 5-pass layers across the
    boundary.
    """

    class TiedLSTMOverlap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.LSTMCell(4, 4)
            self.enc = nn.LSTMCell(4, 4)
            self.dec = nn.LSTMCell(4, 4)
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                setattr(self.dec, name, getattr(self.enc, name))

        def forward(self, x):
            hs = torch.zeros(1, 4)
            cs = torch.zeros(1, 4)
            he = torch.zeros(1, 4)
            ce = torch.zeros(1, 4)
            for _ in range(3):
                hs, cs = self.shared(x, (hs, cs))
                he, ce = self.enc(x, (he, ce))
                x = torch.tanh(hs + he)
            hd = torch.zeros(1, 4)
            cd = torch.zeros(1, 4)
            for _ in range(2):
                hs, cs = self.shared(x, (hs, cs))
                hd, cd = self.dec(x, (hd, cd))
                x = torch.tanh(hs + hd)
            return x

    torch.manual_seed(0)
    traced = trace_fn(TiedLSTMOverlap(), torch.randn(1, 4))
    assert _pass_counts(traced, "lstmcell") == [5, 5, 3, 3, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_three_differing_parent_contexts_split_per_loop() -> None:
    """(g) A param-free op with THREE differing parent contexts splits 2/2/2.

    Three chained 2-iteration loops sharing one reused linear next to three
    site-specific linears: contexts {shared, a} / {shared, b} / {shared, c}
    are pairwise overlapping-but-unequal, and no pair may merge.
    """

    class ThreeContexts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.c = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.a(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.b(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.c(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(ThreeContexts(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [6, 2, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [2, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_untied_overlapping_control_stays_split() -> None:
    """(f) Untied control: distinct shared_a/shared_b flanks keep 3/2 splits."""

    class DisjointControl(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_a = nn.Linear(4, 4)
            self.shared_b = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared_a(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(self.shared_b(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(DisjointControl(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [3, 3, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 23 over-split guards: honest loop-entry saturation must keep merging.
# On pass 1 a recurrent op reads state produced OUTSIDE the loop, so its
# context is a strict subset of every later (saturated) pass. The unequal-
# context veto exempts exactly this shape through a guarded adoption pair;
# these tests are the ALBERT-style counterweight locking the exemption.
# ---------------------------------------------------------------------------


def test_residual_loop_entry_pass_stays_merged() -> None:
    """Top-level residual loop: the first add's unsaturated context still merges.

    ``h = emb(x); for: h = h + attn(h); h = h + mlp(h)`` gives the first add
    context {emb, attn} and every later add {emb, attn, mlp}: strictly-subset,
    singleton, and param-free-reachable -- the flagship honest entry. The adds
    must keep their pre-r23 single 6-pass layer, not shear off a 1-pass entry.
    """

    class ResidualEntry(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = h + self.attn(h)
                h = h + self.mlp(h)
            return h

    torch.manual_seed(0)
    traced = trace_fn(ResidualEntry(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [6], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_single_site_entry_saturation_stays_merged() -> None:
    """One-site loop with a param-free head: pass 1's subset context merges.

    ``h = pre(x); for: t = tanh(h); h = body(t) + t`` gives tanh pass 1 the
    context {pre} and later passes {pre, body} -- honest entry saturation on a
    genuine single-site 3-pass loop. tanh must stay ONE 3-pass layer.
    """

    class SaturatingTanh(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.pre(x)
            for _ in range(3):
                t = torch.tanh(h)
                h = self.body(t) + t
            return h

    torch.manual_seed(0)
    traced = trace_fn(SaturatingTanh(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 23 exemption attacks: each adoption guard is independently load-bearing.
# ---------------------------------------------------------------------------


def test_param_free_bridged_chained_loops_stay_split() -> None:
    """Singleton guard: a 3-member subset class never dissolves into loop 2.

    Loop 2's head reads loop 1's output PARAM-FREE (``tanh(x + dec(x))``), so
    the first loop's tanh contexts {shared, enc} are strict subsets of the
    second's {shared, enc, dec} AND a param-free path crosses the boundary --
    only the unique-carrier condition blocks adoption. Expect tanh 3/2, never
    a 5-pass bridge. (This shape was a FOURTH latent instance of the r22/r23
    defect class on unfixed main.)
    """

    class PfBridgedChained(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(x + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(PfBridgedChained(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_recalled_shared_identity_is_not_adopted() -> None:
    """Flow guard: subset-by-re-call (no param-free path) is a boundary.

    ``x = tanh(shared(x))`` once, then ``tanh(shared(x) + dec(x))`` twice: the
    prelude tanh is a singleton with context {shared} strictly inside
    {shared, dec}, but the containment comes from RE-CALLING ``shared`` behind
    a parametric cut, not from context flow -- no param-free path reaches the
    loop. It must stay a separate 1-pass layer (mirroring the established
    head-of-body precedent), never pass 1 of a 3-pass layer.
    """

    class OneShotThenLoop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(OneShotThenLoop(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [1, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_entry_singleton_cannot_bridge_two_vetoed_loops() -> None:
    """At-most-one guard: one prelude call never fuses two vetoed loops.

    ``x = tanh(shared(x))`` then two param-free-bridged 2-iteration loops
    (``tanh(x + enc(x))``, ``tanh(x + dec(x))``): the prelude's context {shared}
    is a strict subset of BOTH loop classes with param-free paths to both.
    Adoption may attach it to at most ONE loop; the two loops' tanh classes are
    mutually vetoed and must never merge into a single 5-pass layer.
    """

    class SingletonBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(x + self.enc(x))
            for _ in range(2):
                x = torch.tanh(x + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(SingletonBridge(), torch.randn(1, 4))
    tanh_counts = _pass_counts(traced, "tanh")
    assert tanh_counts in ([3, 2], [1, 2, 2]), _layer_passes(traced)
    assert _pass_counts(traced, "add") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_entry_adoption_never_fuses_parallel_vetoed_branches() -> None:
    """At-most-one guard, parallel form: adoption pairs to BOTH branches fuse.

    One prelude ``tanh(shared(x))`` param-free-feeds TWO PARALLEL pf-headed
    loops whose tanh context classes ({shared, enc} vs {shared, dec}) are
    mutually vetoed. If the entry may pair with every reachable superset
    candidate instead of exactly one, union-find fuses one call from EACH
    branch with the prelude into a false cross-branch "3-pass layer" of three
    unrelated calls ([3, 1, 1]), silently invariant-clean. Coherent outcomes
    keep the branches apart: the prelude stays a singleton ([1, 2, 2]) or
    joins exactly one branch ([3, 2]).
    """

    class ParallelBranches(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            t = torch.tanh(self.shared(x))
            a = t
            for _ in range(2):
                a = torch.tanh(a + self.enc(a))
            b = t
            for _ in range(2):
                b = torch.tanh(b + self.dec(b))
            return a + b

    torch.manual_seed(0)
    traced = trace_fn(ParallelBranches(), torch.randn(1, 4))
    tanh_counts = _pass_counts(traced, "tanh")
    assert tanh_counts in ([1, 2, 2], [3, 2]), _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_conditional_in_loop_alternating_arms_stay_per_arm() -> None:
    """Conditional-in-loop control: per-arm interiors keep per-arm pass counts."""

    class ConditionalInLoop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x):
            for i in range(4):
                if i % 2 == 0:
                    x = torch.tanh(self.a(x))
                else:
                    x = torch.tanh(self.b(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(ConditionalInLoop(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
