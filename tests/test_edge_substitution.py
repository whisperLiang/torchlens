"""L6 stage 3: edge substitution — storage-fork honesty, tripwires, save boundary.

Pins: replay-engine-only scoping (typed refusals), edge-provenance gating,
the three-tier storage decision (tier-(ii) store + stamps; tier-(iii)
capture-surface parity), convenience-field coherence (producer truth vs
consumer view), node-level flag untouched, the 4.3 positive invariant +
re-execution acceptance (verdict ``edge_intervention_boundary``, never
"exempted"), the skip-shaped-acceptance meta-test, the v7 persistence
boundary (level-exhaustive refusal, two-conjunct key, precedence over
``artifact_save_level_unsupported``, switch-on round-trip with the
pre-release marker), ``trace.edges``, EDGE algebra rows, and the OBSERVE
masked-read kwarg.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens.selection import SelectionError, edge_address_of
from torchlens.validation.core import _check_edge_intervention_boundary


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


def _capture():
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(1, 1, 12, 12)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    return model, x, trace


@pytest.fixture(scope="module")
def capture():
    model, x, trace = _capture()
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _edge(trace, parent="relu_1_2"):
    return next(e for e in trace.edges if e.parent_label == parent)


def test_trace_edges_family_and_provenance_gate(capture):
    model, x, trace = capture
    labels = [(e.parent_label, e.child_label) for e in trace.edges]
    assert ("relu_1_2", "conv2d_2_3") in labels
    # identity-stable rows: same record objects across reads
    assert trace.edges[0] is trace.edges[0]

    torch.manual_seed(0)
    plain = tl.trace(_Net(), torch.randn(1, 1, 12, 12))
    with pytest.raises(SelectionError) as excinfo:
        plain.edges
    assert excinfo.value.fields["code"] == "edge_provenance_unavailable"


def test_edge_algebra_rows(capture):
    model, x, trace = capture
    a = _edge(trace).__selection__()
    b = _edge(trace, parent="conv2d_2_3").__selection__()
    union = (a | b).resolve(trace)
    assert union.kind == "EDGE" and len(union) == 2
    # complement within the trace's dataflow edge family (well-defined universe)
    complement = ~union
    assert len(complement) == len(trace.edges) - 2
    # ACT x EDGE refuses via the closed matrix
    with pytest.raises(SelectionError) as excinfo:
        a | tl.units("relu_1_2", [(0, 0, 0, 0)])
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_edge_substitution_storage_fork_honesty(capture):
    model, x, trace = capture
    edge = _edge(trace)
    fork = trace.fork()
    baseline_parent = trace["relu_1_2"].out.clone()
    baseline_child_args = [
        a.clone() if isinstance(a, torch.Tensor) else a
        for a in trace["conv2d_2_3"].ops[0].saved_args
    ]
    fork.do(edge.__selection__(), tl.zero_ablate())

    child = fork["conv2d_2_3"].ops[0]
    # only the child's consumption changed; producer truth intact
    assert torch.equal(fork["relu_1_2"].out, baseline_parent)
    assert torch.allclose(child.out, model.c2(torch.zeros_like(baseline_parent)))
    assert not torch.equal(fork["output_1"].out, trace["output_1"].out)

    # tier (ii): consumer view lives in the occurrence-granular store
    store_key = ("positional", (0,))
    assert store_key in child.edge_substitutions
    assert bool((child.edge_substitutions[store_key]["value"] == 0).all())
    assert child.edge_replacement_stamps[store_key]["verdict"] is True

    # tier (iii) capture-surface parity pin: no post-edit value reaches a
    # persisted capture field on the supported engine
    for saved, baseline in zip(child.saved_args, baseline_child_args, strict=True):
        if isinstance(saved, torch.Tensor):
            assert torch.equal(saved, baseline)
    parent_op = fork["relu_1_2"].ops[0]
    assert not (parent_op.out_versions_by_child or {})

    # per-edge intervened marker; node-level key correctly never fires
    edge_records = [r for r in child.interventions if r.edge_address]
    assert edge_records and edge_records[0].edge_address == edge_address_of(edge)
    assert not child.intervention_replaced

    # audit disclosure
    audit = fork.intervention_audit[-1]
    assert audit["kind"] == "EDGE" and audit["edges"][0]["parent"] == "relu_1_2"


def test_engine_scoping_refusals(capture):
    model, x, trace = capture
    selection = _edge(trace).__selection__()
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(selection, tl.zero_ablate(), model=model, x=x, engine="rerun")
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(selection, tl.zero_ablate(), engine="set_only")
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"
    # auto with model+x resolves to rerun -> same refusal
    with pytest.raises(SelectionError) as excinfo:
        trace.fork().do(selection, tl.zero_ablate(), model=model, x=x)
    assert excinfo.value.fields["code"] == "edge_intervention_engine_unsupported"


def _identity_edge_fork(trace):
    """Edge-substitute the edge with its OWN consumed value (identity)."""

    edge = _edge(trace)
    consumed = trace["relu_1_2"].out.clone()
    fork = trace.fork()
    fork.do(edge.__selection__(), consumed)
    return fork


def test_validation_boundary_accepts_corroborated_identity(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None
    assert verdict.decision == "edge_intervention_boundary"  # DISTINCT term, not "exempted"
    assert not verdict.failed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is True


def test_validation_positive_invariant_strip_fire_record(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("interventions", [r for r in child.interventions if not r.edge_address])
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fork.validate_forward_pass(model(x)) is False


def test_validation_positive_invariant_forge_without_stamp(capture):
    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("edge_replacement_stamps", {})
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_substitution_uncorroborated"


def test_validation_skip_shaped_acceptance_meta_test(capture):
    """A WRONG stored child output under a corroborated entry FAILS: the
    boundary is a DIFFERENT check, never NO check."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    child = fork["conv2d_2_3"].ops[0]
    child._internal_set("out", child.out + 5.0)
    verdict = _check_edge_intervention_boundary(fork, child)
    assert verdict is not None and verdict.failed
    assert verdict.reason == "edge_boundary_reexecution_mismatch"


def test_v7_persistence_boundary_level_exhaustive(capture, tmp_path):
    """GATED (switch inactive): every public level refuses with the SAME
    typed code, and the edge refusal PRECEDES artifact_save_level_unsupported
    where both apply (runnable on a bundle artifact)."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    for index, level in enumerate(("audit", "executable_with_callables", "portable", "runnable")):
        with pytest.raises(Exception) as excinfo:
            tl.save(fork, tmp_path / f"edge_{index}.tlspec", level=level)
        assert getattr(excinfo.value, "fields", {}).get("code") == (
            "edge_intervention_save_unsupported"
        ), level
    # non-edge saves are untouched
    tl.save(trace, tmp_path / "clean.tlspec", level="audit")


def test_v7_persistence_boundary_switch_on_round_trip(capture, tmp_path):
    """SWITCH-ON: the guard stands down BY THE STATED KEY (second conjunct
    false), the occurrence carriers persist, the write carries the
    pre-release marker, and the artifact refuses to load once the switch is
    off (test-fixture exercise of the post-bump shape, non-production)."""

    model, x, trace = capture
    fork = _identity_edge_fork(trace)
    path = tmp_path / "edge_switch.tlspec"
    with activate_prerelease_fields():
        tl.save(fork, path, level="portable")
        loaded = tl.load(path)
        child = loaded["conv2d_2_3"].ops[0]
        assert child.edge_substitutions and child.edge_replacement_stamps
    from torchlens._io import PreReleaseArtifactError

    with pytest.raises(PreReleaseArtifactError):
        tl.load(path)  # marker-bearing artifact refuses as a real v7 artifact


def test_tap_masked_values(capture):
    model, x, trace = capture
    selection = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)]).resolve(trace)
    observer = tl.tap(selection)
    fork = trace.fork()
    fork.attach_hooks("relu_1_2", observer, confirm_mutation=True)
    fork.push()
    full = observer.values()
    assert full and tuple(full[0].shape) == tuple(trace["relu_1_2"].out.shape)
    masked = observer.values(masked=True)
    assert masked[0].numel() == 2  # the two selected elements
    # stored mask immutability discipline: fresh copies each read
    assert observer.values(masked=True)[0] is not masked[0]
