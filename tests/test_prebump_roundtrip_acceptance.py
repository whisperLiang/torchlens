"""Pre-bump switched-on round-trip ACCEPTANCE pass over every S3-gated family.

All new persistence this sprint rides the S3 test-only activation switch
(:mod:`torchlens._io.prerelease`), so the first real exercise of the shipped
physical schema would otherwise be deferred to the wave-3 coordinated bump --
the sprint's most congested moment. This module pays that debt early: for
EVERY registered DROP-gated family (enumerated from the registrar itself,
never from memory) it proves, with REAL family data:

1. the switched save -> load round trip preserves the persisted surface
   exactly (bitwise for tensors, exact equality for scalar/dict payloads);
2. a second-generation round trip is stable (no settle oscillation);
3. the pre-release marker rides the switch-on write and the artifact REFUSES
   to load as a real v7 artifact once the switch is off; and
4. the ratified S2 marker-combination table's reachable cells behave: every
   TYPED REFUSE cell refuses, every LEGAL cell works.

The FORGERY_SURFACE_LEDGER below is the tamper census (brief question 2):
families whose load-time validation exists today are proven to refuse, and
families whose validation is deferred to the coordinated bump are pinned as
such -- flipping one to validated at the bump FAILS the pin until the ledger
row is updated, so no field can slip into the bump unvalidated silently.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torchlens as tl
from torchlens._io import FieldPolicy, PreReleaseArtifactError
from torchlens._io.prerelease import (
    PRERELEASE_STATE_KEY,
    activate_prerelease_fields,
    persisted_policy_override,
    registered_prerelease_fields,
)
from torchlens._io.scrub import scrub_for_save

# ---------------------------------------------------------------------------
# Registry-driven completeness ledger (brief: enumerate from the registry,
# not from memory). Every registered owner/field must be claimed by exactly
# the acceptance rows below; a new lane registration fails this file until an
# acceptance row lands for it.
# ---------------------------------------------------------------------------

#: family id -> claimed registrar rows ("Owner.field" or "Trace.annotations.key").
ACCEPTANCE_COVERAGE: dict[str, frozenset[str]] = {
    "l1-grouping": frozenset({"Trace.grouping", "Trace.grouping_policy"}),
    "l1-site-key": frozenset({"Op.site_key"}),
    "l2-episode": frozenset({"Trace.annotations.episode"}),
    "l3-aten-profile": frozenset(
        {"Trace._primitive_op_profile"}
        | {
            f"AtenOp.{name}"
            for name in (
                "algorithmic_flops",
                "autocast_context",
                "backward_epoch_index",
                "capture_phase",
                "decomposition_slot",
                "dispatch_key_context",
                "exception_type",
                "execution_context",
                "flop_formula_source",
                "flop_formula_version",
                "flop_status",
                "forward_pass_index",
                "grad_fn_link_provenance",
                "grad_fn_link_status",
                "grad_fn_ref",
                "input_tensor_facts",
                "label",
                "module_call_stack",
                "mutation_kind",
                "namespace",
                "operator",
                "outcome",
                "output_tensor_facts",
                "overload",
                "owner_func_call_id",
                "owner_status",
                "parent_grad_fn_call_ref",
                "parent_op_refs",
                "schema",
                "schema_fingerprint",
                "sequence",
                "view_copy_kind",
            )
        }
        | {f"OpRef.{name}" for name in ("func_call_id", "op_label", "op_row_index")}
        | {
            f"_AtenExecutionContext.{name}"
            for name in (
                "autocast",
                "backend",
                "compile_stance",
                "completeness_witness_mode",
                "deterministic_algorithms",
                "device_capability",
                "device_model",
                "grad_mode",
                "inference_mode",
                "module_training_summary",
                "owner_thread_coverage",
                "pytorch_version",
                "sdpa_policy",
                "tf32_matmul_policy",
            )
        }
        | {
            f"_AtenTensorFact.{name}"
            for name in (
                "container_path",
                "device",
                "dtype",
                "layout",
                "logical_version",
                "requires_grad",
                "shape",
                "storage_alias_group",
                "stride",
                "tensor_impl_capability",
            )
        }
        | {
            f"_ModePausedInteriorGap.{name}"
            for name in (
                "capture_phase",
                "kind",
                "owner_func_call_id",
                "parent_op_refs",
                "reason",
                "sequence_after",
                "sequence_before",
            )
        }
        | {
            f"_PrimitiveOpProfile.{name}"
            for name in (
                "_event_owner_evidence",
                "aten_event_watermark",
                "mode_paused_interior",
                "primitive_ops",
            )
        }
    ),
    "l3-kernel-telemetry": frozenset(
        {"Trace.annotations._kernel_telemetry"}
        | {
            f"KernelLaunch.{name}"
            for name in (
                "attribution_status",
                "device",
                "duration",
                "launch_name",
                "runtime_correlation",
                "stream",
            )
        }
        | {f"_TelemetryPayload.{name}" for name in ("_available", "_launches", "_relations")}
    ),
    "l6-selection": frozenset({"HelperSpec.selection_recipe", "Trace.intervention_audit"}),
    "l6-edges": frozenset(
        {"Op.edge_substitutions", "Op.edge_replacement_stamps", "FireRecord.edge_address"}
    ),
    "l7a-structure-only": frozenset({"Trace.structure_only"}),
    "l8-distributed-scope": frozenset({"Trace.distributed_scope"}),
    "l9-timing-provenance": frozenset({"Trace.grad_fn_timing_provenance"}),
    "l9-checkpoint-witness": frozenset({"Trace.checkpoint_invocation_witness"}),
}


@pytest.mark.smoke
def test_acceptance_coverage_matches_registrar_inventory() -> None:
    """Every registered gated row is claimed by exactly one acceptance family.

    A lane that registers a new pre-release field/annotations key must add an
    acceptance row (round trip + tamper + marker) to this module in the same
    change; this assertion is the forcing function.
    """

    # Registrations land at their owners' import time; import every lane
    # module explicitly so the inventory is complete regardless of test order
    # (same discipline as tests/test_prerelease_registrar.py).
    import torchlens.data_classes.aten_op  # noqa: F401
    import torchlens.intervention.types  # noqa: F401
    import torchlens.kernel_telemetry  # noqa: F401

    inventory: set[str] = set()
    for owner, fields in registered_prerelease_fields().items():
        for name in fields:
            if owner == "Trace.annotations":
                inventory.add(f"Trace.annotations.{name}")
            else:
                inventory.add(f"{owner}.{name}")
    claimed_lists = [row for rows in ACCEPTANCE_COVERAGE.values() for row in rows]
    claimed = set(claimed_lists)
    assert len(claimed_lists) == len(claimed), "a registrar row is claimed twice"
    assert claimed == inventory, (
        "acceptance coverage and registrar inventory diverged.\n"
        f"registered but unclaimed: {sorted(inventory - claimed)}\n"
        f"claimed but unregistered: {sorted(claimed - inventory)}"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _persisted_eq(a: object, b: object) -> bool:
    """Deep equality over the PERSISTED surface: bitwise tensors, exact scalars.

    Objects declaring ``PORTABLE_STATE_SPEC`` compare field-wise over the
    fields that persist under the active switch (declared non-DROP or
    registrar-overridden); session-time DROP fields are excluded.
    """

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    spec = getattr(type(a), "PORTABLE_STATE_SPEC", None)
    if isinstance(spec, dict) and type(a) is type(b):
        for name, policy in spec.items():
            if policy is FieldPolicy.DROP and persisted_policy_override(type(a), name) is None:
                continue
            if not _persisted_eq(getattr(a, name, None), getattr(b, name, None)):
                return False
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_persisted_eq(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_persisted_eq(x, y) for x, y in zip(a, b))
    return bool(a == b)


def _switched_roundtrip(trace: tl.Trace, tmp_path, name: str) -> tl.Trace:
    """Save under the switch, prove the marker and the off-switch refusal, load.

    Asserts (brief questions 1 and 3): the switch-on scrub state carries the
    marker payload; the artifact loads under the switch; and the SAME artifact
    refuses typed as a real v7 artifact once the switch is off.
    """

    path = tmp_path / f"{name}.tlspec"
    with activate_prerelease_fields():
        state, _, _ = scrub_for_save(trace)
        assert state[PRERELEASE_STATE_KEY]["marker"], "switch-on write must carry the marker"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.save(trace, str(path))
            loaded = tl.load(str(path))
    with pytest.raises(PreReleaseArtifactError):
        tl.load(str(path))
    return loaded


def _second_generation(loaded: tl.Trace, tmp_path, name: str) -> tl.Trace:
    """Re-save the LOADED trace under the switch and load again (stability)."""

    path = tmp_path / f"{name}_gen2.tlspec"
    with activate_prerelease_fields():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.save(loaded, str(path))
            return tl.load(str(path))


def _tiny_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))


class _TwoConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


def _selection_fork() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _TwoConv(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    fork = trace.fork()
    fork.do(tl.units("relu_1_2", [(0, 0, 1, 1)]), tl.zero_ablate())
    return fork


class _EdgeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))) + 1.0)


def _edge_fork() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _EdgeNet(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    edge = next(e for e in trace.edges if e.parent_label == "relu_1_2")
    fork = trace.fork()
    fork.do(edge.__selection__(), trace["relu_1_2"].out.clone())
    return fork


class _BackwardTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l(x))


def _backward_trace() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _BackwardTiny(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    trace.log_backward(trace.output_ops[0].out.sum())
    return trace


class _OneCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.b, torch.relu(self.a(x)), use_reentrant=False)


def _checkpoint_trace() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _OneCheckpoint(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    trace.log_backward(trace.output_ops[0].out.sum())
    return trace


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, width: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class _GreedyRunner(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            logits = self.model(current)
            nxt = logits.argmax(-1, keepdim=True)
            tokens.append(nxt)
            current = torch.cat([current, nxt], dim=1)
        return torch.cat(tokens, dim=1)


def _episode_trace(n_steps: int = 2) -> tl.Trace:
    torch.manual_seed(0)
    lm = _TinyLM()
    runner = _GreedyRunner(lm, n_steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(
            runner,
            torch.zeros(1, 3, dtype=torch.long),
            episode=tl.options.EpisodeSpec(stepped_module=lm, n_steps=n_steps),
        )


def _structure_only_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
        torch.randn(1, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )


def _armed_aten_trace() -> tl.Trace:
    from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests

    torch.manual_seed(0)
    with _activate_aten_recording_for_tests():
        return tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(1, 4))


def _telemetry_trace() -> tl.Trace:
    from torchlens import kernel_telemetry as telemetry

    torch.manual_seed(0)
    return telemetry._profile_trace_with_cuda_kernels(
        lambda: tl.trace(nn.Linear(2, 2), torch.ones(2, 2))
    )


# ---------------------------------------------------------------------------
# Per-family switched round trips (brief questions 1 + 3)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_l1_grouping_family_roundtrip(tmp_path) -> None:
    trace = _tiny_trace()
    loaded = _switched_roundtrip(trace, tmp_path, "grouping")
    assert loaded.grouping == trace.grouping == "structural"
    assert loaded.grouping_policy == trace.grouping_policy
    gen2 = _second_generation(loaded, tmp_path, "grouping")
    assert gen2.grouping_policy == loaded.grouping_policy


@pytest.mark.smoke
def test_l1_site_key_family_roundtrip(tmp_path) -> None:
    trace = _tiny_trace()
    live_keys = [op.site_key for op in trace.ops]
    assert any(key and key.startswith("s1|") for key in live_keys)
    loaded = _switched_roundtrip(trace, tmp_path, "site_key")
    assert [op.site_key for op in loaded.ops] == live_keys
    gen2 = _second_generation(loaded, tmp_path, "site_key")
    assert [op.site_key for op in gen2.ops] == live_keys


def test_l2_episode_family_roundtrip(tmp_path) -> None:
    trace = _episode_trace()
    assert trace.annotations["episode"]["header"]["capture_kind"] == "episode"
    loaded = _switched_roundtrip(trace, tmp_path, "episode")
    assert loaded.annotations["episode"] == trace.annotations["episode"]
    gen2 = _second_generation(loaded, tmp_path, "episode")
    assert gen2.annotations["episode"] == trace.annotations["episode"]


def test_l3_aten_profile_family_roundtrip(tmp_path) -> None:
    trace = _armed_aten_trace()
    live_profile = trace._primitive_op_profile
    assert live_profile is not None and live_profile.primitive_ops
    loaded = _switched_roundtrip(trace, tmp_path, "aten")
    assert _persisted_eq(loaded._primitive_op_profile, live_profile)
    gen2 = _second_generation(loaded, tmp_path, "aten")
    assert _persisted_eq(gen2._primitive_op_profile, live_profile)


def test_l3_kernel_telemetry_family_roundtrip(tmp_path) -> None:
    from torchlens import kernel_telemetry as telemetry

    trace = _telemetry_trace()
    live_payload = trace.annotations["_kernel_telemetry"]
    loaded = _switched_roundtrip(trace, tmp_path, "telemetry")
    assert _persisted_eq(loaded.annotations["_kernel_telemetry"], live_payload)
    telemetry._bind_trace_telemetry(loaded)
    assert loaded.ops[0].gpu_kernels[0].attribution_status in (
        "unavailable",
        "exact",
        "ambiguous",
    )
    gen2 = _second_generation(loaded, tmp_path, "telemetry")
    assert _persisted_eq(gen2.annotations["_kernel_telemetry"], live_payload)


def test_l6_selection_family_roundtrip(tmp_path) -> None:
    fork = _selection_fork()
    live_audit = fork.intervention_audit
    live_recipe = next(
        record.helper.selection_recipe
        for record in fork["relu_1_2"].ops[0].interventions
        if record.helper is not None
    )
    assert live_recipe["resolve_digest"]
    loaded = _switched_roundtrip(fork, tmp_path, "selection")
    assert loaded.intervention_audit == live_audit
    loaded_recipe = next(
        record.helper.selection_recipe
        for record in loaded["relu_1_2"].ops[0].interventions
        if record.helper is not None
    )
    assert loaded_recipe == live_recipe
    gen2 = _second_generation(loaded, tmp_path, "selection")
    assert gen2.intervention_audit == live_audit


def test_l6_edge_family_roundtrip(tmp_path) -> None:
    fork = _edge_fork()
    child = fork["conv2d_2_3"].ops[0]
    store_key = ("positional", (0,))
    assert store_key in child.edge_substitutions
    live_value = child.edge_substitutions[store_key]["value"]
    live_stamps = child.edge_replacement_stamps
    live_edge_address = next(
        record.edge_address for record in child.interventions if record.edge_address
    )
    loaded = _switched_roundtrip(fork, tmp_path, "edges")
    loaded_child = loaded["conv2d_2_3"].ops[0]
    assert torch.equal(loaded_child.edge_substitutions[store_key]["value"], live_value)
    assert _persisted_eq(loaded_child.edge_replacement_stamps, live_stamps)
    assert (
        next(record.edge_address for record in loaded_child.interventions if record.edge_address)
        == live_edge_address
    )
    gen2 = _second_generation(loaded, tmp_path, "edges")
    gen2_child = gen2["conv2d_2_3"].ops[0]
    assert torch.equal(gen2_child.edge_substitutions[store_key]["value"], live_value)


def test_l7a_structure_only_family_roundtrip(tmp_path) -> None:
    trace = _structure_only_trace()
    assert trace.structure_only is True
    loaded = _switched_roundtrip(trace, tmp_path, "structure_only")
    assert loaded.structure_only is True
    gen2 = _second_generation(loaded, tmp_path, "structure_only")
    assert gen2.structure_only is True


@pytest.mark.smoke
def test_l8_distributed_scope_family_roundtrip(tmp_path) -> None:
    from torchlens.distributed import _lifecycle as lifecycle
    from torchlens.distributed._dtensor import RANK_LOCAL_SHARD

    lifecycle.disarm()
    trace = _tiny_trace()
    trace.distributed_scope = RANK_LOCAL_SHARD
    loaded = _switched_roundtrip(trace, tmp_path, "distributed_scope")
    assert loaded.distributed_scope == RANK_LOCAL_SHARD
    gen2 = _second_generation(loaded, tmp_path, "distributed_scope")
    assert gen2.distributed_scope == RANK_LOCAL_SHARD


def test_l9_timing_provenance_family_roundtrip(tmp_path) -> None:
    trace = _backward_trace()
    assert trace.grad_fn_timing_provenance == "perf_counter"
    loaded = _switched_roundtrip(trace, tmp_path, "timing")
    assert loaded.grad_fn_timing_provenance == "perf_counter"
    gen2 = _second_generation(loaded, tmp_path, "timing")
    assert gen2.grad_fn_timing_provenance == "perf_counter"


def test_l9_checkpoint_witness_family_roundtrip(tmp_path) -> None:
    trace = _checkpoint_trace()
    live_witness = trace.checkpoint_invocation_witness
    assert live_witness["token_count"] == 1
    loaded = _switched_roundtrip(trace, tmp_path, "checkpoint")
    assert loaded.checkpoint_invocation_witness == live_witness
    gen2 = _second_generation(loaded, tmp_path, "checkpoint")
    assert gen2.checkpoint_invocation_witness == live_witness


# ---------------------------------------------------------------------------
# Tamper census (brief question 2). Two classes:
#
# VALIDATED-NOW families refuse (or degrade fail-closed, typed) at load; their
# refusals are pinned by the owning lane suites and re-proven here where cheap.
#
# DEFERRED-TO-BUMP families persist tampered values verbatim today: their
# load-validation rows land at the coordinated bump (registrar module
# docstring), and pre-bump the marker refusal keeps every switched artifact
# out of circulation, so the forgery surface is test-scope only. The ledger
# pins the CURRENT truth loudly: when the bump adds a family's validation,
# its pin here fails until the row moves to the validated class -- no family
# can reach the bump with its validation silently forgotten.
# ---------------------------------------------------------------------------

#: field row -> validation status TODAY. Every DEFERRED row is a bump-time
#: obligation (escalated in results/prebump-acceptance.md).
FORGERY_SURFACE_LEDGER: dict[str, str] = {
    "Trace.grouping_policy": "validated_degrade_typed",  # C1-C8, test_grouping_stamp
    "Trace.annotations.episode": "validated_refuse_or_quarantine",  # test_episode_capture
    "Trace._primitive_op_profile": "validated_fk_refusal",  # test_aten_profile
    "Op.edge_substitutions": "validated_at_validation_time",  # boundary check, test_edge_substitution
    "Op.site_key": "DEFERRED_TO_BUMP",
    "Trace.distributed_scope": "DEFERRED_TO_BUMP",
    "Trace.grad_fn_timing_provenance": "DEFERRED_TO_BUMP",
    "Trace.checkpoint_invocation_witness": "DEFERRED_TO_BUMP",
    "Trace.intervention_audit": "DEFERRED_TO_BUMP",
    "Trace.annotations._kernel_telemetry": "DEFERRED_TO_BUMP",
}


def _tampered_loads_verbatim(trace: tl.Trace, tmp_path, name: str, reader, planted) -> None:
    """Pin: the tamper persists verbatim today AND stays marker-contained."""

    path = tmp_path / f"{name}_tampered.tlspec"
    with activate_prerelease_fields():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.save(trace, str(path))
            loaded = tl.load(str(path))
    assert reader(loaded) == planted, (
        f"{name}: load-time validation now fires for this family -- move its "
        "FORGERY_SURFACE_LEDGER row to the validated class and add the typed-"
        "refusal assertion here."
    )
    # Containment: the tampered artifact still refuses as a real v7 artifact.
    with pytest.raises(PreReleaseArtifactError):
        tl.load(str(path))


@pytest.mark.smoke
def test_tamper_site_key_deferred_to_bump(tmp_path) -> None:
    assert FORGERY_SURFACE_LEDGER["Op.site_key"] == "DEFERRED_TO_BUMP"
    trace = _tiny_trace()
    target = trace.ops[1]
    target._internal_set("site_key", "forged-not-a-site-key")
    _tampered_loads_verbatim(
        trace, tmp_path, "site_key", lambda t: t.ops[1].site_key, "forged-not-a-site-key"
    )


@pytest.mark.smoke
def test_tamper_distributed_scope_deferred_to_bump(tmp_path) -> None:
    from torchlens.distributed import _lifecycle as lifecycle

    assert FORGERY_SURFACE_LEDGER["Trace.distributed_scope"] == "DEFERRED_TO_BUMP"
    lifecycle.disarm()
    trace = _tiny_trace()
    trace.distributed_scope = "forged_scope_value"
    _tampered_loads_verbatim(
        trace, tmp_path, "distributed_scope", lambda t: t.distributed_scope, "forged_scope_value"
    )


def test_tamper_timing_provenance_deferred_to_bump(tmp_path) -> None:
    assert FORGERY_SURFACE_LEDGER["Trace.grad_fn_timing_provenance"] == "DEFERRED_TO_BUMP"
    trace = _backward_trace()
    object.__setattr__(trace, "grad_fn_timing_provenance", "forged_clock_source")
    _tampered_loads_verbatim(
        trace, tmp_path, "timing", lambda t: t.grad_fn_timing_provenance, "forged_clock_source"
    )


def test_tamper_checkpoint_witness_deferred_to_bump(tmp_path) -> None:
    assert FORGERY_SURFACE_LEDGER["Trace.checkpoint_invocation_witness"] == "DEFERRED_TO_BUMP"
    trace = _checkpoint_trace()
    witness = dict(trace.checkpoint_invocation_witness)
    witness["token_count"] = "forty-two"  # type-forged: count is not even an int
    object.__setattr__(trace, "checkpoint_invocation_witness", witness)
    _tampered_loads_verbatim(
        trace,
        tmp_path,
        "checkpoint",
        lambda t: t.checkpoint_invocation_witness["token_count"],
        "forty-two",
    )


def test_tamper_intervention_audit_deferred_to_bump(tmp_path) -> None:
    assert FORGERY_SURFACE_LEDGER["Trace.intervention_audit"] == "DEFERRED_TO_BUMP"
    fork = _selection_fork()
    audit = [dict(record) for record in fork.intervention_audit]
    audit[-1]["resolve_digest"] = "f" * 64
    object.__setattr__(fork, "intervention_audit", audit)
    _tampered_loads_verbatim(
        fork, tmp_path, "audit", lambda t: t.intervention_audit[-1]["resolve_digest"], "f" * 64
    )


def test_tamper_kernel_telemetry_deferred_to_bump(tmp_path) -> None:
    assert FORGERY_SURFACE_LEDGER["Trace.annotations._kernel_telemetry"] == "DEFERRED_TO_BUMP"
    trace = _telemetry_trace()
    payload = trace.annotations["_kernel_telemetry"]
    tampered = dict(payload)
    tampered["_relations"] = {"forged_marker": [999999]}
    trace.annotations["_kernel_telemetry"] = tampered
    _tampered_loads_verbatim(
        trace,
        tmp_path,
        "telemetry",
        lambda t: t.annotations["_kernel_telemetry"]["_relations"],
        {"forged_marker": [999999]},
    )


@pytest.mark.smoke
def test_tamper_grouping_policy_validated_degrades_typed(tmp_path) -> None:
    """VALIDATED-NOW representative: the grouping stamp tamper degrades typed
    (fail-closed settlement), re-proven here so the ledger's validated class
    has an in-module witness alongside the owning lane suite."""

    from torchlens.errors import TorchLensWarning
    from torchlens.postprocess._grouping_stamp import degraded_grouping_policy_stamp

    assert FORGERY_SURFACE_LEDGER["Trace.grouping_policy"].startswith("validated")
    trace = _tiny_trace()
    trace.grouping_policy = {**trace.grouping_policy, "policy": "greedy"}
    path = tmp_path / "grouping_tampered.tlspec"
    with activate_prerelease_fields():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.save(trace, str(path))
        with pytest.warns(TorchLensWarning, match="grouping_policy stamp is invalid"):
            loaded = tl.load(str(path))
    assert loaded.grouping_policy == degraded_grouping_policy_stamp("vocabulary")


# ---------------------------------------------------------------------------
# S2 marker-combination table (brief question 4): every reachable TYPED
# REFUSE cell refuses, every reachable LEGAL cell works. The ledger-only
# degrade row (fold arms 2/4 -> episode_unknown) is pinned by the owning
# lane in tests/test_bundle_relations.py; the wave-3 P1 totality test owns
# the full product.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_combination_plain_structure_only_truncation_refuses() -> None:
    """Cell (plain, structure_only=True, truncation present) = TYPED REFUSE.

    A structure-only capture executes no values to truncate; the refusal
    fires at the L7a value-consumer chokepoint before any truncation term.
    """

    trace = _structure_only_trace()
    with pytest.raises(Exception) as excinfo:
        trace.run(inputs=torch.randn(1, 4), until="linear_1_1")
    assert excinfo.value.fields["code"] == "structure_only_replay_unsupported"


def test_combination_episode_structure_only_refuses() -> None:
    """Cell (episode, structure_only=True, *, *) = TYPED REFUSE at entry."""

    lm = _TinyLM()
    runner = _GreedyRunner(lm, 2)
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            runner,
            torch.zeros(1, 3, dtype=torch.long),
            episode=tl.options.EpisodeSpec(stepped_module=lm, n_steps=2),
            capture=tl.options.CaptureOptions(structure_only=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_episode_unsupported"


@pytest.mark.smoke
def test_combination_plain_truncation_legal_run_result_term_only() -> None:
    """Cell (plain, structure_only=False, truncation present) = LEGAL.

    Truncation is a RUN-RESULT term only: the report discloses it and the
    SOURCE capture outcome stays COMPLETE (never a CaptureStatus).
    """

    from torchlens.capture.outcome import CaptureStatus

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 4))
    result = trace.run(inputs=torch.randn(1, 4), until="linear_1_1")
    assert result.report.truncated is True
    assert result.report.truncation.stopped_at == "linear_1_1"
    assert result.output is None  # disclosed, never a fabricated tail
    assert trace.outcome.status is CaptureStatus.COMPLETE
    del model  # keep-alive: the live run needs the strong model reference


def test_combination_episode_core_legal() -> None:
    """Cell (episode, structure_only=False, absent, exact) = LEGAL."""

    trace = _episode_trace()
    header = trace.annotations["episode"]["header"]
    assert header["capture_kind"] == "episode"
    assert header["structure_only"] is False
    rows = trace.annotations["episode"]["rows"]
    assert all(row["status"] == "complete" for row in rows)


@pytest.mark.smoke
def test_combination_plain_structure_only_absent_legal() -> None:
    """Cell (plain, structure_only=True, truncation absent) = LEGAL (L7a core)."""

    trace = _structure_only_trace()
    assert trace.structure_only is True
    assert trace.ops  # structure recorded, values hypothesized
