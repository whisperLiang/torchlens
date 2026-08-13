"""Planted-mutation proofs: every bucket red-capable AND relabel-green.

The five proofs the design-of-record names (section 6.2), plus full-field
planted sensitivity: (a1) within-leg identical-sibling binding swap -> red via
Check B while Check A alone stays green (the singleton-token case that made
one-comparator designs undecidable); (a2) co-location break of a multi-site
token -> red via Check A's partition; (b) pure bijective relabeling -> green
through BOTH checks; (c) deleted handle -> red on presence; (d) cross-lane
aliasing break -> red on partition.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from ._comparator import check_a, check_b
from ._models import scenario_by_name
from ._snapshot import Snapshot, TokenSite, run_scenario

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return run_scenario(
        scenario_by_name("recurrent_exhaustive"),
        tmp_path_factory.mktemp("clean"),
    )


def _event_position(events: Any, event: Any) -> int:
    """Adversarial-plant position finder over the raw op lane.

    The production ``CaptureEvents._event_position`` died with
    ``replace_op_event`` (P4); the plants keep mutating the raw list
    directly BY DESIGN — they model exactly the bypass the two-check
    comparator must convict.
    """

    return next(
        index
        for index, candidate in enumerate(events.op_events)
        if candidate.raw_index == event.raw_index
        and candidate.label_raw == event.label_raw
    )


def _clone_run(run: Any) -> Any:
    snapshot = Snapshot(scenario=run.snapshot.scenario)
    snapshot.journal = copy.deepcopy(run.snapshot.journal)
    snapshot.store = copy.deepcopy(run.snapshot.store)
    snapshot.artifact = copy.deepcopy(run.snapshot.artifact)
    snapshot.token_sites = list(run.snapshot.token_sites)
    snapshot.coherence = list(run.snapshot.coherence)
    snapshot.presence_only = list(run.snapshot.presence_only)
    return dataclasses.replace(run, snapshot=snapshot)


def _grad_fn_anchors(run: Any) -> list[str]:
    """Anchors carrying a singleton grad_fn token, structurally comparable."""

    return sorted(
        {
            site.anchor
            for site in run.snapshot.token_sites
            if site.bucket == "grad_fn_object_id" and site.layer == "journal"
        }
    )


def test_a1_within_leg_sibling_swap_red_via_check_b(tmp_path: Path) -> None:
    """A REAL post-commit journal mutation swapping two sibling grad_fn
    bindings is invisible to Check A but red via Check B."""

    scenario = scenario_by_name("recurrent_exhaustive")

    swapped_labels: list[str] = []

    def swap_two_siblings(events: Any) -> None:
        from torchlens.ir.op_record import OpRecord

        tanh_events = [e for e in events.op_events if e.layer_type == "tanh"]
        assert len(tanh_events) >= 2, "plant needs two structurally identical siblings"
        first, second = tanh_events[0], tanh_events[1]
        handle_first = events.grad_fn_handles_by_label_raw.get(first.label_raw)
        handle_second = events.grad_fn_handles_by_label_raw.get(second.label_raw)
        assert handle_first is not None and handle_second is not None
        # A TRUE singleton swap moves the binding at EVERY within-leg site
        # coherently, so the leg stays structurally self-consistent and only
        # ground truth can convict it. On the decomposed leg the handle index
        # is the ONE within-leg site (records never carry a handle — single
        # ownership); the legacy leg additionally mirrors the journal field.
        for event, handle in ((first, handle_second), (second, handle_first)):
            if not isinstance(event, OpRecord):
                updated = dataclasses.replace(event, grad_fn_handle=handle)
                object.__setattr__(updated, "seq", event.seq)
                position = _event_position(events, event)
                events.op_events[position] = updated
                events.live_index.replace(updated)
            events.grad_fn_handles_by_label_raw[event.label_raw] = handle
        swapped_labels.extend([first.label_raw, second.label_raw])

    clean = run_scenario(scenario, tmp_path / "clean")
    planted = run_scenario(scenario, tmp_path / "planted", journal_mutator=swap_two_siblings)
    assert swapped_labels, "the plant never fired (vacuous proof)"

    b_diffs = check_b(planted)
    assert any(d.kind == "grad_fn_binding" for d in b_diffs), (
        "Check B missed the within-leg sibling binding swap"
    )
    a_diffs = [
        d
        for d in check_a(clean.snapshot, planted.snapshot)
        if d.kind in {"token_partition", "token_site_presence"}
        and "grad_fn" in (d.path or "") + (d.detail or "")
    ]
    assert not a_diffs, (
        "expected the singleton swap to be structurally invisible to Check A; "
        f"got {a_diffs[:5]}"
    )


def test_a2_multi_site_colocation_break_red_via_check_a(clean_run: Any) -> None:
    """Retargeting ONE site of a multi-site token changes the partition."""

    mutated = _clone_run(clean_run)
    tokens: dict[str, list[TokenSite]] = {}
    for site in mutated.snapshot.token_sites:
        if site.bucket == "grad_fn_object_id":
            tokens.setdefault(site.token, []).append(site)
    multi = next((sites for sites in tokens.values() if len(sites) >= 2), None)
    assert multi is not None, "no multi-site grad_fn token found (journal+store join expected)"
    victim = multi[0]
    fresh_token = str(id(object()))
    mutated.snapshot.token_sites = [
        dataclasses.replace(site, token=fresh_token) if site is victim else site
        for site in mutated.snapshot.token_sites
    ]
    diffs = check_a(clean_run.snapshot, mutated.snapshot)
    assert any(d.kind == "token_partition" for d in diffs), "partition break not detected"


def test_b_bijective_relabel_green(clean_run: Any) -> None:
    """A pure bijective relabeling (snapshot AND ground truth, as one real
    relabeled run would produce) passes both checks."""

    mutated = _clone_run(clean_run)
    mapping: dict[str, str] = {}

    def relabel(token: str) -> str:
        if token not in mapping:
            mapping[token] = str(2_000_000_000 + len(mapping))
        return mapping[token]

    mutated.snapshot.token_sites = [
        dataclasses.replace(site, token=relabel(site.token))
        if site.bucket == "grad_fn_object_id"
        else site
        for site in mutated.snapshot.token_sites
    ]
    relabeled_attestation = copy.deepcopy(clean_run.attestation)
    relabeled_attestation.grad_fn_by_anchor = {
        anchor: (int(relabel(str(value))) if value is not None else None)
        for anchor, value in clean_run.attestation.grad_fn_by_anchor.items()
    }
    mutated = dataclasses.replace(mutated, attestation=relabeled_attestation)

    a_diffs = check_a(clean_run.snapshot, mutated.snapshot)
    assert not a_diffs, f"bijective relabel must be green (anti-flakiness): {a_diffs[:5]}"
    b_diffs = check_b(mutated)
    assert not b_diffs, f"relabeled leg vs its own ground truth must be green: {b_diffs[:5]}"


def test_c_deleted_handle_red(clean_run: Any) -> None:
    """Dropping one token site is a presence diff."""

    mutated = _clone_run(clean_run)
    grad_sites = [s for s in mutated.snapshot.token_sites if s.bucket == "grad_fn_object_id"]
    assert grad_sites
    mutated.snapshot.token_sites = [
        site for site in mutated.snapshot.token_sites if site is not grad_sites[0]
    ]
    diffs = check_a(clean_run.snapshot, mutated.snapshot)
    assert any(d.kind == "token_site_presence" for d in diffs), "deleted handle not detected"


def test_d_cross_lane_alias_break_red(clean_run: Any) -> None:
    """Same grad_fn token, one lane retargeted -> partition red (journal row
    keeps the token, the store row moves to another op's token)."""

    mutated = _clone_run(clean_run)
    grad_tokens = sorted(
        {s.token for s in mutated.snapshot.token_sites if s.bucket == "grad_fn_object_id"}
    )
    assert len(grad_tokens) >= 2, "need two grad_fn tokens for the retarget plant"
    donor, victim_token = grad_tokens[0], grad_tokens[1]
    retargeted = False
    new_sites = []
    for site in mutated.snapshot.token_sites:
        if (
            not retargeted
            and site.bucket == "grad_fn_object_id"
            and site.layer == "store"
            and site.token == victim_token
        ):
            new_sites.append(dataclasses.replace(site, token=donor))
            retargeted = True
        else:
            new_sites.append(site)
    assert retargeted, "no store-layer grad_fn site to retarget"
    mutated.snapshot.token_sites = new_sites
    diffs = check_a(clean_run.snapshot, mutated.snapshot)
    assert any(d.kind == "token_partition" for d in diffs), "cross-lane retarget not detected"


def test_barcode_binding_red_capable(tmp_path: Path) -> None:
    """A planted swap of two param barcodes in the journal is red via the
    per-anchor binding join (Check B, tl_barcode bucket)."""

    scenario = scenario_by_name("cnn_exhaustive")
    planted_rows: list[str] = []

    def swap_param_barcodes(events: Any) -> None:
        from torchlens.ir.op_record import OpRecord

        rows = [e for e in events.op_events if len(e.params) >= 1]
        if len(rows) < 2:
            return
        first, second = rows[0], rows[1]
        param_first, param_second = first.params[0], second.params[0]
        pairs = ((first, param_first, param_second), (second, param_second, param_first))
        for event, own, donor in pairs:
            swapped = dataclasses.replace(own, barcode=donor.barcode)
            swapped_params = (swapped, *event.params[1:])
            if isinstance(event, OpRecord):
                updated = dataclasses.replace(
                    event,
                    params_facet=dataclasses.replace(
                        event.params_facet, params=swapped_params
                    ),
                )
            else:
                updated = dataclasses.replace(event, params=swapped_params)
                object.__setattr__(updated, "seq", event.seq)
            events.op_events[_event_position(events, event)] = updated
            events.live_index.replace(updated)
        planted_rows.extend([first.label_raw, second.label_raw])

    planted = run_scenario(scenario, tmp_path, journal_mutator=swap_param_barcodes)
    assert planted_rows, "the plant never fired (vacuous proof)"
    diffs = check_b(planted)
    assert any(d.kind in {"barcode_binding", "equivalence_class_derivation"} for d in diffs), (
        f"barcode swap not detected: {diffs[:5]}"
    )


def test_handle_binding_red_capable(tmp_path: Path) -> None:
    """A planted backend_handle_id retarget is red against the live-tensor
    attestation recorded at the commit boundary."""

    scenario = scenario_by_name("cnn_reference")
    planted_labels: list[str] = []

    def retarget_handle(events: Any) -> None:
        from torchlens.ir.op_record import OpRecord

        rows = [
            e
            for e in events.op_events
            if e.kind == "op" and e.output.tensor.backend_handle_id
        ]
        assert rows, "plant needs a handle-carrying op row"
        event = rows[0]
        tensor_ref = dataclasses.replace(
            event.output.tensor, backend_handle_id=str(id(object()))
        )
        retargeted_output = dataclasses.replace(event.output, tensor=tensor_ref)
        if isinstance(event, OpRecord):
            updated = dataclasses.replace(
                event, core=dataclasses.replace(event.core, output=retargeted_output)
            )
        else:
            updated = dataclasses.replace(event, output=retargeted_output)
            object.__setattr__(updated, "seq", event.seq)
        events.op_events[_event_position(events, event)] = updated
        events.live_index.replace(updated)
        planted_labels.append(event.label_raw)

    planted = run_scenario(scenario, tmp_path, journal_mutator=retarget_handle)
    assert planted_labels, "the plant never fired (vacuous proof)"
    diffs = check_b(planted)
    assert any(d.kind == "handle_binding" for d in diffs), f"handle retarget missed: {diffs[:5]}"


def test_planted_field_sensitivity(clean_run: Any) -> None:
    """Every journal field path is red-capable at the cell level: mutating any
    op-event field in one leg's snapshot produces a Check A diff."""

    from torchlens.ir.events import OpEvent

    field_names = [f.name for f in dataclasses.fields(OpEvent)]
    anchor = next(iter(clean_run.snapshot.journal))
    missed: list[str] = []
    for name in field_names:
        mutated = _clone_run(clean_run)
        mutated.snapshot.journal[anchor][name] = {"__planted__": name}
        diffs = check_a(clean_run.snapshot, mutated.snapshot)
        if not any(d.path == name and d.anchor == anchor for d in diffs):
            missed.append(name)
    assert not missed, f"fields not red-capable: {missed}"
